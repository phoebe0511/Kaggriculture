"""用 ladder 頂端玩家的棋譜訓練 policy / value。

    python -m model.train --data data/dataset --epochs 8

## v5：unit 那一半改成逐格需求

    demand  哪一格要做什麼（11 個 op × 100 格的 multi-label）  ← 主要訊號
    op      到了目標格要做什麼（44 選 1）
    qty     那個動作的數量（12 選 1）
    target  這一段要走去哪一格（100 選 1）  ← v4 的，仍然訓但不再計分
    market  present [21] + qty [21, 18]
    value   期末現金 / 100k

派誰去哪一格由 `agents/gen0._minimum_cost_assignment` 算，走路用
`agents/gen0.step_toward`，兩個都不經過網路。

`demand` 的 loss **只在合法的格子上算**（`contracts.legal_demand_mask`）——
1,100 個格子裡九成多是「沒種東西的地不能澆水」這種白送的負例。

⚠️ demand 標籤只有 `harness.rollout` 產得出來（它讀規劃器的任務清單）。
老師的 replay 沒有任務清單可讀，所以 `data/dataset/` 訓不了 v5。

## ⚠️ 準確率不是 kill switch

v2 拿到 val op 0.9396（dummy 0.1609）、逐類別召回率 0.98 以上，實戰 0 勝 12 負
（journal 2026-08-19 §7d）。真正的驗收是自己打一局比對動作分布：

    python -m eval.runner --a gen2_model --b gen1 --games 40 --workers 16
    python -m tools.action_dist temp/<run 目錄>

## 下面這條仍然有效：贏不過 dummy 就是編碼漏資訊

`workflow.md` §5 排第二的失敗模式是「編碼器漏掉關鍵資訊 → 網路學不到它看不見
的東西，而且**不會報錯**，只會表現成怎麼訓練都不夠強」。

所以每個 epoch 都跟一個 dummy 對照組比：dummy 就是「永遠猜訓練集最常見的
那個動作」。網路明顯贏不過 dummy = 回頭補 `contracts.py` 的 channel、
`ENCODER_VERSION += 1`、重跑資料抽取。**不要調訓練參數硬撐。**

## 切分按局，不按樣本

同一局裡相鄰回合的盤面幾乎一樣。按樣本隨機切的話，驗證集裡會有訓練集的
「隔壁回合」，準確率會虛高。所以整局整局地切。

⚠️ **已知限制**：60 局裡カワシギ佔了大部分（爬蟲從他們的 submission 出發）。
所以驗證集量到的是「對同一套策略的其他對局」的泛化，不是「對不同打法」的泛化。

## 訓練時不遮 mask

replay 裡有 0.12% 的動作連引擎都會忽略（老師自己送錯）。用 mask 遮 loss 的話
那些樣本會變成 -inf。照原標籤學就好，**mask 只在推論時用** —— 那些動作本來
就是 no-op，不送出去嚴格更好。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contracts as C          # noqa: E402
from harness.rollout import unpack_demand                 # noqa: E402
from model.net import KaggricultureNet, count_parameters  # noqa: E402


def demand_f1(pred, truth, legal):
    """只在合法的格子上算 —— 不合法的格子推論時本來就會被遮掉。

    用 F1 不用準確率：11 × 100 個格子裡通常只有幾十個有需求，「永遠說沒有」
    的準確率就有九成多，看不出學到什麼（跟 market present 同一個理由）。
    """
    tp = int((pred & truth & legal).sum())
    fp = int((pred & ~truth & legal).sum())
    fn = int((~pred & truth & legal).sum())
    return tp, fp, fn


class Dataset:
    """把多局的 npz 攤成一份連續記憶體。

    `unit_board` 原本是每局內部的索引，串起來要加上前面幾局的 board 數。
    """

    #: `unit_*` 裡要串起來的欄位 -> 目標屬性名。`labels="target"` 時 op / qty
    #: 換成 segment 的終點版本，`legal_unit_mask` 那一路完全不受影響。
    UNIT_FIELDS = ("unit_board", "unit_pos", "unit_feat",
                   "unit_op", "unit_qty", "unit_target",
                   "unit_term_op", "unit_term_qty")

    def __init__(self, paths, labels="target"):
        boards_spatial, boards_scalar, boards_reward = [], [], []
        market_present, market_qty = [], []
        demand_bits, demand_legal_bits = [], []
        units = {name: [] for name in self.UNIT_FIELDS}
        offset = 0
        for path in paths:
            with np.load(path) as data:
                version = int(data["encoder_version"][0])
                if version != C.ENCODER_VERSION:
                    raise SystemExit(
                        f"{path} 是 ENCODER_VERSION {version}，"
                        f"現在是 {C.ENCODER_VERSION} —— 重跑 harness.build_dataset")
                boards_spatial.append(data["board_spatial"])
                boards_scalar.append(data["board_scalar"])
                boards_reward.append(data["board_reward"])
                for name in ("board_market_present", "board_market_qty"):
                    if name not in data.files:
                        raise SystemExit(
                            f"{path} 沒有 {name} —— 這是 v4 的市場標籤，"
                            "重跑 harness.build_dataset")
                market_present.append(data["board_market_present"])
                market_qty.append(data["board_market_qty"])
                for name in ("board_demand_bits", "board_demand_legal_bits"):
                    if name not in data.files:
                        raise SystemExit(
                            f"{path} 沒有 {name} —— 這是 v5 的逐格需求標籤，"
                            "而它只有 harness.rollout 產得出來"
                            "（老師的 replay 沒有任務清單可讀）")
                demand_bits.append(data["board_demand_bits"])
                demand_legal_bits.append(data["board_demand_legal_bits"])
                for name in self.UNIT_FIELDS:
                    if name not in data.files:
                        raise SystemExit(
                            f"{path} 沒有 {name} —— 這是 v3 的 segment 標籤，"
                            "重跑 harness.build_dataset")
                    units[name].append(data[name])
                # unit_board 原本是每局內部的索引，串起來要加上前面幾局的 board 數
                units["unit_board"][-1] = units["unit_board"][-1].astype(np.int64) + offset
                offset += len(data["board_scalar"])

        self.spatial = np.concatenate(boards_spatial)
        self.scalar = np.concatenate(boards_scalar)
        self.reward = np.concatenate(boards_reward)
        self.market_present = np.concatenate(market_present).astype(np.float32)
        self.market_qty = np.concatenate(market_qty).astype(np.int64)
        # packed 的留 packed（`[N, 11, 13]`，一個盤面 143 byte）。dense 是
        # 1,100 byte，39 萬個盤面就是 427 MB 一份、而且要兩份 —— 這份資料集
        # 光 `spatial` 就已經 2.95 GB。解開的成本在 `make_batch` 分攤掉。
        self.demand_bits = np.concatenate(demand_bits)
        self.demand_legal_bits = np.concatenate(demand_legal_bits)
        self.unit_board = np.concatenate(units["unit_board"])
        self.unit_pos = np.concatenate(units["unit_pos"])
        self.unit_feat = np.concatenate(units["unit_feat"])
        self.unit_target = np.concatenate(units["unit_target"]).astype(np.int64)
        if labels == "target":
            self.unit_op = np.concatenate(units["unit_term_op"]).astype(np.int64)
            self.unit_qty = np.concatenate(units["unit_term_qty"]).astype(np.int64)
        elif labels == "immediate":
            self.unit_op = np.concatenate(units["unit_op"]).astype(np.int64)
            self.unit_qty = np.concatenate(units["unit_qty"]).astype(np.int64)
        else:
            raise SystemExit(f"--labels 只吃 target / immediate，收到 {labels}")

        # unit_board 是非遞減的（抽取時就是照 board 順序寫的），所以每個 board
        # 的 unit 是一段連續區間，用累積計數就能 O(1) 取到。
        counts = np.bincount(self.unit_board, minlength=len(self.scalar))
        self.unit_start = np.concatenate([[0], np.cumsum(counts)])[:-1]
        self.unit_count = counts

    def __len__(self):
        return len(self.scalar)


def make_batch(dataset, board_indices, device):
    """一批盤面 + 它們底下所有的 unit。"""
    starts = dataset.unit_start[board_indices]
    counts = dataset.unit_count[board_indices]
    rows = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)]) \
        if counts.sum() else np.zeros(0, dtype=np.int64)
    # unit_board 要指回「這一批裡的第幾個盤面」，不是全域索引
    local = np.repeat(np.arange(len(board_indices)), counts)

    to = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
    return {
        "spatial": to(dataset.spatial[board_indices].astype(np.float32), torch.float32),
        "scalar": to(dataset.scalar[board_indices], torch.float32),
        "reward": to(dataset.reward[board_indices], torch.float32),
        "unit_board": to(local, torch.long),
        "unit_pos": to(dataset.unit_pos[rows], torch.long),
        "unit_feat": to(dataset.unit_feat[rows], torch.float32),
        "unit_op": to(dataset.unit_op[rows], torch.long),
        "unit_qty": to(dataset.unit_qty[rows], torch.long),
        "unit_target": to(dataset.unit_target[rows], torch.long),
        "market_present": to(dataset.market_present[board_indices], torch.float32),
        "market_qty": to(dataset.market_qty[board_indices], torch.long),
        "demand": to(unpack_demand(dataset.demand_bits[board_indices]), torch.float32),
        "demand_legal": to(
            unpack_demand(dataset.demand_legal_bits[board_indices]), torch.bool),
        # 「目標就是自己現在這格」—— target head 的 dummy 對照組
        "unit_self_cell": to(
            dataset.unit_pos[rows][:, 1].astype(np.int64) * C.BOARD_SIZE
            + dataset.unit_pos[rows][:, 0].astype(np.int64), torch.long),
    }


def evaluate(model, dataset, device, batch_size, limit_batches=0):
    """回傳四個 head 的驗證指標。

    ⚠️ **準確率不是有效的 kill switch**（journal 2026-08-19 §7d）：v2 拿到
    0.94 卻 0 勝 12 負。真正的驗收是 `tools/action_dist.py` —— 自己打一局，
    比對動作分布。這裡的數字只用來確認「有在學」，不用來判定強弱。
    """
    model.eval()
    n_correct = n_total = 0
    op_topk = {3: 0, 5: 0}
    qty_correct = qty_total = 0
    tgt_correct = tgt_total = tgt_dummy = 0
    mk_tp = mk_fp = mk_fn = 0
    mkq_correct = mkq_total = 0
    dm_tp = dm_fp = dm_fn = 0
    dm_all_tp = dm_all_fp = dm_all_fn = 0
    value_sq = 0.0
    order = np.arange(len(dataset))
    with torch.no_grad():
        for b, start in enumerate(range(0, len(order), batch_size)):
            if limit_batches and b >= limit_batches:
                break
            batch = make_batch(dataset, order[start:start + batch_size], device)
            (op_logits, qty_logits, target_logits,
             mk_present, mk_qty, value, demand_logits) = model(
                batch["spatial"], batch["scalar"],
                batch["unit_board"], batch["unit_pos"], batch["unit_feat"])

            # demand：逐格 multi-label，只在合法格上算。
            legal = batch["demand_legal"]
            truth_demand = batch["demand"] > 0.5
            tp, fp, fn = demand_f1(demand_logits > 0, truth_demand, legal)
            dm_tp, dm_fp, dm_fn = dm_tp + tp, dm_fp + fp, dm_fn + fn
            # dummy 對照組是「**所有合法的格子都要做**」（召回率 1，精確率就是
            # 需求佔合法格的比例）。「永遠說沒有」的 F1 是 0，過了不代表學到
            # 判斷；要贏的是這一個 —— 它才是「不做判斷」的下限。
            tp, fp, fn = demand_f1(legal, truth_demand, legal)
            dm_all_tp, dm_all_fp, dm_all_fn = (
                dm_all_tp + tp, dm_all_fp + fp, dm_all_fn + fn)

            # market：present 用 F1 不用準確率 —— 51.7% 的回合一筆訂單都沒有，
            # 「永遠不下單」的準確率就有九成多，看不出學到什麼。
            pred_present = mk_present > 0                      # sigmoid > 0.5
            truth_present = batch["market_present"] > 0.5
            mk_tp += int((pred_present & truth_present).sum().item())
            mk_fp += int((pred_present & ~truth_present).sum().item())
            mk_fn += int((~pred_present & truth_present).sum().item())
            has_mk = batch["market_qty"] >= 0
            if has_mk.any():
                mq_pred = mk_qty[has_mk].argmax(dim=1)
                mkq_correct += (mq_pred == batch["market_qty"][has_mk]).sum().item()
                mkq_total += int(has_mk.sum().item())

            has_op = batch["unit_op"] >= 0
            if has_op.any():
                pred = op_logits[has_op].argmax(dim=1)
                truth_op = batch["unit_op"][has_op]
                n_correct += (pred == truth_op).sum().item()
                n_total += int(has_op.sum().item())
                # top-k 涵蓋率：正確動作有沒有落在前 k 名裡。
                #
                # 🩸 **這是 prior 的驗收指標，argmax 準確率不是。** 離線 search
                # 拿 top-k 當候選集合再用 value head 挑，所以「正確答案在候選裡」
                # 才是它要的；第一名對不對由 search 自己決定。這個 repo 已經量過
                # 三次「準確率不預測實戰」（journal 08-19 §7d、08-20 §7、§11），
                # 不要再拿 op_acc 當門檻。
                topk = op_logits[has_op].topk(5, dim=1).indices
                hit = topk == truth_op[:, None]
                for k in (3, 5):
                    op_topk[k] += int(hit[:, :k].any(dim=1).sum().item())

            has_qty = batch["unit_qty"] >= 0
            if has_qty.any():
                qty_pred = qty_logits[has_qty].argmax(dim=1)
                qty_correct += (qty_pred == batch["unit_qty"][has_qty]).sum().item()
                qty_total += int(has_qty.sum().item())

            has_tgt = batch["unit_target"] >= 0
            if has_tgt.any():
                tgt_pred = target_logits[has_tgt].argmax(dim=1)
                truth = batch["unit_target"][has_tgt]
                tgt_correct += (tgt_pred == truth).sum().item()
                # dummy：永遠猜「目標就是自己現在這格」
                tgt_dummy += (batch["unit_self_cell"][has_tgt] == truth).sum().item()
                tgt_total += int(has_tgt.sum().item())

            value_sq += F.mse_loss(value, batch["reward"], reduction="sum").item()
    model.train()
    precision = mk_tp / max(1, mk_tp + mk_fp)
    recall = mk_tp / max(1, mk_tp + mk_fn)

    def _f1(tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        return 2 * p * r / max(1e-9, p + r), p, r

    demand_score, demand_p, demand_r = _f1(dm_tp, dm_fp, dm_fn)
    demand_dummy, _, _ = _f1(dm_all_tp, dm_all_fp, dm_all_fn)
    return {
        "op_acc": n_correct / max(1, n_total),
        "op_top3": op_topk[3] / max(1, n_total),
        "op_top5": op_topk[5] / max(1, n_total),
        "qty_acc": qty_correct / max(1, qty_total),
        "target_acc": tgt_correct / max(1, tgt_total),
        "target_dummy": tgt_dummy / max(1, tgt_total),
        "market_f1": 2 * precision * recall / max(1e-9, precision + recall),
        "market_precision": precision,
        "market_recall": recall,
        "market_qty_acc": mkq_correct / max(1, mkq_total),
        "demand_f1": demand_score,
        "demand_precision": demand_p,
        "demand_recall": demand_r,
        "demand_dummy": demand_dummy,
        "value_rmse": (value_sq / max(1, len(dataset))) ** 0.5,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dataset",
                    help="npz 目錄，逗號分隔可以給多個。DAgger 的 Aggregation "
                         "就是把每一輪的資料併起來一起訓")
    ap.add_argument("--out", default="model/checkpoints")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-episodes", type=int, default=6)
    ap.add_argument("--train-episodes", type=int, default=0,
                    help="只用前 N 局訓練（驗證集不變）。0 = 全部。"
                         "拿來畫 data-scaling 曲線：曲線平了就代表加資料沒用，"
                         "瓶頸在別的地方")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--qty-weight", type=float, default=0.3)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--market-weight", type=float, default=1.0,
                    help="market present 的 BCE 權重")
    ap.add_argument("--market-qty-weight", type=float, default=0.3)
    ap.add_argument("--target-weight", type=float, default=1.0,
                    help="target head 的 loss 權重。v3 的主要學習訊號，跟 op "
                         "同一個量級")
    ap.add_argument("--demand-weight", type=float, default=2.0,
                    help="逐格需求的 BCE 權重。v5 的主要學習訊號 —— unit 要去"
                         "哪一格是從它算出來的，比 target head 重要，所以預設"
                         "給兩倍")
    ap.add_argument("--labels", choices=("target", "immediate"), default="target",
                    help="target：op 標籤是 segment 的終點動作（v3 預設）。"
                         "immediate：v2 的舊標籤，留著做 A/B，不用重抽資料")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch 的初始化亂數。固定住才分得出「改了參數」跟"
                         "「換了初始化」")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)

    paths = []
    for folder in args.data.split(","):
        folder = folder.strip()
        found = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not found:
            raise SystemExit(f"{folder} 裡沒有 npz")
        paths.extend(found)
    if len(paths) <= args.val_episodes:
        raise SystemExit(f"{args.data} 只有 {len(paths)} 局，不夠切驗證集")

    # 按局切。同一局相鄰回合的盤面幾乎一樣，按樣本切會讓驗證集虛高。
    rng = np.random.default_rng(0)
    order = rng.permutation(len(paths))
    val_paths = [paths[i] for i in order[:args.val_episodes]]
    train_paths = [paths[i] for i in order[args.val_episodes:]]
    if args.train_episodes:
        # 驗證集不動，只砍訓練集 —— 不然不同資料量之間的數字不能比
        train_paths = train_paths[:args.train_episodes]

    print(f"訓練 {len(train_paths)} 局 / 驗證 {len(val_paths)} 局   標籤 {args.labels}")
    train = Dataset(train_paths, labels=args.labels)
    val = Dataset(val_paths, labels=args.labels)
    print(f"  train  board {len(train):,}  unit {len(train.unit_op):,}")
    print(f"  val    board {len(val):,}  unit {len(val.unit_op):,}")
    no_target = float((val.unit_target < 0).mean())
    print(f"  沒有終點動作的 unit（不算 loss）{no_target:.1%}")

    # --- dummy 對照組 ---------------------------------------------------
    op_rows = train.unit_op >= 0
    counts = np.bincount(train.unit_op[op_rows], minlength=C.N_UNIT_OPS)
    dummy_op = int(counts.argmax())
    val_op_rows = val.unit_op >= 0
    dummy_acc = float((val.unit_op[val_op_rows] == dummy_op).mean())
    qty_rows = train.unit_qty >= 0
    dummy_qty = int(np.bincount(train.unit_qty[qty_rows], minlength=C.N_QTY).argmax())
    val_qty_rows = val.unit_qty >= 0
    dummy_qty_acc = float((val.unit_qty[val_qty_rows] == dummy_qty).mean())
    dummy_value_rmse = float(np.sqrt(((val.reward - train.reward.mean()) ** 2).mean()))
    # target 的 dummy 是「目標就是自己現在這格」，不是「猜最常見的格子」——
    # 後者在 100 選 1 上只有 1~2%，過了也不代表學到東西。
    val_tgt_rows = val.unit_target >= 0
    self_cell = (val.unit_pos[:, 1].astype(np.int64) * C.BOARD_SIZE
                 + val.unit_pos[:, 0].astype(np.int64))
    dummy_target_acc = float(
        (self_cell[val_tgt_rows] == val.unit_target[val_tgt_rows]).mean())
    print(f"  dummy  op {dummy_acc:.4f}（永遠猜 {C.UNIT_OPS[dummy_op]}）"
          f"  qty {dummy_qty_acc:.4f}  target {dummy_target_acc:.4f}（猜自己這格）"
          f"  value RMSE {dummy_value_rmse:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KaggricultureNet(
        C.N_SPATIAL, C.N_SCALAR, C.N_UNIT_FEATURES, C.N_UNIT_OPS, C.N_QTY,
        C.N_TARGET_CELLS, C.N_MARKET_OPS, C.N_MARKET_QTY, C.N_TASK_OPS,
        width=args.width, n_blocks=args.blocks).to(device)
    print(f"  device {device}   參數 {count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_stats = {"op_acc": 0.0, "op_top3": 0.0, "op_top5": 0.0,
                  "value_rmse": 0.0, "market_recall": 0.0,
                  "target_acc": 0.0, "target_dummy": 0.0,
                  "market_f1": 0.0}

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        indices = np.random.default_rng(epoch).permutation(len(train))
        running = 0.0
        for step in range(steps_per_epoch):
            batch_idx = indices[step * args.batch:(step + 1) * args.batch]
            batch = make_batch(train, batch_idx, device)
            (op_logits, qty_logits, target_logits,
             mk_present, mk_qty, value, demand_logits) = model(
                batch["spatial"], batch["scalar"],
                batch["unit_board"], batch["unit_pos"], batch["unit_feat"])

            # `ignore_index=-1`：`--labels target` 時，走到一半就日終的那 4.2%
            # 沒有終點動作，標籤是 -1。照 `unit_qty` 一樣遮掉，不要當成類別。
            loss = F.cross_entropy(op_logits, batch["unit_op"], ignore_index=-1)
            has_qty = batch["unit_qty"] >= 0
            if has_qty.any():
                loss = loss + args.qty_weight * F.cross_entropy(
                    qty_logits[has_qty], batch["unit_qty"][has_qty])
            loss = loss + args.target_weight * F.cross_entropy(
                target_logits, batch["unit_target"], ignore_index=-1)

            # market：present 是 multi-label（一回合可以同時下好幾筆），
            # 數量只在有下單的 op 上算。
            loss = loss + args.market_weight * F.binary_cross_entropy_with_logits(
                mk_present, batch["market_present"])
            has_mk = batch["market_qty"] >= 0
            if has_mk.any():
                loss = loss + args.market_qty_weight * F.cross_entropy(
                    mk_qty[has_mk], batch["market_qty"][has_mk])

            # demand：逐格 multi-label。**只在合法的格子上算 loss** ——
            # 不合法的格子（沒種東西的地不能澆水）永遠是 0，把它們算進去的話
            # 1,100 個格子裡有九成多是白送的負例，梯度會被「說沒有」淹掉。
            # 遮掉之後剩下的才是真的要判斷的格子。
            legal = batch["demand_legal"]
            bce = F.binary_cross_entropy_with_logits(
                demand_logits, batch["demand"], reduction="none")
            loss = loss + args.demand_weight * (
                (bce * legal).sum() / legal.sum().clamp(min=1))

            loss = loss + args.value_weight * F.mse_loss(value, batch["reward"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        stats = evaluate(model, val, device, args.batch)
        elapsed = time.perf_counter() - started
        print(f"epoch {epoch:2d}  loss {running / steps_per_epoch:.4f}  "
              f"demand F1 {stats['demand_f1']:.4f} "
              f"(d {stats['demand_dummy']:.4f}, "
              f"P {stats['demand_precision']:.2f} R {stats['demand_recall']:.2f})  "
              f"op {stats['op_acc']:.4f} (d {dummy_acc:.4f}, "
              f"top3 {stats['op_top3']:.4f} top5 {stats['op_top5']:.4f})  "
              f"qty {stats['qty_acc']:.4f} (d {dummy_qty_acc:.4f})  "
              f"target {stats['target_acc']:.4f} (d {stats['target_dummy']:.4f})  "
              f"market F1 {stats['market_f1']:.4f}"
              f"（P {stats['market_precision']:.2f} R {stats['market_recall']:.2f}"
              f" qty {stats['market_qty_acc']:.2f}）  "
              f"value {stats['value_rmse']:.4f} (d {dummy_value_rmse:.4f})  "
              f"{elapsed:.0f}s", flush=True)

        # 選 checkpoint 要把幾個 head 一起看 —— 只看 op 的話，一個「終點動作
        # 猜得準但需求亂指、市場不下單」的模型會被選上，而它上場時整張農場
        # 沒人照顧、什麼都買不到。v5 起 demand 是 unit 那一半的主要訊號，
        # 所以它跟 op / market 同權重進來；target head 仍然訓，但不再計分
        # —— 它已經不決定 unit 去哪裡了（`agents/gen4_demand.py`）。
        #
        # 2026-08-21：`--labels immediate` 也要把 market 算進來。原本它
        # `score = op_acc` 而已，market head 完全不計分 —— 那正是上面這段
        # 註解在警告的情況。端到端那條路（`agents/gen2_model.py`）的動作空間
        # 包含市場訂單，選一個「每個 unit 都動得漂亮但整季不下單」的
        # checkpoint 沒有意義。
        score = stats["op_acc"] + stats["market_f1"]
        if args.labels == "target":
            score += stats["demand_f1"]
        if score > best:
            best, best_stats = score, stats
            torch.save({
                "encoder_version": C.ENCODER_VERSION,
                "state_dict": model.state_dict(),
                "width": args.width,
                "blocks": args.blocks,
                "labels": args.labels,
                # 🩸 存 argv。沒有它就只能靠 rollout 的檔名考古反推當初怎麼跑的
                # —— 2026-08-21 重建 v5-round1 的指令時就卡在這裡，`--expert`
                # 和 `--data` 完全沒有痕跡，只能標成 UNVERIFIED。
                "argv": list(sys.argv[1:]) if argv is None else list(argv),
                "data": args.data,
                "val_op_acc": stats["op_acc"],
                "val_op_top3": stats["op_top3"],
                "val_op_top5": stats["op_top5"],
                "val_value_rmse": stats["value_rmse"],
                "val_market_recall": stats["market_recall"],
                "val_target_acc": stats["target_acc"],
                "val_market_f1": stats["market_f1"],
                "val_demand_f1": stats["demand_f1"],
            }, out_dir / "best.pt")

    # prior 的驗收看 top-k 和 value，不是 argmax —— 理由見 evaluate() 裡那段。
    # 離線 search 拿 top-k 當候選、用 value head 挑，所以「正確答案在候選裡」
    # 才是它要的指標；第一名對不對由 search 自己決定。
    print(f"\n[prior] op top3 {best_stats['op_top3']:.4f}  "
          f"top5 {best_stats['op_top5']:.4f}  "
          f"(argmax {best_stats['op_acc']:.4f})   "
          f"value RMSE {best_stats['value_rmse']:.4f} "
          f"(dummy {dummy_value_rmse:.4f})   "
          f"market recall {best_stats['market_recall']:.4f}")
    print(f"\n存下來的 checkpoint：demand F1 {best_stats['demand_f1']:.4f}"
          f"（dummy {best_stats['demand_dummy']:.4f} —— 所有合法的格子都做）"
          f"  op {best_stats['op_acc']:.4f}（dummy {dummy_acc:.4f}）"
          f"  target {best_stats['target_acc']:.4f}"
          f"（dummy {best_stats['target_dummy']:.4f}）"
          f"  market F1 {best_stats['market_f1']:.4f}（dummy 0 —— 永遠不下單）")
    if best_stats["op_acc"] <= dummy_acc * 1.2:
        print("⚠️  op 贏不過 dummy 多少 —— 先懷疑 contracts.py 漏了資訊，"
              "不要調訓練參數硬撐（workflow.md §5 失敗模式 #2）")
    if best_stats["demand_f1"] <= best_stats["demand_dummy"]:
        print("⚠️  demand 贏不過「所有合法的格子都做」—— 網路沒有學到任何判斷，"
              "只是把 mask 抄了一遍。先看 legal_demand_mask 是不是擋太多"
              "（擋到只剩正解，那 dummy 自然就滿分）")
    if args.labels == "target" and \
            best_stats["target_acc"] <= best_stats["target_dummy"] * 1.2:
        print("⚠️  target 贏不過「猜自己這格」多少 —— 目標可能不是盤面的函數，"
              "先看 harness.build_dataset 的標籤對不對")
    # journal 2026-08-19 §7d：v2 拿到 op 0.94 / dummy 0.16，實戰仍然 0 勝 12 負。
    print("\n上面的數字**不是** kill switch。驗收要自己打一局比對動作分布：")
    print("  python -m eval.runner --a gen2_model --b gen1 --games 40 --workers 16")
    print("  python -m tools.action_dist temp/<run 目錄>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
