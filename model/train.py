"""用 ladder 頂端玩家的棋譜訓練 policy / value。

    python -m model.train --data data/dataset --epochs 8

## kill switch：贏不過 dummy 就是編碼漏資訊

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
from model.net import KaggricultureNet, count_parameters  # noqa: E402


class Dataset:
    """把多局的 npz 攤成一份連續記憶體。

    `unit_board` 原本是每局內部的索引，串起來要加上前面幾局的 board 數。
    """

    def __init__(self, paths):
        boards_spatial, boards_scalar, boards_reward = [], [], []
        units_board, units_pos, units_feat, units_op, units_qty = [], [], [], [], []
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
                units_board.append(data["unit_board"].astype(np.int64) + offset)
                units_pos.append(data["unit_pos"])
                units_feat.append(data["unit_feat"])
                units_op.append(data["unit_op"].astype(np.int64))
                units_qty.append(data["unit_qty"].astype(np.int64))
                offset += len(data["board_scalar"])

        self.spatial = np.concatenate(boards_spatial)
        self.scalar = np.concatenate(boards_scalar)
        self.reward = np.concatenate(boards_reward)
        self.unit_board = np.concatenate(units_board)
        self.unit_pos = np.concatenate(units_pos)
        self.unit_feat = np.concatenate(units_feat)
        self.unit_op = np.concatenate(units_op)
        self.unit_qty = np.concatenate(units_qty)

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
    }


def evaluate(model, dataset, device, batch_size, limit_batches=0):
    model.eval()
    n_correct = n_total = 0
    qty_correct = qty_total = 0
    value_sq = 0.0
    order = np.arange(len(dataset))
    with torch.no_grad():
        for b, start in enumerate(range(0, len(order), batch_size)):
            if limit_batches and b >= limit_batches:
                break
            batch = make_batch(dataset, order[start:start + batch_size], device)
            op_logits, qty_logits, value = model(
                batch["spatial"], batch["scalar"],
                batch["unit_board"], batch["unit_pos"], batch["unit_feat"])
            pred = op_logits.argmax(dim=1)
            n_correct += (pred == batch["unit_op"]).sum().item()
            n_total += len(pred)

            has_qty = batch["unit_qty"] >= 0
            if has_qty.any():
                qty_pred = qty_logits[has_qty].argmax(dim=1)
                qty_correct += (qty_pred == batch["unit_qty"][has_qty]).sum().item()
                qty_total += int(has_qty.sum().item())

            value_sq += F.mse_loss(value, batch["reward"], reduction="sum").item()
    model.train()
    return {
        "op_acc": n_correct / max(1, n_total),
        "qty_acc": qty_correct / max(1, qty_total),
        "value_rmse": (value_sq / max(1, len(dataset))) ** 0.5,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--out", default="model/checkpoints")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-episodes", type=int, default=6)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--qty-weight", type=float, default=0.3)
    ap.add_argument("--value-weight", type=float, default=0.5)
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if len(paths) <= args.val_episodes:
        raise SystemExit(f"{args.data} 只有 {len(paths)} 局，不夠切驗證集")

    # 按局切。同一局相鄰回合的盤面幾乎一樣，按樣本切會讓驗證集虛高。
    rng = np.random.default_rng(0)
    order = rng.permutation(len(paths))
    val_paths = [paths[i] for i in order[:args.val_episodes]]
    train_paths = [paths[i] for i in order[args.val_episodes:]]

    print(f"訓練 {len(train_paths)} 局 / 驗證 {len(val_paths)} 局")
    train = Dataset(train_paths)
    val = Dataset(val_paths)
    print(f"  train  board {len(train):,}  unit {len(train.unit_op):,}")
    print(f"  val    board {len(val):,}  unit {len(val.unit_op):,}")

    # --- dummy 對照組 ---------------------------------------------------
    counts = np.bincount(train.unit_op, minlength=C.N_UNIT_OPS)
    dummy_op = int(counts.argmax())
    dummy_acc = float((val.unit_op == dummy_op).mean())
    qty_rows = train.unit_qty >= 0
    dummy_qty = int(np.bincount(train.unit_qty[qty_rows], minlength=C.N_QTY).argmax())
    val_qty_rows = val.unit_qty >= 0
    dummy_qty_acc = float((val.unit_qty[val_qty_rows] == dummy_qty).mean())
    dummy_value_rmse = float(np.sqrt(((val.reward - train.reward.mean()) ** 2).mean()))
    print(f"  dummy  op {dummy_acc:.4f}（永遠猜 {C.UNIT_OPS[dummy_op]}）"
          f"  qty {dummy_qty_acc:.4f}  value RMSE {dummy_value_rmse:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KaggricultureNet(
        C.N_SPATIAL, C.N_SCALAR, C.N_UNIT_FEATURES, C.N_UNIT_OPS, C.N_QTY,
        width=args.width, n_blocks=args.blocks).to(device)
    print(f"  device {device}   參數 {count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        indices = np.random.default_rng(epoch).permutation(len(train))
        running = 0.0
        for step in range(steps_per_epoch):
            batch_idx = indices[step * args.batch:(step + 1) * args.batch]
            batch = make_batch(train, batch_idx, device)
            op_logits, qty_logits, value = model(
                batch["spatial"], batch["scalar"],
                batch["unit_board"], batch["unit_pos"], batch["unit_feat"])

            loss = F.cross_entropy(op_logits, batch["unit_op"])
            has_qty = batch["unit_qty"] >= 0
            if has_qty.any():
                loss = loss + args.qty_weight * F.cross_entropy(
                    qty_logits[has_qty], batch["unit_qty"][has_qty])
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
              f"val op {stats['op_acc']:.4f} (dummy {dummy_acc:.4f})  "
              f"qty {stats['qty_acc']:.4f} (dummy {dummy_qty_acc:.4f})  "
              f"value RMSE {stats['value_rmse']:.4f} (dummy {dummy_value_rmse:.4f})  "
              f"{elapsed:.0f}s", flush=True)

        if stats["op_acc"] > best:
            best = stats["op_acc"]
            torch.save({
                "encoder_version": C.ENCODER_VERSION,
                "state_dict": model.state_dict(),
                "width": args.width,
                "blocks": args.blocks,
                "val_op_acc": stats["op_acc"],
            }, out_dir / "best.pt")

    print(f"\n最好的 val op 準確率 {best:.4f}，dummy 是 {dummy_acc:.4f}")
    if best <= dummy_acc * 1.2:
        print("⚠️  贏不過 dummy 多少 —— 先懷疑 contracts.py 漏了資訊，"
              "不要調訓練參數硬撐（workflow.md §5 失敗模式 #2）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
