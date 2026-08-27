"""search 的標籤學不學得起來？—— 08-26 §8 的選項 B。

## 要回答的問題

`tools/search_label_probe.py` 量到：把 20 局 search 資料混進 1,016,666 個盤面
（0.059%）之後，網路在「search 改掉 gen0 動作」的盤面上只進步 1.4pp，
比「search 保留 gen0 動作」的 3.4pp 還少。**那 3,803 個標籤就在訓練集裡，
連背都沒背起來。**

那是稀釋造成的，還是這些標籤根本學不起來？這支工具把稀釋拿掉：
**只用 search 盤面微調 round6**，看兩件事。

    訓練組命中率  能不能 fit
    驗證組命中率  能不能**泛化**到沒看過的局

## 三種結果，三個不同的下一步

| 訓練組 | 驗證組 | 結論 | 下一步 |
|---|---|---|---|
| 上得去 | **上得去** | 標籤有可學的規律，問題純粹是混合比例 | 加權（§8 的 A / C） |
| 上得去 | 上不去 | fit 得起來但學不到規律 —— 樣本太少 | 產更多 search 標籤（§8 的 D） |
| **上不去** | 上不去 | 輸入相同、標籤不同 -> **編碼器看不見 search 為什麼選那個** | 回頭補 `contracts.py` 的 channel，`ENCODER_VERSION += 1`。A/C/D 全部白做 |

第三種是 `workflow.md` §5 的第二號失敗模式：「網路學不到它看不見的東西，
而且**不會報錯**」。

## 切分按局，不按盤面

同一局裡相鄰的 search 回合盤面很像。按盤面隨機切的話驗證組會混進訓練組的
鄰居，命中率虛高 —— 跟 `model/train.py` 切訓練/驗證的理由一樣。

## 🩸 這支不產出可用的模型

只在 600 個盤面上微調會災難性遺忘（`--forget-check` 會量給你看）。
**它的產出是一個判斷，不是一個 checkpoint。**

用法：

    python -m tools.search_finetune_probe
    python -m tools.search_finetune_probe --epochs 60 --lr 3e-4
"""
from __future__ import annotations

import argparse
import glob
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._quiet import silenced  # noqa: E402

# 🩸 `contracts` / `agents.gen0` 會連帶 import 引擎，而 open_spiel 掃遊戲
# 清單會噴 336 行到 stderr（fd 層級，`redirect_stderr` 攔不到）。
with silenced():
    import contracts as C  # noqa: E402
    from model.net import KaggricultureNet  # noqa: E402
    from model.train import Dataset, make_batch  # noqa: E402


def _search_boards_per_file(paths):
    """每個檔案的 (全域 board 起點, 那一局裡 search 回合的全域索引)。"""
    out, offset = [], 0
    for p in paths:
        with np.load(p) as d:
            n = d["board_spatial"].shape[0]
            is_search = ~d["board_demand_legal_bits"].any(axis=(1, 2))
            out.append(offset + np.flatnonzero(is_search))
            offset += n
    return out


@torch.no_grad()
def op_accuracy(model, dataset, boards, device, batch=64):
    """這些盤面底下所有 unit 的 op argmax 命中率。"""
    model.eval()
    hit = tot = 0
    for i in range(0, len(boards), batch):
        b = make_batch(dataset, boards[i:i + batch], device)
        if b["unit_op"].numel() == 0:
            continue
        op_logits, _q, _t, _m, _mq, _v, _d = model(
            b["spatial"], b["scalar"], b["unit_board"], b["unit_pos"], b["unit_feat"])
        keep = b["unit_op"] >= 0
        hit += int((op_logits.argmax(dim=1)[keep] == b["unit_op"][keep]).sum())
        tot += int(keep.sum())
    return hit / max(tot, 1), tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search/round7")
    ap.add_argument("--ckpt", default="model/artifacts/ckpt-e2e-round6/best.pt")
    ap.add_argument("--val-games", type=int, default=5,
                    help="留幾局完全不訓練，只拿來看泛化")
    ap.add_argument("--train-games", default="",
                    help="逗號分隔就跑 data-scaling 曲線，例如 3,6,9,12,15。"
                         "空的 = 全部。曲線平了 = 加資料沒用，瓶頸在編碼器")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--forget-check", default="data/dagger/e2e-round0",
                    help="量災難性遺忘用的舊資料夾（抽 3 局）。空字串 = 不量")
    ap.add_argument("--out", default="temp/_finetune.txt")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(f"{args.data}/*.npz"))
    if len(paths) <= args.val_games:
        raise SystemExit(f"{args.data} 只有 {len(paths)} 局")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Dataset(paths, labels="immediate")
    per_file = _search_boards_per_file(paths)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(paths))
    val_files = set(order[:args.val_games].tolist())
    train_b = np.concatenate([per_file[i] for i in range(len(paths))
                              if i not in val_files])
    val_b = np.concatenate([per_file[i] for i in sorted(val_files)])

    def fresh_model():
        m = KaggricultureNet(
            C.N_SPATIAL, C.N_SCALAR, C.N_UNIT_FEATURES, C.N_UNIT_OPS, C.N_QTY,
            C.N_TARGET_CELLS, C.N_MARKET_OPS, C.N_MARKET_QTY, C.N_TASK_OPS,
            width=128, n_blocks=8).to(device)
        m.load_state_dict(torch.load(args.ckpt, map_location=device)["state_dict"])
        return m

    lines = [f"微調起點 {args.ckpt}   device {device}",
             f"search 盤面：訓練 {len(train_b)}（{len(paths) - args.val_games} 局）"
             f"／驗證 {len(val_b)}（{args.val_games} 局，完全沒訓練過）",
             f"lr {args.lr}   epochs {args.epochs}   batch {args.batch}",
             ""]

    train_files = [i for i in range(len(paths)) if i not in val_files]

    forget_ds = forget_b = None
    if args.forget_check:
        old = sorted(glob.glob(f"{args.forget_check}/*.npz"))[:3]
        if old:
            forget_ds = Dataset(old, labels="immediate")
            forget_b = np.arange(len(forget_ds))

    def measure(m):
        a_old = None
        if forget_ds is not None:
            a_old = op_accuracy(m, forget_ds, forget_b, device, args.batch)[0]
        return (op_accuracy(m, dataset, val_b, device, args.batch)[0], a_old)

    def run_one(n_games, log_rows):
        """只用前 n_games 局的 search 盤面微調。回傳最後和最好的驗證命中率。"""
        tb = np.concatenate([per_file[i] for i in train_files[:n_games]])
        m = fresh_model()
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
        a_tr = op_accuracy(m, dataset, tb, device, args.batch)[0]
        a_va, a_old = measure(m)
        best_va = a_va
        for ep in range(1, args.epochs + 1):
            m.train()
            perm = np.random.default_rng(ep).permutation(len(tb))
            for i in range(0, len(perm), args.batch):
                b = make_batch(dataset, tb[perm[i:i + args.batch]], device)
                if b["unit_op"].numel() == 0:
                    continue
                op_logits, qty_logits, _t, _m, _mq, _v, _d = m(
                    b["spatial"], b["scalar"], b["unit_board"],
                    b["unit_pos"], b["unit_feat"])
                # 只訓 op（+ qty）—— 診斷的量測對象就是 op argmax，目標函數要對齊。
                # target / demand 在 search 盤面上是 -1 / 全 0，本來就不算 loss。
                loss = F.cross_entropy(op_logits, b["unit_op"], ignore_index=-1)
                has_qty = b["unit_qty"] >= 0
                if has_qty.any():
                    loss = loss + 0.3 * F.cross_entropy(
                        qty_logits[has_qty], b["unit_qty"][has_qty])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if ep % max(1, args.epochs // 20) == 0 or ep == args.epochs:
                a_tr = op_accuracy(m, dataset, tb, device, args.batch)[0]
                a_va, a_old = measure(m)
                best_va = max(best_va, a_va)
                if log_rows is not None:
                    row = f"epoch {ep:>3}   訓練組 {a_tr:.4f}   驗證組 {a_va:.4f}"
                    if a_old is not None:
                        row += f"   舊資料 {a_old:.4f}"
                    log_rows.append(row)
        return len(tb), a_tr, a_va, best_va, a_old

    base_m = fresh_model()
    a_va0, a_old0 = measure(base_m)
    a_tr0 = op_accuracy(base_m, dataset, train_b, device, args.batch)[0]
    base = f"起點（round6）   訓練組 {a_tr0:.4f}   驗證組 {a_va0:.4f}"
    if a_old0 is not None:
        base += f"   舊資料 {a_old0:.4f}"
    lines.append(base)
    lines.append(f"  (unit 標籤 訓練 {len(train_b)} 盤面 / 驗證 {len(val_b)} 盤面)")
    lines.append("")
    del base_m

    if args.train_games:
        sizes = [int(s) for s in args.train_games.split(",") if s.strip()]
        lines.append("data-scaling 曲線（驗證組永遠是同樣那 "
                     f"{args.val_games} 局，完全沒訓練過）")
        lines.append(f"{'訓練局數':>8}{'search 盤面':>12}{'訓練組':>10}"
                     f"{'驗證組(最後)':>14}{'驗證組(最好)':>14}{'舊資料':>10}")
        lines.append(f"{0:>8}{len(train_b):>12}{a_tr0:>10.4f}"
                     f"{a_va0:>14.4f}{a_va0:>14.4f}"
                     f"{(a_old0 if a_old0 is not None else 0):>10.4f}")
        for n in sizes:
            nb, a_tr, a_va, best_va, a_old = run_one(n, None)
            lines.append(f"{n:>8}{nb:>12}{a_tr:>10.4f}"
                         f"{a_va:>14.4f}{best_va:>14.4f}"
                         f"{(a_old if a_old is not None else 0):>10.4f}")
        lines += ["",
                  "🩸 看的是「驗證組(最好)」那一欄隨訓練局數的走勢：",
                  "  往上爬 -> 加資料有用，瓶頸是樣本數 -> 產更多 search 標籤",
                  "  平的   -> 加資料沒用，瓶頸在編碼器 -> 補 contracts.py 的 channel"]
    else:
        rows = []
        run_one(len(train_files), rows)
        lines += rows
        lines += ["",
                  "判讀（08-26 journal §8）：",
                  "  訓練↑ 驗證↑  -> 標籤有可學的規律，問題是混合比例 -> 加權",
                  "  訓練↑ 驗證平 -> fit 得起來但學不到規律 -> 樣本太少，要產更多",
                  "  訓練平        -> 輸入相同標籤不同 -> **編碼器看不見**，要補 channel",
                  "",
                  "🩸 「舊資料」那欄掉下去是災難性遺忘，預期之內 —— "
                  "這支不產出可用的模型。"]
    io.open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
