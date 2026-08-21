"""拿一份 checkpoint 去考另一批資料 —— **不訓練**。

    python -m tools.prior_eval --ckpt model/artifacts/ckpt-e2e-round5/best.pt `
           --data data/dagger/e2e-val

## 為什麼需要這支

`model/train.py` 的 `[prior]` 那一行是訓練當下算的，驗證集固定在
`--val-from data/dagger/e2e-round0`。那份量尺有兩個問題：

1. **洩漏。** 1,200 局裡切 6 局當驗證，那 6 局來自 round0（seed 0~199），
   而**同一批 seed 在 round1~round5 全部是訓練資料**。開車的 policy 不同
   所以盤面不同，市場走勢與對手行為卻是同一條。輪與輪的趨勢仍然可比
   （每輪都同樣偏），但絕對數字偏樂觀。
2. **量錯對象。** round0 是 **gen1 走出來的盤面**。而 search 要用 prior 和
   value 的地方是**網路自己走出來的盤面**。

要修這兩個問題只要換一批資料重考一次，不必重訓 —— 這支就是幹這個的。

    python -m harness.rollout --policy e2e --expert gen1 `
           --seed0 200 --games 100 --workers 16 --out data/dagger/e2e-val

`--seed0 200` 是全新 seed（訓練資料用的是 0~199），`--policy e2e` 讓盤面
來自網路自己。

## dummy 怎麼算的

⚠️ **這裡的 dummy 跟 `train.py` 的不完全一樣。** `train.py` 的
value dummy 是「永遠猜**訓練集**的平均」，這支沒有訓練集可以看，改成
「永遠猜**這份資料自己**的平均」—— 那是比較嚴格的對照組（它偷看了答案的
平均值），所以網路要贏過它更難。op / qty 的 dummy 同理，用這份資料裡
最常見的類別。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contracts as C                                    # noqa: E402
from model.net import KaggricultureNet                   # noqa: E402
from model.train import Dataset, evaluate                # noqa: E402


def load_model(ckpt_path, device):
    """讀 checkpoint 並還原網路。回傳 `(model, ckpt)`。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("encoder_version") != C.ENCODER_VERSION:
        raise SystemExit(
            f"{ckpt_path} 是 ENCODER_VERSION {ckpt.get('encoder_version')}，"
            f"contracts.py 是 {C.ENCODER_VERSION} —— 輸入語意不一樣，不能比")
    model = KaggricultureNet(
        C.N_SPATIAL, C.N_SCALAR, C.N_UNIT_FEATURES, C.N_UNIT_OPS, C.N_QTY,
        C.N_TARGET_CELLS, C.N_MARKET_OPS, C.N_MARKET_QTY, C.N_TASK_OPS,
        width=ckpt["width"], n_blocks=ckpt["blocks"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def dummies(data):
    """這份資料自己的對照組（見 docstring 的警告）。"""
    op_rows = data.unit_op >= 0
    op = np.bincount(data.unit_op[op_rows], minlength=C.N_UNIT_OPS).argmax()
    qty_rows = data.unit_qty >= 0
    qty = np.bincount(data.unit_qty[qty_rows], minlength=C.N_QTY).argmax()
    self_cell = (data.unit_pos[:, 1].astype(np.int64) * C.BOARD_SIZE
                 + data.unit_pos[:, 0].astype(np.int64))
    tgt_rows = data.unit_target >= 0
    return {
        "op": float((data.unit_op[op_rows] == op).mean()),
        "op_name": C.UNIT_OPS[int(op)],
        "qty": float((data.unit_qty[qty_rows] == qty).mean()),
        "target": float((self_cell[tgt_rows] == data.unit_target[tgt_rows]).mean()),
        "value": float(data.reward.std()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="要考的 checkpoint（best.pt）")
    ap.add_argument("--data", required=True,
                    help="要考的資料目錄，逗號分隔可以給多個")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=0,
                    help="只讀前 N 局（0 = 全部）。純粹是為了快")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    paths = []
    for folder in args.data.split(","):
        found = sorted(glob.glob(os.path.join(folder.strip(), "*.npz")))
        if not found:
            raise SystemExit(f"{folder} 裡沒有 npz")
        paths.extend(found)
    if args.episodes:
        paths = paths[:args.episodes]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.ckpt, device)

    # 🩸 `labels` 一定要跟訓練時一致。`immediate`（這一步做什麼）和
    #    `target`（走到終點做什麼）用同一個 op head 但語意不同 ——
    #    拿錯的那個去考會得到一個很低而且沒有意義的數字。
    labels = ckpt.get("labels", "target")
    data = Dataset(paths, labels=labels)
    d = dummies(data)

    print(f"checkpoint  {args.ckpt}")
    print(f"  訓練指令  {' '.join(ckpt.get('argv') or []) or '(沒存)'}")
    print(f"  labels    {labels}   width {ckpt['width']}  blocks {ckpt['blocks']}")
    print(f"考卷        {args.data}   {len(paths)} 局 / "
          f"board {len(data):,} / unit {len(data.unit_op):,}")

    stats = evaluate(model, data, device, args.batch)
    print(f"\n  op       argmax {stats['op_acc']:.4f}  "
          f"top3 {stats['op_top3']:.4f}  top5 {stats['op_top5']:.4f}"
          f"   (dummy {d['op']:.4f}，永遠猜 {d['op_name']})")
    print(f"  target   {stats['target_acc']:.4f}   (dummy {d['target']:.4f}，猜自己這格)")
    print(f"  qty      {stats['qty_acc']:.4f}   (dummy {d['qty']:.4f})")
    print(f"  market   F1 {stats['market_f1']:.4f}  "
          f"P {stats['market_precision']:.4f}  R {stats['market_recall']:.4f}  "
          f"qty {stats['market_qty_acc']:.4f}")
    print(f"  demand   F1 {stats['demand_f1']:.4f}   (dummy {stats['demand_dummy']:.4f}"
          f"，所有合法的格子都做)")
    # value 的單位是「期末現金 / 100k」（`model/train.py` 的 docstring），
    # 換回錢比較好判斷「差這麼多要不要緊」。
    print(f"  value    RMSE {stats['value_rmse']:.4f} = ${stats['value_rmse'] * 1e5:,.0f}"
          f"   (dummy {d['value']:.4f} = ${d['value'] * 1e5:,.0f})")

    print(f"\n  對照：訓練當下在 {ckpt.get('data', '?')} 的驗證集上是")
    print(f"    op argmax {ckpt.get('val_op_acc', float('nan')):.4f}  "
          f"top3 {ckpt.get('val_op_top3', float('nan')):.4f}  "
          f"top5 {ckpt.get('val_op_top5', float('nan')):.4f}  "
          f"market recall {ckpt.get('val_market_recall', float('nan')):.4f}  "
          f"value RMSE {ckpt.get('val_value_rmse', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
