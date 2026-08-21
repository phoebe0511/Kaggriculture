"""從資料算出 market head 的**逐 op 門檻**，不用重訓。

    python -m tools.market_calibrate --ckpt model/artifacts/ckpt-e2e-round5/best.pt `
           --data data/dagger/e2e-val --out config/opponents/e2e-cal.json

## 為什麼要這支

2026-08-21 在網路**自己走出來的**盤面上量到（`data/dagger/e2e-val`，100 局
新 seed）：market head 的逐 op AUC 是 **0.847~0.999**（排序幾乎完美），但
門檻 0 的召回率只有 **0.02~0.44**。正例的 logit 中位數幾乎全是負的 ——
`BUY_SEED CARROT` −2.08、`SELL WOOL` −3.01、`BUY_ANIMAL GOOSE` −3.41。

**排序對、切點錯。** 這不是「沒學會」，是類別不平衡把操作點推歪了
（正例只佔 6%）。修法有兩條：

1. **推論時逐 op 定門檻** —— 這支。零重訓，而且門檻是從資料算的，
   不是手轉的（`agents/gen2_model.RESTOCK_OPS` 那個 −2.0 是手轉出來的）。
2. 訓練時加 `--market-pos-weight` —— 正解，但要重訓，而且要挑倍數。

## 🩸 門檻會隨著 policy 漂

門檻是從「這份權重走出來的盤面」算的。門檻一改，policy 就變，盤面分布跟著
變 —— 下一輪要重新算。**不要把算出來的數字當成常數搬到別的權重上。**

## 切點怎麼挑

預設是**逐 op 最大化 F1**。F1 不是遊戲目標（漏買一次種子和多賣一次的代價
不對稱），但它是唯一不用先假設代價比例就能算的東西。想偏向「敢下單」就用
`--beta`（F-beta，beta > 1 偏 recall）。

資料切兩半：前半算門檻，後半驗。**兩半的數字差很多就是門檻在過擬合**，
那時候要嘛加資料、要嘛回去用 `--market-pos-weight`。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contracts as C                                       # noqa: E402
from model.train import Dataset, make_batch                 # noqa: E402
from tools.prior_eval import load_model                     # noqa: E402


def forward(model, paths, device, batch=512):
    """回傳 `(present_logits, truth)`，形狀都是 `[board, N_MARKET_OPS]`。"""
    ds = Dataset(paths, labels="immediate")
    idx = np.arange(len(ds))
    logits, truth = [], []
    with torch.no_grad():
        for s in range(0, len(ds), batch):
            b = make_batch(ds, idx[s:s + batch], device)
            out = model(b["spatial"], b["scalar"], b["unit_board"],
                        b["unit_pos"], b["unit_feat"])
            logits.append(out[3].cpu().numpy())
            truth.append(b["market_present"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(truth).astype(bool)


def best_threshold(score, label, beta=1.0, grid=None):
    """掃出 F-beta 最大的切點。回傳 `(門檻, F-beta, precision, recall)`。"""
    if label.sum() == 0:
        return 0.0, float("nan"), float("nan"), float("nan")
    # 候選切點取正例分數的百分位 —— 比固定格點省，而且解析度自動跟著分布走。
    grid = grid if grid is not None else np.percentile(
        score[label], np.linspace(0, 100, 101))
    b2 = beta * beta
    best = (0.0, -1.0, 0.0, 0.0)
    for t in np.unique(grid):
        pred = score > t
        tp = int((pred & label).sum())
        if tp == 0:
            continue
        p = tp / int(pred.sum())
        r = tp / int(label.sum())
        f = (1 + b2) * p * r / max(1e-9, b2 * p + r)
        if f > best[1]:
            best = (float(t), f, p, r)
    return best


def score_at(score, label, th):
    """給定門檻的 (F1, precision, recall)。"""
    pred = score > th
    tp = int((pred & label).sum())
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / int(pred.sum())
    r = tp / int(label.sum())
    return 2 * p * r / max(1e-9, p + r), p, r


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="**網路自己走出來的**盤面")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="F-beta 的 beta。>1 偏 recall（敢下單），<1 偏 precision")
    ap.add_argument("--min-pos", type=int, default=30,
                    help="正例少於這個數就不動那個 op 的門檻（留 0）")
    ap.add_argument("--out", help="寫成 config/opponents 的 json")
    ap.add_argument("--name", default="e2e-cal", help="--out 時的對手名字")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    paths = sorted(glob.glob(str(Path(args.data) / "*.npz")))
    if len(paths) < 4:
        raise SystemExit(f"{args.data} 只有 {len(paths)} 局，切不出兩半")
    # 🩸 **隔一個取一個，不能對半切。** 檔名是
    # `{policy}-{對手}-{seed}.npz`，`sorted()` 之後是按對手分群的 ——
    # 對半切會讓前半全是 `ladder-top-a`、後半全是 `starter`，兩半的盤面
    # 分布完全不同。第一次跑就踩到：holdout 的 F1 反而比 fit 高。
    fit_paths, hold_paths = paths[0::2], paths[1::2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.ckpt, device)
    if ckpt.get("labels") != "immediate":
        raise SystemExit(f"{args.ckpt} 的 labels 是 {ckpt.get('labels')}，"
                         f"這支只處理 immediate")

    fit_lg, fit_tr = forward(model, fit_paths, device)
    hold_lg, hold_tr = forward(model, hold_paths, device)
    print(f"checkpoint  {args.ckpt}")
    print(f"算門檻      {len(fit_paths)} 局 / {len(fit_lg):,} 盤面")
    print(f"驗證        {len(hold_paths)} 局 / {len(hold_lg):,} 盤面   beta {args.beta}")

    thresholds = np.zeros(C.N_MARKET_OPS)
    print(f"\n{'op':<14}{'item':<12}{'正例率':>8}{'門檻':>8}"
          f"{'算的 F1':>10}{'驗的 F1':>10}{'驗 P':>8}{'驗 R':>8}{'原 R':>8}")
    for j, (op, item) in enumerate(C.MARKET_OPS):
        lab = fit_tr[:, j]
        if lab.sum() < args.min_pos:
            # 正例太少，切點會被幾個樣本綁架。留 0，行為跟現在一樣。
            print(f"{op:<14}{str(item):<12}{lab.mean():>7.2%}"
                  f"{'—':>8}{'(正例太少)':>20}")
            continue
        th, f_fit, _, _ = best_threshold(fit_lg[:, j], lab, args.beta)
        thresholds[j] = th
        f_h, p_h, r_h = score_at(hold_lg[:, j], hold_tr[:, j], th)
        _, _, r0 = score_at(hold_lg[:, j], hold_tr[:, j], 0.0)
        print(f"{op:<14}{str(item):<12}{lab.mean():>7.2%}{th:>8.2f}"
              f"{f_fit:>10.3f}{f_h:>10.3f}{p_h:>8.3f}{r_h:>8.3f}{r0:>8.3f}")

    # 整體：門檻 0 vs 校準過的，在**驗證那一半**上比
    for label, th in (("門檻 0", np.zeros(C.N_MARKET_OPS)), ("校準後", thresholds)):
        pred = hold_lg > th[None, :]
        tp = int((pred & hold_tr).sum())
        p = tp / max(1, int(pred.sum()))
        r = tp / max(1, int(hold_tr.sum()))
        print(f"\n  {label:<8}整體 F1 {2*p*r/max(1e-9,p+r):.4f}  "
              f"P {p:.4f}  R {r:.4f}  每盤面下單 {pred.sum()/len(pred):.3f} 筆"
              f"（真實 {hold_tr.sum()/len(hold_tr):.3f} 筆）")

    if args.out:
        spec = {
            "name": args.name,
            "entry": "agents.gen2_model:act",
            "params": {"market_threshold_ops": [round(float(t), 3)
                                                for t in thresholds]},
            "note": (
                f"market head 的逐 op 門檻，由 `python -m tools.market_calibrate` "
                f"從 {args.data} 算出來（beta {args.beta}）。"
                f"依據：那份資料上 AUC 是 0.847~0.999（排序幾乎完美）但門檻 0 "
                f"的召回率只有 0.02~0.44 —— 正例只佔 6%，操作點被不平衡推歪了。"
                f"🩸 門檻是從 `{Path(args.ckpt).parent.name}` 走出來的盤面算的，"
                f"**換權重就要重算**，不要當常數搬。"),
        }
        Path(args.out).write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
