"""search 的標籤有沒有被學進去？—— Stage 4 的第一個判準。

## 為什麼不看總 loss

`data/search/round7/` 只有 600 個 search 盤面，混進 1,016,666 個訓練盤面裡是
**0.059%**。總 loss 和驗證 `op` 準確率幾乎不可能動 —— 動了才奇怪。所以要看的是
**那 600 個盤面上的命中率**，而且要跟沒有這批資料的 round6 比。

## 兩組盤面

    search 盤面    這 600 個回合的動作是 rollout 搜出來的，不是 gen0 的
    對照盤面       同一批 npz 裡其餘的回合，動作仍然是 gen0 的

對照組是防均值回歸用的：如果 round7 在兩組上都升，那是整體訓練的效果，
不是「學到 search」。要的是 **search 組升得比對照組多**。

## 怎麼判讀

    search 命中率 round7 明顯 > round6   標籤學進去了 -> 去打局驗收
    兩者差不多                           被稀釋淹掉 -> 先解決稀釋，不是多產資料

用法：

    python -m tools.search_label_probe \
        --a model/weights-e2e-round6.npz --b model/artifacts/weights-e2e-round7.npz
"""
from __future__ import annotations

import argparse
import glob
import io
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.npz_forward import NumpyPolicy  # noqa: E402


def _rows_by_board(unit_board, n_boards):
    """每個盤面對應的 unit 列。一次算完，不要在迴圈裡對整個陣列做布林比較。"""
    order = np.argsort(unit_board, kind="stable")
    sorted_board = unit_board[order]
    starts = np.searchsorted(sorted_board, np.arange(n_boards), side="left")
    ends = np.searchsorted(sorted_board, np.arange(n_boards), side="right")
    return [order[s:e] for s, e in zip(starts, ends)]


def probe(policy, boards, spatial, scalar, rows, unit_pos, unit_feat, unit_op):
    hit = tot = 0
    for i in boards:
        r = rows[i]
        if len(r) == 0:
            continue
        feats = policy.trunk(spatial[i].astype(np.float32), scalar[i])
        op_l, _q, _t = policy.unit_logits(feats, unit_pos[r], unit_feat[r])
        hit += int((op_l.argmax(axis=1) == unit_op[r]).sum())
        tot += len(r)
    return hit, tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="model/weights-e2e-round6.npz")
    ap.add_argument("--b", default="model/artifacts/weights-e2e-round7.npz")
    ap.add_argument("--data", default="data/search/round7")
    ap.add_argument("--control", type=int, default=600,
                    help="對照組抽幾個非 search 盤面（0 = 全部，很慢）")
    ap.add_argument("--out", default="temp/_probe.txt")
    args = ap.parse_args(argv)

    pols = [(Path(args.a).stem, NumpyPolicy(args.a)),
            (Path(args.b).stem, NumpyPolicy(args.b))]
    rng = np.random.default_rng(0)
    acc = {name: {"search": [0, 0], "control": [0, 0]} for name, _p in pols}

    files = sorted(glob.glob(f"{args.data}/*.npz"))
    for f in files:
        d = np.load(f)
        spatial, scalar = d["board_spatial"], d["board_scalar"]
        n = spatial.shape[0]
        rows = _rows_by_board(d["unit_board"], n)
        # search 的回合：demand_legal 全 0（search 給不出 demand 標籤）
        is_search = ~d["board_demand_legal_bits"].any(axis=(1, 2))
        s_idx = np.flatnonzero(is_search)
        c_pool = np.flatnonzero(~is_search)
        k = len(s_idx) if args.control else len(c_pool)
        c_idx = rng.choice(c_pool, size=min(k, len(c_pool)), replace=False)
        for name, p in pols:
            for tag, idx in (("search", s_idx), ("control", c_idx)):
                h, t = probe(p, idx, spatial, scalar, rows,
                             d["unit_pos"], d["unit_feat"], d["unit_op"])
                acc[name][tag][0] += h
                acc[name][tag][1] += t

    lines = [f"資料 {args.data}（{len(files)} 局）",
             f"{'權重':<28}{'search 盤面':>16}{'對照盤面':>16}"]
    for name, _p in pols:
        s = acc[name]["search"]
        c = acc[name]["control"]
        lines.append(f"{name:<28}{s[0]/s[1]:>15.4f} {c[0]/c[1]:>15.4f}"
                     f"   (n={s[1]:,} / {c[1]:,})")
    a, b = pols[0][0], pols[1][0]
    ds = acc[b]["search"][0]/acc[b]["search"][1] - acc[a]["search"][0]/acc[a]["search"][1]
    dc = acc[b]["control"][0]/acc[b]["control"][1] - acc[a]["control"][0]/acc[a]["control"][1]
    lines += ["",
              f"{b} − {a}：search {ds:+.4f}   對照 {dc:+.4f}   差的差 {ds - dc:+.4f}",
              "",
              "「差的差」> 0 才代表學到的是 search 特有的東西，",
              "而不是整體訓練變好。接近 0 = 被稀釋淹掉。"]
    text = "\n".join(lines)
    io.open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
