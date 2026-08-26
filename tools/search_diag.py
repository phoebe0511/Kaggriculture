"""search 的決策點健康度 —— 從 progress JSONL 分析，不用重跑。

## 為什麼需要這支

2026-08-26 seed 4 死在 `ValueError: 候選裡沒有基準線`，而**在那之前沒有任何
紀錄看得出來**：progress 只記了 `label` / `q` / `n_cands`，看不到「候選被擋掉
幾個」「基準線合不合法」。修完之後 `tools/search_game.py` 會多記
`blocked` / `n_raw` / `base_illegal` / `dropped_orders` / `gain` / `runner_up`，
這支負責把它們讀成人看得懂的東西。

## 三個要回答的問題

1. **基準線非法有多常見、落在哪裡？** 一局 30~60 個決策點，只要中一次舊版就整局
   死掉。修完之後不會死，但頻率仍然是分佈偏移的指標。
2. **`gain` 夠不夠大？** `gain` = 贏家的 Q 減基準線的 Q。如果它的中位數遠小於
   **局間配對差的 sd**（2026-08-26 量到 10,704），那 60 個候選的 argmax 就是被
   長 horizon 的混沌決定的 —— **標籤本質上學不起來**，加權/換編碼/換對手全都
   沒用。
3. **候選集合健康嗎？** `blocked / n_raw` 太高代表大多數候選根本沒進到評分。

用法：

    python -m tools.search_diag temp/search-game-<時間戳>
    python -m tools.search_diag temp/search-game-A temp/search-game-B
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: 2026-08-26 實測：20 局配對差的 sd。`gain` 要跟它比才有意義 ——
#: 比它小很多的話，「最佳候選」是雜訊選出來的。
PAIRED_SD = 10704.0


def load(run_dirs):
    rows = []
    for d in run_dirs:
        prog = Path(d)
        if prog.name != "progress":
            prog = prog / "progress"
        for f in sorted(glob.glob(str(prog / "*.jsonl"))):
            seed = int(Path(f).stem[4:])
            opponent = "?"
            for line in io.open(f, encoding="utf-8"):
                r = json.loads(line)
                if r.get("phase") == "baseline":
                    opponent = r.get("opponent", "?")
                elif r.get("phase") == "search":
                    r["seed"], r["opponent"], r["run"] = seed, opponent, Path(d).name
                    rows.append(r)
    return rows


def _pct(part, whole):
    return f"{part / whole:.1%}" if whole else "—"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="temp/_diag.txt")
    args = ap.parse_args(argv)

    rows = load(args.run_dirs)
    if not rows:
        raise SystemExit("這些目錄裡沒有 search 那種行 —— 路徑對嗎？")
    L = [f"決策點 {len(rows)} 個，來自 {len({(r['run'], r['seed']) for r in rows})} 局"]

    # --- 1. 基準線非法 ---
    ill = [r for r in rows if r.get("base_illegal")]
    L += ["", f"## 基準線非法 {len(ill)}/{len(rows)}  {_pct(len(ill), len(rows))}"]
    if not any("base_illegal" in r for r in rows):
        L.append("  （這批是 2026-08-26 22:50 之前跑的，沒有這個欄位）")
    elif ill:
        by_day = collections.Counter(r["day"] for r in ill)
        by_hour = collections.Counter(r.get("hour", "?") for r in ill)
        by_opp = collections.Counter(r["opponent"] for r in ill)
        L.append(f"  落在哪些 day ：{dict(sorted(by_day.items()))}")
        L.append(f"  落在哪些 hour：{dict(sorted(by_hour.items(), key=str))}")
        # 🩸 只搜一個 hour 的話這一列不帶資訊 —— 講明白，不要讓人誤讀。
        hours_searched = {r.get("hour") for r in rows}
        if len(hours_searched) <= 1:
            L.append(f"    ⚠️ 這批只搜 hour {hours_searched} —— "
                     "所有決策點都在同一個 hour，這一列判不出 hour 相關性")
        L.append(f"  落在哪些對手：{dict(by_opp.most_common())}")
        why = collections.Counter(str(r["base_illegal"])[:60] for r in ill)
        L.append("  原因：")
        for k, v in why.most_common(5):
            L.append(f"    {v:>4}  {k}")
        L.append(f"  平均丟掉的市場單 "
                 f"{statistics.fmean(r.get('dropped_orders', 0) for r in ill):.2f} 筆")

    # --- 2. gain ---
    gains = [r["gain"] for r in rows if r.get("gain") is not None]
    L += ["", f"## 贏家比基準線好多少（gain），n={len(gains)}"]
    if gains:
        changed = [g for g in gains if g > 0]
        L.append(f"  中位數 {statistics.median(gains):>10,.0f}"
                 f"   平均 {statistics.fmean(gains):>10,.0f}"
                 f"   最大 {max(gains):>10,.0f}")
        L.append(f"  gain = 0（沒改）{len(gains) - len(changed)}/{len(gains)}"
                 f"  {_pct(len(gains) - len(changed), len(gains))}")
        if changed:
            L.append(f"  只看有改的：中位數 {statistics.median(changed):>10,.0f}"
                     f"   平均 {statistics.fmean(changed):>10,.0f}")
        for q in (0.5, 0.75, 0.9):
            v = statistics.quantiles(gains, n=100)[int(q * 100) - 1]
            L.append(f"  第 {q:.0%} 百分位 {v:>10,.0f}"
                     f"   = 局間 sd 的 {v / PAIRED_SD:.2f} 倍")
        big = sum(1 for g in gains if g >= PAIRED_SD)
        L.append("")
        L.append(f"  🩸 gain >= 局間 sd（{PAIRED_SD:,.0f}）的有 "
                 f"{big}/{len(gains)}  {_pct(big, len(gains))}")
        L.append("     這個比例低 = 多數決策點的「最佳候選」是雜訊選出來的，")
        L.append("     標籤本質上學不起來（加權/換編碼/換對手都沒用）。")

    # --- 3. 第一名跟第二名的距離 ---
    pairs = [(r["q"], r["runner_up"]) for r in rows
             if r.get("runner_up") is not None and r.get("q") is not None]
    if pairs:
        margins = [a - b for a, b in pairs]
        tie = sum(1 for m in margins if m == 0)
        L += ["", f"## 第一名 − 第二名，n={len(margins)}",
              f"  中位數 {statistics.median(margins):>10,.0f}"
              f"   平均 {statistics.fmean(margins):>10,.0f}",
              f"  完全並列 {tie}/{len(margins)}  {_pct(tie, len(margins))}",
              f"  差距 < 局間 sd 的 "
              f"{sum(1 for m in margins if m < PAIRED_SD)}/{len(margins)}"
              f"  {_pct(sum(1 for m in margins if m < PAIRED_SD), len(margins))}",
              "  🩸 差距小 = argmax 不穩，換一組 rollout 就會選到別的。"]

    # --- 4. 候選集合健康度 ---
    have = [r for r in rows if r.get("n_raw")]
    if have:
        bl = [r["blocked"] / r["n_raw"] for r in have]
        nc = [r["n_cands"] for r in have]
        L += ["", "## 候選集合",
              f"  進到評分的 中位數 {statistics.median(nc):.0f} 個"
              f"（最少 {min(nc)}、最多 {max(nc)}）",
              f"  被擋掉的比例 中位數 {statistics.median(bl):.1%}"
              f"（最高 {max(bl):.1%}）"]
        thin = [r for r in have if r["n_cands"] <= 3]
        if thin:
            L.append(f"  🩸 候選 <= 3 個的決策點有 {len(thin)} 個 —— 那些等於沒搜")
            for r in thin[:5]:
                L.append(f"     seed {r['seed']} day {r['day']}"
                         f"  {r['n_cands']}/{r['n_raw']}"
                         f"  base_illegal={str(r.get('base_illegal'))[:40]}")

    # --- 5. 選了什麼 ---
    labs = collections.Counter(r["label"] for r in rows)
    L += ["", "## 選中的候選（前 10）"]
    for k, v in labs.most_common(10):
        L.append(f"  {k:<12}{v:>5}  {_pct(v, len(rows))}")

    text = "\n".join(L)
    io.open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
