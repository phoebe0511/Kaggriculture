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
2. **`gain` 夠不夠大？** `gain` = 贏家的 Q 減基準線的 Q。

   🩸 **`gain` 恆 ≥ 0，這是恆等式不是量測結果。** 基準線自己就在候選裡
   （`search/rollout_search.py:114-118` 會強制），取 max 不可能比成員小。
   而且 `q_t` 就等於下一個決策點的 `q_base`，相鄰項抵消之後
   **Σgain == 期末現金 − 對照組現金**（2026-08-27 實測 10 局誤差全部是 0）。
   所以 `search_game` 印的「N/N 局都贏」「Wilcoxon p」也全是恆等式，
   **不要拿它們當 search 有效的證據**。有內容的是「現金比」那一欄 ——
   對手會跟著動，那個沒有結構保證。

   `gain` 是**一個決策點的貢獻**，所以不能拿它跟局間配對差的 sd 比
   （那是整季 30 天累積起來的量）。合理的對照是同一批的
   「整局差 / 有改的決策點數」。

3. **候選集合健康嗎？** `blocked / n_raw` 太高代表大多數候選根本沒進到評分。

⚠️ 這支**回答不了**「標籤學不學得起來」。`gain` 再大也可能只是 51 個一樣好的
候選裡挑到「這一局最走運的那條鏈」（winner's curse）。要判定那件事得換一個
延續策略重評 —— `tools/search_curse_probe.py`。

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

#: 2026-08-26 實測：20 局配對差的 sd。這是**整季**層級的量。
#: 🩸 不要拿它跟單一決策點的 `gain` 比 —— 2026-08-27 發現那是比錯尺度，
#: 一步的貢獻當然幾乎都小於 30 天累積起來的抖動（1200 個點只有 11 個過線，
#: 但那 11/1200 什麼都證明不了）。留著只當「整季尺度」的參照。
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
            L.append(f"  第 {q:.0%} 百分位 {v:>10,.0f}")
        L.append("")
        L.append(f"  Σgain = {sum(gains):,.0f}  = 這批「期末現金 − 對照組」的"
                 "總和（恆等式，2026-08-27 驗過 10 局誤差全 0）")
        L.append(f"  整季尺度參照：局間配對差 sd {PAIRED_SD:,.0f}"
                 f"（🩸 那是整季的量，不要拿來跟單點的 gain 比大小）")

    # --- 3. 第一名跟第二名的距離 ---
    # 🩸 一定要拆成「有改」跟「沒改」。混在一起算的話，沒改的那些點（贏家就是
    # base，一堆候選跟它打平）會把並列率灌爆 —— 2026-08-27 踩過：全部一起算
    # 並列 68%，只看有改的其實是 22%。
    scored = [r for r in rows
              if r.get("runner_up") is not None and r.get("q") is not None]
    for name, sub in (("有改（search 換掉 base）",
                       [r for r in scored if r.get("gain", 0) > 0]),
                      ("沒改（贏家就是 base）",
                       [r for r in scored if r.get("gain", 0) == 0])):
        if not sub:
            continue
        margins = [r["q"] - r["runner_up"] for r in sub]
        tie = sum(1 for m in margins if m == 0)
        L += ["", f"## 第一名 − 第二名 · {name}，n={len(margins)}",
              f"  中位數 {statistics.median(margins):>10,.0f}"
              f"   平均 {statistics.fmean(margins):>10,.0f}",
              f"  完全並列 {tie}/{len(margins)}  {_pct(tie, len(margins))}"]
        for q in (0.25, 0.75):
            v = statistics.quantiles(margins, n=100)[int(q * 100) - 1]
            L.append(f"  第 {q:.0%} 百分位 {v:>10,.0f}")
    if scored:
        L += ["", "  🩸 「有改」那組的差距若跟 gain 本身同一個量級，代表 argmax "
                  "可能只是挑到走運的鏈。",
              "     判定要跑 tools/search_curse_probe.py（換延續策略重評），"
              "這支判不出來。"]

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
