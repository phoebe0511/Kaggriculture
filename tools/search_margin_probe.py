"""減掉對手的期末現金，能不能把「跨延續的變異」壓下來？

## 為什麼問這個

2026-08-27 量到：同一個決策點，換一個延續（換我方接手的策略、或換對手），
「贏家 − base」的期末現金差會抖 sd 4,572（同對手）/ 18,140（跨對手），而
動作本身的差距只有幾百元。所以 `argmax(我方期末現金)` 挑到的是走運的鏈
（journal 08-27 §6~§8）。

550 個回合岔開之後的變化**有一部分是兩家共同的** —— market 是共用的
（一份 inventory、一份 prices，`kaggriculture.py:181-182`），整體供需走高走低
兩家都受影響。那一塊在「我方 − 對手」相減之後會抵消。

所以問題是：**共同的那一塊佔多少？** 佔大宗的話，把 search 的目標從
`我方期末現金` 換成 `我方 − 對手` 就能大幅降低變異，argmax 才有意義。

    sd(margin 版) 明顯小於 sd(現金版)   -> 換目標有用
    兩者差不多                          -> 岔開是各走各的，減對手沒幫助

## 用法

    python -m tools.search_margin_probe temp/curse-gen0-margin

吃的是 `tools/search_curse_probe.py` 的輸出（要有 `o_win` / `o_base` 欄位，
2026-08-27 之後才記）。
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load(run_dir):
    rows = []
    for f in sorted(glob.glob(str(Path(run_dir) / "seed*.jsonl"))):
        rows += [json.loads(line) for line in io.open(f, encoding="utf-8")]
    return [r for r in rows if "o_win" in r]


def _block(name, d):
    win = sum(1 for x in d if x > 0)
    lose = sum(1 for x in d if x < 0)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    out = [f"  {name}",
           f"    勝率（不含平手）{win / max(1, win + lose):>7.1%}"
           f"   贏 {win}  平 {len(d) - win - lose}  輸 {lose}",
           f"    中位 {statistics.median(d):>9,.0f}"
           f"   平均 {statistics.fmean(d):>9,.0f}"
           f"   sd {sd:>9,.0f}"]
    for sig in (500, 2000):
        out.append(f"    要偵測 {sig:>5,} 元的真實差距，需要 "
                   f"{(2.8 * sd / sig) ** 2:>10,.0f} 個不同的延續")
    return out, sd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="temp/_margin.txt")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    L = []
    for run in args.run_dirs:
        rows = load(run)
        if not rows:
            L.append(f"{run}：沒有 o_win 欄位 —— 那批是加對手現金之前跑的")
            continue
        cash = [r["delta"] for r in rows]
        marg = [r["delta_margin"] for r in rows]
        L += [f"{Path(run).name}   n={len(rows)}", ""]
        b1, sd1 = _block("目標 = 我方期末現金（現在的做法）", cash)
        b2, sd2 = _block("目標 = 我方 − 對手（margin）", marg)
        L += b1 + [""] + b2 + [""]
        L.append(f"  sd 比值 margin / 現金 = {sd2 / sd1:.2f}"
                 f"   -> 需要的延續數變成 {(sd2 / sd1) ** 2:.2f} 倍")

        # 我方和對手在同一次 rollout 裡是不是一起動？
        # 🩸 `我方 − 對手` 是「係數固定 = 1」的 control variate，那**不一定**比較好：
        #     Var(X − Y) = σx² + σy² − 2ρσxσy
        # 對手抖得比我方兇（σy > 2ρσx）的話，相減反而放大變異。
        # 最佳係數是 β = ρ·σx/σy，此時 Var(X − βY) = σx²(1 − ρ²)，
        # 也就是**最多**只能把 sd 降到 sqrt(1 − ρ²) 倍 —— 這是相關性給的上限。
        dm = [r["o_win"] - r["o_base"] for r in rows]
        sd_x, sd_y = statistics.stdev(cash), statistics.stdev(dm)
        L.append(f"  對手 delta 的 sd {sd_y:,.0f}（我方是 {sd_x:,.0f}）")
        try:
            from scipy.stats import pearsonr, spearmanr
            rho = pearsonr(cash, dm).statistic
            L.append(f"  我方 delta vs 對手 delta   Pearson r={rho:+.3f}"
                     f"   Spearman r={spearmanr(cash, dm).statistic:+.3f}")
            beta = rho * sd_x / sd_y
            best = sd_x * (1 - rho ** 2) ** 0.5
            L.append(f"  最佳係數 β = {beta:.3f}"
                     f"   -> sd 最好只能降到 {best:,.0f}"
                     f"（{best / sd_x:.2f} 倍，需要的延續數 {(best / sd_x) ** 2:.2f} 倍）")
            L.append(f"    🩸 這是**上限**。要靠 control variate 解決問題，"
                     f"|r| 得非常接近 1，現在是 {abs(rho):.2f}")
        except ImportError:
            pass
        L.append("")

    text = "\n".join(L)
    io.open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
