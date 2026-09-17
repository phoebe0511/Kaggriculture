"""2 seed × 2 λ 設定的 2×2：把 λ 效果與 seed 效果分開。

    seed 4382   qty-base  (λ=0.998)   lam-split  (0.998/240/0.95)
    seed 8156   lam2-base (λ=0.998)   lam2-split (0.998/240/0.95)

🩸 λ 效果的估計是「兩個 within-seed 差」的平均，n=2。這個 n 不足以做顯著性
宣稱，只能看方向是否一致、量級是否接近。seed 效果同理。
"""
from __future__ import annotations
import json, os, sys
import statistics as st

os.chdir(r"C:\_phoebe_priv\Kaggriculture")
CELLS = {("4382", "base"): "qty-base", ("4382", "split"): "lam-split",
         ("8156", "base"): "lam2-base", ("8156", "split"): "lam2-split"}

def margins(tag):
    p = f"model/artifacts/{tag}/train.jsonl"
    if not os.path.exists(p):
        return None
    R = [json.loads(l) for l in open(p, encoding="utf-8")]
    return {r["iter"]: r["greedy_margin"] for r in R if "greedy_margin" in r}

M = {k: margins(v) for k, v in CELLS.items()}
missing = [CELLS[k] for k, v in M.items() if not v]
if missing:
    print("還沒有資料：", ", ".join(missing)); sys.exit(0)
its = sorted(set.intersection(*(set(v) for v in M.values())))
print(f"共同評估點 n={len(its)}  {its}\n")

mean = {k: st.mean(M[k][i] for i in its) for k in M}
print(f"{'':12}{'λ=0.998':>12}{'phased':>12}{'λ 效果':>12}")
for s in ("4382", "8156"):
    b, t = mean[(s, "base")], mean[(s, "split")]
    print(f"seed {s:<7}{b:>12,.0f}{t:>12,.0f}{t - b:>+12,.0f}")
print(f"{'seed 效果':<12}{mean[('8156','base')] - mean[('4382','base')]:>+12,.0f}"
      f"{mean[('8156','split')] - mean[('4382','split')]:>+12,.0f}")

d = [mean[(s, "split")] - mean[(s, "base")] for s in ("4382", "8156")]
sd = [mean[("8156", c)] - mean[("4382", c)] for c in ("base", "split")]
print(f"\nλ 效果   兩個 seed: {d[0]:+,.0f} / {d[1]:+,.0f}"
      f"   平均 {st.mean(d):+,.0f}   同向: {'是' if d[0]*d[1] > 0 else '否'}")
print(f"seed 效果 兩個設定: {sd[0]:+,.0f} / {sd[1]:+,.0f}   平均 {st.mean(sd):+,.0f}")

import math
eff = st.mean(d)
sd_d = abs(d[0] - d[1]) / math.sqrt(2)          # 兩個 within-seed 估計的 SD
print()
print("[!] λ 效果的 run-to-run SD（由兩個估計反推，n=2，本身極不精確）"
      " = %s" % format(sd_d, ",.0f"))
print("    兩個估計相差 %s，符號 %s"
      % (format(abs(d[0] - d[1]), ",.0f"), "相同" if d[0] * d[1] > 0 else "相反"))
if eff:
    n = 8 * (sd_d / abs(eff)) ** 2
    print("    要偵測 %s 的效果、SD %s，80%% power 每臂約需 %.0f 條 run"
          "（粗估，約 %.0f 小時）" % (format(abs(eff), ",.0f"),
                                    format(sd_d, ",.0f"), n, n * 2))
