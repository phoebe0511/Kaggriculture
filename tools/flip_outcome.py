"""140 -> 200 被改掉的 decision，跟實際 outcome 的關係。

用 `tools/argmax_changed.py` 收的 npz（預設 temp/changed.npz）——
兩個 checkpoint 在**同一批盤面**上各重算一次前向，argmax 差異是純 policy 差異。

可以算：argmax flip、|Δ 信心|（|b_p1 − a_p1|）。
🩸 **算不了 KL** —— npz 只存 top1/top2 的機率，沒有完整分布。

outcome 一律在軌跡內做：
    cfut     fut − mean_j(fut)
    R_resid  cfut − MA_21(cfut)        （§116 的主定義）
    margin   該局最終 zero-sum 差額（trajectory-level）

🩸 因果性：軌跡由 driver（ckpt-200）取樣驅動。A 和 B 不同意的那些 decision，
觀測不到「照 A 做會怎樣」。所以這裡全部是相關，不是 causal。
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")
SRC = sys.argv[1] if len(sys.argv) > 1 else "temp/changed.npz"
OUT = open("temp/flip_outcome.txt", "w", encoding="utf-8")

def P(*a):
    print(*a, file=OUT); OUT.flush(); print(*a)

def ma_centred(c, W=10):
    cs = np.concatenate([[0.0], np.cumsum(c)])
    i = np.arange(len(c))
    lo = np.maximum(0, i - W); hi = np.minimum(len(c), i + W + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)

z = np.load(SRC)
us = z["unit_step"]; traj = z["traj"]; pos = z["pos"]
adv = z["adv"].astype(np.float64); margin = z["margin"]
fut = z["fut"]; step_rew = z["step_rew"].astype(np.float64)
starts = list(z["traj_start"]) + [len(step_rew)]

# 步層的 R_resid，再 broadcast 到 unit 層
n = len(step_rew)
sfut = np.zeros(n); rres = np.zeros(n)
for a, e in zip(starts[:-1], starts[1:]):
    f = np.cumsum(step_rew[a:e][::-1])[::-1]
    c = f - f.mean()
    sfut[a:e] = f; rres[a:e] = c - ma_centred(c)
u_rres = rres[us]

flip_op = z["a_op_top1"] != z["b_op_top1"]
flip_tg = z["a_tgt_top1"] != z["b_tgt_top1"]
flip_any = flip_op | flip_tg
dconf = np.abs(z["b_op_p1"].astype(np.float64) - z["a_op_p1"].astype(np.float64))

P(f"來源 {SRC}   {len(np.unique(traj))} 局 / {n:,} 步 / {len(us):,} 個 unit 因子")
P("")
P("  1. 改了多少")
for tag, m in [("op argmax flip", flip_op), ("target argmax flip", flip_tg),
               ("任一 flip", flip_any)]:
    P(f"    {tag:<22}{100*m.mean():>7.2f}%   n={m.sum():>8,}")
P(f"    |Δ 信心| mean {dconf.mean():.4f}  中位 {np.median(dconf):.4f}"
  f"  p90 {np.percentile(dconf,90):.4f}")

P("")
P("  2. flip 的 decision 落在哪裡（對照：沒 flip 的）")
P(f"    {'':16}{'n':>10}{'|adv|':>10}{'adv':>10}{'pos':>9}{'a 的 top1-top2 差':>18}")
gap_a = z["a_op_p1"].astype(np.float64) - z["a_op_p2"].astype(np.float64)
for tag, m in [("op flip", flip_op), ("op 沒 flip", ~flip_op)]:
    P(f"    {tag:16}{m.sum():>10,}{np.abs(adv[m]).mean():>10.4f}"
      f"{adv[m].mean():>+10.4f}{pos[m].mean():>9.3f}{gap_a[m].mean():>18.4f}")

P("")
P("  3. flip vs outcome（unit 層，軌跡內殘差）")
P(f"    {'':16}{'n':>10}{'R_resid':>12}{'naive SE':>11}{'margin':>12}")
for tag, m in [("op flip", flip_op), ("op 沒 flip", ~flip_op)]:
    P(f"    {tag:16}{m.sum():>10,}{u_rres[m].mean():>+12.5f}"
      f"{u_rres[m].std()/np.sqrt(m.sum()):>11.5f}{margin[m].mean():>12,.0f}")
P(f"    差 {u_rres[flip_op].mean()-u_rres[~flip_op].mean():+.5f}")
P("    🩸 unit 層的 n 不是有效樣本數 —— 同一步的 unit 共用同一個 adv 和 outcome。")

P("")
P("  4. 步層（每步取該步有無 flip），早中晚分段")
step_flip = np.zeros(n, bool)
np.logical_or.at(step_flip, us, flip_op)
spos = np.zeros(n)
for a, e in zip(starts[:-1], starts[1:]):
    spos[a:e] = np.arange(e - a) / max(e - a - 1, 1)
P(f"    {'段':<10}{'有 flip 的步':>13}{'R_resid(flip)':>15}{'R_resid(無)':>14}{'差':>11}")
for tag, lo, hi in [("early", 0, 1/3), ("mid", 1/3, 2/3), ("late", 2/3, 1.01)]:
    s = (spos >= lo) & (spos < hi)
    f_, nf = s & step_flip, s & ~step_flip
    P(f"    {tag:<10}{100*step_flip[s].mean():>12.1f}%{rres[f_].mean():>+15.5f}"
      f"{rres[nf].mean():>+14.5f}{rres[f_].mean()-rres[nf].mean():>+11.5f}")
s = slice(None)
P(f"    {'全部':<10}{100*step_flip.mean():>12.1f}%{rres[step_flip].mean():>+15.5f}"
  f"{rres[~step_flip].mean():>+14.5f}{rres[step_flip].mean()-rres[~step_flip].mean():>+11.5f}")

P("")
P("  5. |Δ 信心| 分桶 vs outcome（unit 層）")
P(f"    {'桶':<12}{'n':>10}{'|Δ信心| 下界':>14}{'R_resid':>12}{'adv':>10}")
order = np.argsort(-dconf); N = len(dconf)
for lo, hi, tag in [(0,.10,"top 10%"),(.10,.20,"10-20%"),(.20,.50,"20-50%"),(.50,1.,"後 50%")]:
    idx = order[int(lo*N):int(hi*N)]
    P(f"    {tag:<12}{len(idx):>10,}{dconf[idx].min():>14.4f}"
      f"{u_rres[idx].mean():>+12.5f}{adv[idx].mean():>+10.4f}")
OUT.close()
