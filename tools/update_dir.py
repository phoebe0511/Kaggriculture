"""PPO 實際做出的 policy 移動，方向對不對得上實際結果。臂名吃 argv。

用 `--dump-batch` 的 `old_logp` / `new_logp`（**逐步**，整個聯合動作的 logprob，
`ppo_train.py:208-226`）。Δlogp = new_logp − old_logp 就是「PPO 把這一步實際
採取的動作機率調高還是調低」。

🩸 做不到的（dump 沒存 observation）：|ΔP(top1)|、KL、argmax flip、跨 checkpoint
   的 140->200 對比。那些都要重新前向。

outcome 一律在**軌跡內**做，避免跨 trajectory 比 raw margin：
    fr        沿軌跡後加的 MC future reward（gamma=1）
    cent_fr   fr − mean_j(fr)
    R_resid   fr − mean_j(fr) − MA_21(fr − mean_j(fr))   （§116 的主定義）
    cent_rew  rew − mean_j(rew)                          （§116 的對照）
"""
from __future__ import annotations
import glob, os, sys
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(r"C:\_phoebe_priv\Kaggriculture")
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
ARMS = sys.argv[1:] or ["qty-base"]
OUT = open(os.path.join(OUTDIR, "update_dir_" + "_".join(ARMS) + ".txt"),
           "w", encoding="utf-8")

def P(*a):
    print(*a, file=OUT); OUT.flush(); print(*a)

def pe(x, y):
    s = x.std() * y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / s) if s else np.nan

def sp(x, y):
    r = lambda v: np.argsort(np.argsort(v)).astype(float)
    return pe(r(x), r(y))

def ma_centred(c, W=10):
    cs = np.concatenate([[0.0], np.cumsum(c)])
    i = np.arange(len(c))
    lo = np.maximum(0, i - W); hi = np.minimum(len(c), i + W + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)

def load(name):
    fs = sorted(glob.glob(f"model/artifacts/{name}/batch-*.npz"))
    keys = ("dlogp", "adv", "rew", "fr", "cfr", "rres", "crew", "t", "traj", "margin")
    D = {k: [] for k in keys}
    tj = 0
    for f in fs:
        z = np.load(f)
        ol = z["old_logp"].astype(np.float64); nl = z["new_logp"].astype(np.float64)
        adv = z["adv"].astype(np.float64); rew = z["rew"].astype(np.float64)
        cash = z["traj_cash"]
        b = list(z["traj_start"]) + [len(ol)]
        for j, (a, e) in enumerate(zip(b[:-1], b[1:])):
            seg = rew[a:e]
            fr = np.cumsum(seg[::-1])[::-1]
            cfr = fr - fr.mean()
            D["dlogp"].append(nl[a:e] - ol[a:e])
            D["adv"].append(adv[a:e]); D["rew"].append(seg)
            D["fr"].append(fr); D["cfr"].append(cfr)
            D["rres"].append(cfr - ma_centred(cfr))
            D["crew"].append(seg - seg.mean())
            D["t"].append(np.arange(e - a))
            D["traj"].append(np.full(e - a, tj))
            D["margin"].append(np.full(e - a, float(cash[j, 0] - cash[j, 1])))
            tj += 1
    return {k: np.concatenate(v) for k, v in D.items()}, len(fs), tj

def report(d, name, NPERM=200, seed=20260917):
    dl, rr, cr, adv = d["dlogp"], d["rres"], d["crew"], d["adv"]
    traj, t = d["traj"], d["t"]
    n = len(dl)
    P("")
    P("=" * 78)
    P(f"  {name}   {n:,} 步")
    P("=" * 78)
    P(f"  Δlogp   mean {dl.mean():+.5f}   sd {dl.std():.5f}   中位 {np.median(dl):+.5f}")
    P(f"          調高 {100*(dl>0).mean():.1f}%   調低 {100*(dl<0).mean():.1f}%")

    P("")
    P("  1. 按 |Δlogp| 分桶（被改最多的在最前面）")
    P(f"    {'桶':<14}{'n':>9}{'|Δlogp| 下界':>14}{'mean R_resid':>14}{'mean cent_rew':>15}{'mean adv':>11}")
    q = np.abs(dl); order = np.argsort(-q)
    for lo, hi, tag in [(0, .10, "top 10%"), (.10, .20, "10-20%"),
                        (.20, .50, "20-50%"), (.50, 1.0, "後 50%")]:
        idx = order[int(lo * n):int(hi * n)]
        P(f"    {tag:<14}{len(idx):>9,}{q[idx].min():>14.5f}"
          f"{rr[idx].mean():>14.5f}{cr[idx].mean():>15.5f}{adv[idx].mean():>11.4f}")

    P("")
    P("  2. 按 Δlogp 分三組（校準檢查：是否單調）")
    lo_c, hi_c = np.percentile(dl, [33.333, 66.667])
    P(f"    {'組':<16}{'n':>9}{'Δlogp 平均':>13}{'mean R_resid':>14}{'mean cent_rew':>15}{'mean adv':>11}")
    vals = []
    for tag, m in [("負向 (下 1/3)", dl <= lo_c), ("近零 (中 1/3)", (dl > lo_c) & (dl < hi_c)),
                   ("正向 (上 1/3)", dl >= hi_c)]:
        P(f"    {tag:<16}{m.sum():>9,}{dl[m].mean():>13.5f}"
          f"{rr[m].mean():>14.5f}{cr[m].mean():>15.5f}{adv[m].mean():>11.4f}")
        vals.append(rr[m].mean())
    mono = (vals[0] < vals[1] < vals[2]) or (vals[0] > vals[1] > vals[2])
    P(f"    R_resid 單調: {'是' if mono else '否'}"
      f"   （{vals[0]:+.5f} -> {vals[1]:+.5f} -> {vals[2]:+.5f}）")

    P("")
    P("  3. Δlogp vs outcome 的相關")
    P(f"    {'':16}{'Pearson':>10}{'Spearman':>10}{'early':>10}{'mid':>10}{'late':>10}")
    segs = [(t < 240), (t >= 240) & (t < 480), (t >= 480)]
    for tag, y in [("R_resid", rr), ("cent_rew", cr), ("cent_fr", d["cfr"]),
                   ("(對照) adv", adv)]:
        row = [pe(dl, y), sp(dl, y)] + [pe(dl[s], y[s]) for s in segs]
        P(f"    {tag:16}" + "".join(f"{v:>+10.4f}" for v in row))

    P("")
    P("  4. 調高 vs 調低（Δlogp 的符號）")
    up, dn = dl > 0, dl < 0
    P(f"    {'':18}{'n':>10}{'R_resid':>12}{'SE':>10}{'cent_rew':>12}{'adv':>11}")
    for tag, m in [("調高 Δlogp>0", up), ("調低 Δlogp<0", dn)]:
        se = rr[m].std() / np.sqrt(m.sum())
        P(f"    {tag:18}{m.sum():>10,}{rr[m].mean():>+12.5f}{se:>10.5f}"
          f"{cr[m].mean():>+12.5f}{adv[m].mean():>+11.4f}")
    diff = rr[up].mean() - rr[dn].mean()
    sed = np.sqrt(rr[up].var() / up.sum() + rr[dn].var() / dn.sum())
    P(f"    差（調高 − 調低） {diff:+.5f}   naive SE {sed:.5f}")
    P("    🩸 naive SE 假設獨立，軌跡內自相關會讓它低估。用下面的 permutation。")

    b = np.searchsorted(traj, np.arange(traj[-1] + 2))
    rng = np.random.default_rng(seed)
    P("")
    P("  5. permutation null（軌跡內環狀位移 Δlogp，保留其自相關）")
    obs = pe(dl, rr)
    null = np.empty(NPERM)
    for i in range(NPERM):
        x = dl.copy()
        for a, e in zip(b[:-1], b[1:]):
            if e > a:
                x[a:e] = np.roll(dl[a:e], int(rng.integers(e - a)))
        null[i] = pe(x, rr)
    P(f"    觀測 {obs:+.4f}   null mean {null.mean():+.5f}   sd {null.std():.5f}"
      f"   |obs|/sd {abs(obs - null.mean()) / null.std():.1f}")
    pv = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (NPERM + 1)
    P(f"    empirical p = {pv:.4f}  (NPERM={NPERM}，下限 {1/(NPERM+1):.4f})")

    P("")
    P("  6. 軌跡層級：Δlogp 平均 vs 該局最終 zero-sum margin")
    tdl, tmg = [], []
    for a, e in zip(b[:-1], b[1:]):
        if e > a:
            tdl.append(dl[a:e].mean()); tmg.append(d["margin"][a])
    tdl, tmg = np.asarray(tdl), np.asarray(tmg)
    P(f"    n={len(tdl)} 條軌跡   Pearson {pe(tdl, tmg):+.4f}   Spearman {sp(tdl, tmg):+.4f}")
    P("    🩸 這是 trajectory-level，不能當 action-level 證據。")

if __name__ == "__main__":
    for nm in ARMS:
        d, nr, ntj = load(nm)
        report(d, f"{nm}  {nr} 輪 / {ntj} 條軌跡")
    OUT.close()
