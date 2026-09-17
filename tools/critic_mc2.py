"""同 critic_mc.py，但同時報「逐輪算再平均」與「全部合併」兩種。

§113.2 用的是哪一種可以由此判斷：MAE / bias 是均值，兩種算法相同；
RMSE / 相關 / sd 是非線性的，兩種不同。
"""
from __future__ import annotations
import glob, os, sys
import numpy as np

os.chdir(r"C:\_phoebe_priv\Kaggriculture")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
ARMS = sys.argv[1:] or ["qty-base", "lam-split"]
OUT = open(os.path.join(OUTDIR, "critic_mc2_" + "_".join(ARMS) + ".txt"), "w",
           encoding="utf-8")

def P(*a):
    print(*a, file=OUT); OUT.flush(); print(*a)

def pe(x, y):
    s = x.std() * y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / s) if s else np.nan

def sp(x, y):
    r = lambda v: np.argsort(np.argsort(v)).astype(float)
    return pe(r(x), r(y))

def ev(pred, target):
    return 1.0 - ((target - pred).var() / target.var())

def stats(v, r, ret):
    return dict(MAE=np.abs(v - r).mean(),
                RMSE=np.sqrt(((v - r) ** 2).mean()),
                bias=(v - r).mean(),
                pearson=pe(v, r), spearman=sp(v, r),
                ev_mc=ev(v, r), ev_gae=ev(v, ret), Rsd=r.std())

def analyse(name):
    fs = sorted(glob.glob(f"model/artifacts/{name}/batch-*.npz"))
    if not fs:
        P(f"!! {name}: 找不到 dump"); return
    per, V, R, RET = [], [], [], []
    for f in fs:
        z = np.load(f)
        v = z["old_value"].astype(np.float64)
        rew = z["rew"].astype(np.float64)
        adv = z["adv"].astype(np.float64)
        b = list(z["traj_start"]) + [len(v)]
        fr = np.zeros(len(v))
        for a, e in zip(b[:-1], b[1:]):
            seg = rew[a:e]
            fr[a:e] = np.cumsum(seg[::-1])[::-1]
        per.append(stats(v, fr, adv + v))
        V.append(v); R.append(fr); RET.append(adv + v)
    pool = stats(np.concatenate(V), np.concatenate(R), np.concatenate(RET))
    keys = ["MAE", "RMSE", "bias", "pearson", "spearman", "ev_mc", "ev_gae", "Rsd"]
    P(f"\n{'=' * 74}\n  {name}   {len(fs)} 輪\n{'=' * 74}")
    P(f"  {'':12}{'逐輪算再平均':>14}{'全部合併':>12}")
    for k in keys:
        a = float(np.mean([p[k] for p in per]))
        P(f"  {k:12}{a:>14.4f}{pool[k]:>12.4f}")

if __name__ == "__main__":
    for nm in ARMS:
        analyse(nm)
    OUT.close()
