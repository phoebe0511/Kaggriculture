"""critic 對 **MC future reward** 的準度，不是對 GAE target。臂名吃 argv。

🩸 為什麼要另外算：train.jsonl 的 explained_var 是 `1 - Var(ret-value)/Var(ret)`
（ppo.py:497），而 compute_gae 回傳 `ret = adv + value`（ppo.py:88），所以
`ret - value` 恆等於 `adv`：

    explained_var ≡ 1 - Var(adv)/Var(ret)

降 λ 會直接縮小 Var(adv)，explained_var 必然上升 —— 那是定義，不是 critic 變好。
這支用沿軌跡後加的實際 reward（gamma=1）當 target 重算，數字可以跟 §113.2 的
0.7284 / 0.6999 直接比。
"""
from __future__ import annotations
import glob, os, sys
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(r"C:\_phoebe_priv\Kaggriculture")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
ARMS = sys.argv[1:] or ["qty-base", "lam-split"]
OUT = open(os.path.join(OUTDIR, "critic_mc_" + "_".join(ARMS) + ".txt"), "w",
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

def analyse(name):
    fs = sorted(glob.glob(f"model/artifacts/{name}/batch-*.npz"))
    if not fs:
        P(f"!! {name}: 找不到 dump"); return
    V, R, RET, T = [], [], [], []
    for f in fs:
        z = np.load(f)
        v = z["old_value"].astype(np.float64)
        rew = z["rew"].astype(np.float64)
        adv = z["adv"].astype(np.float64)
        b = list(z["traj_start"]) + [len(v)]
        fr = np.zeros(len(v)); t = np.zeros(len(v))
        for a, e in zip(b[:-1], b[1:]):
            seg = rew[a:e]
            fr[a:e] = np.cumsum(seg[::-1])[::-1]
            t[a:e] = np.arange(e - a)
        V.append(v); R.append(fr); RET.append(adv + v); T.append(t)
    v = np.concatenate(V); r = np.concatenate(R)
    ret = np.concatenate(RET); t = np.concatenate(T)

    P(f"\n{'=' * 74}\n  {name}   {len(fs)} 輪   {len(v):,} 步\n{'=' * 74}")
    P(f"  MAE                       {np.abs(v - r).mean():.4f}")
    P(f"  RMSE                      {np.sqrt(((v - r) ** 2).mean()):.4f}")
    P(f"  bias (V-R)               {(v - r).mean():+.4f}")
    P(f"  Pearson(V,R)             {pe(v, r):+.4f}")
    P(f"  Spearman(V,R)            {sp(v, r):+.4f}")
    P(f"  explained var vs MC 報酬  {ev(v, r):+.4f}   <- 跟 §113.2 的 0.7284/0.6999 比")
    P(f"  explained var vs GAE ret {ev(v, ret):+.4f}   <- train.jsonl 報的是這個")
    P(f"  R 的 sd                    {r.std():.4f}")
    P(f"  分段（720 步一局）")
    P(f"    {'':14}{'MAE':>9}{'RMSE':>9}{'r(V,R)':>10}{'ev vs MC':>11}")
    for tag, lo, hi in [("early  0-239", 0, 240), ("mid  240-479", 240, 480),
                        ("late 480-719", 480, 720)]:
        m = (t >= lo) & (t < hi)
        P(f"    {tag:14}{np.abs(v[m]-r[m]).mean():9.4f}"
          f"{np.sqrt(((v[m]-r[m])**2).mean()):9.4f}{pe(v[m], r[m]):+10.4f}"
          f"{ev(v[m], r[m]):+11.4f}")

if __name__ == "__main__":
    for nm in ARMS:
        analyse(nm)
    OUT.close()
