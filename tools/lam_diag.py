"""advantage 自相關 / 有效獨立樣本數 / 三成分拆解。臂名吃 argv。

跟 critic3.py 的差別：移動平均用**置中、邊界縮窗**（跟 resid_signal.py 一致），
不是 np.convolve(mode="same") 的零填充 —— 後者會在頭尾 10 步把殘差灌大。
所以這支的數字可以跟 resid_signal.txt 對，不能直接跟 §113.5 對。

另外補 train.jsonl 沒有的 advantage mean（ppo.py:122 每個 minibatch 會正規化掉）。
"""
from __future__ import annotations
import glob, os, sys
import numpy as np

os.chdir(r"C:\_phoebe_priv\Kaggriculture")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
ARMS = sys.argv[1:] or ["qty-base", "qty-on"]
OUT = open(os.path.join(OUTDIR, "lam_diag_" + "_".join(ARMS) + ".txt"), "w",
           encoding="utf-8")

def P(*a):
    print(*a, file=OUT); OUT.flush(); print(*a)

def ma_centred(c, W=10):
    """置中、邊界縮窗的移動平均（窗寬 2W+1）。"""
    cs = np.concatenate([[0.0], np.cumsum(c)])
    i = np.arange(len(c))
    lo = np.maximum(0, i - W); hi = np.minimum(len(c), i + W + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)

def analyse(name):
    fs = sorted(glob.glob(f"model/artifacts/{name}/batch-*.npz"))
    if not fs:
        P(f"\n!! {name}: 找不到 dump"); return
    tot, bet, slow, fast, neff, nraw, lag1, means = [], [], [], [], [], [], [], []
    lag1p = []
    for f in fs:
        z = np.load(f)
        adv = z["adv"].astype(np.float64)
        b = list(z["traj_start"]) + [len(adv)]
        tm = np.zeros_like(adv); sl = np.zeros_like(adv); l1 = []
        for a, e in zip(b[:-1], b[1:]):
            seg = adv[a:e]; mu = seg.mean()
            tm[a:e] = mu
            sl[a:e] = ma_centred(seg - mu)
            l1.append(np.corrcoef(seg[:-1], seg[1:])[0, 1])
        fs_ = adv - tm - sl
        tot.append(adv.var()); bet.append(tm.var())
        slow.append(sl.var()); fast.append(fs_.var())
        lag1.append(np.mean(l1)); means.append(adv.mean())
        rho = []
        for k in range(1, 361):
            xs, ys = [], []
            for a, e in zip(b[:-1], b[1:]):
                s = adv[a:e]
                if len(s) > k:
                    xs.append(s[:-k]); ys.append(s[k:])
            x, y = np.concatenate(xs), np.concatenate(ys)
            r = np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std())
            if r <= 0:
                break
            rho.append(r)
        lag1p.append(rho[0] if rho else 0.0)
        tau = 1 + 2 * sum(rho)
        neff.append(len(adv) / tau); nraw.append(len(adv))

    t = np.mean(tot)
    P(f"\n{'=' * 74}\n  {name}   {len(fs)} 輪\n{'=' * 74}")
    P(f"  Var(adv) 拆解（{len(fs)} 輪平均，置中縮窗 MA_21）")
    P(f"    軌跡平均              {100*np.mean(bet)/t:>6.2f}%   sd {np.sqrt(np.mean(bet)):.4f}")
    P(f"    軌跡內慢速 ±10 步      {100*np.mean(slow)/t:>6.2f}%   sd {np.sqrt(np.mean(slow)):.4f}")
    P(f"    逐步殘差              {100*np.mean(fast)/t:>6.2f}%   sd {np.sqrt(np.mean(fast)):.4f}")
    nd = np.sqrt(np.mean(bet) + np.mean(slow))
    P(f"  advantage  mean {np.mean(means):+.4f}   sd {np.sqrt(t):.4f}"
      f"   （mean 是正規化前；train.jsonl 只有 adv_std）")
    P(f"  不區分動作成分 sd {nd:.4f}   （軌跡平均 ⊕ 慢速）")
    P(f"  lag1 自相關  軌跡內平均 {np.mean(lag1):+.4f}"
      f"   合併全部軌跡 {np.mean(lag1p):+.4f}  （§113.5 報的是後者）")
    P(f"  積分自相關時間 {np.mean(nraw)/np.mean(neff):.0f} 步"
      f"  ->  有效獨立 advantage 值每輪 {np.mean(neff):,.0f} 個"
      f"（原始 {np.mean(nraw):,.0f} 步）")

if __name__ == "__main__":
    for nm in ARMS:
        analyse(nm)
    OUT.close()
