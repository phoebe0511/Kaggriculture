"""residual advantage 裡到底有沒有 action-level outcome signal。

用既有的 120 個 dump，不收新資料、不訓練、不改 PPO code。
ground truth 一律用沿軌跡後加的 MC future reward（gamma=1），不用 `ret`。

分解（對每條軌跡 j 各自做）：
    adv        = m_j                     軌跡平均
               + slow_j(t)               軌跡內慢速 = MA_21(adv − m_j)
               + resid_j(t)              逐步殘差
MA_21 是**置中、邊界縮窗**的移動平均（不是零填充；零填充會在頭尾 10 步
把殘差灌大）。critic3.py 那版用 np.convolve mode="same"，這裡修掉了。

action-level ground truth 的定義（第 3 項要求選一個並寫清楚）：

    R_resid[t] = fr[t] − mean_j(fr) − MA_21(fr − mean_j(fr))

理由：
  1  對 adv 與 fr 施加**完全相同**的線性濾波，比較的是同一個頻段，
     不會因為濾波器不同而製造或抹掉相關。
  2  沒有 leakage —— R_resid 只由實際 reward 造出來，完全沒用到 V 或 adv。
  3  它剛好是「軌跡層級與慢速成分解釋不了的那一部分結果」，也就是
     留給 action-level 訊號去解釋的部分。
  4  fr 是尾和、天生高度自相關；高通濾波把這個共有結構移掉。

限制（要一起講）：fr[t] − fr[t+1] = rew[t]，所以 fr 的高頻段主要就是逐步
reward 的起伏。這個檢定測得到「立即結果」的訊號；**純延遲**的動作效果落在慢速
段，用觀測資料跟狀態效果分不開。

對照定義（第二個，交叉檢查）：cent_rew[t] = rew[t] − mean_j(rew)。

null test：在每條軌跡內對 resid advantage 做
  (a) 隨機環狀位移 —— 保留殘差自身的自相關結構，只打掉與結果的對齊。**主要**
  (b) 完全打散 —— 連自相關一起打掉，null 會偏窄，當次要參考
"""
from __future__ import annotations

import glob
import os

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(r"C:\_phoebe_priv\Kaggriculture")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
OUT = open(os.path.join(
    OUTDIR,
    os.path.splitext(os.path.basename(__file__))[0]
    + "_" + "_".join(sys.argv[1:] or ["qty"]) + ".txt"),
    "w", encoding="utf-8")
RNG = np.random.default_rng(20260910)
W = 21
NPERM = 200


def P(*a):
    print(*a, file=OUT)
    OUT.flush()


def pe(x, y):
    sx, sy = np.std(x), np.std(y)
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)) \
        if sx > 0 and sy > 0 else np.nan


def sp(x, y):
    return pe(np.argsort(np.argsort(x)).astype(float),
              np.argsort(np.argsort(y)).astype(float))


def ma(x, w=W):
    """置中移動平均，邊界縮窗（不零填充）。"""
    half = w // 2
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(0, i - half)
    hi = np.minimum(n, i + half + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def decompose(x):
    """回傳 (平均, 慢速, 殘差)。"""
    m = x.mean()
    slow = ma(x - m)
    return m, slow, x - m - slow


def load(path):
    z = np.load(path)
    adv = z["adv"].astype(np.float64)
    rew = z["rew"].astype(np.float64)
    b = list(z["traj_start"]) + [len(adv)]
    out = []
    for a, e in zip(b[:-1], b[1:]):
        seg_rew = rew[a:e]
        fr = np.cumsum(seg_rew[::-1])[::-1]
        out.append(dict(adv=adv[a:e], fr=fr, rew=seg_rew,
                        t=np.arange(e - a)))
    return out


def arm(name):
    trajs = []
    for f in sorted(glob.glob(f"model/artifacts/{name}/batch-*.npz")):
        trajs += load(f)
    return trajs


def build(trajs):
    """每條軌跡各自分解，然後把各成分攤平。"""
    cols = {k: [] for k in ("a_mean", "a_slow", "a_res", "f_mean", "f_slow",
                            "f_res", "fr", "adv", "cent_rew", "t", "tid")}
    for i, tr in enumerate(trajs):
        am, aslow, ares = decompose(tr["adv"])
        fm, fslow, fres = decompose(tr["fr"])
        n = len(tr["adv"])
        cols["a_mean"].append(np.full(n, am))
        cols["a_slow"].append(aslow)
        cols["a_res"].append(ares)
        cols["f_mean"].append(np.full(n, fm))
        cols["f_slow"].append(fslow)
        cols["f_res"].append(fres)
        cols["fr"].append(tr["fr"])
        cols["adv"].append(tr["adv"])
        cols["cent_rew"].append(tr["rew"] - tr["rew"].mean())
        cols["t"].append(tr["t"])
        cols["tid"].append(np.full(n, i))
    return {k: np.concatenate(v) for k, v in cols.items()}


SEGS = (("early  0-239", 0, 240), ("mid  240-479", 240, 480),
        ("late  480-719", 480, 720))


def run(name):
    trajs = arm(name)
    d = build(trajs)
    P(f"\n{'#' * 78}\n#  {name}   {len(trajs)} 條軌跡 / {len(d['adv']):,} 步"
      f"\n{'#' * 78}")

    P("\n① advantage 分解（置中、邊界縮窗的 MA_21）")
    tot = d["adv"].var()
    for lab, key in (("軌跡平均", "a_mean"), ("軌跡內慢速 ±10 步", "a_slow"),
                     ("逐步殘差", "a_res")):
        v = d[key].var()
        P(f"  {lab:<20} 佔 Var(adv) {100 * v / tot:>6.2f}%   sd {np.sqrt(v):.4f}")
    P(f"  advantage 總 sd {np.sqrt(tot):.4f}"
      f"   （三項相加 {100 * sum(d[k].var() for k in ('a_mean', 'a_slow', 'a_res')) / tot:.1f}%，"
      f"差額是三者非完全正交）")

    P("\n② 三個成分 vs ground-truth MC future reward（fr）")
    P(f"  {'成分':<22}{'Pearson':>10}{'Spearman':>10}"
      f"{'early r':>10}{'mid r':>10}{'late r':>10}")
    for lab, key in (("軌跡平均", "a_mean"), ("軌跡內慢速", "a_slow"),
                     ("逐步殘差", "a_res"), ("(對照) 完整 adv", "adv")):
        cells = ""
        for _lab2, lo, hi in SEGS:
            m = (d["t"] >= lo) & (d["t"] < hi)
            cells += f"{pe(d[key][m], d['fr'][m]):>+10.4f}"
        P(f"  {lab:<22}{pe(d[key], d['fr']):>+10.4f}"
          f"{sp(d[key], d['fr']):>+10.4f}{cells}")
    P("  🩸「軌跡平均 vs fr」是 trajectory-level 的，不能當 action-level 證據。")

    P("\n③④ residual advantage vs action-level outcome")
    P(f"    主定義 R_resid = fr − mean_j(fr) − MA_21(fr − mean_j(fr))")
    P(f"    對照   cent_rew = rew − mean_j(rew)")
    for glab, gkey in (("R_resid（主）", "f_res"), ("cent_rew（對照）", "cent_rew")):
        P(f"\n  {glab}")
        P(f"    {'':<16}{'Pearson':>10}{'Spearman':>10}"
          f"{'early r':>10}{'mid r':>10}{'late r':>10}")
        cells = ""
        for _l, lo, hi in SEGS:
            m = (d["t"] >= lo) & (d["t"] < hi)
            cells += f"{pe(d['a_res'][m], d[gkey][m]):>+10.4f}"
        P(f"    {'a_res':<16}{pe(d['a_res'], d[gkey]):>+10.4f}"
          f"{sp(d['a_res'], d[gkey]):>+10.4f}{cells}")

    P("\n⑤ permutation null（軌跡內）")
    bounds = np.concatenate([[0], np.cumsum([len(t['adv']) for t in trajs])])
    for gkey, glab in (("f_res", "R_resid"), ("cent_rew", "cent_rew")):
        obs = pe(d["a_res"], d[gkey])
        for mode in ("roll", "shuffle"):
            null = np.empty(NPERM)
            x = d["a_res"].copy()
            for p in range(NPERM):
                z = x.copy()
                for a, e in zip(bounds[:-1], bounds[1:]):
                    if mode == "roll":
                        z[a:e] = np.roll(x[a:e], int(RNG.integers(1, e - a)))
                    else:
                        z[a:e] = RNG.permutation(x[a:e])
                null[p] = pe(z, d[gkey])
            pv = (1 + np.sum(np.abs(null) >= abs(obs))) / (NPERM + 1)
            P(f"  {glab:<10} null={mode:<8} 觀測 {obs:+.4f}"
              f"   null mean {null.mean():+.5f}  sd {null.std():.5f}"
              f"   |obs|/sd {abs(obs) / null.std():>7.1f}"
              f"   empirical p {pv:.4f}  (NPERM={NPERM})")


if __name__ == "__main__":
    import sys
    arms = sys.argv[1:] or ["qty-base", "qty-on"]
    for nm in arms:
        run(nm)
    OUT.close()
    print("done")
