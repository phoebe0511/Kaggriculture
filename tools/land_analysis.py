"""空地率的 behavioural signature：高分 vs 目前 PPO。

資料：
  temp/curve_episodes.npz   67 個 Kaggle episode 雙方重播（逐位元重現原局）
                            = 134 條真實軌跡，期末現金 36,335 ~ 165,141
  temp/curve_ppo140.npz     ppo phi-200/ckpt-00140 vs cma5-g175，20 局同 seed
  temp/curve_ppolam2.npz    ppo lam2-base/last.pt  vs cma5-g175，20 局同 seed
  temp/curve_cma1.npz       cma1-g50-wt            vs cma5-g175，20 局同 seed

空地率 = tiles 是 None 的數 / 已解鎖 tile 數。LOCKED 不進分母，所以晚買的地
不會被 LOCKED 期間稀釋（跟 eval/runner._farm_history 同一個定義）。
WEED 不算空地，另外報 unproductive = empty + weed。

一局 720 步 = 30 天 × 24 turn，所以用**一天 24 步**當 bin。
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")
OUT = open("temp/land_analysis.txt", "w", encoding="utf-8")

def P(*a):
    print(*a, file=OUT); OUT.flush(); print(*a)

def pe(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    s = x.std() * y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / s) if s else np.nan

def sp(x, y):
    r = lambda v: np.argsort(np.argsort(v)).astype(float)
    return pe(r(x), r(y))

def flat(path, tag):
    """攤成 [trajectory, step] 的空地率 + 每條的期末現金/margin。"""
    z = np.load(path)
    op, em, wd = z["open"], z["empty"], z["weed"]
    er = em / np.maximum(op, 1)
    un = (em + wd) / np.maximum(op, 1)
    cash = z["cash"]                                   # [game, 2]
    n, npl, T = er.shape
    ER = er.reshape(n * npl, T); UN = un.reshape(n * npl, T)
    money = z["money"].reshape(n * npl, T)
    C = cash.reshape(n * npl)
    MG = np.stack([cash[:, 0] - cash[:, 1], cash[:, 1] - cash[:, 0]], 1).reshape(n * npl)
    return dict(tag=tag, er=ER, un=UN, cash=C, margin=MG, money=money, n=n * npl)

EP = flat("temp/curve_episodes.npz", "kaggle 67 局")
P(f"kaggle: {EP['n']} 條軌跡   期末現金 {EP['cash'].min():,.0f} ~ {EP['cash'].max():,.0f}")
AG = {}
for p, t in [("temp/curve_ppo140.npz", "PPO ckpt-140"),
             ("temp/curve_ppolam2.npz", "PPO lam2-base"),
             ("temp/curve_cma1.npz", "cma1-g50-wt")]:
    z = np.load(p)
    op, em, wd = z["open"][:, 0], z["empty"][:, 0], z["weed"][:, 0]   # 只取 a 方
    AG[t] = dict(er=em / np.maximum(op, 1), un=(em + wd) / np.maximum(op, 1),
                 cash=z["cash"][:, 0], margin=z["cash"][:, 0] - z["cash"][:, 1],
                 money=z["money"][:, 0])
    if t == "PPO ckpt-140":
        AG["cma5-g175"] = dict(
            er=z["empty"][:, 1] / np.maximum(z["open"][:, 1], 1),
            un=(z["empty"][:, 1] + z["weed"][:, 1]) / np.maximum(z["open"][:, 1], 1),
            cash=z["cash"][:, 1], margin=z["cash"][:, 1] - z["cash"][:, 0],
            money=z["money"][:, 1])

# ---------- 1. 時間曲線 ----------
BIN = 24
nb = 720 // BIN
def curve(er, stat=np.mean):
    return np.array([stat(er[:, i*BIN:(i+1)*BIN]) for i in range(nb)])

q = np.percentile(EP["cash"], [25, 50, 75])
grp = [("kaggle 最高 25%", EP["cash"] >= q[2]), ("kaggle 中間 50%", (EP["cash"] > q[0]) & (EP["cash"] < q[2])),
       ("kaggle 最低 25%", EP["cash"] <= q[0])]
P("")
P("1. 空地率隨天數（一天 24 step；mean，括號是 median）")
P(f"   {'':18}" + "".join(f"{d:>7}" for d in ("d1","d3","d5","d8","d11","d15","d20","d25","d30")))
sel = [0, 2, 4, 7, 10, 14, 19, 24, 29]
def row(tag, er):
    m = curve(er); md = curve(er, np.median)
    P(f"   {tag:18}" + "".join(f"{m[i]:>7.3f}" for i in sel))
    P(f"   {'':18}" + "".join(f"{'('+format(md[i],'.2f')+')':>7}" for i in sel))
for tag, m in grp:
    row(f"{tag} n={m.sum()}", EP["er"][m])
for tag in ("cma5-g175", "cma1-g50-wt", "PPO ckpt-140", "PPO lam2-base"):
    row(f"{tag} n={len(AG[tag]['er'])}", AG[tag]["er"])

# ---------- 2. 摘要統計 ----------
def summ(er):
    mean = er.mean(1)
    auc = er.sum(1) / 720
    fin = er[:, -1]
    def first_below(th):
        hit = er < th
        idx = np.where(hit.any(1), hit.argmax(1), 720)
        return idx
    return mean, auc, fin, first_below(0.10), first_below(0.20)

P("")
P("2. 摘要統計（first<X 的 720 代表整局從未達到）")
P(f"   {'':22}{'整局平均':>10}{'AUC':>9}{'期末':>9}{'首次<10%':>11}{'首次<20%':>11}")
def s_row(tag, er):
    mean, auc, fin, f10, f20 = summ(er)
    P(f"   {tag:22}{mean.mean():>10.4f}{auc.mean():>9.4f}{fin.mean():>9.4f}"
      f"{np.median(f10):>11.0f}{np.median(f20):>11.0f}")
for tag, m in grp:
    s_row(tag, EP["er"][m])
for tag in ("cma5-g175", "cma1-g50-wt", "PPO ckpt-140", "PPO lam2-base"):
    s_row(tag, AG[tag]["er"])

# ---------- 3. 對照表現 ----------
P("")
P("3. 對照表現（kaggle 134 條，同一個母體內）")
mean, auc, fin, f10, f20 = summ(EP["er"])
for nm, v in [("整局平均空地率", mean), ("AUC", auc), ("期末空地率", fin),
              ("首次<10% 的步", f10.astype(float)), ("首次<20% 的步", f20.astype(float))]:
    P(f"   {nm:18} vs 期末現金 {pe(v, EP['cash']):+.4f}   vs margin {pe(v, EP['margin']):+.4f}"
      f"   (Spearman {sp(v, EP['cash']):+.4f})")

# ---------- 4. 控制遊戲進程 ----------
P("")
P("4. 控制進程：每個 bin 內，空地率與該局期末現金的相關（kaggle 134 條）")
P(f"   {'':6}" + "".join(f"{'d'+str(i+1):>7}" for i in sel))
r_in = [pe(EP["er"][:, i*BIN:(i+1)*BIN].mean(1), EP["cash"]) for i in range(nb)]
P(f"   {'r':6}" + "".join(f"{r_in[i]:>+7.3f}" for i in sel))
P("   🩸 每個 bin 內部只比同一時間的不同 trajectory，沒有進程 confound。")

# ---------- 5. 領先落後 ----------
P("")
P("5. 領先落後：空地率下降 vs 現金領先擴大（kaggle 134 條，逐日差分）")
d_er = np.diff(curve_all := np.stack([EP["er"][:, i*BIN:(i+1)*BIN].mean(1) for i in range(nb)], 1), axis=1)
mny = np.stack([EP["money"][:, i*BIN:(i+1)*BIN].mean(1) for i in range(nb)], 1)
d_mn = np.diff(mny, axis=1)
P(f"   {'lag':>5}{'r(Δ空地率[t], Δ現金[t+lag])':>30}")
for lag in (-3, -2, -1, 0, 1, 2, 3):
    if lag < 0:
        x, y = d_er[:, -lag:], d_mn[:, :lag]
    elif lag == 0:
        x, y = d_er, d_mn
    else:
        x, y = d_er[:, :-lag], d_mn[:, lag:]
    P(f"   {lag:>5}{pe(x.ravel(), y.ravel()):>30.4f}")
P("   lag>0 = 空地率先動；lag<0 = 現金先動。負相關代表空地率下降伴隨現金上升。")
OUT.close()
