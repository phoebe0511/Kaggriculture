"""策略鏈 divergence map：PPO 與高分 trajectory 從哪一天開始分叉、差距怎麼傳遞。

phase = 一天 = 24 步（引擎的 `_end_of_day` 結算週期）。

資料：
  temp/kaggle_days.npz   134 條 Kaggle 軌跡（雙方重播，逐位元重現原局）
  temp/day_ppo140.npz    PPO ckpt-140  vs cma5-g175，20 局
  temp/day_ppolam2.npz   PPO lam2-base vs cma5-g175，20 局
  temp/day_cma1.npz      cma1-g50-wt   vs cma5-g175，20 局

🩸 這張圖不是 causal proof。它找的是「最早出現、而且能解釋後續一連串差距的
   divergence」，不是宣稱因果。
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")
from tools.behav_ops import IDX, FIELDS                      # noqa: E402

OUT = open("temp/chain_div.txt", "w", encoding="utf-8")
def P(*a):
    print(*a, file=OUT); OUT.flush(); print(*a)

ST = {n: i for i, n in enumerate(
    ("open", "empty", "crop", "structure", "weed", "money", "d_money"))}

K = np.load("temp/kaggle_days.npz")
kact, kst, kcash = K["act"], K["st"], K["cash"]
q = np.percentile(kcash, [25, 75])
HI, LO = kcash >= q[1], kcash <= q[0]

def load_local(path):
    z = np.load(path)
    op, em, cr, stc, wd = (z[k][:, 0] for k in ("open", "empty", "crop", "structure", "weed"))
    mn = z["money"][:, 0]
    n = op.shape[0]
    st = np.zeros((n, 30, 7))
    for k, arr in enumerate((op, em, cr, stc, wd)):
        st[:, :, k] = arr.reshape(n, 30, 24).mean(2)
    st[:, :, 5] = mn.reshape(n, 30, 24)[:, :, -1]
    st[:, :, 6] = np.diff(np.concatenate([mn[:, :1], st[:, :, 5]], 1), axis=1)
    return z["ops"][:, 0], st, z["cash"][:, 0]

GROUPS = [("kaggle 高 25%", kact[HI], kst[HI]), ("kaggle 低 25%", kact[LO], kst[LO])]
for path, tag in [("temp/day_cma1.npz", "cma1-g50-wt"),
                  ("temp/day_ppo140.npz", "PPO ckpt-140"),
                  ("temp/day_ppolam2.npz", "PPO lam2-base")]:
    if os.path.exists(path):
        a, s, _ = load_local(path)
        GROUPS.append((tag, a, s))

DAYS = list(range(0, 30))
SHOW = [0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 29]

def series(act, st, name):
    if name in ST:
        return st[:, :, ST[name]]
    return act[:, :, IDX[name]]

P(f"分組：kaggle 高 25% n={HI.sum()}  低 25% n={LO.sum()}"
  f"（期末現金門檻 {q[0]:,.0f} / {q[1]:,.0f}）")
P("")
P("=" * 96)
P("  1. 策略鏈逐日表（每格是該組的平均）")
P("=" * 96)
CHAIN = [("crop", "作物佔比"), ("empty", "空地率"), ("PLANT", "PLANT"),
         ("WATER", "WATER"), ("HARVEST", "HARVEST"), ("BUY_SEED_qty", "買種子量"),
         ("SELL_qty", "賣出量"), ("money", "現金"), ("PASS", "PASS")]
for key, label in CHAIN:
    P("")
    P(f"  {label}")
    P(f"    {'':16}" + "".join(f"{'d'+str(d):>8}" for d in SHOW))
    for tag, act, st in GROUPS:
        v = series(act, st, key).mean(0)
        fmt = "{:>8.3f}" if key in ("crop", "empty") else ("{:>8,.0f}" if key == "money" else "{:>8.1f}")
        P(f"    {tag:16}" + "".join(fmt.format(v[d]) for d in SHOW))

P("")
P("=" * 96)
P("  2. divergence map：PPO ckpt-140 相對 kaggle 高 25%，以高分組的日內 sd 標準化")
P("=" * 96)
ppo = next((g for g in GROUPS if g[0] == "PPO ckpt-140"), None)
if ppo:
    _, pact, pst = ppo
    P(f"    {'量':16}" + "".join(f"{'d'+str(d):>7}" for d in SHOW) + "   首次|z|>2")
    rows = []
    for key, label in CHAIN + [("structure", "建物佔比"), ("PICKUP", "PICKUP"),
                               ("PICKUP_qty", "PICKUP 量"), ("FERTILIZE", "FERTILIZE"),
                               ("COLLECT_FERTILIZER", "COLL_FERT"), ("FEED", "FEED"),
                               ("BUY_SEED", "BUY_SEED"), ("HIRE", "HIRE"),
                               ("d_money", "當日現金變化")]:
        h = series(kact[HI], kst[HI], key)
        p = series(pact, pst, key)
        sd = h.std(0); sd[sd == 0] = np.nan
        z = (p.mean(0) - h.mean(0)) / sd
        first = next((d for d in DAYS if abs(z[d]) > 2), None)
        rows.append((label, z, first))
    for label, z, first in sorted(rows, key=lambda r: (r[2] is None, r[2])):
        P(f"    {label:16}" + "".join(f"{z[d]:>+7.1f}" for d in SHOW)
          + ("      d" + str(first) if first is not None else "         —"))

P("")
P("=" * 96)
P("  3. kaggle 內部：高 25% 減 低 25%，以低分組的 sd 標準化")
P("=" * 96)
P(f"    {'量':16}" + "".join(f"{'d'+str(d):>7}" for d in SHOW) + "   首次|z|>0.5")
rows = []
for key, label in CHAIN + [("structure", "建物佔比"), ("PICKUP_qty", "PICKUP 量"),
                           ("FERTILIZE", "FERTILIZE"), ("HIRE", "HIRE"),
                           ("BUY_LAND", "BUY_LAND"), ("d_money", "當日現金變化")]:
    h = series(kact[HI], kst[HI], key); l = series(kact[LO], kst[LO], key)
    sd = l.std(0); sd[sd == 0] = np.nan
    z = (h.mean(0) - l.mean(0)) / sd
    first = next((d for d in DAYS if abs(z[d]) > 0.5), None)
    rows.append((label, z, first))
for label, z, first in sorted(rows, key=lambda r: (r[2] is None, r[2])):
    P(f"    {label:16}" + "".join(f"{z[d]:>+7.2f}" for d in SHOW)
      + ("      d" + str(first) if first is not None else "         —"))
OUT.close()
