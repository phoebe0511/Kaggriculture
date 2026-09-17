"""作物 cohort 的 divergence chain：PLANT -> lifecycle -> HARVEST -> SELL。

資料由 tools/cohort_run.py 錄製（決定性重播，不是新資料）。
🩸 分組修正：同一局兩方的期末現金相關 +0.96，所以「高現金」不是強玩家組。
   PPO 的對照用**全部 134 條 Kaggle**；強弱另外用 margin 分組報。
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")
from tools.crop_cohort import CROPS                          # noqa: E402

OUT = open("temp/cohort_analysis.txt", "w", encoding="utf-8")
def P(*a):
    print(*a, file=OUT); OUT.flush(); print(*a)

# 欄位：0 traj 1 tile 2 crop 3 plant_step 4 end_step 5 end_reason
#       6 peak_yield 7 end_yield 8 n_water_day 9 n_fert_day
#       10 n_yield_drop 11 sum_yield_drop
CROPSET = {"WHEAT": 0, "CARROT": 1, "TOMATO": 2, "STRAWBERRY": 3, "MELON": 4}
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER")

def load(path, sel=None):
    z = np.load(path, allow_pickle=True)
    C, S, H, CA = z["cohort"], z["sell"], z["harvest"], z["cash"]
    if sel is not None:
        keep = np.where(sel)[0]
        remap = {t: i for i, t in enumerate(keep)}
        m = np.isin(C[:, 0].astype(int), keep)
        C = C[m].copy()
        C[:, 0] = [remap[int(t)] for t in C[:, 0]]
        S, H, CA = S[keep], H[keep], CA[keep]
    return C, S, H, CA

KG = load("temp/coh_kaggle.npz")
GROUPS = [("kaggle 全部 134", KG)]
for p, t in [("temp/coh_cma1.npz", "cma1-g50-wt"),
             ("temp/coh_ppo140.npz", "PPO ckpt-140"),
             ("temp/coh_ppolam2.npz", "PPO lam2-base")]:
    z = np.load(p, allow_pickle=True)
    n = len(z["cash"])
    GROUPS.append((t, load(p, np.arange(n) % 2 == 0)))   # 只取 a 方

def stats(C, S, H, CA):
    ntj = len(CA)
    life = (C[:, 4] - C[:, 3]) / 24.0
    hv = C[:, 5] == 0
    wd = C[:, 5] == 1
    d = dict(
        n_plant=len(C) / ntj,
        harv_rate=hv.mean(),
        weed_rate=wd.mean(),
        life_mean=life.mean(), life_med=np.median(life),
        life_h_mean=life[hv].mean(), life_h_med=np.median(life[hv]),
        cropdays=life.sum() / ntj,
        peak_mean=C[hv, 6].mean(), peak_med=np.median(C[hv, 6]),
        peak_p25=np.percentile(C[hv, 6], 25), peak_p75=np.percentile(C[hv, 6], 75),
        water_d=C[:, 8].mean(), fert_d=C[:, 9].mean(),
        yield_per_plant=C[hv, 6].sum() / len(C),
        harvest_act=H.mean(),
        sell_crop=S[:, :5].sum(1).mean(), sell_animal=S[:, 5:8].sum(1).mean(),
        sell_fert=S[:, 8].mean(), cash=CA.mean())
    return d

P("=" * 90)
P("  1. cohort 總表（每條軌跡平均；life 單位=天）")
P("=" * 90)
S_ = {}
for tag, g in GROUPS:
    S_[tag] = stats(*g)
KEYS = [("n_plant", "cohort 數/局"), ("harv_rate", "收穫結束率"), ("weed_rate", "變雜草率"),
        ("life_mean", "壽命平均"), ("life_med", "壽命中位"),
        ("life_h_mean", "收穫者壽命平均"), ("cropdays", "作物格日/局"),
        ("peak_mean", "peak_yield 平均"), ("peak_med", "peak_yield 中位"),
        ("peak_p25", "peak p25"), ("peak_p75", "peak p75"),
        ("water_d", "澆水天數/cohort"), ("fert_d", "施肥天數/cohort"),
        ("yield_per_plant", "產量/cohort"), ("harvest_act", "HARVEST 動作/局"),
        ("sell_crop", "賣作物量"), ("sell_animal", "賣動物產品量"),
        ("sell_fert", "賣肥料量"), ("cash", "期末現金")]
P(f"  {'':18}" + "".join(f"{t[:14]:>16}" for t, _ in GROUPS))
for k, label in KEYS:
    row = [S_[t][k] for t, _ in GROUPS]
    fmt = "{:>16,.0f}" if k == "cash" else ("{:>16.3f}" if max(abs(v) for v in row) < 10 else "{:>16.2f}")
    P(f"  {label:18}" + "".join(fmt.format(v) for v in row))

P("")
P("=" * 90)
P("  2. 只看 WHEAT（主力作物；引擎：加產窗口 age 2~4 天，無肥上限 4、全肥 6）")
P("=" * 90)
P(f"  {'':18}" + "".join(f"{t[:14]:>16}" for t, _ in GROUPS))
for tag, (C, S, H, CA) in GROUPS:
    pass
rows = {}
for tag, (C, S, H, CA) in GROUPS:
    w = C[C[:, 2] == 0]
    hv = w[:, 5] == 0
    life = (w[:, 4] - w[:, 3]) / 24.0
    rows[tag] = dict(n=len(w) / len(CA), share=len(w) / max(len(C), 1),
                     life_med=np.median(life[hv]) if hv.any() else np.nan,
                     peak_mean=w[hv, 6].mean() if hv.any() else np.nan,
                     peak_med=np.median(w[hv, 6]) if hv.any() else np.nan,
                     water=w[:, 8].mean(), fert=w[:, 9].mean(),
                     harv=hv.mean(), weed=(w[:, 5] == 1).mean())
for k, label in [("n", "WHEAT cohort/局"), ("share", "佔全部 cohort"),
                 ("life_med", "收穫時壽命中位(天)"), ("peak_mean", "peak_yield 平均"),
                 ("peak_med", "peak_yield 中位"), ("water", "澆水天數"),
                 ("fert", "施肥天數"), ("harv", "收穫結束率"), ("weed", "變雜草率")]:
    P(f"  {label:18}" + "".join(f"{rows[t][k]:>16.3f}" for t, _ in GROUPS))

P("")
P("=" * 90)
P("  3. divergence chain：每一環的比值（PPO ckpt-140 / kaggle 全部）")
P("=" * 90)
k, p = S_["kaggle 全部 134"], S_["PPO ckpt-140"]
CHAIN = [("cohort 數/局", "n_plant"), ("收穫結束率", "harv_rate"),
         ("壽命(收穫者,天)", "life_h_mean"), ("作物格日/局", "cropdays"),
         ("澆水天數/cohort", "water_d"), ("施肥天數/cohort", "fert_d"),
         ("peak_yield 平均", "peak_mean"), ("產量/cohort", "yield_per_plant"),
         ("HARVEST 動作/局", "harvest_act"), ("賣作物量", "sell_crop"),
         ("賣動物產品量", "sell_animal"), ("期末現金", "cash")]
P(f"  {'環節':22}{'kaggle':>12}{'PPO':>12}{'比值':>8}")
for label, key in CHAIN:
    P(f"  {label:22}{k[key]:>12.3f}{p[key]:>12.3f}{p[key]/max(abs(k[key]),1e-9):>8.2f}")
OUT.close()
