"""critic 診斷：value prediction 有沒有足夠解析度提供正確排序的 baseline。

不改 code、不訓練、**不收新資料** —— 用訓練當下 `--dump-batch` 存下來的 60 個
npz（兩臂各 60 輪，每輪 11,504 步 / 16 條軌跡）。裡面有 old_value / rew /
traj_start / adv / ret，足夠算 ground-truth 未來報酬並跟 value 比。

ground truth 用 `rew` 沿軌跡往後加（gamma=1 所以就是未折扣的尾和），
不是 `ret` —— ret = adv + old_value 兩邊都含 advantage
（model/ppo.py:450 的註解）。
🩸 train.jsonl 的 explained_var 用的是 `ret`（GAE 目標）不是 MC 報酬，
   所以 0.82~0.84 跟這裡對 MC 報酬算的解釋力不是同一個量，兩個都報。

第 ④ 項（qty=1 vs qty>1）要 qty_idx，dump 沒存，用先前診斷 rollout 存下來的
qtyadv_{init,last}.npz（那一批的設定與訓練相同）。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

REPO = r"C:\_phoebe_priv\Kaggriculture"
os.chdir(REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
OUT = open(os.path.join(OUTDIR, "critic_report.txt"), "w", encoding="utf-8")


def P(*a):
    print(*a, file=OUT)
    OUT.flush()


def pe(x, y):
    s = np.std(x) * np.std(y)
    return float(np.mean((x - x.mean()) * (y - y.mean())) / s) if s else np.nan


def sp(x, y):
    if len(x) < 3:
        return np.nan
    return pe(np.argsort(np.argsort(x)).astype(float),
              np.argsort(np.argsort(y)).astype(float))


def load(path):
    z = np.load(path)
    v = z["old_value"].astype(np.float64)
    rew = z["rew"].astype(np.float64)
    adv = z["adv"].astype(np.float64)
    ret = z["ret"].astype(np.float64)
    b = list(z["traj_start"]) + [len(v)]
    fr = np.zeros(len(v))
    t = np.zeros(len(v), np.int64)
    tj = np.zeros(len(v), np.int64)
    for k, (a, e) in enumerate(zip(b[:-1], b[1:])):
        fr[a:e] = np.cumsum(rew[a:e][::-1])[::-1]
        t[a:e] = np.arange(e - a)
        tj[a:e] = k
    return dict(v=v, fr=fr, adv=adv, ret=ret, rew=rew, t=t, traj=tj,
                bounds=np.array(b, np.int64))


def arm(name):
    fs = sorted(glob.glob(f"model/artifacts/{name}/batch-*.npz"))
    return [load(f) for f in fs]


def stats(v, fr):
    e = v - fr
    return dict(mae=np.abs(e).mean(), rmse=float(np.sqrt((e ** 2).mean())),
                bias=e.mean(), r=pe(v, fr), rho=sp(v, fr),
                ev=1 - e.var() / fr.var())


def section(name, ds):
    P(f"\n{'#' * 78}\n#  {name}   {len(ds)} 輪 × {len(ds[0]['v']):,} 步"
      f" / 每輪 {len(ds[0]['bounds']) - 1} 條軌跡\n{'#' * 78}")

    P("\n① value prediction vs 實際未來累積 reward（逐輪算，再看 60 輪的分布）")
    rows = [stats(d["v"], d["fr"]) for d in ds]
    P(f"  {'量':<28}{'mean':>10}{'sd':>9}{'第1輪':>10}{'第30輪':>10}{'第60輪':>10}")
    for k, lab in (("mae", "MAE"), ("rmse", "RMSE"), ("bias", "bias (V−R)"),
                   ("r", "Pearson(V,R)"), ("rho", "Spearman(V,R)"),
                   ("ev", "explained var vs MC 報酬")):
        a = np.array([r[k] for r in rows])
        P(f"  {lab:<28}{a.mean():>+10.4f}{a.std():>9.4f}"
          f"{a[0]:>+10.4f}{a[29]:>+10.4f}{a[-1]:>+10.4f}")
    evr = np.array([1 - (d["ret"] - d["v"]).var() / d["ret"].var() for d in ds])
    P(f"  {'explained var vs GAE ret':<28}{evr.mean():>+10.4f}{evr.std():>9.4f}"
      f"{evr[0]:>+10.4f}{evr[29]:>+10.4f}{evr[-1]:>+10.4f}"
      f"   <- train.jsonl 報的是這個")
    frsd = np.array([d["fr"].std() for d in ds])
    vsd = np.array([d["v"].std() for d in ds])
    P(f"  R 的 sd {frsd.mean():.4f}   V 的 sd {vsd.mean():.4f}"
      f"   R 的 mean {np.mean([d['fr'].mean() for d in ds]):+.4f}")

    P("\n② 最接近『同 observation 多 action』的可辨識分析")
    P("   直接版做不到：每個 observation 在 rollout 裡只出現一次、只抽一個聯合動作，")
    P("   沒有同狀態的第二個動作可比。用兩個代用：")
    P("   (a) 同一個 timestep、跨同一輪的 16 條軌跡 —— critic 能不能排對「哪一局此刻較好」")
    sr, pr_, ratio = [], [], []
    for d in ds[::6]:
        for tt in range(0, 720, 20):
            m = d["t"] == tt
            if m.sum() >= 8:
                sr.append(sp(d["v"][m], d["fr"][m]))
                pr_.append(pe(d["v"][m], d["fr"][m]))
                ratio.append(np.abs(d["v"][m] - d["fr"][m]).mean()
                             / (d["fr"][m].std() + 1e-12))
    sr, pr_, ratio = map(np.array, (sr, pr_, ratio))
    P(f"      {len(sr)} 個 (輪, timestep) 桶")
    P(f"      桶內 Spearman(V,R) mean {np.nanmean(sr):+.4f}"
      f"  median {np.nanmedian(sr):+.4f}  P(>0) {100 * np.nanmean(sr > 0):.1f}%")
    P(f"      桶內 Pearson(V,R)  mean {np.nanmean(pr_):+.4f}")
    P(f"      桶內 |V−R| / 桶內 R 的 sd  mean {np.nanmean(ratio):.2f}"
      f"   （>1 表示 critic 的誤差比它要分辨的差異還大）")
    P("   (b) 軌跡內相鄰步 —— 一個動作的效果就落在這一步的轉移上")
    dv, dr = [], []
    for d in ds[::6]:
        for a, e in zip(d["bounds"][:-1], d["bounds"][1:]):
            dv.append(np.diff(d["v"][a:e])); dr.append(np.diff(d["fr"][a:e]))
    dv, dr = np.concatenate(dv), np.concatenate(dr)
    P(f"      ΔV vs ΔR  Pearson {pe(dv, dr):+.4f}  Spearman {sp(dv, dr):+.4f}"
      f"  同號 {100 * np.mean(np.sign(dv) == np.sign(dr)):.2f}%")
    P(f"      ΔV sd {dv.std():.4f}   ΔR sd {dr.std():.4f}")

    P("\n③ 分段（軌跡內第幾步；一局 720 步）")
    P(f"  {'段':<14}{'MAE':>9}{'RMSE':>9}{'bias':>10}{'r(V,R)':>10}"
      f"{'ρ(V,R)':>10}{'r(adv,R)':>11}{'ρ(adv,R)':>11}")
    for lab, lo, hi in (("early  0-239", 0, 240), ("mid  240-479", 240, 480),
                        ("late  480-719", 480, 720)):
        acc = []
        for d in ds:
            m = (d["t"] >= lo) & (d["t"] < hi)
            s = stats(d["v"][m], d["fr"][m])
            acc.append([s["mae"], s["rmse"], s["bias"], s["r"], s["rho"],
                        pe(d["adv"][m], d["fr"][m]), sp(d["adv"][m], d["fr"][m])])
        a = np.nanmean(np.array(acc), axis=0)
        P(f"  {lab:<14}{a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>+10.4f}{a[3]:>+10.4f}"
          f"{a[4]:>+10.4f}{a[5]:>+11.4f}{a[6]:>+11.4f}")

    P("\n⑤ critic error 的尺度 vs advantage 的尺度")
    advsd = np.array([d["adv"].std() for d in ds])
    errsd = np.array([(d["v"] - d["fr"]).std() for d in ds])
    errrmse = np.array([np.sqrt(((d["v"] - d["fr"]) ** 2).mean()) for d in ds])
    corr = np.array([pe(d["adv"], -(d["v"] - d["fr"])) for d in ds])
    P(f"  advantage sd {advsd.mean():.4f}   critic error (V−R) sd"
      f" {errsd.mean():.4f}   RMSE {errrmse.mean():.4f}")
    P(f"  corr(adv, −(V−R)) {corr.mean():+.4f}"
      f"   —— gamma=1、lam=0.998 之下 adv 本來就接近 R−V")
    bet, wit = [], []
    for d in ds:
        gm = np.array([d["adv"][d["traj"] == k].mean()
                       for k in range(int(d["traj"].max()) + 1)])
        bet.append(np.var(gm[d["traj"]]) / d["adv"].var())
        wit.append(np.var(d["adv"] - gm[d["traj"]]) / d["adv"].var())
    P(f"  Var(adv) 拆解：跨軌跡（哪一局）{100 * np.mean(bet):.1f}%"
      f"   軌跡內（局內哪一步）{100 * np.mean(wit):.1f}%")
    line = "  軌跡內 advantage 自相關： "
    for lag in (1, 5, 20, 100, 300):
        xs, ys = [], []
        for d in ds[::6]:
            for a, e in zip(d["bounds"][:-1], d["bounds"][1:]):
                s = d["adv"][a:e]
                if len(s) > lag:
                    xs.append(s[:-lag]); ys.append(s[lag:])
        line += f"lag{lag} {pe(np.concatenate(xs), np.concatenate(ys)):+.3f}  "
    P(line)
    rsd = np.array([d["rew"].std() for d in ds])
    P(f"  逐步 reward 的 sd {rsd.mean():.4f}"
      f"  （單一步的動作能直接影響的量級）"
      f"   vs advantage sd {advsd.mean():.4f}"
      f"   比 {advsd.mean() / rsd.mean():.1f}x")


def qty_section():
    P(f"\n{'#' * 78}\n#  ④ qty=1 vs qty>1（只當一類 state 查 critic 偏差，"
      f"不當 causal factor）\n{'#' * 78}")
    for tag in ("init", "last"):
        f = os.path.join(OUTDIR, f"qtyadv_{tag}.npz")
        if not os.path.exists(f):
            P(f"  {tag}: 找不到 {f}")
            continue
        z = np.load(f)
        q, adv, fr, v = z["q"], z["adv"].astype(float), z["fr"], z["val"]
        err = v - fr
        one, big = q == 1, q > 1
        P(f"\n  {tag}   PICKUP 決策 n={len(q):,}")
        P(f"  {'':<20}{'qty=1':>12}{'qty>1':>12}{'差':>12}")
        P(f"  {'n':<20}{int(one.sum()):>12,}{int(big.sum()):>12,}"
          f"{int(big.sum() - one.sum()):>12,}")
        for lab, a in (("未來實際報酬 R", fr), ("value 預測 V", v),
                       ("advantage", adv), ("value error V−R", err),
                       ("|V−R|", np.abs(err))):
            P(f"  {lab:<20}{a[one].mean():>+12.4f}{a[big].mean():>+12.4f}"
              f"{a[big].mean() - a[one].mean():>+12.4f}")


if __name__ == "__main__":
    for nm in ("qty-base", "qty-on"):
        section(nm, arm(nm))
    qty_section()
    OUT.close()
    print("done")
