"""無 observation 的預測下限：只用「第幾步 + 目前為止的累積 reward」預測 MC 報酬。

🩸 為什麼不是 log 要的那個實驗：`--dump-batch` 只存
adv / ret / old_value / old_logp / new_logp / unit_step / op_idx / tgt_idx /
rew / traj_start / traj_cash（model/ppo_train.py:224），**沒有 observation**
（spatial / scalar / unit_pos / unit_feats）。434 KB / 11,504 步 = 38 bytes/步，
只有純量。所以「同架構的離線 critic」在現有資料上訓不了 —— 有 target、沒有 input。

這裡做的是最接近而且做得到的事：量一個**完全不看盤面**的基準能到多準。
線上 critic 高出這個基準多少，才是它真正從 observation 學到的部分。

target 一律用沿軌跡後加的 MC future reward（gamma=1），不用 GAE ret。
切分以**軌跡**為單位，不是逐 step random split。兩種切法都報：
  (a) 同一輪內 16 條軌跡切 12 訓練 / 4 驗證
  (b) 第 k 輪訓練 -> 第 k+1 輪驗證（跨 policy 位移，線上 critic 真正面對的情況）
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
OUT = open(os.path.join(OUTDIR, "critic_floor.txt"), "w", encoding="utf-8")


def P(*a):
    print(*a, file=OUT)
    OUT.flush()


def pe(x, y):
    s = np.std(x) * np.std(y)
    return float(np.mean((x - x.mean()) * (y - y.mean())) / s) if s else np.nan


def sp(x, y):
    return pe(np.argsort(np.argsort(x)).astype(float),
              np.argsort(np.argsort(y)).astype(float))


def load(path):
    z = np.load(path)
    v = z["old_value"].astype(np.float64)
    rew = z["rew"].astype(np.float64)
    b = list(z["traj_start"]) + [len(v)]
    fr = np.zeros(len(v)); t = np.zeros(len(v)); tj = np.zeros(len(v), np.int64)
    cum = np.zeros(len(v))
    for k, (a, e) in enumerate(zip(b[:-1], b[1:])):
        seg = rew[a:e]
        fr[a:e] = np.cumsum(seg[::-1])[::-1]
        cum[a:e] = np.cumsum(seg) - seg          # 到 t 為止（不含 t）
        t[a:e] = np.arange(e - a)
        tj[a:e] = k
    return dict(v=v, fr=fr, t=t, cum=cum, traj=tj, n_traj=len(b) - 1)


def feats(d):
    """只用時間與「到目前為止拿到多少」—— 兩者在 t 時刻都已知，沒有 leakage。
    R(t) = 總報酬 − cum(t)，但預測時看不到總報酬。"""
    tn = d["t"] / 720.0
    return np.column_stack([np.ones_like(tn), tn, tn ** 2, tn ** 3,
                            d["cum"], d["cum"] * tn])


def ridge(X, y, lam=1e-6):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def metrics(pred, y):
    e = pred - y
    return dict(mae=np.abs(e).mean(), rmse=float(np.sqrt((e ** 2).mean())),
                bias=e.mean(), r=pe(pred, y), rho=sp(pred, y),
                ev=1 - e.var() / y.var())


def seg_metrics(pred, y, t):
    out = {}
    for lab, lo, hi in (("early", 0, 240), ("mid", 240, 480),
                        ("late", 480, 720)):
        m = (t >= lo) & (t < hi)
        out[lab] = metrics(pred[m], y[m])
    return out


def run(name):
    ds = [load(f) for f in
          sorted(glob.glob(f"model/artifacts/{name}/batch-*.npz"))]
    P(f"\n{'#' * 78}\n#  {name}   {len(ds)} 輪 × {len(ds[0]['v']):,} 步"
      f" / 每輪 {ds[0]['n_traj']} 條軌跡\n{'#' * 78}")

    for split, lab in (("within", "(a) 同一輪內 16 條軌跡切 12 訓練 / 4 驗證"),
                       ("next", "(b) 第 k 輪訓練 -> 第 k+1 輪驗證（跨 policy 位移）")):
        acc_base, acc_on, segs_b, segs_o = [], [], [], []
        for i, d in enumerate(ds):
            if split == "within":
                va = np.isin(d["traj"], [3, 7, 11, 15])
                tr_d, va_d, va_m = d, d, va
            else:
                if i + 1 >= len(ds):
                    continue
                tr_d, va_d = d, ds[i + 1]
                va_m = np.ones(len(va_d["v"]), bool)
            trm = (~np.isin(tr_d["traj"], [3, 7, 11, 15])
                   if split == "within" else np.ones(len(tr_d["v"]), bool))
            w = ridge(feats(tr_d)[trm], tr_d["fr"][trm])
            pred = feats(va_d) @ w
            y, t = va_d["fr"][va_m], va_d["t"][va_m]
            acc_base.append(metrics(pred[va_m], y))
            acc_on.append(metrics(va_d["v"][va_m], y))
            segs_b.append(seg_metrics(pred[va_m], y, t))
            segs_o.append(seg_metrics(va_d["v"][va_m], y, t))

        def avg(rows, k):
            return float(np.nanmean([r[k] for r in rows]))

        P(f"\n{lab}   驗證 {len(acc_base)} 組")
        P(f"  {'':<26}{'MAE':>9}{'RMSE':>9}{'bias':>10}{'Pearson':>10}"
          f"{'Spearman':>10}{'expl.var':>10}")
        for tag, rows in (("無 observation 基準", acc_base),
                          ("線上 critic（同批驗證步）", acc_on)):
            P(f"  {tag:<26}{avg(rows, 'mae'):>9.4f}{avg(rows, 'rmse'):>9.4f}"
              f"{avg(rows, 'bias'):>+10.4f}{avg(rows, 'r'):>+10.4f}"
              f"{avg(rows, 'rho'):>+10.4f}{avg(rows, 'ev'):>+10.4f}")
        P(f"  {'差（線上 − 基準）':<26}"
          f"{avg(acc_on, 'mae') - avg(acc_base, 'mae'):>+9.4f}"
          f"{avg(acc_on, 'rmse') - avg(acc_base, 'rmse'):>+9.4f}"
          f"{'':>10}{avg(acc_on, 'r') - avg(acc_base, 'r'):>+10.4f}"
          f"{avg(acc_on, 'rho') - avg(acc_base, 'rho'):>+10.4f}"
          f"{avg(acc_on, 'ev') - avg(acc_base, 'ev'):>+10.4f}")

        P(f"  分段 RMSE / Pearson")
        P(f"  {'':<26}{'early':>18}{'mid':>18}{'late':>18}")
        for tag, rows in (("無 observation 基準", segs_b),
                          ("線上 critic", segs_o)):
            cells = ""
            for lab2 in ("early", "mid", "late"):
                rm = float(np.nanmean([r[lab2]["rmse"] for r in rows]))
                pr = float(np.nanmean([r[lab2]["r"] for r in rows]))
                cells += f"{rm:>10.4f}/{pr:>+7.4f}"
            P(f"  {tag:<26}{cells}")


if __name__ == "__main__":
    P("target = 沿軌跡後加的 MC future reward（gamma=1），不是 GAE ret。")
    P("切分以軌跡為單位。無 observation 基準的特徵："
      "[1, t, t², t³, cum_rew(t), cum_rew(t)·t]，全部在 t 時刻可見。")
    for nm in ("qty-base", "qty-on"):
        run(nm)
    OUT.close()
    print("done")
