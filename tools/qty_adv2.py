"""診斷：PPO 的 learning signal 有沒有在系統性壓低較大的 PICKUP quantity。

不改 code、不訓練。既有的 `--dump-batch` npz **沒有存 qty_idx**（只有 op_idx，
見 model/ppo_train.py:dump_batch），所以分不出數量 —— 這裡另收診斷 rollout：

  ckpt   init = model/artifacts/phi-200/ckpt-00140.pt（兩臂共同起點，PPO 前）
         last = model/artifacts/qty-on/last.pt（60 輪後）
  解碼   sampling（跟訓練 rollout 一致，不是 greedy）
  GAE/Φ  gamma 1.0, lam 0.998, zero_sum, phi=assets, phi_weight 1e-4（同訓練）
  對手   config/params/cma5-g175.json，32 envs，seed0 902240

🩸 advantage 的粒度是**一步**（整個聯合動作），不是一個 unit。
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = r"C:\_phoebe_priv\Kaggriculture"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")

from tools._quiet import silenced

with silenced():
    import torch

import contracts as C
from model import ppo
from harness.ppo_rollout import VecRollout, build_net, load_league

DEV = "cuda"
N_ENVS = 32
SEED0 = 902_240
OUT = open("qty_adv_report.txt", "w", encoding="utf-8")


def P(*a):
    print(*a, file=OUT)


def collect(ckpt):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = build_net(int(blob["width"]), int(blob["blocks"]))
    net.load_state_dict(blob["state_dict"])
    net = net.to(DEV).eval()
    opp, _ = load_league("config/params/cma5-g175.json")
    vec = VecRollout(net, n_envs=N_ENVS, seed0=SEED0, device=DEV,
                     episode_steps=None, opponent=opp, phi="assets",
                     recognise="strict", greedy=False, qty_factor=True)
    _s, _c, trajs = vec.run(collect=True)
    batch = ppo.RolloutBatch(trajs, gamma=1.0, lam=0.998, zero_sum=True,
                             phi_weight=1e-4)
    return net, batch


def qty_probs(net, batch, chunk=256):
    out = np.zeros((len(batch.op_idx), C.N_QTY), np.float32)
    with torch.no_grad():
        for i in range(0, batch.n_steps, chunk):
            idx = np.arange(i, min(i + chunk, batch.n_steps))
            mb = batch._pack(idx, DEV)
            _o, q_logits, _t, _mp, _mq, _v, _d = net(
                mb["spatial"], mb["scalar"], mb["unit_board"],
                mb["unit_pos"], mb["unit_feats"])
            p = torch.softmax(q_logits, dim=-1).cpu().numpy()
            u0 = int(batch.unit_start[idx[0]])
            out[u0:u0 + len(p)] = p
    return out


def future_return(batch):
    fr = np.zeros(batch.n_steps, np.float64)
    b = list(batch.traj_start) + [batch.n_steps]
    for a, z in zip(b[:-1], b[1:]):
        fr[a:z] = np.cumsum(batch.rew[a:z][::-1])[::-1]
    return fr


def step_meta(batch):
    t_in = np.zeros(batch.n_steps, np.int64)
    traj = np.zeros(batch.n_steps, np.int64)
    b = list(batch.traj_start) + [batch.n_steps]
    for k, (a, z) in enumerate(zip(b[:-1], b[1:])):
        t_in[a:z] = np.arange(z - a)
        traj[a:z] = k
    return t_in, traj, batch.traj_cash[:, 0] - batch.traj_cash[:, 1]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(tag, ckpt):
    net, batch = collect(ckpt)
    fam = np.array([C.UNIT_OPS[int(i)][0] for i in batch.op_idx])
    q = np.array(C.QTY_CHOICES, np.int64)[batch.qty_idx]
    step = batch.unit_step
    t_in, traj, margin = step_meta(batch)
    fr = future_return(batch)
    a_raw = batch.adv
    a_std = (a_raw - a_raw.mean()) / (a_raw.std() + 1e-8)
    pi = qty_probs(net, batch)

    P(f"\n{'#' * 74}\n#  {tag}   {ckpt}\n{'#' * 74}")
    P(f"批次 {batch.n_steps:,} 步 / {len(batch.op_idx):,} 個 unit 決策 / "
      f"{len(batch.traj_start)} 條軌跡；每步平均 {batch.unit_count.mean():.2f}"
      f" 個 unit")
    P(f"adv 原始 mean {a_raw.mean():+.4f} std {a_raw.std():.4f}；"
      f"期末零和差額 mean {margin.mean():+,.0f}")

    res = {}
    for OP in ("PICKUP", "PLACE"):
        m = (fam == OP) & batch.qty_used
        qs, st = q[m], step[m]
        av, avs, pp = a_raw[st], a_std[st], pi[m]
        P(f"\n① {OP} quantity × advantage   n={int(m.sum()):,}"
          f"（真的進引擎 {int((m & batch.op_exec).sum()):,}）"
          f"  平均 qty {qs.mean():.2f}  P(qty>1) {100 * (qs > 1).mean():.2f}%")
        P(f"{'qty':>4} {'n':>7} {'佔比':>8} {'mean adv':>10} {'median adv':>11}"
          f" {'mean adv(std)':>14} {'logit 一階漂移':>15}")
        for j, qq in enumerate(C.QTY_CHOICES):
            sel = qs == qq
            n = int(sel.sum())
            drift = float(avs[sel].sum() - (avs * pp[:, j]).sum())
            if n == 0:
                P(f"{qq:>4} {n:>7} {'-':>8} {'-':>10} {'-':>11} {'-':>14}"
                  f" {drift:>+15.1f}")
            else:
                P(f"{qq:>4} {n:>7} {100 * n / len(qs):>7.2f}% "
                  f"{av[sel].mean():>+10.4f} {np.median(av[sel]):>+11.4f} "
                  f"{avs[sel].mean():>+14.4f} {drift:>+15.1f}")
        P(f"  Spearman(qty,adv) {spearman(qs, av):+.4f}"
          f"   Pearson {float(np.corrcoef(qs, av)[0, 1]):+.4f}")
        res[OP] = (qs, st, av, avs, pp)

    qs, st, av, avs, pp = res["PICKUP"]
    P(f"\n② PICKUP 分段（軌跡內第幾步，一局 720 步）")
    t_unit = t_in[st]
    for name, lo, hi in (("early  0-239", 0, 240), ("mid  240-479", 240, 480),
                         ("late  480-719", 480, 720)):
        sm = (t_unit >= lo) & (t_unit < hi)
        qsg, avg = qs[sm], av[sm]
        P(f"\n  {name}  n={int(sm.sum()):,}  qty 平均 {qsg.mean():.2f}"
          f"  P(qty>1) {100 * (qsg > 1).mean():.2f}%"
          f"  Spearman(qty,adv) {spearman(qsg, avg):+.4f}"
          f"  mean adv(qty=1) {avg[qsg == 1].mean():+.4f}"
          f"  mean adv(qty>1) {avg[qsg > 1].mean():+.4f}")
        line = "    "
        for qq in C.QTY_CHOICES:
            s2 = qsg == qq
            if s2.sum():
                line += f"q{qq}:n={int(s2.sum())},adv={avg[s2].mean():+.3f}  "
        P(line)

    P(f"\n③ PICKUP qty=1 vs qty>1")
    one, big = qs == 1, qs > 1
    rows = [("n", float(one.sum()), float(big.sum()), "{:>14,.0f}"),
            ("mean advantage", av[one].mean(), av[big].mean(), "{:>+14.4f}"),
            ("median advantage", float(np.median(av[one])),
             float(np.median(av[big])), "{:>+14.4f}"),
            ("mean 未來實際報酬", fr[st][one].mean(), fr[st][big].mean(),
             "{:>+14.4f}"),
            ("mean value 預測", batch.old_value[st][one].mean(),
             batch.old_value[st][big].mean(), "{:>+14.4f}"),
            ("mean 期末零和差額", margin[traj[st][one]].mean(),
             margin[traj[st][big]].mean(), "{:>14,.0f}"),
            ("mean 軌跡內步數", t_in[st][one].mean(), t_in[st][big].mean(),
             "{:>14.1f}")]
    P(f"  {'':<22}{'qty=1':>14}{'qty>1':>14}{'差':>14}")
    for name, x, y, f in rows:
        P(f"  {name:<22}{f.format(x)}{f.format(y)}{f.format(y - x)}")

    P(f"\n④ 因果性檢查")
    cnt = batch.unit_count[st]
    npick = np.bincount(st, minlength=batch.n_steps)[st]
    P(f"  PICKUP 那一步平均 {cnt.mean():.2f} 個 unit 因子共用同一個 adv；"
      f"同一步的 PICKUP 平均 {npick.mean():.2f} 個"
      f"（佔那一步 unit 因子的 {100 * (npick / cnt).mean():.2f}%）")
    np.savez_compressed(
        f"qtyadv_{tag}.npz", q=qs, adv=av, advs=avs, t_in=t_in[st],
        fr=fr[st], val=batch.old_value[st], margin=margin[traj[st]], pi=pp)


if __name__ == "__main__":
    analyse("init", "model/artifacts/phi-200/ckpt-00140.pt")
    analyse("last", "model/artifacts/qty-on/last.pt")
    OUT.close()
    print("written qty_adv_report.txt")
