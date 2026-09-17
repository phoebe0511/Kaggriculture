"""qty_push 的重複量測：一次 update 把 qty 分布推去哪，跑 5 對。

同一批診斷 rollout，只換 update 的 seed。每一對是：
  A  qty_factor=True   （qty 在 policy loss 裡）
  B  qty_factor=False  （對照：qty 不在 loss，等於訓練前的原狀）
A − B 就是 qty learning signal 自己的貢獻，B 自己是共用軀幹的漂移。

單次量測有噪音（CUDA backward 的 index_add_ / conv 用 atomics，不保證可重現），
所以要配對重複才看得出符號穩不穩。
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

DEV, N_ENVS, SEED0, REPS = "cuda", 32, 902_240, 5
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
OUT = open(os.path.join(OUTDIR, "qty_push_rep.txt"), "w", encoding="utf-8")


def P(*a):
    print(*a, file=OUT)
    OUT.flush()


def load(ckpt):
    b = torch.load(ckpt, map_location="cpu", weights_only=False)
    n = build_net(int(b["width"]), int(b["blocks"]))
    n.load_state_dict(b["state_dict"])
    return n.to(DEV).eval()


def qty_probs(net, batch, chunk=256):
    out = np.zeros((len(batch.op_idx), C.N_QTY), np.float32)
    net.eval()
    with torch.no_grad():
        for i in range(0, batch.n_steps, chunk):
            idx = np.arange(i, min(i + chunk, batch.n_steps))
            mb = batch._pack(idx, DEV)
            _o, ql, _t, _mp, _mq, _v, _d = net(
                mb["spatial"], mb["scalar"], mb["unit_board"],
                mb["unit_pos"], mb["unit_feats"])
            p = torch.softmax(ql, dim=-1).cpu().numpy()
            u0 = int(batch.unit_start[idx[0]])
            out[u0:u0 + len(p)] = p
    return out


def stat(pi, rows):
    p = pi[rows]
    mp = p.mean(0)
    qc = np.array(C.QTY_CHOICES, np.float64)
    return float(1 - mp[0]) * 100, float((p * qc).sum(1).mean())


def go(tag, ckpt):
    net0 = load(ckpt)
    opp, _ = load_league("config/params/cma5-g175.json")
    vec = VecRollout(net0, n_envs=N_ENVS, seed0=SEED0, device=DEV,
                     episode_steps=None, opponent=opp, phi="assets",
                     recognise="strict", greedy=False, qty_factor=True)
    _s, _c, trajs = vec.run(collect=True)
    batch = ppo.RolloutBatch(trajs, gamma=1.0, lam=0.998, zero_sum=True,
                             phi_weight=1e-4)
    fam = np.array([C.UNIT_OPS[int(i)][0] for i in batch.op_idx])
    rows = np.nonzero((fam == "PICKUP") & batch.qty_used)[0]
    g0, e0 = stat(qty_probs(net0, batch), rows)

    P(f"\n{'#' * 70}\n#  {tag}   {ckpt}")
    P(f"#  {batch.n_steps:,} 步 / PICKUP 決策 {len(rows):,}"
      f" / 更新前 P(qty>1) {g0:.3f}%  E[qty] {e0:.4f}\n{'#' * 70}")
    P(f"{'seed':>5} {'ΔP 有loss':>11} {'ΔP 對照':>10} {'淨 signal':>11}"
      f" {'ΔE 有loss':>11} {'ΔE 對照':>10} {'淨 signal':>11}")
    A, B, EA, EB = [], [], [], []
    for s in range(REPS):
        row = []
        for flag in (True, False):
            net = load(ckpt)
            opt = torch.optim.Adam(net.parameters(), lr=3e-5)
            net.train()
            ppo.update(net, opt, batch, epochs=8, minibatch=512, seed=s,
                       device=DEV, target_kl=0.02, ent_coef=0.0,
                       qty_factor=flag)
            g, e = stat(qty_probs(net, batch), rows)
            row.append((g - g0, e - e0))
        (da, ea), (db, eb) = row
        A.append(da); B.append(db); EA.append(ea); EB.append(eb)
        P(f"{s:>5} {da:>+11.3f} {db:>+10.3f} {da - db:>+11.3f}"
          f" {ea:>+11.4f} {eb:>+10.4f} {ea - eb:>+11.4f}")
    A, B, EA, EB = map(np.array, (A, B, EA, EB))

    def line(name, v):
        se = v.std(ddof=1) / len(v) ** 0.5
        P(f"  {name:<26} mean {v.mean():>+9.4f}  sd {v.std(ddof=1):>7.4f}"
          f"  SE {se:>7.4f}  t {v.mean() / (se + 1e-12):>+7.2f}")
    P("")
    line("ΔP(qty>1) 有 loss (pp)", A)
    line("ΔP(qty>1) 對照/軀幹 (pp)", B)
    line("ΔP(qty>1) 淨 signal (pp)", A - B)
    line("ΔE[qty] 有 loss", EA)
    line("ΔE[qty] 對照/軀幹", EB)
    line("ΔE[qty] 淨 signal", EA - EB)


if __name__ == "__main__":
    go("init  (PPO 前，兩臂共同起點)", "model/artifacts/phi-200/ckpt-00140.pt")
    go("last  (qty-on 60 輪後)", "model/artifacts/qty-on/last.pt")
    OUT.close()
