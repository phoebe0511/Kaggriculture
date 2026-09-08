"""qty intervention 的 sanity check。跑 PPO 之前先過這一關。

    python -m tools.qty_sanity --games 4

1  qty head 拿到梯度了嗎（grad norm、參數變動）
2  rollout 的 logprob vs update 重算的 logprob
3  quantity round-trip：policy 選的數量 == 送進引擎的數量
4  引擎會夾多少（sampled vs executed，分 op）
5  動作頻率：PICKUP 的數量分布，對照固定 qty=1 的 baseline

🩸 旗標會改變**取樣**（多抽一次 qty），所以兩臂的軌跡本來就不同 —— 這一支的
比較是「同一個 checkpoint、同一組 seed、開關旗標」，不是逐位元比對。
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")

from tools._quiet import silenced                       # noqa: E402

with silenced():
    import torch

import contracts as C                                   # noqa: E402
from model import ppo                                   # noqa: E402
from harness.ppo_rollout import (QTY_OPS, VecRollout,    # noqa: E402
                                 build_net, load_league)


class QtyProbe(VecRollout):
    """每一步把「policy 選的數量」跟「引擎真正吃到的數量」對起來。

    引擎的規則（`kaggriculture.py:358-410`）：
      PICKUP  n = min(要的, shed 現有量)
      PLACE   放動物固定 1（忽略第三個元素）；放回 shed 時
              min(要的, 手上的量, shed 剩餘容量)
    這裡照那三條重算一次，**不改引擎**，只是把夾了多少量出來。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows = []          # (op_index, want, got)

    def _policy_batch(self, items, collect=False):
        out = super()._policy_batch(items, collect=collect)
        for (ei, p, obs, cfg), (_e, _p, action, rec) in zip(items, out):
            emitted = [action["farmer"]] + list(action["hands"] or [])
            priv = obs["private"]
            player = int(obs["player"])
            farm = obs["farms"][player]
            shed = dict(priv.get("shed") or {})
            for i, act in enumerate(emitted):
                if not isinstance(act, (list, tuple)) or len(act) < 3:
                    continue
                op, item, want = act[0], act[1], int(act[2])
                enc = C.encode_unit_action(act)
                if enc is None:
                    continue
                if op == "PICKUP":
                    got = min(want, int(shed.get(item, 0)))
                elif op == "PLACE":
                    inv = (priv.get("inventories") or [{}])
                    have = int((inv[i] if i < len(inv) else {}).get(item, 0))
                    room = max(0, 100 - sum(shed.values()))
                    got = min(want, have, room)
                else:
                    continue
                self.rows.append((enc[0], want, max(got, 0)))
        return out


def roll(net, args, flag):
    opp, _ = load_league(args.opponent)
    vec = QtyProbe(net, n_envs=args.games, seed0=args.seed0,
                   device=args.device, opponent=opp, phi="assets",
                   recognise="strict", qty_factor=flag)
    _s, _c, trajs = vec.run(collect=True)
    batch = ppo.RolloutBatch(trajs, gamma=1.0, lam=0.998, zero_sum=True,
                             phi_weight=1e-4)
    return vec, batch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default="model/artifacts/phi-200/ckpt-00140.pt")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=902_240)
    ap.add_argument("--opponent", default="config/params/cma5-g175.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    def fresh():
        n = build_net(int(blob["width"]), int(blob["blocks"]))
        n.load_state_dict(blob["state_dict"])
        return n.to(args.device).eval()

    print("== 現有的 quantity representation（沒有新增任何 mapping）==")
    print(f"  QTY_CHOICES {C.QTY_CHOICES}   N_QTY {C.N_QTY}")
    print(f"  帶數量的 op：{int(QTY_OPS.sum())} / {C.N_UNIT_OPS}"
          f"（PICKUP 12 + PLACE 12），PLANT 與 15 個無參數 op 不帶")

    vb, batch_b = roll(fresh(), args, False)
    vq, batch_q = roll(fresh(), args, True)

    # ---- 1  qty head 的梯度 ----
    print("\n== 1  qty head 有沒有梯度 ==")
    for tag, batch, flag in (("baseline", batch_b, False),
                             ("qty-factor", batch_q, True)):
        net = fresh()
        before = {k: v.detach().clone() for k, v in net.state_dict().items()}
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        ppo.update(net, opt, batch, epochs=1, minibatch=256, seed=0,
                   device=args.device, target_kl=0.0, qty_factor=flag)
        g, d = [], []
        for name, prm in net.named_parameters():
            if name.startswith("qty_out"):
                g.append(0.0 if prm.grad is None else float(prm.grad.norm()))
                d.append((prm.detach() - before[name]).abs())
        gn = float(np.sqrt(sum(x * x for x in g)))
        mx = float(max(x.max() for x in d))
        mean = float(np.mean([float(x.mean()) for x in d]))
        print(f"  {tag:<12}qty_out grad norm {gn:.6e}   "
              f"參數 max Δ {mx:.6e}   mean Δ {mean:.6e}")

    # ---- 2  logprob 一致性 ----
    print("\n== 2  rollout 的 logprob vs update 重算 ==")
    for tag, batch, flag in (("baseline", batch_b, False),
                             ("qty-factor", batch_q, True)):
        lp = ppo.all_logp(fresh(), batch, args.device, qty_factor=flag)
        d = np.abs(lp - batch.old_logp)
        print(f"  {tag:<12}max |Δ| {d.max():.3e}   mean {d.mean():.3e}"
              f"   （float16 觀測的量級是 1e-5）")

    # ---- 3  round-trip ----
    print("\n== 3  quantity round-trip：policy 選的 == 送進引擎的 ==")
    bad = []
    for qi in range(C.N_QTY):
        for op_i in np.nonzero(QTY_OPS)[0][:2]:
            act = C.decode_unit(int(op_i), qi)
            if len(act) < 3 or int(act[2]) != C.QTY_CHOICES[qi]:
                bad.append((int(op_i), qi, act))
    print(f"  12 個數量桶 × 2 個 op 檢查 decode_unit："
          f"{'全部一致' if not bad else bad[:3]}")
    used = batch_q.qty_used.astype(bool)
    print(f"  這批 rollout：qty_used 的比例 {100 * used.mean():.2f}%"
          f"（= 取樣到 PICKUP / PLACE 的比例）")
    print(f"  baseline 的 qty_used 比例 "
          f"{100 * batch_b.qty_used.astype(bool).mean():.2f}%（應該是 0）")

    # ---- 4  引擎夾多少 ----
    print("\n== 4  sampled vs executed（引擎自己的夾，沒有改它）==")
    for tag, vec in (("baseline", vb), ("qty-factor", vq)):
        rows = np.array(vec.rows, dtype=np.int64) if vec.rows else np.zeros(
            (0, 3), np.int64)
        if not len(rows):
            print(f"  {tag}: 這批沒有 PICKUP / PLACE")
            continue
        fam = np.array([C.UNIT_OPS[int(i)][0] for i in rows[:, 0]])
        print(f"  {tag}")
        for name in sorted(set(fam.tolist())):
            m = fam == name
            w, g = rows[m, 1], rows[m, 2]
            print(f"    {name:<8}n={int(m.sum()):>6,}  想要平均 {w.mean():>5.2f}"
                  f"  實際平均 {g.mean():>5.2f}  被夾 "
                  f"{100 * (g != w).mean():>5.2f}%  夾成 0 "
                  f"{100 * (g == 0).mean():>5.2f}%")

    # ---- 5  PICKUP 的數量分布 ----
    print("\n== 5  PICKUP 的數量分布 ==")
    for tag, vec in (("baseline", vb), ("qty-factor", vq)):
        rows = np.array(vec.rows, dtype=np.int64) if vec.rows else np.zeros(
            (0, 3), np.int64)
        if not len(rows):
            continue
        fam = np.array([C.UNIT_OPS[int(i)][0] for i in rows[:, 0]])
        w = rows[fam == "PICKUP", 1]
        if not len(w):
            continue
        c = collections.Counter(w.tolist())
        print(f"  {tag:<12}n={len(w):,}  平均 {w.mean():.2f}  中位 "
              f"{np.median(w):.0f}  p90 {np.percentile(w, 90):.0f}  "
              f"最大 {w.max()}  P(qty>1) {100 * (w > 1).mean():.2f}%")
        print(f"    分布 {sorted(c.items())[:12]}")
    print("  對照：老師 cma1-g50-wt（data/dagger/cma1-round4，5,411 筆）"
          "平均 6.86、中位 9、qty=1 佔 17.8%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
