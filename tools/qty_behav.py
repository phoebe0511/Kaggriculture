"""行為側比較：用 greedy_eval 的同一套 protocol 重播兩臂的 last.pt。

protocol 跟 model/ppo_train.py:173 greedy_eval 一致：
  greedy=True, n_envs=20, seed0=900000(--eval-seed0 預設), opponent cma5-g175,
  base_policy=None, base_side="units", episode_steps=None(引擎預設 720)
每一臂帶自己的 qty_factor（qty 會改變送進引擎的動作，旗標要跟部署一致）。

🩸 為了拿到 `op_exec`（unit 有沒有站到目標格上、op 到底有沒有進引擎）這裡用
`collect=True`，但同時 `half_obs=False` —— `_policy_batch` 只在
`collect and half_obs` 都成立時才把 spatial 轉 float16 再轉回去。關掉之後
網路吃到的輸入跟 `greedy_eval`（collect=False）逐位元相同，動作不會變。
"""
from __future__ import annotations

import collections
import os
import sys
from pathlib import Path

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
from harness.ppo_rollout import VecRollout, build_net, load_league


class BehavProbe(VecRollout):
    """記錄每個 unit 這一步實際送出什麼，以及那個 op 有沒有真的進引擎。

    引擎的夾照 `kaggriculture.py:358-410` 重算（跟 `tools/qty_sanity.py`
    同一份，沒有改引擎）：
      PICKUP  n = min(要的, shed 現有量)
      PLACE   min(要的, 手上的量, shed 剩餘容量)
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows = []      # (op_index, want, got, op_exec)

    def _policy_batch(self, items, collect=False):
        out = super()._policy_batch(items, collect=collect)
        for (ei, p, obs, cfg), (_e, _p, action, rec) in zip(items, out):
            if not self._ours(ei, p):
                continue
            emitted = [action["farmer"]] + list(action["hands"] or [])
            exec_mask = rec.get("op_exec")
            priv = obs["private"]
            shed = dict(priv.get("shed") or {})
            for i, act in enumerate(emitted):
                ex = bool(exec_mask[i]) if exec_mask is not None else True
                if not isinstance(act, (list, tuple)) or not act:
                    continue
                enc = C.encode_unit_action(act)
                if enc is None:
                    continue
                op = act[0]
                want = int(act[2]) if len(act) >= 3 else 0
                got = want
                if op == "PICKUP":
                    got = min(want, int(shed.get(act[1], 0)))
                elif op == "PLACE":
                    inv = (priv.get("inventories") or [{}])
                    have = int((inv[i] if i < len(inv) else {}).get(act[1], 0))
                    room = max(0, 100 - sum(shed.values()))
                    got = min(want, have, room)
                self.rows.append((enc[0], want, max(got, 0), ex))
        return out


def run(ckpt, flag, games=20, seed0=900_000, device="cuda"):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = build_net(int(blob["width"]), int(blob["blocks"]))
    net.load_state_dict(blob["state_dict"])
    net = net.to(device).eval()
    opp, _ = load_league("config/params/cma5-g175.json")
    vec = BehavProbe(net, n_envs=games, seed0=seed0, device=device,
                     episode_steps=None, opponent=opp, base_policy=None,
                     greedy=True, base_side="units", qty_factor=flag,
                     half_obs=False)
    _steps, cash, _t = vec.run(collect=True)
    pairs = vec.our_cash(cash)
    ours = np.array([a for a, _ in pairs], float)
    theirs = np.array([b for _, b in pairs], float)
    return vec, ours, theirs


def summarise(vec, ours, theirs, games):
    rows = np.array(vec.rows, dtype=np.int64)
    ex = rows[:, 3].astype(bool)
    real = rows[ex]                       # 真的送進引擎的 op
    walk = rows[~ex]                      # step_toward，那一步的 op 沒進引擎
    fam = np.array([C.UNIT_OPS[int(i)][0] for i in real[:, 0]])
    return {"rows": rows, "real": real, "walk": walk, "fam": fam,
            "counts": collections.Counter(fam.tolist()),
            "margin": float((ours - theirs).mean()),
            "cash": float(ours.mean()),
            "opp": float(theirs.mean()),
            "win": float(np.mean(ours > theirs)),
            "games": games}


def report(tag, s):
    rows, real, fam, G = s["rows"], s["real"], s["fam"], s["games"]
    print(f"\n===== {tag} =====")
    print(f"  greedy 現金 {s['cash']:>10,.0f}   對手 {s['opp']:>10,.0f}"
          f"   margin {s['margin']:>10,.0f}   勝率 {s['win']:.2f}")
    print(f"  unit-step 合計 {len(rows):,}   走路(op 沒進引擎) {len(s['walk']):,}"
          f" ({100 * len(s['walk']) / len(rows):.2f}%)   真的送 op {len(real):,}")
    print("  op 家族（有進引擎的，每局平均）：")
    for name, n in sorted(s["counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:<20} {n:>7,}   {n / G:>8.1f}/局")
    for name in ("PICKUP", "PLACE"):
        m = fam == name
        if not m.any():
            continue
        w, g = real[m, 1], real[m, 2]
        c = collections.Counter(w.tolist())
        print(f"  {name}: n={int(m.sum()):,}  想要平均 {w.mean():.2f}  中位 "
              f"{np.median(w):.0f}  p90 {np.percentile(w, 90):.0f}  最大 {w.max()}"
              f"  P(qty>1) {100 * (w > 1).mean():.2f}%")
        print(f"       實際平均 {g.mean():.2f}  被夾 {100 * (g != w).mean():.2f}%"
              f"  夾成 0 {100 * (g == 0).mean():.2f}%"
              f"  搬運總量 {int(g.sum()):,}（{g.sum() / G:.1f}/局）")
        print(f"       分布 {sorted(c.items())}")


if __name__ == "__main__":
    G = 20
    arms = []
    for tag, ckpt, flag in (
            ("baseline  qty-base/last.pt (qty off)",
             "model/artifacts/qty-base/last.pt", False),
            ("treatment qty-on/last.pt   (qty on)",
             "model/artifacts/qty-on/last.pt", True)):
        vec, ours, theirs = run(ckpt, flag, games=G)
        s = summarise(vec, ours, theirs, G)
        report(tag, s)
        arms.append(s)

    a, b = arms
    print("\n===== 兩臂對照（每局平均）=====")
    keys = sorted(set(a["counts"]) | set(b["counts"]))
    print(f"  {'op':<20} {'baseline':>10} {'qty-on':>10} {'差':>10}")
    for k in keys:
        print(f"  {k:<20} {a['counts'].get(k, 0) / G:>10.1f}"
              f" {b['counts'].get(k, 0) / G:>10.1f}"
              f" {(b['counts'].get(k, 0) - a['counts'].get(k, 0)) / G:>10.1f}")
    for name in ("PICKUP", "PLACE"):
        ga = a["real"][a["fam"] == name][:, 2]
        gb = b["real"][b["fam"] == name][:, 2]
        print(f"  {name} 搬運總量/局 {ga.sum() / G:>12.1f} {gb.sum() / G:>10.1f}"
              f" {(gb.sum() - ga.sum()) / G:>10.1f}")
    print(f"  走路佔比 {100 * len(a['walk']) / len(a['rows']):>17.2f}%"
          f" {100 * len(b['walk']) / len(b['rows']):>9.2f}%")
    print(f"  greedy margin {a['margin']:>17,.0f} {b['margin']:>10,.0f}"
          f" {b['margin'] - a['margin']:>10,.0f}")
