"""value head 排得動候選動作嗎？—— Phase 2 開工前的 gate。

    python -m tools.value_probe --weights model/weights-e2e-round5.npz

## 在測什麼

search 的核心假設是「產一堆候選 -> 用 value 挑最好的」。這支直接驗那個假設：

1. 把一局推到某一天的 hour 12，**存快照**
2. 產 K 個候選聯合動作（把某個 unit 換成它第 2 名的合法動作、強迫下買種子單）
3. 每個候選走一步之後，在 H = 1 / 8 / 24 讀 value
4. 然後**把那局真的打完**，拿期末現金當答案
5. 比三件事：prior argmax、用 value 挑的、oracle（事後最好的）

## 2026-08-21 的結果（round5 權重，12 個盤面）

    prior argmax  79,561
    H=1           78,485   -1.35%   贏 5 輸 7
    H=8           78,422   -1.43%   贏 7 輸 5
    H=24          77,989   -1.98%   贏 3 輸 7
    oracle        81,644   +2.62%

🚫 **三個 horizon 都比不搜還差。** 候選之間的真實差距約 2,000~5,000，
而 value 的 RMSE 是 $12,394（`tools/prior_eval.py` 量的）—— 訊噪比小於 1。

⚠️ 而且 oracle 只有 +2.62%：**改一個回合的動作，上限就是這麼多。**
2026-08-21 量到我們離 gen1 差 15.6%，而那個差距是「day 12 之後種子持續買少
17.5%」造成的 —— 單回合 rerank 修不了系統性的偏差。

## 前提（這支順便驗過）

引擎的 forward model **是決定性的**：同一個快照 + 同一個 policy 跑三次，
期末現金完全一樣。回溯的做法是 `deepcopy(env.state)` + 砍掉 `env.steps`
多出來的部分 + 把 `status` 設回 `ACTIVE`（`env.done` 是唯讀 property）。
`env.clone()` 要 13.75 ms，`deepcopy(env.state)` 只要 2.14 ms。

🩸 **快照不要取在 hour 0** —— 那時 `hands` 是空的（`engine-notes.md` §1），
「換第 N 個 unit 的動作」會全部變成同一個候選。第一次跑就踩到。
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup(weights):
    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    os.environ["KAGGRI_WEIGHTS"] = weights
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
    return make


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="model/weights-e2e-round5.npz")
    ap.add_argument("--seeds", default="7,21,33")
    ap.add_argument("--days", default="12,16,20,24")
    ap.add_argument("--horizons", default="1,8,24")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    make = _setup(args.weights)
    import contracts as C
    from agents import gen2_model as G
    from agents.gen0 import act as gen0_act

    def heads(obs, cfg):
        sp, sc = C.encode(obs, cfg)
        pos, uf = C.encode_units(obs, cfg)
        return G._policy()(sp, sc, pos, uf)

    def value_of(obs, cfg):
        return float(np.ravel(heads(obs, cfg)[5])[0]) * 1e5

    seed_ops = [j for j, (op, _i) in enumerate(C.MARKET_OPS) if op == "BUY_SEED"]

    def build(obs, cfg, swap_unit=None, force_market=None):
        op_l, qty_l, _t, mk, mkq, _v, _d = heads(obs, cfg)
        mask = C.legal_unit_mask(obs, cfg)
        units = G._choose(op_l, qty_l, mask, obs)
        if swap_unit is not None and swap_unit < len(units):
            s = np.where(mask[swap_unit], op_l[swap_unit], -np.inf)
            order = np.argsort(-s)
            if len(order) > 1 and np.isfinite(s[order[1]]):
                units[swap_unit] = C.decode_unit(
                    int(order[1]), int(np.argmax(qty_l[swap_unit])))
        th = G.market_thresholds()
        if force_market is not None:
            th = th.copy()
            th[list(force_market)] = -99.0
        return {"farmer": units[0], "hands": units[1:],
                "market": C.decode_market_orders(mk, mkq, obs, cfg, threshold=th)}

    cands = ([("argmax", {})]
             + [(f"u{k}", {"swap_unit": k}) for k in range(6)]
             + [("seed", {"force_market": seed_ops})])
    hs = tuple(int(h) for h in args.horizons.split(","))
    rows, t0 = [], time.perf_counter()
    for seed in (int(s) for s in args.seeds.split(",")):
        for day in (int(d) for d in args.days.split(",")):
            env = make("kaggriculture", configuration={"seed": seed}, debug=False)
            cfg = env.configuration
            # 🩸 +12 = hour 12。hour 0 的 hands 是空的，候選會全部一樣。
            for _ in range(day * 24 + 12):
                env.step([G.act(env.state[0].observation, cfg),
                          gen0_act(env.state[1].observation, cfg)])
            snap, nst = copy.deepcopy(env.state), len(env.steps)
            vals, true = {h: [] for h in hs}, []
            for _name, kw in cands:
                env.state = copy.deepcopy(snap)
                del env.steps[nst:]
                for s in env.state:
                    s.status = "ACTIVE"
                env.step([build(env.state[0].observation, cfg, **kw),
                          gen0_act(env.state[1].observation, cfg)])
                for h in range(1, max(hs) + 1):
                    if h in hs:
                        vals[h].append(value_of(env.state[0].observation, cfg))
                    if h < max(hs):
                        env.step([G.act(env.state[0].observation, cfg),
                                  gen0_act(env.state[1].observation, cfg)])
                while env.state[0].observation["day"] < 30 and not env.done:
                    env.step([G.act(env.state[0].observation, cfg),
                              gen0_act(env.state[1].observation, cfg)])
                true.append(env.state[0].observation["farms"][0]["money"])
            rows.append({"true": true, **{f"v{h}": vals[h] for h in hs}})
            print(f"  seed {seed} day {day}  argmax {true[0]:,.0f}  "
                  f"best {max(true):,.0f}  worst {min(true):,.0f}", flush=True)

    print(f"\n{len(rows)} 個盤面，{time.perf_counter() - t0:.0f} 秒")
    base = np.array([r["true"][0] for r in rows])
    best = np.array([max(r["true"]) for r in rows])
    print(f"  prior argmax 平均 {base.mean():,.0f}")
    print(f"  oracle       平均 {best.mean():,.0f}   "
          f"（+{best.mean() / base.mean() - 1:.2%} = 改一個回合的上限）")
    for h in hs:
        pick = np.array([r["true"][int(np.argmax(r[f"v{h}"]))] for r in rows])
        win = int((pick > base).sum())
        lose = int((pick < base).sum())
        print(f"  H={h:<3} value 挑的 平均 {pick.mean():,.0f}   "
              f"{pick.mean() / base.mean() - 1:+.2%}   "
              f"贏 {win} 平 {len(rows) - win - lose} 輸 {lose}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
