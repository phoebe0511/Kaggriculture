"""搜「多回合的承諾」有沒有搞頭？—— Phase 2 第二版的 gate。

    python -m tools.commit_probe --weights model/weights-e2e-round5.npz

## 為什麼要換這個問法

`tools/value_probe.py` 量到：改**一個回合**的動作，事後最好的候選也只比
prior argmax 好 **+2.62%**，而用 value head 去挑反而是 -1.4%。
我們離 gen1 差 15.6%，而那個差距是「day 12 之後種子持續買少 17.5%」造成的
系統性偏差（journal §15）—— 單回合 rerank 修不了。

所以改問：**如果一次決定「接下來 24 個回合的採購傾向」，上限是多少？**

## 做法

在 day D 的 hour 0 存快照，對每個候選：

1. 套用該候選的 market 門檻，跑 `--hold` 個回合（預設 24 = 一整天）
2. 之後**恢復預設門檻**，用同一支 policy 打到最後
3. 記期末現金

評估器是**打到底的真實結果，不是 value head** —— day 級決策點只有 18 個，
一次 rollout 從 day 12 起約 3.1 秒，付得起。這樣就把 value 的雜訊完全排除。

## 讀這份輸出

看 `oracle` 那一欄相對 baseline 的百分比。那是「每天都選對」的上限，
不是可達成的分數（真的要達成得有一個選得準的準則）。
**上限如果還是只有 3% 左右，這條路跟單回合一樣不通。**
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="model/weights-e2e-round5.npz")
    ap.add_argument("--seeds", default="7,21,33")
    ap.add_argument("--days", default="12,15,18,21,24")
    ap.add_argument("--hold", type=int, default=24,
                    help="承諾維持幾個回合（24 = 一整天）")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    os.environ["KAGGRI_WEIGHTS"] = args.weights
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
    import contracts as C
    from agents import gen2_model as G
    from agents.gen0 import act as gen0_act

    def ops_of(*names):
        return [j for j, (op, _i) in enumerate(C.MARKET_OPS) if op in names]

    SEED_OPS = ops_of("BUY_SEED")
    BUY_OPS = ops_of("BUY_SEED", "BUY_PRODUCT")
    SELL_OPS = ops_of("SELL")
    ANIMAL_OPS = ops_of("BUY_ANIMAL")

    def th(**kw):
        """`{op 群: 門檻}` -> `[N_MARKET_OPS]`。沒提到的維持 0。"""
        t = np.zeros(C.N_MARKET_OPS)
        for group, v in kw.items():
            t[{"seed": SEED_OPS, "buy": BUY_OPS,
               "sell": SELL_OPS, "animal": ANIMAL_OPS}[group]] = v
        return t

    VARIANTS = [
        ("預設", None),
        ("種子 -1.0", th(seed=-1.0)),
        ("種子 -2.0", th(seed=-2.0)),
        ("種子+飼料 -1.5", th(buy=-1.5)),
        ("種子 +1.0（少買）", th(seed=+1.0)),
        ("賣 -1.5（多賣）", th(sell=-1.5)),
        ("動物 -1.5", th(animal=-1.5)),
    ]

    def policy(thresholds):
        p = {} if thresholds is None else {"market_threshold_ops": thresholds}
        return lambda obs, cfg: G.act(obs, cfg, p)

    base_pol = policy(None)
    rows, t0 = [], time.perf_counter()
    for seed in (int(s) for s in args.seeds.split(",")):
        for day in (int(d) for d in args.days.split(",")):
            env = make("kaggriculture", configuration={"seed": seed}, debug=False)
            cfg = env.configuration
            for _ in range(day * 24):
                env.step([base_pol(env.state[0].observation, cfg),
                          gen0_act(env.state[1].observation, cfg)])
            snap, nst = copy.deepcopy(env.state), len(env.steps)
            finals = []
            for _name, t in VARIANTS:
                env.state = copy.deepcopy(snap)
                del env.steps[nst:]
                for s in env.state:
                    s.status = "ACTIVE"
                pol = policy(t)
                for _ in range(args.hold):
                    if env.done:
                        break
                    env.step([pol(env.state[0].observation, cfg),
                              gen0_act(env.state[1].observation, cfg)])
                while env.state[0].observation["day"] < 30 and not env.done:
                    env.step([base_pol(env.state[0].observation, cfg),
                              gen0_act(env.state[1].observation, cfg)])
                finals.append(env.state[0].observation["farms"][0]["money"])
            rows.append({"seed": seed, "day": day, "finals": finals})
            b, best = finals[0], max(finals)
            print(f"  seed {seed} day {day:>2}  預設 {b:>9,.0f}  "
                  f"最好 {best:>9,.0f} ({VARIANTS[int(np.argmax(finals))][0]})  "
                  f"{best / b - 1:+.2%}", flush=True)

    print(f"\n{len(rows)} 個決策點，{time.perf_counter() - t0:.0f} 秒   "
          f"承諾維持 {args.hold} 回合")
    base = np.array([r["finals"][0] for r in rows])
    best = np.array([max(r["finals"]) for r in rows])
    print(f"  預設（不搜）  {base.mean():>10,.0f}")
    print(f"  oracle        {best.mean():>10,.0f}   "
          f"**{best.mean() / base.mean() - 1:+.2%}** = 每天都選對的上限")
    print(f"\n  每個候選單獨用（整段都套同一個）相對預設：")
    for i, (name, _t) in enumerate(VARIANTS):
        col = np.array([r["finals"][i] for r in rows])
        win = int((col > base).sum())
        print(f"    {name:<18}{col.mean():>10,.0f}   {col.mean()/base.mean()-1:+.2%}"
              f"   贏 {win}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
