"""離線 search 的第一個 gate：**單回合到底有多少 headroom？**

    python -m tools.search_probe --weights model/weights-e2e-round6.npz

## 在問什麼

journal 08-24 §5 記著「改一個回合的動作 oracle 上限只有 +2.42%」，`docs/CLAUDE.md`
也照抄了那句。但那個數字是在 **8 個候選**上量的
（[value_probe.py:118-120](tools/value_probe.py#L118)）：不改、6 個「把單一 unit
換成它的**第 2 名**」、1 個強迫買種子。而聯合動作空間是 13 units × 44 個
`UNIT_OPS` × 目標格 + 市場訂單。

這支把候選擴大到約 60 個，並且**把基準線從網路的 argmax 換成 `gen0` 自己的動作**
—— 後者才是我們真正想超越的對象，也才量得到「網路和 gen0 意見不同時誰對」。

## 跟 value_probe 的差別

| | value_probe（08-21） | 這支 |
|---|---|---|
| 基準線 | 網路 argmax | **`gen0` 的動作** |
| 候選數 | 8 | **約 57** |
| 評估 | value head（已證明沒用）+ 打到底 | **只打到底** |
| rollout policy | 網路（3.54 ms/act） | **`gen0`（0.46 ms/act）** |

value head 完全不用 —— 08-24 已經量到三個 horizon 全部比不搜差，那條路關了。

## 判讀

`oracle = max(Q) / Q(gen0 的動作) − 1`。

    < 3%   跟 08-21 一樣，單回合真的封死 -> 停
    3~8%   有一點，要看 Stage 2 累積起來夠不夠
    > 8%   有明顯 headroom -> 進 Stage 2

還會印**哪一類候選貢獻最多**（單 unit / 多 unit / 市場 / 網路的完整意見），
那決定 Stage 2 要把預算花在哪一類上。

🩸 盤面一律取 **hour 12**。hour 0 的 `hands` 是空的，候選會全部一樣
（value_probe 的 docstring 踩過這個）。
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup(weights):
    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    os.environ["KAGGRI_WEIGHTS"] = weights
    os.environ.pop("KAGGRI_LOG_FILE", None)
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
    return make


def _kind(label):
    """候選標籤 -> 類別，用來看哪一類在貢獻。"""
    if label in ("gen0", "net", "net-units"):
        return label
    if label.startswith("mk-"):
        return "market"
    if label.startswith("u"):
        return "single"
    return "multi"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="model/weights-e2e-round6.npz")
    ap.add_argument("--seeds", default="7,21,33")
    ap.add_argument("--days", default="6,12,18,24")
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--opponent", default="gen0",
                    help="gen0 或 config/opponents/<name>.json 的名字")
    ap.add_argument("--rng-seed", type=int, default=20260825)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    make = _setup(args.weights)
    import numpy as np

    from agents.gen0 import act as gen0_act
    from search import candidates as CAND
    from search import rollout_search as RS

    if args.opponent == "gen0":
        def act_them(obs):
            return gen0_act(obs, cfg)
    else:
        from eval.runner import build_agent, load_spec
        _opp = build_agent(load_spec(args.opponent))

        def act_them(obs):
            return _opp(obs, cfg)

    def rollout_act(obs):
        return gen0_act(obs, cfg)

    rng = np.random.default_rng(args.rng_seed)
    seeds = [int(s) for s in args.seeds.split(",")]
    days = [int(d) for d in args.days.split(",")]

    rows = []
    t_all = time.perf_counter()
    print(f"權重 {args.weights}   對手 {args.opponent}   "
          f"rollout policy gen0\n")

    for seed in seeds:
        for day in days:
            with contextlib.redirect_stderr(io.StringIO()):
                env = make("kaggriculture", configuration={"seed": seed},
                           debug=False)
            cfg = env.configuration

            # 用 gen0 推到取樣點。基準線是 gen0，所以走到那裡的也該是 gen0。
            for _ in range(day * 24 + args.hour):
                env.step([gen0_act(env.state[0].observation, cfg),
                          act_them(env.state[1].observation)])

            obs = env.state[0].observation
            base = gen0_act(obs, cfg)
            t_snap = time.perf_counter()
            snap = RS.snapshot(env)
            snap_ms = (time.perf_counter() - t_snap) * 1000

            cands, blocked, _info = CAND.generate(obs, cfg, base, rng)
            t0 = time.perf_counter()
            scored = RS.evaluate_all(env, snap, cands, act_them, rollout_act)
            wall = time.perf_counter() - t0

            by_label = dict(scored)
            baseline = by_label["gen0"]
            best_label, best = max(scored, key=lambda kv: kv[1])
            rows.append({
                "seed": seed, "day": day, "baseline": baseline,
                "best": best, "best_label": best_label,
                "n_cands": len(cands), "blocked": blocked,
                "snapshot_ms": snap_ms, "wall_secs": wall,
                "scored": scored,
            })
            # 🩸 這個遊戲對小改動是崎嶇的（08-25 量到相鄰參數的逐 seed 相關
            # 係數 0.083）。如果候選分數散得很開而 gen0 落在中間，「取最大」
            # 的意義就要重新解讀 —— 所以把分布也印出來，不要只看 oracle。
            vals = sorted((c for _l, c in scored), reverse=True)
            rank = sum(1 for v in vals if v > baseline) + 1
            rows[-1].update({"rank_of_gen0": rank,
                             "n_better": rank - 1,
                             "spread_sd": statistics.stdev(vals),
                             "median": statistics.median(vals)})
            gain = best / baseline - 1 if baseline else float("nan")
            print(f"  seed {seed:>3} day {day:>2}   基準 {baseline:>9,.0f}   "
                  f"最佳 {best:>9,.0f}  {gain:>+7.2%}  ({best_label})   "
                  f"gen0 排 {rank}/{len(vals)}   "
                  f"候選中位 {statistics.median(vals):>9,.0f}   "
                  f"sd {statistics.stdev(vals):>8,.0f}   {wall:.0f}s")

    print(f"\n{len(rows)} 個盤面，{time.perf_counter() - t_all:.0f} 秒")

    base_mean = statistics.fmean(r["baseline"] for r in rows)
    best_mean = statistics.fmean(r["best"] for r in rows)
    gains = [r["best"] / r["baseline"] - 1 for r in rows if r["baseline"]]
    print(f"\n  gen0 的動作  平均 {base_mean:>10,.0f}")
    print(f"  oracle       平均 {best_mean:>10,.0f}   "
          f"+{best_mean / base_mean - 1:.2%}   "
          f"（逐盤面中位 +{statistics.median(gains):.2%}）")
    print(f"  每個盤面都贏？ {sum(1 for g in gains if g > 0)}/{len(gains)}")

    # gen0 在候選裡的位置。排很前面 = 它的動作本來就好，只有少數能贏它；
    # 排中間 = 候選之間的差主要是軌跡發散，不是「這步比較高明」。
    ranks = [r["rank_of_gen0"] for r in rows]
    n_better = [r["n_better"] for r in rows]
    med_gap = [r["baseline"] / r["median"] - 1 for r in rows if r["median"]]
    print(f"  gen0 在候選裡的名次   中位 {statistics.median(ranks):.0f}/"
          f"{statistics.median(r['n_cands'] for r in rows):.0f}"
          f"   （贏過 gen0 的候選數 中位 {statistics.median(n_better):.0f}）")
    print(f"  gen0 vs 候選中位數    {statistics.fmean(med_gap):+.2%}"
          f"   （>0 = gen0 比隨便挑一個好，這步不是隨機的）")
    print(f"  候選分數的 sd（逐盤面中位） "
          f"{statistics.median(r['spread_sd'] for r in rows):>10,.0f}")

    # 哪一類候選在貢獻
    wins = collections.Counter(_kind(r["best_label"]) for r in rows)
    print("\n  最佳候選的類別分布：")
    for kind, n in wins.most_common():
        print(f"    {kind:10} {n:>3}/{len(rows)}")

    # 各類別單獨能拿到多少（把該類別當成唯一可選的候選）
    print("\n  只用某一類候選的話，oracle 是多少：")
    for kind in ("single", "multi", "market", "net", "net-units"):
        vals = []
        for r in rows:
            same = [c for label, c in r["scored"] if _kind(label) == kind]
            vals.append(max(same) if same else r["baseline"])
        if vals:
            m = statistics.fmean(vals)
            print(f"    {kind:10} {m:>10,.0f}   +{m / base_mean - 1:>6.2%}")

    print("\n=== Gate ===")
    oracle = best_mean / base_mean - 1
    if oracle > 0.08:
        print(f"✅ oracle +{oracle:.2%} > 8% —— 有明顯 headroom，進 Stage 2。")
    elif oracle > 0.03:
        print(f"⚠️ oracle +{oracle:.2%}（3~8%）—— 有一點，但要看 Stage 2 "
              f"在 60 個決策點累積起來夠不夠。")
    else:
        print(f"❌ oracle +{oracle:.2%} < 3% —— 跟 08-21 的 8 個候選一樣，"
              f"單回合真的封死。停下來重想形式。")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"argv": vars(args), "rows": rows}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n逐盤面結果寫到 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
