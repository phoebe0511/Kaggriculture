"""CMA-ES 開工前的 gate：每次評估要幾個 seed？

    python -m tools.noise_probe --games 40 --workers 16

## 在測什麼

CMA-ES 每一代要把 popsize 個候選**排出順序**。排得動的條件是

    候選之間的真實差距（訊號） > 2 x 測量誤差（SEM）

這支直接量那兩個數字，然後反推每次評估要幾個 seed。

## 為什麼不是「量噪音」

整條評估鏈是**決定性的**：`agents/gen0.py` 和 `agents/replay.py` 都沒有用亂數
（grep 全檔無 `random`），引擎的 forward model 也是決定性的
（`tools/value_probe.py` 驗過）。同一組參數、同一組 seed 跑兩次，期末現金**逐位
元組相同**（`tests/test_param_space.py::test_same_params_same_seed_give_identical_cash`）。

所以這裡沒有「重複評估的噪音」可以量，只有 **seed 抽樣誤差** —— 「這 M 個 seed 上
比較好」能不能推廣到別的 seed。這也直接推出兩件事：

1. 重複評估同一個候選是純浪費，一次就夠
2. **CMA-ES 一定會 overfit 那組固定 seed**，holdout seed 是必要的不是加分的

## 為什麼擾動要用 CMA-ES 自己的分布

`docs/plan.md` §4 建議拿 `cash_reserve_days: 3 -> 2` 當擾動。**那個做法會給出
沒有意義的數字** —— 決定性的系統裡，如果那個小改動沒有改變任何一個決策，配對差
會**恰好是 0**、sd 也是 0，看起來像「噪音超小、20 局就夠」，實際上什麼都沒量到。

這裡改成從 `N(x0, sigma0^2)` 抽候選，也就是 CMA-ES 第一代真正會產生的擾動量。
（cma 套件處理 bounds 用的是轉換不是夾取，這裡用夾取近似，量級一致。）

## 順便驗兩個 Stage 2 要用的假設

- **固定 slot 可不可以**：同一批候選在 slot 0 和 slot 1 各排一次序，看排序翻不翻。
  不翻的話 Stage 2 就能用 `swap=False`，成本直接砍一半。
  （實測自我對戰同一組參數在兩個位置的期末現金並不相同，所以這件事要驗不能假設。）
- **pool 開銷**：`eval.runner.run()` 每次呼叫都開一個 `mp.Pool`。Windows 是 spawn，
  開銷不小。量出來才知道 Stage 2 要不要繞過公開 API 自己批次化。

產物寫在 `temp/noise-probe-<時間戳>/`，**刻意多一層目錄**，這樣
`tools/eval_table.py`（掃的是 `temp/*/result.json`）不會把這幾個探測 run 收進
`docs/eval-results.md`。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from eval.runner import load_spec, run
from tools.param_space import DIM, decode, x0

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 要排得動候選，訊號至少要是測量誤差的幾倍。
SNR_TARGET = 2.0


def sample_candidates(n, sigma0, rng_seed):
    """從 `N(x0, sigma0^2)` 抽 n 個候選，夾回 [0, 1]。"""
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    centre = np.asarray(x0(), dtype=float)
    raw = rng.normal(centre, sigma0, size=(n, DIM))
    return [list(map(float, row)) for row in raw.clip(0.0, 1.0)]


def evaluate(x, name, opponent, games, workers, seed0):
    """一個候選對 `opponent` 跑 `games` 個 seed x 兩個 slot。

    回傳 `(cash_by_key, wall, cpu_secs)`。`cash_by_key` 是
    `{(seed, a_slot): 候選的期末現金}`。
    """
    spec_a = {"name": name, "entry": "agents.gen0:act", "params": decode(x)}
    spec_b = load_spec(opponent)

    t0 = time.perf_counter()
    summary, results = run(spec_a, spec_b, games, workers,
                           seed0=seed0, swap=True, progress=False)
    wall = time.perf_counter() - t0

    cash = {}
    for row in results:
        bad = (row.get("error") or row.get("cash_a") is None
               or row.get("status_a") != "DONE" or row.get("status_b") != "DONE")
        if not bad:
            cash[(row["seed"], row["a_slot"])] = float(row["cash_a"])
    cpu = sum(r["elapsed"] for r in results)
    return cash, wall, cpu, summary, results


def _slot(cash, slot):
    """抽出某一個 slot 的 `{seed: cash}`。"""
    return {seed: v for (seed, a_slot), v in cash.items() if a_slot == slot}


def analyse(scores_by_slot, slot):
    """一個 slot 的訊號 / 誤差分析。`scores_by_slot[name] = {seed: cash}`。"""
    names = list(scores_by_slot)
    seeds = sorted(set.intersection(*(set(scores_by_slot[n]) for n in names)))
    means = {n: statistics.fmean(scores_by_slot[n][s] for s in seeds) for n in names}

    # 訊號與誤差都用**同一組兩兩配對**算，統計量才對得齊：
    #   訊號 = |兩個候選平均分數的差|
    #   誤差 = 那一對逐 seed 配對差的 sd / sqrt(M)
    # 用中位數不用平均 —— 少數幾個「擾動到爛掉」的候選會把平均整個拉走，
    # 而 CMA-ES 真正卡住的地方是**相鄰候選排不排得動**，那是中位數的那一段。
    m_used = len(seeds)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diffs = [scores_by_slot[a][s] - scores_by_slot[b][s] for s in seeds]
            if len(diffs) < 2:
                continue
            sd = statistics.stdev(diffs)
            gap = abs(statistics.fmean(diffs))
            sem_pair = sd / math.sqrt(m_used)
            pairs.append({
                "a": a, "b": b, "gap": gap, "sd": sd, "sem": sem_pair,
                "resolvable": gap > SNR_TARGET * sem_pair,
                # 這一對要排得動需要幾個 seed
                "seeds_required": ((SNR_TARGET * sd / gap) ** 2) if gap else float("inf"),
            })

    gaps = [p["gap"] for p in pairs]
    sds = [p["sd"] for p in pairs]
    reqs = sorted(p["seeds_required"] for p in pairs)
    signal = statistics.median(gaps) if gaps else 0.0
    sd_pair = statistics.median(sds) if sds else 0.0
    sem = sd_pair / math.sqrt(m_used) if m_used else float("inf")

    return {
        "slot": slot,
        "seeds": m_used,
        "means": means,
        "ranking": sorted(names, key=lambda n: -means[n]),
        "n_pairs": len(pairs),
        "resolvable_pairs": sum(1 for p in pairs if p["resolvable"]),
        "median_gap": signal,
        "sd_pairdiff_median": sd_pair,
        "sem_at_m_used": sem,
        "snr_at_m_used": (signal / sem) if sem else float("inf"),
        # 中位數那一對要幾個 seed；以及要讓四分之三的配對都排得動要幾個
        "seeds_required": reqs[len(reqs) // 2] if reqs else float("inf"),
        "seeds_required_p75": reqs[int(len(reqs) * 0.75)] if reqs else float("inf"),
        "pairs": pairs,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="CMA-ES 每次評估要幾個 seed？")
    ap.add_argument("--games", type=int, default=40, help="每個候選幾個 seed（跑 2x 局）")
    ap.add_argument("--candidates", type=int, default=6, help="抽幾個候選")
    ap.add_argument("--sigma0", type=float, default=0.2, help="CMA-ES 的初始步長")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=0, help="起始 seed")
    ap.add_argument("--rng-seed", type=int, default=20260825, help="抽候選用的 RNG seed")
    ap.add_argument("--opponent", default="ladder-top-a")
    args = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "temp" / f"noise-probe-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [("baseline", x0())]
    for i, x in enumerate(sample_candidates(args.candidates, args.sigma0,
                                            args.rng_seed)):
        configs.append((f"cand{i}", x))

    total_games = len(configs) * args.games * 2
    print(f"{len(configs)} 個設定 x {args.games} seed x 2 slot = {total_games} 局，"
          f"對手 {args.opponent}，sigma0={args.sigma0}\n")

    scores = {}
    overheads = []
    t_all = time.perf_counter()
    for name, x in configs:
        cash, wall, cpu, summary, results = evaluate(
            x, name, args.opponent, args.games, args.workers, args.seed0)
        scores[name] = cash
        ideal = cpu / args.workers
        overheads.append(wall - ideal)
        print(f"  {name:9} 平均 {statistics.fmean(cash.values()):>10,.0f}   "
              f"wall {wall:6.1f}s   CPU/workers {ideal:6.1f}s   "
              f"多出 {wall - ideal:5.1f}s")
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump({"name": name, "x": x, "summary": summary,
                       "results": results}, f, ensure_ascii=False, indent=2)
    wall_all = time.perf_counter() - t_all

    report = {}
    for slot in (0, 1):
        by_slot = {n: _slot(c, slot) for n, c in scores.items()}
        report[f"slot{slot}"] = analyse(by_slot, slot)

    a0, a1 = report["slot0"], report["slot1"]

    print(f"\n總 wall {wall_all:.1f}s，{total_games / wall_all * 3600:,.0f} 局/小時")
    print(f"每次 run() 多出的 pool 開銷：中位 {statistics.median(overheads):.1f}s "
          f"（{len(configs)} 次呼叫）")

    print("\n=== slot 0（Stage 2 打算用的那個位置）===")
    print(f"訊號（兩兩平均分差的中位）     {a0['median_gap']:>12,.0f}")
    print(f"配對差 sd（兩兩中位）          {a0['sd_pairdiff_median']:>12,.0f}")
    print(f"SEM（{a0['seeds']} 個 seed）"
          f"{a0['sem_at_m_used']:>{max(1, 26 - len(str(a0['seeds'])))},.0f}")
    print(f"訊噪比                         {a0['snr_at_m_used']:>12.2f}   "
          f"（要 > {SNR_TARGET}）")
    print(f"排得動的配對                   {a0['resolvable_pairs']:>7}/{a0['n_pairs']}")
    print(f"→ 每次評估需要的 seed 數       {a0['seeds_required']:>12.0f}   "
          f"（要讓 3/4 的配對排得動：{a0['seeds_required_p75']:.0f}）")

    print("\n=== 固定 slot 可不可以 ===")
    print(f"slot 0 排序  {' > '.join(a0['ranking'])}")
    print(f"slot 1 排序  {' > '.join(a1['ranking'])}")
    same = a0["ranking"] == a1["ranking"]
    top_same = a0["ranking"][0] == a1["ranking"][0]
    print(f"排序{'相同' if same else '不同'}；"
          f"第一名{'相同' if top_same else '不同'}")
    if not same:
        print("⚠️ 排序會翻 —— Stage 2 不要用 swap=False，成本省不下來。")

    print("\n=== Gate ===")
    need = a0["seeds_required"]
    if need <= 30:
        print(f"✅ 需要 {need:.0f} 個 seed（<= 30）—— 進 Stage 2。")
    elif need <= 100:
        print(f"⚠️ 需要 {need:.0f} 個 seed（30~100）—— 進得去但成本要重算。")
    else:
        print(f"❌ 需要 {need:.0f} 個 seed（> 100）—— 先做敏感度分析降維，不要硬上。")

    with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "argv": vars(args),
            "wall_secs": wall_all,
            "pool_overhead_median": statistics.median(overheads),
            "slot0": a0,
            "slot1": a1,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n產物寫在 {out_dir}")


if __name__ == "__main__":
    main()
