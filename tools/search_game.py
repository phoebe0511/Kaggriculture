"""Stage 2 gate：**整局套用 search，累積起來贏得了 gen0 嗎？**

    python -m tools.search_game --seeds 0-23 --workers 24

## 為什麼單回合的 oracle 不夠

Stage 1 量的是「在**一個**盤面改一步能好多少」。那是上限，不是可達成的分數，
而且沒有回答真正的問題：**在幾十個決策點各改一次，累積起來會怎樣。**

反過來也成立 —— 就算單回合只有 3%，改 30 次也可能很可觀。這是 policy
improvement 的標準論證，08-21 / 08-24 從來沒量過。

## 做法

跑一整局，在取樣的決策點（預設每天 hour 12）用搜出來的動作取代 `gen0` 的；
其餘回合照常 `gen0`。同一個 seed、同一個對手再跑一局純 `gen0` 當對照。

🩸 **搜的時候 rollout 用 `gen0`，但實際往下走的時候後面的決策點也會被搜。**
所以搜尋當下估的 Q 不等於最後真的拿到的分數 —— 那是這個做法內建的近似，
不是 bug。真正算數的是**最後的期末現金**。

## 平行化放在「局」這一層

一局之內的決策點是**循序**的（後面的狀態取決於前面選了什麼）。同一個決策點的
候選互不相依，但那要把 `env.state` 送進別的行程，而它能不能 pickle 沒驗過。
所以直接一個 seed 一個 worker，局內循序 —— 簡單而且不會踩到未知數。

## 輸出

`temp/search-game-<時間戳>/result.json`，欄位做成 `eval.runner` 的形狀
（`seed` / `a_slot` / `cash_a` / `cash_b` / `status_*` / `error`）。
**`cash_a` 是搜尋版、`cash_b` 是同一個 seed 的純 `gen0`**，已經配對好了，
所以判定直接在這支裡面算（Wilcoxon + MDE），不需要 `tools.paired_stats`
（那支是比兩個不同 run 的 `cash_a`）。

⚠️ **不要看平均差就下結論。** 08-24 §1 踩過：20 局 p=0.123、MDE 8,332，
實測差 5,185 落在測不出的範圍裡，差點被當成「沒有差別」。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_seeds(text):
    """`0-23` 或 `0,3,7` 都吃。"""
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return out


def watch(run_dir):
    """把 progress/*.jsonl 讀成一張表。跑到一半隨時可以看。"""
    import glob

    prog = Path(run_dir)
    if prog.name != "progress":
        prog = prog / "progress"
    if not prog.is_dir():
        raise SystemExit(f"找不到 {prog} —— 路徑對嗎？")
    files = sorted(glob.glob(str(prog / "*.jsonl")))
    if not files:
        # 開跑後的頭幾分鐘都在算 gen0 對照組，還沒有決策點可以記。
        print(f"{prog} 還沒有 .jsonl —— 十之八九是還在跑 gen0 對照組，等一下再看。")
        return 0

    print(f"（10 局同時跑，一個 seed 一個 worker；"
          f"「搜到第幾天」是那一局之內的進度，不是第幾局）")
    print(f"{'seed':>5}{'gen0 基準':>11}{'搜到第幾天':>12}{'目前 Q':>11}"
          f"{'vs gen0':>9}{'改過':>6}{'已花':>8}{'更新':>10}")
    done_rows, running = [], 0
    for f in files:
        seed = int(Path(f).stem[4:])
        base = last = final = None
        for line in open(f, encoding="utf-8"):
            row = json.loads(line)
            if row["phase"] == "baseline":
                base = row["cash"]
            elif row["phase"] == "search":
                last = row
            elif row["phase"] == "done":
                final = row
        if final:
            done_rows.append((final["cash"], final["baseline"]))
            print(f"{seed:>5}{final['baseline']:>11,.0f}{'30/30 完成':>12}"
                  f"{final['cash']:>11,.0f}"
                  f"{final['cash'] / final['baseline'] - 1:>8.1%}"
                  f"{last['changed'] if last else 0:>6}"
                  f"{(last['secs'] if last else 0) / 60:>7.0f}分{final['t']:>10}")
        elif last:
            running += 1
            print(f"{seed:>5}{base:>11,.0f}{last['done']:>9}/30"
                  f"{last['q']:>11,.0f}{last['q'] / base - 1:>8.1%}"
                  f"{last['changed']:>6}{last['secs'] / 60:>7.0f}分{last['t']:>10}")
        else:
            running += 1
            print(f"{seed:>5}{(base or 0):>11,.0f}{'還在算對照組':>12}")

    if done_rows:
        diffs = [a - b for a, b in done_rows]
        print()
        print(f"  整局跑完的有 {len(done_rows)}/{len(files)} 局：配對差平均 "
              f"{statistics.fmean(diffs):>+,.0f}，"
              f"search 較好 {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
    if running:
        print(f"  還在跑 {running} 局（都在同時進行）")
    print()
    print("⚠️ 「目前 Q」是「搜到這一天、之後全用 gen0 打到底」的估值，"
          "不是最後的期末現金。")
    return 0


def _play_one(job):
    """跑一個 seed：搜尋版 + 純 gen0 對照。top-level 才 pickle 得動。"""
    (seed, hours, opponent, rng_seed, days, max_cands, a_slot, prog_dir) = job

    def note(**row):
        """一個決策點一行。worker 之間各寫各的檔，不會互相蓋。"""
        if not prog_dir:
            return
        row["t"] = time.strftime("%H:%M:%S")
        with open(Path(prog_dir) / f"seed{seed:04d}.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    os.environ.pop("KAGGRI_LOG_FILE", None)
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make

    import numpy as np

    from agents.gen0 import act as gen0_act
    from search import candidates as CAND
    from search import rollout_search as RS

    def new_env():
        with contextlib.redirect_stderr(io.StringIO()):
            return make("kaggriculture", configuration={"seed": seed},
                        debug=False)

    t0 = time.perf_counter()
    err = None
    searched = baseline = opp_searched = opp_baseline = None
    n_searched = n_changed = 0
    try:
        # --- 對照組：純 gen0 ---
        env = new_env()
        cfg = env.configuration

        if opponent == "gen0":
            def act_them(obs):
                return gen0_act(obs, cfg)
        else:
            from eval.runner import build_agent, load_spec
            _opp = build_agent(load_spec(opponent))

            def act_them(obs):
                return _opp(obs, cfg)

        def act_gen0(obs):
            return gen0_act(obs, cfg)

        baseline = RS.play_to_end(env, act_gen0, act_them, a_slot, days)
        opp_baseline = RS.final_cash(env, 1 - a_slot)
        note(phase="baseline", cash=baseline, opp=opp_baseline)

        # --- 搜尋版 ---
        rng = np.random.default_rng(rng_seed + seed)
        env = new_env()
        cfg = env.configuration
        while env.state[0].observation["day"] < days and not env.done:
            obs = env.state[a_slot].observation
            base = gen0_act(obs, cfg)
            if int(obs.get("hour", 0)) in hours:
                snap = RS.snapshot(env)
                cands, _blocked = CAND.generate(obs, cfg, base, rng)
                if max_cands and len(cands) > max_cands:
                    keep = [c for c in cands if c[0] == "gen0"]
                    rest = [c for c in cands if c[0] != "gen0"]
                    idx = rng.choice(len(rest), size=max_cands - 1, replace=False)
                    cands = keep + [rest[int(i)] for i in idx]
                label, action, _q, _scored = RS.search_action(
                    env, snap, cands, act_them, act_gen0, a_slot, days,
                    base_label="gen0")
                RS.restore(env, snap)
                n_searched += 1
                n_changed += int(label != "gen0")
                note(phase="search", day=int(obs["day"]), label=label,
                     q=round(_q), n_cands=len(cands),
                     changed=n_changed, done=n_searched,
                     secs=round(time.perf_counter() - t0))
            else:
                action = base
            actions = [None, None]
            actions[a_slot] = action
            actions[1 - a_slot] = act_them(env.state[1 - a_slot].observation)
            env.step(actions)
        searched = RS.final_cash(env, a_slot)
        # 🩸 對手的期末現金一定要記。市場是兩家共用的，我方分數變高有兩種可能：
        # 自己賺更多，或是壓低對手。不記的話分不出來 —— 而對開迴路 replay 來說
        # 「讓它的 BUY_*/HIRE 失敗」是不會轉移到真實對手身上的假增益。
        opp_searched = RS.final_cash(env, 1 - a_slot)
        note(phase="done", cash=searched, baseline=baseline,
             opp=opp_searched, opp_baseline=opp_baseline)
    except Exception as exc:                          # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    return {
        "seed": seed,
        "a_slot": a_slot,
        "cash_a": searched,
        "cash_b": baseline,
        "status_a": "DONE" if err is None else "ERROR",
        "status_b": "DONE" if err is None else "ERROR",
        "opp_cash_searched": opp_searched,
        "opp_cash_baseline": opp_baseline,
        "decisions": n_searched,
        "changed": n_changed,
        "elapsed": time.perf_counter() - t0,
        "error": err,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="model/weights-e2e-round6.npz")
    ap.add_argument("--seeds", default="0-23")
    ap.add_argument("--hours", default="12",
                    help="每天的哪幾個 hour 要搜（逗號分隔）。12 = 一天一次")
    ap.add_argument("--opponent", default="gen0")
    # 🩸 這台是 20 實體核心 / 28 邏輯核心。對局是純 CPU 的 Python 模擬，
    # 行程數超過實體核心就互搶、總時間反而變長，而且整台機器會被吃滿。
    # 2026-08-25 用 24 跑過一次，使用者回報變慢。要開更多之前先問。
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-cands", type=int, default=0,
                    help="每個決策點最多幾個候選，0 = 全部（約 57）")
    ap.add_argument("--a-slot", type=int, default=0, choices=(0, 1))
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rng-seed", type=int, default=20260825)
    ap.add_argument("--watch", metavar="RUN_DIR",
                    help="只看某一輪的進度，不跑新的")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.watch:
        return watch(args.watch)

    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    os.environ["KAGGRI_WEIGHTS"] = args.weights

    seeds = _parse_seeds(args.seeds)
    hours = frozenset(int(h) for h in args.hours.split(","))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = REPO_ROOT / "temp" / f"search-game-{stamp}"
    prog_dir = run_dir / "progress"
    prog_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(s, hours, args.opponent, args.rng_seed, args.days,
             args.max_cands, args.a_slot, str(prog_dir)) for s in seeds]

    print(f"{len(seeds)} 個 seed   每天搜 {sorted(hours)}   對手 {args.opponent}   "
          f"權重 {args.weights}   {args.workers} workers")
    print(f"產物 {run_dir}")
    print(f"看進度：python -m tools.search_game --watch {run_dir}")

    t0 = time.perf_counter()
    if args.workers <= 1:
        results = [_play_one(j) for j in jobs]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = []
            for r in pool.imap_unordered(_play_one, jobs):
                results.append(r)
                mark = "⚠️" if r["error"] else "  "
                print(f"  {mark} seed {r['seed']:>4}   搜尋 "
                      f"{(r['cash_a'] or 0):>9,.0f}   gen0 "
                      f"{(r['cash_b'] or 0):>9,.0f}   "
                      f"{((r['cash_a'] or 0) - (r['cash_b'] or 0)):>+9,.0f}   "
                      f"改了 {r['changed']}/{r['decisions']}   "
                      f"{r['elapsed']:.0f}s"
                      + (f"   {r['error']}" if r["error"] else ""), flush=True)
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["error"] is None]
    diffs = [r["cash_a"] - r["cash_b"] for r in ok]
    summary = {
        "a": "search", "b": "gen0",
        "games": len(results), "failures": len(results) - len(ok),
        "mean_cash_a": statistics.fmean(r["cash_a"] for r in ok) if ok else 0,
        "mean_cash_b": statistics.fmean(r["cash_b"] for r in ok) if ok else 0,
        "mean_diff": statistics.fmean(diffs) if diffs else 0,
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
        "draws": sum(1 for d in diffs if d == 0),
        "wall_secs": wall,
        "workers": args.workers,
    }

    payload = {
        "a": {"name": "search", "note": f"gen0 + 離線 search，每天 hour {sorted(hours)}"},
        "b": {"name": "gen0", "note": "同一個 seed、同一個對手的純 gen0"},
        "summary": summary,
        "results": sorted(results, key=lambda r: r["seed"]),
        "argv": vars(args),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n{len(ok)}/{len(results)} 局完成，{wall / 60:.0f} 分鐘")
    if ok:
        print(f"  search 平均 {summary['mean_cash_a']:>10,.0f}")
        print(f"  gen0   平均 {summary['mean_cash_b']:>10,.0f}")
        print(f"  配對差      {summary['mean_diff']:>+10,.0f}   "
              f"（中位 {statistics.median(diffs):>+,.0f}）")
        print(f"  search 較好 {summary['wins']}/{len(ok)}")
        print(f"  平均改掉    "
              f"{statistics.fmean(r['changed'] for r in ok):.1f}/"
              f"{statistics.fmean(r['decisions'] for r in ok):.0f} 個決策點")
    # search 和 gen0 已經配對在同一份 result.json 的 cash_a / cash_b 裡，
    # 所以不需要 tools.paired_stats（那支是比**兩個** run 的 cash_a）。
    # 這裡直接用同一組公式：Wilcoxon + MDE（α=0.05、power=0.8）。
    if len(diffs) > 1:
        try:
            from scipy.stats import wilcoxon
            nz = [d for d in diffs if d]
            if nz:
                p = float(wilcoxon(nz).pvalue)
                sd = statistics.stdev(diffs)
                mde = 2.8016 * sd / len(diffs) ** 0.5
                print(f"\n  Wilcoxon p = {p:.5g}   sd {sd:,.0f}   "
                      f"MDE {mde:,.0f}（α=0.05、power=0.8）")
                if abs(summary["mean_diff"]) < mde:
                    print(f"  ⚪ 判不出來 —— 實測差 {abs(summary['mean_diff']):,.0f} "
                          f"< MDE，要加 seed。")
                elif p < 0.05:
                    print("  ✅ 顯著較強 —— 進 Stage 3。" if summary["mean_diff"] > 0
                          else "  ❌ 顯著較弱。")
                else:
                    print(f"  ⚪ 判不出來（p={p:.3g} ≥ 0.05）。")
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
