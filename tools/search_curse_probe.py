"""贏家動作換一個延續策略還贏嗎 —— winner's curse 的判定。

## 要回答什麼

`tools/search_game.py` 的 `gain = q(贏家) − q(base)` **恆 ≥ 0**，因為 base
自己就在候選裡（`search/rollout_search.py:114-118` 會強制，不在就拋
`ValueError`）。所以「N/N 局 search 都贏」「配對差 +53,303」「Wilcoxon
p=0.002」全部是恆等式，不帶資訊 —— 2026-08-27 實測確認：兩批共 1,770 次
相鄰決策點比較，`q` 一次都沒下降，而且每局最後一個決策點的 `q` 就等於期末現金。

真正沒被回答的是：**贏家那個動作本身比較好，還是只是這一局剛好走運？**

51 個候選就算好壞完全一樣，30 天的連鎖反應也會讓它們的期末現金散開幾百到
幾千塊；取 max 就是挑「這一局最走運的那條鏈」。那種 gain 是真的錢（rollout
是決定性的，我們真的走上那條鏈），但**不是可以蒸餾的知識** —— 換一個延續，
優勢就不見了。這跟 2026-08-26 §7/§8 兩次蒸餾失敗、data-scaling 曲線是平的
互相吻合。

## 做法

重播 progress 裡記下的動作回到每一個決策點（引擎是決定性的，
`tools/search_game.py --resume` 已經驗過逐位元組相同）。在 search 真的改掉
動作的那些點上（`gain > 0`），把**贏家**和 **base** 各自套下去，然後用
**另一個延續策略**打到底，比大小：

    勝率明顯 > 50%  -> 動作真的比較好，標籤有內容 -> 問題在混訓比例
    勝率 ≈ 50%      -> winner's curse，argmax 選的是雜訊 -> search 的形狀要改

## 用法

    python -m tools.search_curse_probe temp/search-game-20260826-233232
    python -m tools.search_curse_probe <run> --continuation net   # 自我檢查

🩸 `--continuation net` 是**自我檢查**：延續策略跟原本那次一模一樣，所以
`q_win − q_base` 必須逐位元組等於 progress 裡記的 `gain`。對不上就代表重播
沒回到同一個狀態，這支的結論全部不可信。
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


def _read_seed(prog_dir, seed):
    """一個 seed 的 progress -> (對手, [search 那些行])。"""
    f = Path(prog_dir) / f"seed{seed:04d}.jsonl"
    if not f.is_file():
        return None, []
    opponent, rows, done = "?", [], False
    for line in io.open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["phase"] == "baseline":
            opponent = r.get("opponent", "?")
        elif r["phase"] == "search":
            rows.append(r)
        elif r["phase"] == "done":
            done = True
    if not done or any("action" not in r for r in rows):
        return None, []          # 沒跑完、或是舊格式沒記動作 -> 重播不了
    return opponent, rows


def _probe_one(job):
    (seed, hours, opponent, days, a_slot, rows, continuation, out_dir,
     roll_opponent) = job
    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make

    from agents.gen0 import act as gen0_act
    from agents.gen2_model import act as net_act
    from search import rollout_search as RS

    def note(**row):
        row["seed"] = seed
        with open(Path(out_dir) / f"seed{seed:04d}.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    t0 = time.perf_counter()
    n_probed = 0
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            env = make("kaggriculture", configuration={"seed": seed},
                       debug=False)
        cfg = env.configuration

        import inspect

        from eval.runner import build_agent, load_spec

        def make_them(name):
            # 引擎內建的對手回傳字串、而且只吃 obs 一個參數
            # （`tools/search_game.py` 同一段註解）。
            f = build_agent(load_spec(name))
            if isinstance(f, str):
                f = env.agents[f]
            if len(inspect.signature(f).parameters) == 1:
                return lambda obs: f(obs)
            return lambda obs: f(obs, cfg)

        act_them = make_them(opponent)
        # 🩸 走盤面用原本的對手（才會回到同一個狀態），只有**重評的 rollout**
        # 換對手。這是為了排除「gen0 是另一套策略，本來就不會接手贏家鋪的路」
        # 那個混淆 —— 換對手的話我方延續完全不變。
        roll_them = make_them(roll_opponent) if roll_opponent else act_them

        def act_me(obs):                     # 走盤面的：跟原本那次一樣是網路
            return net_act(obs, cfg)

        if continuation == "gen0":
            def cont_act(obs):
                return gen0_act(obs, cfg)
        else:
            cont_act = act_me

        queue = list(rows)
        while env.state[0].observation["day"] < days and not env.done:
            obs = env.state[a_slot].observation
            base = act_me(obs)
            action = base
            if int(obs.get("hour", 0)) in hours and queue:
                row = queue.pop(0)
                action = row["action"]
                # 🩸 對齊檢查：重播到的決策點必須跟紀錄的 day/hour 相同。
                # 對不上就代表狀態岔開了，硬跑下去只會產出安靜的垃圾。
                if (int(obs["day"]), int(obs.get("hour", 0))) != \
                        (row["day"], row["hour"]):
                    raise RuntimeError(
                        f"重播對不上：走到 day {obs['day']} hour "
                        f"{obs.get('hour')}，紀錄是 day {row['day']} hour "
                        f"{row['hour']}")
                if row.get("gain", 0) > 0:
                    snap = RS.snapshot(env)
                    q_win = RS.evaluate(env, snap, action, roll_them,
                                        cont_act, a_slot, days)
                    # 🩸 一定要在下一次 evaluate 之前讀 —— env 會被 restore 覆蓋。
                    o_win = RS.final_cash(env, 1 - a_slot)
                    q_base = RS.evaluate(env, snap, base, roll_them,
                                         cont_act, a_slot, days)
                    o_base = RS.final_cash(env, 1 - a_slot)
                    RS.restore(env, snap)
                    n_probed += 1
                    note(day=row["day"], hour=row["hour"], label=row["label"],
                         gain_orig=row["gain"], q_win=round(q_win),
                         q_base=round(q_base), delta=round(q_win - q_base),
                         # 對手的期末現金 —— 用來測「減掉對手能不能降低變異」
                         o_win=round(o_win), o_base=round(o_base),
                         delta_margin=round((q_win - o_win)
                                            - (q_base - o_base)),
                         n_cands=row.get("n_cands"),
                         roll_opponent=roll_opponent or opponent,
                         margin_orig=(row["q"] - row["runner_up"]
                                      if row.get("runner_up") is not None
                                      else None),
                         secs=round(time.perf_counter() - t0))
            actions = [None, None]
            actions[a_slot] = action
            actions[1 - a_slot] = act_them(env.state[1 - a_slot].observation)
            env.step(actions)
        return {"seed": seed, "opponent": opponent, "probed": n_probed,
                "elapsed": time.perf_counter() - t0, "error": None}
    except Exception as exc:                                  # noqa: BLE001
        import traceback
        return {"seed": seed, "opponent": opponent, "probed": n_probed,
                "elapsed": time.perf_counter() - t0,
                "error": f"{exc}\n{traceback.format_exc()}"}


def summarise(out_dir, continuation, orig_run, roll_opponent=None):
    import glob
    rows = []
    for f in sorted(glob.glob(str(Path(out_dir) / "seed*.jsonl"))):
        rows += [json.loads(l) for l in io.open(f, encoding="utf-8")]
    if not rows:
        return "沒有任何探測點 —— 那批的 gain 全都是 0？"
    # 🩸 自我檢查只在「延續沒換、對手也沒換」時才該成立。換了對手還拿它比
    # gain_orig 會全部對不上，那不是 bug 而是預期。
    selfcheck = continuation == "net" and not roll_opponent
    check = "   （自我檢查：delta 必須等於 gain_orig）" if selfcheck else ""
    L = [f"原始 run  {orig_run}",
         f"延續策略  {continuation}{check}",
         f"重評對手  {roll_opponent or '（不換，同原局）'}",
         f"探測點    {len(rows)} 個（search 真的改掉動作的決策點）", ""]

    if selfcheck:
        bad = [r for r in rows if r["delta"] != r["gain_orig"]]
        L.append(f"## 自我檢查：delta == gain_orig  "
                 f"{len(rows) - len(bad)}/{len(rows)}")
        for r in bad[:5]:
            L.append(f"   seed {r['seed']} day {r['day']} hour {r['hour']}"
                     f"  delta {r['delta']:,} vs gain {r['gain_orig']:,}")
        if not bad:
            L.append("   全部相同 -> 重播回到了同一個狀態，其他模式的結論可信")
        L.append("")

    d = [r["delta"] for r in rows]
    win = sum(1 for x in d if x > 0)
    tie = sum(1 for x in d if x == 0)
    lose = sum(1 for x in d if x < 0)
    L += ["## 換了延續之後，贏家還贏 base 嗎",
          f"  贏 {win}  平 {tie}  輸 {lose}",
          f"  勝率（不含平手）{win / max(1, win + lose):.1%}",
          f"  delta 中位數 {statistics.median(d):,.0f}"
          f"   平均 {statistics.fmean(d):,.0f}"]
    if len(d) > 1:
        se = statistics.stdev(d) / len(d) ** 0.5
        L.append(f"  平均的標準誤 {se:,.0f}"
                 f"   -> 平均 {statistics.fmean(d):,.0f} ± {1.96 * se:,.0f}（95%）")
    try:
        from scipy.stats import binomtest, wilcoxon
        nz = [x for x in d if x != 0]
        if nz:
            L.append(f"  Wilcoxon p = {wilcoxon(nz).pvalue:.4g}")
        if win + lose:
            L.append(f"  勝率 vs 50% 的 binomial p = "
                     f"{binomtest(win, win + lose).pvalue:.4g}")
    except ImportError:
        pass

    # 原始 gain 大的那些點，是不是比較站得住？
    L += ["", "## 依原始 gain 分桶"]
    buckets = [(0, 500), (500, 2000), (2000, 10000), (10000, float("inf"))]
    L.append(f"  {'原始 gain':<18}{'n':>5}{'勝率':>8}{'delta 中位':>12}"
             f"{'delta 平均':>12}")
    for lo, hi in buckets:
        b = [r for r in rows if lo <= r["gain_orig"] < hi]
        if not b:
            continue
        bd = [r["delta"] for r in b]
        w = sum(1 for x in bd if x > 0)
        l_ = sum(1 for x in bd if x < 0)
        name = f"{lo:,} ~ {hi:,.0f}" if hi != float("inf") else f"{lo:,}+"
        L.append(f"  {name:<18}{len(b):>5}{w / max(1, w + l_):>8.1%}"
                 f"{statistics.median(bd):>12,.0f}{statistics.fmean(bd):>12,.0f}")

    L += ["", "## 判讀",
          "  勝率明顯 > 50% -> 贏家動作本身比較好，標籤有內容，",
          "                    問題在混訓比例 -> 加權 / 提高 search 盤面比例",
          "  勝率 ≈ 50%     -> winner's curse：argmax 挑的是這一局走運的鏈，",
          "                    不是比較好的動作。加資料/加權都沒用，",
          "                    要改 search 的形狀（候選變少、延續變多、取平均）"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="tools.search_game 產出的 temp/search-game-*")
    ap.add_argument("--continuation", default="gen0", choices=("gen0", "net"),
                    help="重評時用誰打到底。net = 自我檢查（要復現 gain）")
    ap.add_argument("--rollout-opponent", default=None, metavar="NAME",
                    help="重評的 rollout 換這個對手（走盤面仍用原本的）。"
                         "配 --continuation net 就是「我方延續完全不變、"
                         "只換對手」的對照組")
    ap.add_argument("--seeds", default=None, help="預設吃 run 裡全部的")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarise-only", metavar="OUT_DIR",
                    help="不重跑，只把既有的輸出重新彙總")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.summarise_only:
        text = summarise(args.summarise_only, args.continuation, args.run_dir,
                         args.rollout_opponent)
        io.open(Path(args.summarise_only) / "summary.txt", "w",
                encoding="utf-8").write(text + "\n")
        print(text)
        return 0

    run_dir = Path(args.run_dir)
    saved = json.loads(io.open(run_dir / "config.json", encoding="utf-8").read())
    if saved.get("policy") != "net":
        print(f"⚠️ 這批的 policy 是 {saved.get('policy')!r} 不是 'net' —— "
              "走盤面的是 gen0，這支目前只支援 net")
        return 2
    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    os.environ["KAGGRI_WEIGHTS"] = saved["weights"]

    hours = frozenset(int(h) for h in str(saved["hours"]).split(","))
    days, a_slot = saved["days"], saved["a_slot"]
    prog = run_dir / "progress"
    if args.seeds:
        from tools.search_game import _parse_seeds
        seeds = _parse_seeds(args.seeds)
    else:
        seeds = sorted(int(p.stem[4:]) for p in prog.glob("seed*.jsonl"))

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "temp" / f"curse-{args.continuation}"
                             f"{'-vs-' + args.rollout_opponent if args.rollout_opponent else ''}"
                             f"-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, skipped = [], []
    for s in seeds:
        opp, rows = _read_seed(prog, s)
        if opp is None:
            skipped.append(s)
            continue
        jobs.append((s, hours, opp, days, a_slot, rows, args.continuation,
                     str(out_dir), args.rollout_opponent))
    if not jobs:
        raise SystemExit("沒有可重播的局 —— 那批沒跑完，或是舊格式沒記動作")
    n_pts = sum(sum(1 for r in j[5] if r.get("gain", 0) > 0) for j in jobs)
    print(f"重評 {run_dir.name}   延續 {args.continuation}   "
          f"{len(jobs)} 局 / {n_pts} 個探測點   {args.workers} workers")
    if skipped:
        print(f"  跳過 {len(skipped)} 局（沒跑完或舊格式沒記動作）：{skipped}")
    print(f"  輸出 {out_dir}")

    t0 = time.perf_counter()
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for res in pool.imap_unordered(_probe_one, jobs):
            tag = "✗" if res["error"] else "✓"
            print(f"  {tag} seed {res['seed']:>3} {res['opponent']:<11}"
                  f" 探測 {res['probed']:>3} 個  {res['elapsed'] / 60:>5.1f} 分",
                  flush=True)
            if res["error"]:
                print(res["error"], flush=True)
    print(f"總共 {(time.perf_counter() - t0) / 60:.1f} 分")

    text = summarise(out_dir, args.continuation, run_dir.name,
                     args.rollout_opponent)
    io.open(out_dir / "summary.txt", "w", encoding="utf-8").write(text + "\n")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
