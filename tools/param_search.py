"""CMA-ES 聯合最佳化 `gen0` 的 34 個連續參數。

    python -m tools.param_search --seeds 20 --workers 24            # 開新的
    python -m tools.param_search --resume temp/cma/<時間戳>          # 續跑
    python -m tools.param_search --watch  temp/cma/<時間戳>          # 只看進度
    python -m tools.param_search --smoke                            # 2 代小 smoke

## 目標函數

固定 seed 集上、對固定對手（預設 `ladder-top-a`）的平均分數。CMA-ES 是最小化，
所以實際餵進去的是負值。`--objective` 兩種寫法的實測訊噪比見 `OBJECTIVES`
——**預設是 `margin`（我方減對手）**，因為 `cash` 要 197 個 seed 才排得動候選，
margin 只要 54 個。

🩸 **每一代都用同一組 seed**（common random numbers）。換了的話不同代的分數
不可比，CMA-ES 會把 seed 換掉造成的差當成參數造成的差。

## 決定性 -> 兩個後果

整條鏈沒有亂數（`agents/gen0.py`、`agents/replay.py` 都沒有 `random`，引擎的
forward model 由 `tools/value_probe.py` 驗過）。所以：

1. **同一個候選不需要重複評估**，跑一次就是準確值
2. **一定會 overfit 那組 seed** —— 所以有 holdout。`--checkpoint-every` 代拿當代
   最佳去打沒用過的 seed 和另外三支隊伍，train 升而 holdout 不升就是 overfit

## 為什麼一代開一個 pool，不是一個候選開一個

`eval.runner.run()` 每次呼叫都開一個 `mp.Pool`。Windows 是 spawn，開銷實測不小，
一代 14 個候選就要付 14 次。這裡改成把一整代的 jobs 併起來丟進同一個 pool。

代價是要用 `eval.runner._play`（私有）。`harness/rollout.py:237` 已經在 import
`eval.runner._quiet_make`，所以這在這個 repo 是既有做法，不是新開的洞。
不想用私有 API 的話，`--no-batch` 會退回逐候選呼叫 `run()`。

## 停止

- `es.stop()`（cma 自己的收斂判準）
- `--generations` 上限
- **在 state 目錄放一個 `STOP` 檔**：當代跑完就乾淨收工，不會弄壞存檔
- Ctrl-C：當代的存檔可能不完整，用上一代的 pickle 續跑

## 產出

    temp/cma/<時間戳>/
        config.json      這一輪的設定（seed 集、popsize、對手、維度）
        log.jsonl        一代一行：最佳/平均分數、sigma、wall
        gen_NNN.pkl      CMA-ES 狀態（mean / covariance / sigma），續跑用
        best.json        目前最佳的 x 與 params
        checkpoints.jsonl  每次 holdout 驗收的結果
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import statistics
import time
from pathlib import Path

from eval.runner import _play, build_jobs, load_spec, run
from tools.param_space import DIM, decode, to_json_params, x0

REPO_ROOT = Path(__file__).resolve().parents[1]

#: checkpoint 打的隊伍。挑打法明顯不同的 —— tetsuya 是 8 支裡唯一養鵝的、
#: recursion 是唯一種瓜的、utkarsh 的動物配置跟主流不同。
#: 🩸 **不要用 `ladder-top-b`** —— 它跟 `ladder-top-a` 是同一局（episode 93916293）
#: 的兩個 player，config note 自己寫「不是兩個獨立樣本」，拿它驗收等於沒驗。
#:
#: 🩸 **`gen1` 是刻意放進來的、唯一一支會反應的。** 前三支跟訓練用的
#: `ladder-top-a` 一樣都是**開迴路 replay**，所以「靠讓對手的 BUY_*/HIRE 失敗
#: 來壓低它的現金」這種招數對它們全部有效，holdout 抓不到（見 `OBJECTIVES`
#: 的誘因問題）。`gen1` 會依盤面改動作，被卡住就換做別的 —— train 升而 gen1
#: 那一欄不升，就是在吃 replay 的被動性，不是真的變強。
#: （`gen1` 不是另一支 agent，是 `agents.gen0:act` 不帶 params 走 DEFAULT_PARAMS，
#: `config/opponents/gen1.json`；`agents/gen1.py` 2026-08-21 就併掉了。）
HOLDOUT_TEAMS = ("tetsuya", "recursion", "utkarsh", "gen1")


def load_team_specs(names=HOLDOUT_TEAMS):
    """撈出指定隊伍的 spec。先找 `config/ladder-top.json`，沒有就走 `load_spec`。"""
    path = REPO_ROOT / "config" / "ladder-top.json"
    with open(path, encoding="utf-8") as f:
        pool = json.load(f)["opponents"]
    by_name = {o["name"]: o for o in pool}
    out = []
    for n in names:
        if n in by_name:
            out.append(by_name[n])
        else:
            try:
                out.append(load_spec(n))
            except (FileNotFoundError, SystemExit) as exc:
                raise SystemExit(
                    f"holdout 隊伍 {n!r} 在 {path} 和 config/opponents/ 都找不到"
                    f"；ladder-top 有的是 {sorted(by_name)}（{exc}）") from exc
    return out


def candidate_spec(x, name):
    return {"name": name, "entry": "agents.gen0:act", "params": decode(x)}


# --------------------------------------------------------------------------
# 評估
# --------------------------------------------------------------------------

#: 目標函數。2026-08-25 實測（`temp/noise-probe-20260825-105910`，sigma=0.05、
#: 40 seed、7 個相鄰候選兩兩比）：
#:
#:     cash     訊號 3,125   配對差 sd 21,932   要 197 個 seed 才排得動
#:     margin   訊號 7,017   配對差 sd 25,880   要  54 個 seed
#:
#: margin 的訊號大三倍，因為市場是兩家共用的 —— 我方變強會同時壓低對手，
#: 差值等於把同一個效果算兩次。
#:
#: 🩸 **margin 對開迴路 replay 有一個誘因問題**：壓低對手的現金可以靠讓它的
#: `BUY_*` / `HIRE` 失敗達成，那是在利用 replay 不會反應的弱點，不會轉移到
#: 真正的對手身上。checkpoint 打另外三支隊伍是唯一的防線。
OBJECTIVES = {
    "cash": lambda a, b: a,
    "margin": lambda a, b: a - b,
}


def _score_rows(rows, objective="cash"):
    """一批 `_play` 的結果 -> (平均分數, 作廢局數)。

    作廢的局算 0 分。參數搞到 agent 拋例外的候選就是沒有價值的候選，但**不要
    安靜地跳過** —— 跳過的話「跑掛一半的局」會因為只平均剩下的而看起來很好。
    """
    score = OBJECTIVES[objective]
    values, dropped = [], 0
    for row in rows:
        bad = (row.get("error") or row.get("cash_a") is None
               or row.get("cash_b") is None
               or row.get("status_a") != "DONE" or row.get("status_b") != "DONE")
        if bad:
            values.append(0.0)
            dropped += 1
        else:
            values.append(float(score(row["cash_a"], row["cash_b"])))
    return (statistics.fmean(values) if values else 0.0), dropped


def evaluate_batch(xs, opponent_spec, games, seed0, workers, swap, tag="cand",
                   objective="cash"):
    """一整代的候選，共用一個 pool。回傳 `[(平均分數, 作廢局數), ...]`。"""
    per = games * (2 if swap else 1)
    jobs = []
    for i, x in enumerate(xs):
        jobs.extend(build_jobs(candidate_spec(x, f"{tag}{i}"), opponent_spec,
                               games, seed0, swap, None))

    if workers <= 1:
        results = [_play(job) for job in jobs]
    else:
        with mp.Pool(processes=workers) as pool:
            results = list(pool.map(_play, jobs))   # map 保序，才切得回各候選

    return [_score_rows(results[i * per:(i + 1) * per], objective)
            for i in range(len(xs))]


def evaluate_one(x, opponent_spec, games, seed0, workers, swap, name="cand",
                 objective="cash"):
    """單一候選走 `eval.runner.run()`（公開 API，但每次都開新的 pool）。"""
    _summary, results = run(candidate_spec(x, name), opponent_spec,
                            games, workers, seed0=seed0, swap=swap, progress=False)
    return _score_rows(results, objective)


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------

def checkpoint(x, args, opponent_spec, teams):
    """當代最佳去打 holdout seed + 另外三支隊伍。"""
    out = {}
    out["holdout"], out["holdout_dropped"] = evaluate_one(
        x, opponent_spec, args.holdout_seeds, args.holdout_seed0,
        args.workers, args.swap, name="best-holdout", objective=args.objective)

    per_team = {}
    for team in teams:
        score, _dropped = evaluate_one(
            x, team, args.team_seeds, args.holdout_seed0,
            args.workers, args.swap, name=f"best-vs-{team['name']}",
            objective=args.objective)
        per_team[team["name"]] = score
    out["teams"] = per_team
    out["teams_mean"] = statistics.fmean(per_team.values()) if per_team else 0.0
    return out


# --------------------------------------------------------------------------
# 主迴圈
# --------------------------------------------------------------------------

def watch(state_dir):
    """把一輪的進度讀成一張表。跑到一半隨時可以看。

        python -m tools.param_search --watch temp/cma/<時間戳>

    要盯的兩件事印在最後面 —— `gen1` 那一欄跟 train 同不同向（不同向就是在吃
    replay 的被動性，不會轉移到榜上），以及 `歷來最佳` 有沒有卡住（卡住代表
    sigma 已經小到 `--seeds` 排不動相鄰候選，2026-08-25 量到 margin 要 54 個）。
    """
    d = Path(state_dir)
    cfg = {}
    if (d / "config.json").is_file():
        with open(d / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
    a = cfg.get("argv", {})
    rows = []
    if (d / "log.jsonl").is_file():
        with open(d / "log.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    seed0, nseed = a.get("seed0") or 0, a.get("seeds") or 0
    print(f"{d}   對手 {cfg.get('opponent')}   目標 {a.get('objective')}"
          f"   popsize {a.get('popsize')}   train seed {seed0}~{seed0 + nseed - 1}")
    if not rows:
        print("  還沒有完整的一代。")
        return 0
    total = a.get("generations", 0)
    per = statistics.fmean(r["wall_secs"] for r in rows)
    left = max(0, total - len(rows))
    print(f"  {len(rows)}/{total} 代   一代平均 {per:.0f} 秒"
          f"   還剩 {left} 代 ≈ {left * per / 3600:.1f} 小時")
    if (d / "STOP").exists():
        print("  ⚠️ STOP 檔還在 —— 當代跑完就會收工")

    print()
    print(f"## 每一代（最近 15 代；分數是 {a.get('objective')}，越接近 0 越好）")
    print(f"  {'代':>4}{'最佳':>11}{'平均':>11}{'歷來最佳':>11}"
          f"{'sigma':>8}{'秒':>6}{'作廢局':>7}")
    for r in rows[-15:]:
        print(f"  {r['generation']:>4}{r['best']:>11,.0f}{r['mean']:>11,.0f}"
              f"{r['best_so_far']:>11,.0f}{r['sigma']:>8.4f}"
              f"{r['wall_secs']:>6.0f}{r['dropped_games']:>7}")

    # 🩸 歷來最佳卡住 = sigma 已經小到 `--seeds` 排不動相鄰候選。那時候該重開
    # 一輪把 seeds 拉高，繼續等不會有東西。
    best = [r["best_so_far"] for r in rows]
    stale = 0
    for x in reversed(best[:-1]):
        if x != best[-1]:
            break
        stale += 1
    if stale >= 20:
        print(f"  🩸 歷來最佳已經 {stale} 代沒動（sigma {rows[-1]['sigma']:.4f}）"
              f" —— 考慮重開一輪把 --seeds 從 {a.get('seeds')} 拉高")

    cks = []
    if (d / "checkpoints.jsonl").is_file():
        with open(d / "checkpoints.jsonl", encoding="utf-8") as f:
            cks = [json.loads(line) for line in f if line.strip()]
    print()
    print(f"## holdout（每 {a.get('checkpoint_every')} 代）")
    if not cks:
        print(f"  還沒到第 {a.get('checkpoint_every')} 代")
    else:
        teams = list(cks[-1].get("teams", {}))
        print(f"  {'代':>4}{'train':>10}{'holdout':>10}{'差':>9}  "
              + "".join(f"{t:>11}" for t in teams))
        for c in cks:
            print(f"  {c['generation']:>4}{c['train']:>10,.0f}"
                  f"{c['holdout']:>10,.0f}{c['train'] - c['holdout']:>9,.0f}  "
                  + "".join(f"{c['teams'].get(t, float('nan')):>11,.0f}"
                            for t in teams))
        print("  「差」= train − holdout（都是打 "
              f"{cfg.get('opponent')}，只是 seed 不同）。擴大就是在背 seed")
        if len(cks) >= 2:
            # 🩸 一定要把 `holdout` 也列進來 —— 它是「同一個對手、沒訓練過的
            # seed」，最直接的過擬合訊號。只列 train + 各隊伍的話，
            # 「train 升但 holdout 崩」這件事看不出來（2026-08-27 漏過一次）。
            #
            # 🩸 也一定要有「相對前一次」那一欄。只看頭到尾的話，最近才發生的
            # 反轉會被前面的進步蓋掉。
            def _get(c, k):
                return c["teams"].get(k, 0.0) if k in teams else c.get(k, 0.0)

            # 🩸 checkpoint 打的是「歷來最佳」那組參數。歷來最佳沒動的話，
            # 連續兩次 checkpoint 會**逐位元組相同**，拿它們相減永遠是 0 ——
            # 「train 升但其它全退」就永遠不會觸發（2026-08-27 漏過）。
            # 所以要往回找到最近一次**數值不同**的 checkpoint。
            prev = cks[-2]
            for c in reversed(cks[:-1]):
                if c["train"] != cks[-1]["train"] or c["holdout"] != cks[-1]["holdout"]:
                    prev = c
                    break
            if prev is not cks[-2]:
                print()
                print(f"  ⚠️ 第 {cks[-2]['generation']} 代之後歷來最佳沒再更新 ——"
                      f" 後面的 checkpoint 是同一組參數，數字一模一樣。"
                      f"「相對前一次」改跟第 {prev['generation']} 代比。")
            g0, gp, g1 = (cks[0]["generation"], prev["generation"],
                          cks[-1]["generation"])

            print()
            print(f"  變化（正 = 變好）      {f'第{g0}->第{g1}':>12}"
                  f"{f'第{gp}->第{g1}':>12}")
            for k in ["train", "holdout"] + list(teams):
                d_all = _get(cks[-1], k) - _get(cks[0], k)
                d_last = _get(cks[-1], k) - _get(prev, k)
                tag = ""
                if k == "holdout":
                    tag = "  ← 同一個對手、沒訓練過的 seed"
                elif k == "gen1":
                    tag = "  ← 唯一會反應的（其餘三支都是開迴路 replay）"
                print(f"    {k:<18}{d_all:>+12,.0f}{d_last:>+12,.0f}{tag}")

            # 判讀：train 升、其它全跌 = 在背 train 的 seed
            others = ["holdout"] + list(teams)
            d_tr = _get(cks[-1], "train") - _get(prev, "train")
            down = [k for k in others if _get(cks[-1], k) - _get(prev, k) < 0]
            if d_tr >= 0 and len(down) == len(others):
                print(f"  🩸 第 {gp} -> {g1} 代：train 沒退步但**其它 "
                      f"{len(others)} 欄全部退**（{', '.join(down)}）"
                      " —— 那是在背 train 的 seed，不是變強")
            gaps = [c["train"] - c["holdout"] for c in cks]
            if gaps[-1] == max(gaps) and len(gaps) >= 3:
                print(f"  🩸 train − holdout 的差 {gaps[-1]:,.0f} 是歷來最大"
                      f"（之前 {min(gaps):,.0f} ~ {sorted(gaps)[-2]:,.0f}）")

    bp = d / "best.json"
    if bp.is_file():
        from tools.param_space import KEYS, decode, x0
        with open(bp, encoding="utf-8") as f:
            b = json.load(f)
        base, cur = decode(x0()), decode(b["x"])
        diffs = []
        for k in KEYS:
            v0, v1 = base.get(k), cur.get(k)
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)) and v0:
                diffs.append((abs((v1 - v0) / v0), k, v0, v1))
        diffs.sort(reverse=True)
        print()
        print(f"## 目前最佳（第 {b['generation']} 代，{b['score']:,.0f}）"
              "動最多的 8 個參數")
        print(f"  {'參數':<28}{'預設':>12}{'現在':>12}{'變化':>9}")
        for frac, k, v0, v1 in diffs[:8]:
            print(f"  {k:<28}{v0:>12,.4g}{v1:>12,.4g}{(v1 - v0) / v0:>+9.0%}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="CMA-ES 最佳化 gen0 的連續參數")
    ap.add_argument("--seeds", type=int, default=20, help="每次評估幾個 train seed")
    ap.add_argument("--seed0", type=int, default=0, help="train seed 的起點")
    ap.add_argument("--holdout-seeds", type=int, default=40, help="holdout 幾個 seed")
    ap.add_argument("--holdout-seed0", type=int, default=1000)
    ap.add_argument("--team-seeds", type=int, default=20, help="每支隊伍幾個 seed")
    ap.add_argument("--popsize", type=int, default=14)
    ap.add_argument("--sigma0", type=float, default=0.2)
    ap.add_argument("--generations", type=int, default=243, help="最多跑幾代")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--opponent", default="ladder-top-a")
    ap.add_argument("--objective", choices=sorted(OBJECTIVES), default="margin",
                    help="cash = 我方期末現金；margin = 我方減對手。"
                         "2026-08-25 實測 margin 的訊噪比高 1.9 倍（54 vs 197 "
                         "個 seed），但對開迴路 replay 有誘因問題，見 OBJECTIVES。")
    ap.add_argument("--cma-seed", type=int, default=20260825)
    ap.add_argument("--swap", action="store_true",
                    help="兩個 slot 都跑（貴一倍）。預設只跑 slot 0 —— "
                         "候選之間是相對比較，位置偏差對所有候選一樣。"
                         "tools/noise_probe.py 會驗這個假設。")
    ap.add_argument("--no-batch", action="store_true",
                    help="退回逐候選呼叫 eval.runner.run()，不碰私有 API")
    ap.add_argument("--resume", help="續跑：指向 temp/cma/<時間戳>")
    ap.add_argument("--watch", metavar="STATE_DIR",
                    help="只看某一輪的進度，不跑新的")
    ap.add_argument("--smoke", action="store_true",
                    help="2 代、popsize 4、2 seed —— 驗存檔續跑走得通")
    ap.add_argument("--log-level", default="0",
                    help="agent 的 KAGGRI_LOG_LEVEL。🩸 預設 0 —— "
                         "`agents/gen0.py:66` 沒設的話是 **3**，每回合把一個大 "
                         "dict 序列化噴到 stdout。2026-08-27 實測（popsize 4 / "
                         "4 seed / 1 代）：輸出 50,651 bytes vs 14,699,160 bytes。"
                         "⚠️ **牆鐘時間沒有變**（一代 221.3 -> 218.6 秒，在雜訊"
                         "範圍內）—— 關掉是為了不要洗版和寫爆磁碟，不是為了快。")
    args = ap.parse_args(argv)

    if args.watch:
        return watch(args.watch)

    # 🩸 一定要在開 Pool 之前設。Windows 是 spawn，child 拿的是這一刻的環境。
    os.environ["KAGGRI_LOG_LEVEL"] = str(args.log_level)

    if args.smoke:
        # 🩸 明確傳進來的旗標不要被 smoke 蓋掉 —— 續跑時 `--generations` 被蓋回 2
        # 會讓迴圈變成空的 range，看起來像「續跑壞了」但其實什麼都沒跑。
        import sys as _sys
        given = {a.split("=")[0] for a in (argv if argv is not None else _sys.argv)}
        if "--generations" not in given:
            args.generations = 2
        args.popsize, args.seeds = 4, 2
        args.checkpoint_every, args.holdout_seeds, args.team_seeds = 1, 2, 2

    import cma

    opponent_spec = load_spec(args.opponent)
    teams = load_team_specs()

    # --- 建立或載入狀態 ---
    if args.resume:
        state_dir = Path(args.resume)
        pkls = sorted(state_dir.glob("gen_*.pkl"))
        if not pkls:
            raise SystemExit(f"{state_dir} 裡沒有 gen_*.pkl")
        with open(pkls[-1], "rb") as f:
            es = pickle.load(f)
        gen0_idx = int(pkls[-1].stem.split("_")[1])
        print(f"從 {pkls[-1].name} 續跑（已完成 {gen0_idx} 代）")
    else:
        state_dir = REPO_ROOT / "temp" / "cma" / time.strftime("%Y%m%d-%H%M%S")
        state_dir.mkdir(parents=True, exist_ok=True)
        es = cma.CMAEvolutionStrategy(
            x0(), args.sigma0,
            {"bounds": [0, 1], "popsize": args.popsize, "seed": args.cma_seed,
             "verbose": -9},
        )
        gen0_idx = 0
        with open(state_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"argv": vars(args), "dim": DIM,
                       "opponent": opponent_spec.get("name"),
                       "teams": [t["name"] for t in teams]},
                      f, ensure_ascii=False, indent=2)

    games_per_gen = args.popsize * args.seeds * (2 if args.swap else 1)
    print(f"維度 {DIM}   popsize {args.popsize}   train seed {args.seed0}~"
          f"{args.seed0 + args.seeds - 1}   對手 {opponent_spec.get('name')}"
          f"   目標 {args.objective}")
    print(f"一代 {games_per_gen} 局   狀態在 {state_dir}\n")

    # 🩸 續跑一定要把歷來最佳讀回來。不讀的話 best_score 是 -inf，續跑後第一代
    # 不管多爛都會覆寫 best.json —— 重啟一次就把先前找到的最佳解弄丟，而且不報錯。
    prev_holdout = None
    best_score, best_x = float("-inf"), None
    best_path = state_dir / "best.json"
    if args.resume and best_path.is_file():
        with open(best_path, encoding="utf-8") as f:
            prev = json.load(f)
        best_score, best_x = prev["score"], prev["x"]
        print(f"歷來最佳讀回 {best_score:,.0f}（第 {prev['generation']} 代）")

    log_path = state_dir / "log.jsonl"
    ck_path = state_dir / "checkpoints.jsonl"

    for gen in range(gen0_idx + 1, args.generations + 1):
        if (state_dir / "STOP").exists():
            print("看到 STOP 檔，收工。")
            break
        if es.stop():
            print(f"CMA-ES 自己收斂了：{es.stop()}")
            break

        t0 = time.perf_counter()
        xs = [list(map(float, x)) for x in es.ask()]

        if args.no_batch:
            scored = [evaluate_one(x, opponent_spec, args.seeds, args.seed0,
                                   args.workers, args.swap, f"cand{i}",
                                   objective=args.objective)
                      for i, x in enumerate(xs)]
        else:
            scored = evaluate_batch(xs, opponent_spec, args.seeds, args.seed0,
                                    args.workers, args.swap,
                                    objective=args.objective)

        scores = [s for s, _d in scored]
        dropped = sum(d for _s, d in scored)
        es.tell(xs, [-s for s in scores])            # CMA-ES 最小化
        wall = time.perf_counter() - t0

        gen_best = max(scores)
        if gen_best > best_score:
            best_score, best_x = gen_best, xs[scores.index(gen_best)]
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump({"generation": gen, "score": best_score, "x": best_x,
                           "params": to_json_params(decode(best_x))},
                          f, ensure_ascii=False, indent=2)

        with open(state_dir / f"gen_{gen:04d}.pkl", "wb") as f:
            pickle.dump(es, f)
        row = {"generation": gen, "best": gen_best,
               "mean": statistics.fmean(scores), "worst": min(scores),
               "best_so_far": best_score, "sigma": float(es.sigma),
               "dropped_games": dropped, "wall_secs": wall}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        flag = f"  ⚠️ 作廢 {dropped} 局" if dropped else ""
        print(f"  gen {gen:4}  最佳 {gen_best:>10,.0f}  平均 "
              f"{row['mean']:>10,.0f}  歷來 {best_score:>10,.0f}  "
              f"sigma {es.sigma:.4f}  {wall:5.1f}s{flag}")

        if args.checkpoint_every and gen % args.checkpoint_every == 0:
            ck = checkpoint(best_x, args, opponent_spec, teams)
            ck.update({"generation": gen, "train": best_score})
            with open(ck_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ck, ensure_ascii=False) + "\n")
            teams_txt = "  ".join(f"{k} {v:,.0f}" for k, v in ck["teams"].items())
            print(f"    checkpoint  train {best_score:>10,.0f}   "
                  f"holdout {ck['holdout']:>10,.0f}   三隊平均 "
                  f"{ck['teams_mean']:>10,.0f}   ({teams_txt})")
            if prev_holdout is not None and ck["holdout"] <= prev_holdout:
                print(f"    ⚠️ holdout 沒有進步（上次 {prev_holdout:,.0f}）—— "
                      f"train 還在升的話就是 overfit 那組 seed，考慮收工。")
            prev_holdout = ck["holdout"]

    if best_x is not None:
        print(f"\n最佳 {best_score:,.0f}（train seed {args.seed0}~"
              f"{args.seed0 + args.seeds - 1}）")
        print(f"參數寫在 {state_dir / 'best.json'}")
    print(f"狀態在 {state_dir}")


if __name__ == "__main__":
    main()
