"""平行對戰 runner。

一次跑多局，用多行程把 20 核吃滿。設計依據 `docs/rules.md` §2.3：

- **配對種子**：同一個 seed 跑兩局，第二局把兩邊位置對調。市場行情、雜草運氣
  這些共同的隨機因素直接抵消，同樣的信心水準所需局數降到 1/3 以下。
- **不看單局**：輸出勝率 + Wilson 95% 信賴區間，不是一句「我這版比較強」。

對手可以是內建名字、`config/opponents/*.json` 的檔名、或任意路徑。
凍結的對手池放在 `config/ladder.json`（見 `docs/division-of-labor.md` §6）。

用法：

    python -m eval.runner --a gen0 --b starter --games 30 --workers 10
    python -m eval.runner --a gen0 --b config/opponents/gen0-wheat.json --games 30
    python -m eval.runner --a gen0 --ladder            # 打整個對手池
    python -m eval.runner --a gen0 --b random --games 100 --json out.json

`--games N` 是「幾個 seed」，開了配對（預設）實際會跑 2N 局。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
OPPONENT_DIR = CONFIG_DIR / "opponents"
DEFAULT_LADDER = CONFIG_DIR / "ladder.json"

#: 沒有 config 檔時的後備對照表。`builtin:` 是引擎內建的，直接把名字交給 env.run。
FALLBACK_REGISTRY = {
    "pass": {"builtin": "pass"},
    "random": {"builtin": "random"},
    "starter": {"builtin": "starter"},
    "gen0": {"entry": "agents.gen0:act"},
}


# --------------------------------------------------------------------------
# 對手載入
# --------------------------------------------------------------------------

def load_spec(name):
    """把 `--a` / `--b` 的值變成 spec dict。

    依序嘗試：
      1. 直接是存在的檔案路徑          → 讀 JSON
      2. `config/opponents/<name>.json` → 讀 JSON
      3. `FALLBACK_REGISTRY`
      4. 含 `:` 的話當成 `module:attr`
    """
    candidates = [Path(name), OPPONENT_DIR / f"{name}.json", REPO_ROOT / name]
    for path in candidates:
        if path.suffix == ".json" and path.is_file():
            with open(path, encoding="utf-8") as f:
                spec = json.load(f)
            spec.setdefault("name", path.stem)
            spec["_source"] = str(path)
            return spec

    if name in FALLBACK_REGISTRY:
        spec = dict(FALLBACK_REGISTRY[name])
        spec.setdefault("name", name)
        spec["_source"] = "fallback registry"
        return spec

    if ":" in name:
        return {"name": name, "entry": name, "_source": "inline"}

    raise SystemExit(
        f"找不到對手 {name!r}。\n"
        f"  可用的 config：{', '.join(sorted(p.stem for p in OPPONENT_DIR.glob('*.json'))) or '（config/opponents/ 是空的）'}\n"
        f"  內建名字：{', '.join(sorted(FALLBACK_REGISTRY))}\n"
        f"  或給一個 JSON 路徑，或 module:attr"
    )


def build_agent(spec):
    """把 spec 變成 env.run 吃得下的東西（內建名字字串，或 callable）。

    在 worker 行程裡呼叫 —— callable 不會被 pickle，所以帶 params 的 closure 沒問題。
    """
    if "builtin" in spec:
        return spec["builtin"]

    entry = spec["entry"]
    module_path, _, attr = entry.partition(":")
    attr = attr or "agent"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    module = __import__(module_path, fromlist=[attr])
    fn = getattr(module, attr)

    params = spec.get("params") or {}
    if not params:
        return fn

    def agent(obs, config):
        return fn(obs, config, params)

    return agent


def load_ladder(path=None):
    """讀對手池，回傳 (specs, meta)。

    `opponents` 的每個元素可以是：

    - 字串 —— 走 `load_spec` 查 `config/opponents/<name>.json`
    - dict —— 直接就是一個完整的 spec，不用另外開檔

    後者讓一個檔就能定義一整組參數掃描。
    """
    path = Path(path) if path else DEFAULT_LADDER
    if not path.is_file():
        raise SystemExit(f"找不到對手池 {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    specs = []
    for i, item in enumerate(data.get("opponents", [])):
        if isinstance(item, str):
            specs.append(load_spec(item))
        elif isinstance(item, dict):
            spec = dict(item)
            if "builtin" not in spec and "entry" not in spec:
                raise SystemExit(
                    f"{path} 的第 {i} 個對手缺 entry 或 builtin：{item!r}"
                )
            spec.setdefault("name", spec.get("entry", f"opponent-{i}"))
            spec["_source"] = f"{path} (inline)"
            specs.append(spec)
        else:
            raise SystemExit(f"{path} 的第 {i} 個對手格式不對：{item!r}")

    names = [s["name"] for s in specs]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise SystemExit(f"{path} 有重複的對手名字：{', '.join(sorted(dupes))}")

    return specs, data


# --------------------------------------------------------------------------
# 跑局
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _silenced():
    """把 fd 1/2 導到 devnull。

    `contextlib.redirect_stdout` 只換 Python 層的 `sys.stdout`，擋不住
    C++ extension 直接寫檔案描述子 —— open_spiel 就是這樣，一次噴 300+ 行
    遊戲清單。必須在 fd 層級擋。
    """
    with open(os.devnull, "w") as devnull:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def _quiet_make():
    """import kaggle_environments 時 open_spiel 會噴幾百行，擋掉。"""
    with _silenced():
        from kaggle_environments import make
    return make


#: 這個 worker 行程在 pool 裡的序號。由 `_init_worker` 在行程啟動時發一次。
_WORKER_IDX = 0


def _init_worker(counter):
    """Pool 的 initializer：發一個序號給這個行程，之後 log 檔名用得到。"""
    global _WORKER_IDX
    with counter.get_lock():
        _WORKER_IDX = counter.value
        counter.value += 1


def _play(job):
    """跑一局。這個函式必須是 top-level，Windows 的 spawn 才 pickle 得動。"""
    spec0, spec1, seed, a_slot, log_dir = job
    make = _quiet_make()
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)

    # 一個 worker 一個 log 檔：`worker-<對手>-<序號>.jsonl`。
    # 不讓多個行程共寫一個檔，因為同時 append 會交錯、把行切斷。
    # 一個檔裡混著這個 worker 跑過的所有局，靠每筆記錄的 `tag` 分辨是哪一局哪一邊。
    # agent 讀 KAGGRI_LOG_FILE 決定寫哪，沒設的話寫 stderr（會被引擎攔截丟掉）。
    if log_dir:
        opponent = (spec1 if a_slot == 0 else spec0)["name"]
        os.environ["KAGGRI_LOG_TAG"] = (
            f"seed{seed:04d}_{spec0['name']}_vs_{spec1['name']}")
        os.environ["KAGGRI_LOG_FILE"] = str(
            Path(log_dir) / f"worker-{opponent}-{_WORKER_IDX:02d}.jsonl")
    else:
        os.environ.pop("KAGGRI_LOG_FILE", None)
        os.environ.pop("KAGGRI_LOG_TAG", None)

    t0 = time.perf_counter()
    err = None
    try:
        env.run([build_agent(spec0), build_agent(spec1)])
    except Exception as exc:  # 產線要能跑完 400 局，單局炸掉不該拖垮整批
        err = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0

    rewards = [s.reward for s in env.state]
    statuses = [s.status for s in env.state]
    b_slot = 1 - a_slot
    land = _land_history(env)
    return {
        "seed": seed,
        "a_slot": a_slot,
        "cash_a": rewards[a_slot],
        "cash_b": rewards[b_slot],
        "status_a": statuses[a_slot],
        "status_b": statuses[b_slot],
        "land_a": land[a_slot],
        "land_b": land[b_slot],
        "steps": len(env.steps),
        "elapsed": elapsed,
        "error": err,
    }


def _land_history(env):
    """掃 `env.steps` 找出每一方買地的時機。

    回傳 [[{quadrant, day, cash}, ...], ...]，索引是 player id。
    `cash` 是買下去**之後**的現金 —— 引擎在同一回合先處理 HIRE / BUY_SEED
    再處理 BUY_LAND，所以這個數字是「買完還剩多少」。
    """
    out = [[], []]
    seen = [set(), set()]
    for step in env.steps:
        try:
            obs = step[0]["observation"]
            farms = obs["farms"]
            day = obs.get("day", 0)
        except (KeyError, TypeError, IndexError):
            continue
        if not farms:
            continue
        for pid, farm in enumerate(farms[:2]):
            for q in farm.get("unlocked_quadrants", []):
                if q not in seen[pid]:
                    seen[pid].add(q)
                    if len(seen[pid]) > 1:      # NW 是一開始就有的，不算買
                        out[pid].append(
                            {"quadrant": q, "day": day,
                             "cash": round(farm.get("money", 0))})
    return out


def build_jobs(spec_a, spec_b, games, seed0, swap=True, log_dir=None):
    """配對種子：每個 seed 跑一局正手、一局反手。"""
    jobs = []
    for i in range(games):
        seed = seed0 + i
        jobs.append((spec_a, spec_b, seed, 0, log_dir))
        if swap:
            jobs.append((spec_b, spec_a, seed, 1, log_dir))
    return jobs


def make_run_dir(label):
    """每次跑開一個 temp/<時間戳>_<標籤>/ 資料夾放這一輪的所有產物。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:60]
    run_dir = REPO_ROOT / "temp" / f"{stamp}_{safe}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------
# 統計
# --------------------------------------------------------------------------

def wilson(successes, n, z=1.96):
    """勝率的 Wilson 信賴區間。n 小的時候比常態近似可靠得多。"""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - spread) / denom, (centre + spread) / denom)


def _land_stats(buys_per_game):
    """買地統計。每一塊分開算 —— 只看第一塊的話，看不出第二、三塊有沒有買成。"""
    bought = [b for b in buys_per_game if b]

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    # 第 n 塊：幾局買到、平均第幾天、買完剩多少現金
    max_plots = max((len(b) for b in buys_per_game), default=0)
    plots = []
    for i in range(max_plots):
        got = [b[i] for b in buys_per_game if len(b) > i]
        plots.append({
            "n": i + 1,
            "games": len(got),
            "quadrant": got[0]["quadrant"] if got else None,
            "day": mean([g["day"] for g in got]),
            "cash": mean([g["cash"] for g in got]),
        })

    return {
        "games_bought": len(bought),
        "games": len(buys_per_game),
        "mean_plots": mean([len(b) for b in buys_per_game]),
        "first_day": mean([b[0]["day"] for b in bought]),
        "first_cash": mean([b[0]["cash"] for b in bought]),
        "quadrants": sorted({q["quadrant"] for b in bought for q in b}),
        "plots": plots,
    }


def summarise(results, name_a, name_b):
    wins = draws = losses = 0
    failures = []
    cash_a, cash_b, diffs, times = [], [], [], []
    land_a, land_b = [], []

    for r in results:
        times.append(r["elapsed"])
        land_a.append(r.get("land_a") or [])
        land_b.append(r.get("land_b") or [])
        bad = (
            r["error"]
            or r["cash_a"] is None
            or r["cash_b"] is None
            or r["status_a"] != "DONE"
            or r["status_b"] != "DONE"
        )
        if bad:
            failures.append(r)
            # reward 為 None 代表 TIMEOUT / ERROR，該局作廢，不計入勝負
            continue
        ca, cb = r["cash_a"], r["cash_b"]
        cash_a.append(ca)
        cash_b.append(cb)
        diffs.append(ca - cb)
        if ca > cb:
            wins += 1
        elif ca < cb:
            losses += 1
        else:
            draws += 1

    n = wins + draws + losses
    score = wins + 0.5 * draws
    lo, hi = wilson(score, n)

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    return {
        "a": name_a,
        "b": name_b,
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": (score / n) if n else float("nan"),
        "ci95": [lo, hi],
        "mean_cash_a": mean(cash_a),
        "mean_cash_b": mean(cash_b),
        "mean_diff": mean(diffs),
        "mean_secs_per_game": mean(times),
        "land_a": _land_stats(land_a),
        "land_b": _land_stats(land_b),
        "failures": len(failures),
        "failure_detail": failures[:5],
    }


def run(spec_a, spec_b, games, workers, seed0=0, swap=True, progress=None, log_dir=None):
    # 進度計數只在互動終端有用。重導到檔案時印出來只是幾百行垃圾。
    if progress is None:
        progress = sys.stderr.isatty()
    jobs = build_jobs(spec_a, spec_b, games, seed0, swap, log_dir)
    results = []
    wall0 = time.perf_counter()

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            results.append(_play(job))
            if progress:
                print(f"\r  {i}/{len(jobs)}", end="", file=sys.stderr, flush=True)
    else:
        counter = mp.Value("i", 0)
        with mp.Pool(processes=workers, initializer=_init_worker,
                     initargs=(counter,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_play, jobs), 1):
                results.append(res)
                if progress:
                    print(f"\r  {i}/{len(jobs)}", end="", file=sys.stderr, flush=True)
    if progress:
        print("\r" + " " * 20 + "\r", end="", file=sys.stderr, flush=True)

    wall = time.perf_counter() - wall0
    summary = summarise(results, spec_a["name"], spec_b["name"])
    summary["wall_secs"] = wall
    summary["games_per_hour"] = (len(jobs) / wall * 3600) if wall > 0 else float("inf")
    summary["workers"] = workers
    return summary, results


# --------------------------------------------------------------------------
# 輸出
# --------------------------------------------------------------------------

def verdict(s):
    lo, hi = s["ci95"]
    if lo > 0.5:
        return "✅ 確實較強"
    if hi < 0.5:
        return "❌ 確實較弱"
    return "⚪ 判不出來"


def _fmt_land(ls):
    """買地：幾局有買 / 平均幾塊 / 每一塊分別在第幾天買、買完剩多少現金。"""
    if not ls["games_bought"]:
        return "沒買地"
    per_plot = "   ".join(
        f"第{p['n']}塊 {p['quadrant']} {p['games']}/{ls['games']}局 "
        f"day {p['day']:.1f} 剩 ${p['cash']:,.0f}"
        for p in ls.get("plots", [])
    )
    return (f"平均 {ls['mean_plots']:.2f} 塊   {per_plot}")


def format_summary(s):
    lo, hi = s["ci95"]
    lines = [
        f"{s['a']}  vs  {s['b']}",
        f"  局數      {s['games']}"
        + (f"   ⚠️ 作廢 {s['failures']} 局" if s["failures"] else ""),
        f"  勝/和/負  {s['wins']} / {s['draws']} / {s['losses']}",
        f"  得分率    {s['score_rate']:.1%}   95% CI [{lo:.1%}, {hi:.1%}]",
        f"  平均現金  {s['a']} {s['mean_cash_a']:,.0f}"
        f"   vs   {s['b']} {s['mean_cash_b']:,.0f}"
        f"   （差 {s['mean_diff']:+,.0f}）",
        f"  買地 {s['a']:<10} {_fmt_land(s['land_a'])}",
        f"  買地 {s['b']:<10} {_fmt_land(s['land_b'])}",
        f"  吞吐      {s['games_per_hour']:,.0f} 局/小時"
        f"   （{s['workers']} workers，單局 {s['mean_secs_per_game']:.2f} 秒）",
        f"  判定      {verdict(s)}",
    ]
    for f in s["failure_detail"]:
        lines.append(
            f"  ⚠️ seed {f['seed']}: {f['error'] or f['status_a'] + '/' + f['status_b']}"
        )
    return "\n".join(lines)


def _plots_cell(ls, max_plots=3):
    """`20/8/0` = 20 局買到第一塊、8 局買到第二塊、沒人買到第三塊。"""
    got = {p["n"]: p["games"] for p in ls.get("plots", [])}
    return "/".join(str(got.get(i + 1, 0)) for i in range(max_plots))


def _dw(text):
    """顯示寬度。CJK 是雙寬，`str.__format__` 的 `:<20` 卻按字元數補，
    含中文的欄位標題會跟底下的數字對不齊。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _cell(text, width, align=">"):
    pad = " " * max(0, width - _dw(text))
    return pad + text if align == ">" else text + pad


def format_ladder(summaries, name_a="A"):
    # 欄位標題直接寫 A 是誰 —— 一張表裡 A 固定、B 每列不同，只寫 A/B 會看不懂。
    a = name_a[:12]
    # 買地欄顯示「第1塊/第2塊/第3塊 各有幾局買到」，例如 20/8/0 =
    # 20 局都買了第一塊、8 局買到第二塊、沒有人買到第三塊。
    # 原本只印 games_bought/games，那是「有沒有買地」，看不出買了幾塊。
    cols = [
        ("對手 (B)", 18, "<"),
        ("勝/和/負", 10, ">"),
        ("得分率", 8, ">"),
        ("95% CI", 14, ">"),
        (a + " 現金", 13, ">"),
        ("對手現金", 12, ">"),
        ("差", 11, ">"),
        (a + " 塊數", 13, ">"),
        ("對手塊數", 12, ">"),
        ("首塊", 6, ">"),
        ("判定", 12, "<"),
    ]
    head = " ".join(_cell(t, w, al) for t, w, al in cols)
    lines = [f"A = {name_a}（勝負、現金差都是從 A 的角度看）", head,
             "-" * _dw(head)]
    for s in summaries:
        lo, hi = s["ci95"]
        la, lb = s["land_a"], s["land_b"]
        cells = [
            s["b"],
            "{}/{}/{}".format(s["wins"], s["draws"], s["losses"]),
            f"{s['score_rate']:.1%}",
            "[{:.0%}, {:.0%}]".format(lo, hi),
            f"{s['mean_cash_a']:,.0f}",
            f"{s['mean_cash_b']:,.0f}",
            f"{s['mean_diff']:+,.0f}",
            _plots_cell(la),
            _plots_cell(lb),
            (f"d{la['first_day']:.0f}" if la["games_bought"] else "—"),
            verdict(s),
        ]
        lines.append(" ".join(_cell(v, w, al)
                              for v, (_t, w, al) in zip(cells, cols)))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="平行跑多局對戰並做統計判定",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    ap.add_argument("--a", default="gen0", help="A 方：內建名字 / config 檔名 / JSON 路徑 / module:attr")
    ap.add_argument("--b", help="B 方，同上。跟 --ladder 二選一")
    ap.add_argument("--ladder", nargs="?", const=str(DEFAULT_LADDER),
                    help="打整個對手池（預設 config/ladder.json）")
    ap.add_argument("--games", type=int, default=20, help="幾個 seed（開配對時實際局數 ×2）")
    ap.add_argument("--workers", type=int, default=0, help="行程數，0 = CPU 數 - 2")
    ap.add_argument("--seed0", type=int, default=0, help="起始 seed")
    ap.add_argument("--no-swap", action="store_true", help="關掉配對種子（不對調位置）")
    ap.add_argument("--json", help="額外把完整結果複製到這個路徑")
    ap.add_argument("--log-level", type=int,
                    help="agent 的 KAGGRI_LOG_LEVEL：0 安靜 / 2 每回合 / 3 決策細節。"
                         "非 0 時每一局會在 run 資料夾的 logs/ 下產一個 .jsonl")
    args = ap.parse_args(argv)

    # Windows 重導到檔案時 Python 會改用系統 locale（正體中文是 cp950），
    # 判定欄的 ⚪ ✅ ❌ 編不出來就整個 print 拋 UnicodeEncodeError。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):     # 不是真的 TextIOWrapper 就算了
            pass

    if not args.b and not args.ladder:
        ap.error("要給 --b 或 --ladder")

    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)
    spec_a = load_spec(args.a)

    if args.log_level is not None:
        os.environ["KAGGRI_LOG_LEVEL"] = str(args.log_level)

    label = f"{spec_a['name']}_vs_" + (
        Path(args.ladder).stem if args.ladder else load_spec(args.b)["name"])
    run_dir = make_run_dir(label)
    print(f"run 目錄  {run_dir.relative_to(REPO_ROOT)}\n")
    log_dir = run_dir / "logs"

    if args.ladder:
        specs_b, meta = load_ladder(args.ladder)
        # 不濾掉跟 A 同名的對手 —— 鏡像對局是有用的檢查：
        # 開了配對種子的話結果應該剛好 50%、現金差 0，不然就是有不對稱的東西。
        print(f"對手池 {meta.get('name', args.ladder)}"
              + (f"（凍結於 {meta['frozen']}）" if meta.get("frozen") else ""))
        print(f"A = {spec_a['name']}   每個對手 {args.games} seeds"
              f"{' ×2（配對）' if not args.no_swap else ''}   {workers} workers\n")
        summaries, all_results = [], {}
        for spec_b in specs_b:
            s, r = run(spec_a, spec_b, args.games, workers, args.seed0,
                       swap=not args.no_swap, log_dir=log_dir)
            summaries.append(s)
            all_results[spec_b["name"]] = r
            if sys.stderr.isatty():
                print(f"  {s['b']:<22} 完成 ({s['games']} 局)", file=sys.stderr)
        print(format_ladder(summaries, spec_a["name"]))
        payload = {"a": spec_a, "ladder": meta, "summaries": summaries,
                   "results": all_results}
        table = format_ladder(summaries, spec_a["name"])
    else:
        spec_b = load_spec(args.b)
        s, r = run(spec_a, spec_b, args.games, workers, args.seed0,
                   swap=not args.no_swap, log_dir=log_dir)
        payload = {"a": spec_a, "b": spec_b, "summary": s, "results": r}
        table = format_summary(s)

    payload["run_dir"] = str(run_dir)
    payload["argv"] = sys.argv[1:]
    payload["log_level"] = os.environ.get("KAGGRI_LOG_LEVEL", "(agent 預設)")

    # ⚠️ 先落地再印。跑完 240 局之後 print 掛掉（cp950 編不出 ⚪）整輪白跑過一次。
    targets = [run_dir / "result.json"]
    if args.json:
        targets.append(Path(args.json))
    for out in targets:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(table + "\n")

    print(table)
    n_logs = len(list(log_dir.glob("*.jsonl")))
    size_mb = sum(p.stat().st_size for p in run_dir.rglob("*")) / 2 ** 20
    print(f"\n產物 → {run_dir.relative_to(REPO_ROOT)}"
          f"   （result.json + summary.txt + logs/ 共 {n_logs} 個 .jsonl，"
          f"{size_mb:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
