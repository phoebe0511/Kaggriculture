"""search 到底改了什麼？—— 把「有 search」和「沒 search」同一局的帳攤開來比。

    python -m tools.search_ledger temp/search-game-20260826-233232 --seed 7
    python -m tools.search_ledger <run> --seed 7 --plot

## 要回答什麼

2026-08-27 量到三件事（journal §3）：整局現金變高是恆等式；**8/10 局對手的
現金也上升**，平均 +15,161；seed 7 甚至對手漲得比我方還多（+57,917 對
+50,265）。那時只知道「市場是共用的、對手是反應式的」兩條可能的路，
**沒有量過哪一條是主因**。

這支把同一個 seed 跑兩次 —— 一次重播 progress 裡記下的 search 動作，一次純
round6 不搜 —— 兩局都 hook `_commit_unit`（`kaggriculture.py:652`，每一單位
成交的唯一入口，`tools/revenue.py` 用同一招），記下**雙方**每一筆實際成交價。
然後比差額。

於是可以直接看到：

- search 讓我方多賣了什麼、多買了什麼
- **對手多賺的那筆錢從哪裡來** —— 它多賣了東西？還是同樣的東西賣得更貴？
- 市價被推去哪裡（價格是庫存的函數，`kaggriculture.py:192-206`）

## 為什麼不能用 `tools/revenue.py`

那支是 `env.run([a, b])` 一路跑完，塞不進「在某些回合改用搜出來的動作」。
這支自己驅動迴圈，重播的做法跟 `tools/search_curse_probe.py` 一樣
（引擎決定性，自我檢查 41/41 逐位元組復現過）。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.revenue import Ledger, _silenced  # noqa: E402  帳本與 fd 靜音共用

with _silenced():
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from kaggle_environments import make
    from eval.runner import build_agent, load_spec


PRODUCTS_SHOWN = 9


def _read(run_dir, seed):
    """progress -> (對手, [search 那些行], done 那一行)。"""
    f = Path(run_dir) / "progress" / f"seed{seed:04d}.jsonl"
    if not f.is_file():
        raise SystemExit(f"{f} 不存在")
    opponent, rows, done = None, [], None
    for line in io.open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["phase"] == "baseline":
            opponent = r.get("opponent")
        elif r["phase"] == "search":
            rows.append(r)
        elif r["phase"] == "done":
            done = r
    if done is None:
        raise SystemExit(f"seed {seed} 沒跑完（progress 沒有 done 那一行）")
    if any("action" not in r for r in rows):
        raise SystemExit(f"seed {seed} 是舊格式，沒記動作，重播不了")
    return opponent, rows, done


def play(seed, opponent, hours, days, a_slot, replay, act_me):
    """跑一局，回傳 `(兩個 Ledger, 逐日現金, 逐日市價)`。

    `replay` 給 `None` 就是純 policy（不搜）；給 progress 的那些行就重播。
    """
    books = [Ledger(), Ledger()]
    farm_ids = {}
    st_day = {"day": 0}
    orig_pm, orig_cu = K._process_market, K._commit_unit

    def pm(state, env):
        # farm dict 每個 step 都重建，所以每次進市場都要重新對照身分
        # （`tools/revenue.py` 的註解：快取 id 會拿到 337 個對不上的 id）。
        farm_ids.clear()
        for pid, farm in enumerate(state[0].observation.farms):
            farm_ids[id(farm)] = pid
        st_day["day"] = state[0].observation.day
        return orig_pm(state, env)

    def cu(op, item, price, farm, private, market, shed_capacity=100):
        ok = orig_cu(op, item, price, farm, private, market, shed_capacity)
        pid = farm_ids.get(id(farm))
        if ok and pid is not None:
            bk = books[pid]
            if op == "SELL":
                bk.sell_amt[item] += price
                bk.sell_qty[item] += 1
            else:
                bk.buy_amt[(op, item)] += price
                bk.buy_qty[(op, item)] += 1
            bk.day_amt[(st_day["day"], item, op)] += price
            bk.day_qty[(st_day["day"], item, op)] += 1
        return ok

    K._process_market, K._commit_unit = pm, cu
    cash, prices = [], []
    try:
        with _silenced():
            env = make("kaggriculture", configuration={"seed": seed}, debug=False)
        cfg = env.configuration
        import inspect
        _opp = build_agent(load_spec(opponent))
        if isinstance(_opp, str):
            _opp = env.agents[_opp]
        if len(inspect.signature(_opp).parameters) == 1:
            def act_them(obs):
                return _opp(obs)
        else:
            def act_them(obs):
                return _opp(obs, cfg)

        queue = list(replay or [])
        seen_day = -1
        books[0].start = books[1].start = float(
            env.state[0].observation["farms"][0]["money"])
        while env.state[0].observation["day"] < days and not env.done:
            obs = env.state[a_slot].observation
            day = int(obs["day"])
            if day != seen_day:
                seen_day = day
                cash.append((day,
                             float(obs["farms"][a_slot]["money"]),
                             float(obs["farms"][1 - a_slot]["money"])))
                prices.append((day, dict(obs["market"]["prices"]),
                               dict(obs["market"]["inventory"]),
                               list(obs["town"].get("unlocked_shops", []))))
            action = act_me(obs, cfg)
            if int(obs.get("hour", 0)) in hours and queue:
                row = queue.pop(0)
                if (day, int(obs.get("hour", 0))) != (row["day"], row["hour"]):
                    raise RuntimeError(
                        f"重播對不上：走到 day {day} hour {obs.get('hour')}，"
                        f"紀錄是 day {row['day']} hour {row['hour']}")
                action = row["action"]
            actions = [None, None]
            actions[a_slot] = action
            actions[1 - a_slot] = act_them(
                env.state[1 - a_slot].observation)
            env.step(actions)
        for pid, bk in enumerate(books):
            bk.final = float(
                env.state[0].observation["farms"][pid]["money"])
    finally:
        K._process_market, K._commit_unit = orig_pm, orig_cu
    return books, cash, prices


def _diff_table(title, a, b, top=PRODUCTS_SHOWN):
    """`a` 是 search 版、`b` 是不搜的。列出金額差最大的品項。"""
    L = ["", title,
         f"  {'品項':<14}{'search 賣':>12}{'不搜 賣':>12}{'差':>12}"
         f"{'search 量':>11}{'不搜 量':>10}{'量差':>8}"]
    items = set(a.sell_amt) | set(b.sell_amt)
    for it in sorted(items, key=lambda i: -abs(a.sell_amt[i] - b.sell_amt[i]))[:top]:
        L.append(f"  {it:<14}{a.sell_amt[it]:>12,.0f}{b.sell_amt[it]:>12,.0f}"
                 f"{a.sell_amt[it] - b.sell_amt[it]:>+12,.0f}"
                 f"{a.sell_qty[it]:>11,}{b.sell_qty[it]:>10,}"
                 f"{a.sell_qty[it] - b.sell_qty[it]:>+8,}")
    L += ["", f"  {'採購':<22}{'search':>12}{'不搜':>12}{'差':>12}"]
    keys = set(a.buy_amt) | set(b.buy_amt)
    for k in sorted(keys, key=lambda k: -abs(a.buy_amt[k] - b.buy_amt[k]))[:6]:
        name = f"{k[0]} {k[1]}"
        L.append(f"  {name:<22}{a.buy_amt[k]:>12,.0f}{b.buy_amt[k]:>12,.0f}"
                 f"{a.buy_amt[k] - b.buy_amt[k]:>+12,.0f}")
    L += ["",
          f"  {'':<22}{'search':>12}{'不搜':>12}{'差':>12}",
          f"  {'起始現金':<22}{a.start:>12,.0f}{b.start:>12,.0f}"
          f"{a.start - b.start:>+12,.0f}",
          f"  {'+ 賣出':<22}{a.revenue:>12,.0f}{b.revenue:>12,.0f}"
          f"{a.revenue - b.revenue:>+12,.0f}",
          f"  {'- 採購':<22}{a.purchases:>12,.0f}{b.purchases:>12,.0f}"
          f"{a.purchases - b.purchases:>+12,.0f}",
          f"  {'- 工資與買地':<22}{a.other:>12,.0f}{b.other:>12,.0f}"
          f"{a.other - b.other:>+12,.0f}   （差額推得）",
          f"  {'= 期末現金':<22}{a.final:>12,.0f}{b.final:>12,.0f}"
          f"{a.final - b.final:>+12,.0f}"]
    return L


def _avg_price_table(pa, pb):
    """兩局的市價：全季平均，以及差最大的那幾天。"""
    items = sorted(pa[0][1])
    # 🩸 價格是庫存的函數（`kaggriculture.py:192-206`：庫存低於 I0 價漲、
    # 高於 I0 價跌）。所以價格要跟庫存一起看，不然會把因果講反。
    L = ["", "## 市價與市場庫存（全季平均）",
         f"  {'品項':<12}{'價 search':>11}{'價 不搜':>10}{'差 %':>8}"
         f"{'   庫存 search':>14}{'庫存 不搜':>11}{'庫存差':>9}"]
    for it in items:
        ma = sum(p[it] for _d, p, _i, _s in pa) / len(pa)
        mb = sum(p[it] for _d, p, _i, _s in pb) / len(pb)
        ia = sum(i.get(it, 0) for _d, _p, i, _s in pa) / len(pa)
        ib = sum(i.get(it, 0) for _d, _p, i, _s in pb) / len(pb)
        L.append(f"  {it:<12}{ma:>11,.1f}{mb:>10,.1f}"
                 f"{(ma - mb) / mb * 100 if mb else 0:>+8.1f}"
                 f"{ia:>14,.0f}{ib:>11,.0f}{ia - ib:>+9,.0f}")
    L.append("  庫存低於 I0 -> 價漲；高於 I0 -> 價跌（kaggriculture.py:192-206）")
    # 🩸 庫存被鎮上的店消耗（`_town_consume`，:728-749），而**店是環境 RNG
    # 抽的**（:891 `rng.choice(sorted(SHOPS))`），不是玩家蓋的。我方動作改變
    # 之後如果 RNG 的消耗次數不同，抽到的店就會不同 -> 消耗的品項不同 ->
    # 價格不同。那是**純運氣的通道**，一定要驗。
    sa = pa[-1][3]
    sb = pb[-1][3]
    L += ["", "## 鎮上解鎖的店（環境 RNG 抽的，不是玩家蓋的）",
          f"  search  {sa}",
          f"  不搜    {sb}",
          f"  -> {'一樣' if sa == sb else '🩸 不一樣 —— 我方動作改變了 RNG 的走向'}"]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--plot", action="store_true", help="另外存一張 PNG")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    run_dir = Path(args.run_dir)
    saved = json.loads(io.open(run_dir / "config.json", encoding="utf-8").read())
    if saved.get("policy") != "net":
        raise SystemExit(f"這批的 policy 是 {saved.get('policy')!r}，這支只支援 net")
    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    os.environ["KAGGRI_WEIGHTS"] = saved["weights"]
    # 🩸 `agents/gen2_model.py:111` 的 WEIGHTS_PATH 是 import 當下決定的，
    # 一定要在設好環境變數之後才 import。
    from agents.gen2_model import act as net_act

    hours = frozenset(int(h) for h in str(saved["hours"]).split(","))
    days, a_slot = saved["days"], saved["a_slot"]
    opponent, rows, done = _read(run_dir, args.seed)

    changed = [r for r in rows if r.get("gain", 0) > 0]
    L = [f"{run_dir.name}   seed {args.seed}   對手 {opponent}",
         f"決策點 {len(rows)} 個，search 改掉 {len(changed)} 個"
         f"（{len(changed) / max(1, len(rows)):.1%}）"]

    bk_s, cash_s, price_s = play(args.seed, opponent, hours, days, a_slot,
                                 rows, net_act)
    bk_b, cash_b, price_b = play(args.seed, opponent, hours, days, a_slot,
                                 None, net_act)

    # 🩸 對帳：重播出來的期末現金必須等於 progress 記的，不然重播沒回到原軌跡。
    L += ["", "## 對帳（重播 vs progress 記錄）",
          f"  我方 search  {bk_s[a_slot].final:>12,.0f}"
          f"   progress {done['cash']:>12,.0f}"
          f"   {'✓' if abs(bk_s[a_slot].final - done['cash']) < 0.5 else '✗ 對不上'}",
          f"  我方 不搜    {bk_b[a_slot].final:>12,.0f}"
          f"   progress {done['baseline']:>12,.0f}"
          f"   {'✓' if abs(bk_b[a_slot].final - done['baseline']) < 0.5 else '✗ 對不上'}",
          f"  對手 search  {bk_s[1 - a_slot].final:>12,.0f}"
          f"   progress {done['opp']:>12,.0f}"
          f"   {'✓' if abs(bk_s[1 - a_slot].final - done['opp']) < 0.5 else '✗ 對不上'}",
          f"  對手 不搜    {bk_b[1 - a_slot].final:>12,.0f}"
          f"   progress {done['opp_baseline']:>12,.0f}"
          f"   {'✓' if abs(bk_b[1 - a_slot].final - done['opp_baseline']) < 0.5 else '✗ 對不上'}"]

    L += _diff_table("## 我方", bk_s[a_slot], bk_b[a_slot])
    L += _diff_table("## 對手（它多賺的錢從哪來）", bk_s[1 - a_slot],
                     bk_b[1 - a_slot])
    L += _avg_price_table(price_s, price_b)

    L += ["", "## search 改掉的決策點（gain 前 12 大）",
          f"  {'day':>4}{'hour':>5}  {'候選':<10}{'gain':>10}   動作"]
    for r in sorted(changed, key=lambda r: -r["gain"])[:12]:
        act = r["action"]
        brief = (f"farmer={act.get('farmer')} "
                 f"hands={act.get('hands')} market={act.get('market')}")
        L.append(f"  {r['day']:>4}{r['hour']:>5}  {r['label']:<10}"
                 f"{r['gain']:>10,}   {brief[:110]}")

    text = "\n".join(L)
    out = Path(args.out) if args.out else Path(
        f"temp/_ledger-seed{args.seed:04d}.txt")
    io.open(out, "w", encoding="utf-8").write(text + "\n")
    print(text)
    print(f"\n-> {out}")

    if args.plot:
        _plot(args.seed, cash_s, cash_b, price_s, price_b,
              out.with_suffix(".png"))
        print(f"-> {out.with_suffix('.png')}")
    return 0


def _plot(seed, cash_s, cash_b, price_s, price_b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(price_s[0][1])
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    ax = axes[0][0]
    d = [x[0] for x in cash_s]
    ax.plot(d, [x[1] for x in cash_s], label="我方 search")
    ax.plot(d, [x[1] for x in cash_b], "--", label="我方 不搜")
    ax.plot(d, [x[2] for x in cash_s], label="對手 search")
    ax.plot(d, [x[2] for x in cash_b], "--", label="對手 不搜")
    ax.set_title(f"seed {seed} 現金")
    ax.set_xlabel("day")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    # 🩸 九個產品的 base 差 10 倍（WHEAT $25 vs MELON $250），一張圖畫九條線
    # 低價的會被壓平（`tools/plot_prices.py` 的同一個理由）。所以一個產品一格。
    for i, it in enumerate(items[:9]):
        a = axes[(i + 1) // 5][(i + 1) % 5]
        a.plot(d, [p[it] for _x, p, _i, _s in price_s], label="search")
        a.plot(d, [p[it] for _x, p, _i, _s in price_b], "--", label="不搜")
        a.set_title(it, fontsize=9)
        a.grid(alpha=.3)
        if i == 0:
            a.legend(fontsize=7)
    for f in ("Microsoft JhengHei", "Microsoft YaHei", "SimHei"):
        if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [f]
            plt.rcParams["axes.unicode_minus"] = False
            break
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
