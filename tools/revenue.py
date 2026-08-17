"""把一局（或多局）的收支**按金額**拆開，不是按數量。

    python tools/revenue.py --a ref-v4 --b ref-v4 --seeds 0
    python tools/revenue.py --a gen1 --b ref-v4 --seeds 0-9      # 多局取平均
    python tools/revenue.py --a gen1 --b ref-v4 --seeds 0 --side both
    python tools/revenue.py --a gen1 --b ref-v4 --seeds 0 --by-day WHEAT

## 為什麼不能從 log 算

log 只記得送出去的 SELL 訂單（品項 + 數量），**沒有實收金額**。引擎是
逐單位重新報價的（`kaggriculture.py:583-627`），賣 40 個 WHEAT 這 40 個
單價都不一樣，而且兩個 player 交錯結算 —— 拿當回合的 `prices` 乘數量會錯。

所以這支工具**自己跑一局**，hook `_commit_unit`（`kaggriculture.py:652`，
每一單位成交的唯一入口）記下實際成交價。輸出的帳會對平：

    起始現金 + 賣出 - 採購 - (工資 + 買地) = 期末現金

「工資 + 買地」是差額推出來的，不是直接量的 —— `_commit_unit` 只處理
SELL / BUY_*，HIRE 和 BUY_LAND 走別的路徑。

驗算過：gen1 與 ref-v4 在 seed 0~3 全都是 $10,538.00 整
= 買地 $3,000（`LAND_PRICES[0:2]` = 1000 + 2000）+ 工資 $7,538。
兩邊相同是因為雇工排程和買地時機不受 seed 影響（80 局統計：第 1 塊
day 9 是 80/80，第 2 塊 day 11 是 80/80），不是把兩個 player 算混了。

## 為什麼要看金額

實測 seed 0（ref-v4 對打自己）：WHEAT 賣了 583 個是全場數量第一，佔總
出貨量 44%，但只佔收入 19.8%（均價 $38.4）。而且同一局買回 465 個
（$17,542，採購支出第一名），**淨貢獻只剩 $4,867 = 總收入的 4.3%**。
只看數量會把 WHEAT 當成主力，看金額才知道主力是 STRAWBERRY 和 MILK。
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@contextlib.contextmanager
def _silenced():
    """fd 層級靜音。open_spiel 是 C++ extension，直接寫檔案描述子，
    `redirect_stdout` 只換 Python 層的 `sys.stdout`，攔不到。"""
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


with _silenced():
    from kaggle_environments import make
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from eval.runner import load_spec, build_agent


class Ledger:
    """一個 player 的收支明細。金額全部是引擎實際成交的價格。"""

    def __init__(self):
        self.sell_amt = Counter()      # item -> 實收金額
        self.sell_qty = Counter()
        self.buy_amt = Counter()       # (op, item) -> 支出金額
        self.buy_qty = Counter()
        # (day, item, op) -> 金額。op 保留原樣（SELL / BUY_PRODUCT / BUY_SEED …）
        # —— WHEAT 同時是作物和飼料，把 BUY_SEED 和 BUY_PRODUCT 併在一起會
        # 讓逐日的買量對不上採購表（實測 615 vs 465，差的正是 150 個種子）。
        self.day_amt = Counter()
        self.day_qty = Counter()
        self.final = 0.0
        self.start = 0.0

    @property
    def revenue(self):
        return sum(self.sell_amt.values())

    @property
    def purchases(self):
        return sum(self.buy_amt.values())

    @property
    def other(self):
        """工資 + 買地。差額推出來的 —— 見模組說明。"""
        return self.start + self.revenue - self.purchases - self.final

    def net(self, item):
        """這個品項的淨貢獻：賣掉的錢減去買回同一品項的錢。

        WHEAT 是飼料也是作物，會同時出現在兩邊。
        """
        back = sum(v for (op, it), v in self.buy_amt.items()
                   if it == item and op == "BUY_PRODUCT")
        return self.sell_amt[item] - back


def play(spec_a, spec_b, seed):
    """跑一局，回傳兩個 player 的 `Ledger`。"""
    books = [Ledger(), Ledger()]
    farm_ids = {}                       # id(farm) -> player_id
    state = {"day": 0}

    orig_pm, orig_cu = K._process_market, K._commit_unit

    def pm(st, env):
        # farm dict 每個 step 都重建，所以每次進市場都要重新對照身分。
        # 實測直接快取 id 會拿到 337 個不同的 id，全部對不上。
        farm_ids.clear()
        for pid, farm in enumerate(st[0].observation.farms):
            farm_ids[id(farm)] = pid
        state["day"] = st[0].observation.day
        return orig_pm(st, env)

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
            bk.day_amt[(state["day"], item, op)] += price
            bk.day_qty[(state["day"], item, op)] += 1
        return ok

    K._process_market, K._commit_unit = pm, cu
    try:
        with _silenced():
            a = build_agent(spec_a)
            b = build_agent(spec_b)
            env = make("kaggriculture", configuration={"seed": seed}, debug=False)
            start = float(env.state[0].observation.farms[0]["money"])
            env.run([a, b])
        for pid, bk in enumerate(books):
            bk.start = start
            bk.final = float(env.state[pid].reward)
    finally:
        K._process_market, K._commit_unit = orig_pm, orig_cu
    return books


def merge(books):
    """把多局的 Ledger 相加，金額之後再除以局數就是平均。"""
    out = Ledger()
    for bk in books:
        out.sell_amt += bk.sell_amt
        out.sell_qty += bk.sell_qty
        out.buy_amt += bk.buy_amt
        out.buy_qty += bk.buy_qty
        out.day_amt += bk.day_amt
        out.day_qty += bk.day_qty
        out.start += bk.start
        out.final += bk.final
    return out


def _bar(frac, width=18):
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def report(bk, name, games, by_day=None):
    div = max(1, games)
    rev = bk.revenue
    lines = []
    lines.append(f"### {name}"
                 + (f"（{games} 局平均）" if games > 1 else ""))
    lines.append("")
    lines.append("品項          賣出量     實收金額    佔比  均價     "
                 "買回金額     淨貢獻  淨佔比")
    lines.append("-" * 88)
    for item, amt in bk.sell_amt.most_common():
        q = bk.sell_qty[item]
        back = sum(v for (op, it), v in bk.buy_amt.items()
                   if it == item and op == "BUY_PRODUCT")
        net = amt - back
        lines.append(
            f"  {item:11s} {q/div:7.0f} {amt/div:11,.0f} {amt/rev*100:6.1f}% "
            f"${amt/max(1,q):6.1f} {back/div:11,.0f} {net/div:10,.0f} "
            f"{net/rev*100:6.1f}%  {_bar(net/rev)}")
    lines.append("")
    lines.append("採購           數量       金額")
    lines.append("-" * 40)
    for (op, item), amt in bk.buy_amt.most_common():
        lines.append(f"  {op:12s} {item:11s} "
                     f"{bk.buy_qty[(op, item)]/div:6.0f} {amt/div:10,.0f}")
    lines.append("")
    lines.append(f"  起始現金        {bk.start/div:11,.0f}")
    lines.append(f"+ 賣出          {bk.revenue/div:11,.0f}")
    lines.append(f"- 採購          {bk.purchases/div:11,.0f}")
    lines.append(f"- 工資與買地     {bk.other/div:11,.0f}   （差額推得，非直接量測）")
    lines.append(f"= 期末現金      {bk.final/div:11,.0f}")

    if by_day:
        item = by_day.upper()
        lines.append("")
        # 只比 BUY_PRODUCT 和 SELL —— 那兩個才是同一個市場的來回。
        # BUY_SEED / BUY_ANIMAL 是固定牌價，不進市場庫存。
        lines.append(f"### {item} 逐 3 天（{'平均' if games > 1 else '單局'}）"
                     f"  買 = BUY_PRODUCT，不含種子與動物")
        lines.append("天       買量      買金額      賣量      賣金額        淨額")
        lines.append("-" * 64)
        days = sorted({d for (d, it, _) in bk.day_amt if it == item})
        for lo in range(0, (max(days) + 1 if days else 0), 3):
            rng = range(lo, lo + 3)
            bq = sum(bk.day_qty[(d, item, "BUY_PRODUCT")] for d in rng)
            ba = sum(bk.day_amt[(d, item, "BUY_PRODUCT")] for d in rng)
            sq = sum(bk.day_qty[(d, item, "SELL")] for d in rng)
            sa = sum(bk.day_amt[(d, item, "SELL")] for d in rng)
            lines.append(f"{lo:2d}-{lo+2:2d} {bq/div:9.0f} {ba/div:11,.0f} "
                         f"{sq/div:9.0f} {sa/div:11,.0f} {(sa-ba)/div:11,.0f}")
    return "\n".join(lines)


def parse_seeds(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="按金額拆解一局的收支（不是按數量）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--a", default="ref-v4", help="player 0：名字 / config / .py / module:attr")
    ap.add_argument("--b", default="ref-v4", help="player 1，同上")
    ap.add_argument("--seeds", default="0", help="例：0 或 0-9 或 0,3,7")
    ap.add_argument("--side", choices=["a", "b", "both"], default="a",
                    help="要看哪一邊的帳（預設 a）")
    ap.add_argument("--by-day", metavar="ITEM",
                    help="額外印這個品項的逐 3 天買賣，例：WHEAT")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec_a, spec_b = load_spec(args.a), load_spec(args.b)
    seeds = parse_seeds(args.seeds)

    got = [[], []]
    for seed in seeds:
        books = play(spec_a, spec_b, seed)
        got[0].append(books[0])
        got[1].append(books[1])

    names = [spec_a.get("name", args.a), spec_b.get("name", args.b)]
    sides = {"a": [0], "b": [1], "both": [0, 1]}[args.side]
    print(f"# 收支拆解  seed {args.seeds}   A = {names[0]}   B = {names[1]}")
    print()
    for pid in sides:
        print(report(merge(got[pid]),
                     f"{'A' if pid == 0 else 'B'} = {names[pid]}",
                     len(seeds), args.by_day))
        print()


if __name__ == "__main__":
    main()
