"""比對兩邊送出的 unit 動作分布 —— 模仿學習用的 kill switch。

    python -m tools.action_dist temp/<run_dir> [<run_dir> ...]

## 為什麼要這支

2026-08-19（journal §7d）記的教訓：**驗證準確率不是有效的 kill switch。**
那一版網路對老師局面的 unit-turn 準確率 0.94（dummy 0.16）、逐類別召回率
0.98 以上，兩個門檻都過了，實戰仍然 0 勝 12 負。

真正顯示問題的是「自己打一局，把動作分布跟老師的擺在一起」。2026-08-20 對
`ladder-top-a` 的實測（seed 1）：

    MOVE                網路 75.6%   老師 51.8%
    FEED                     7 次        313 次
    CARE                     8 次        310 次
    COLLECT_FERTILIZER      25 次        306 次
    BUILD_PASTURE           22 次         14 次

這些差距準確率完全看不出來，因為它們發生在**老師沒走過的局面**上，
而準確率是在老師走過的局面上量的。

**不需要標籤，一局就跑得出來**（4 workers 約 1,463 局/小時）。

## 紅線

| 條件 | 意思 |
|---|---|
| MOVE 佔比差 > 10 個百分點 | unit 在路上的時間過長，該做的事做不完 |
| 任一生產動作少一個數量級（< 1/10） | 那條產線整條斷掉，不是做得少而已 |

紅線是判定用的門檻，不是統計檢定 —— 上面那些差距是 5~40 倍，不需要檢定。
要比較兩個版本的**強弱**仍然要用 `eval.runner` 的勝率與信賴區間。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")

#: 產線動作。少一個數量級就是那條線斷了。
PRODUCTION = ("WATER", "HARVEST", "PLANT", "FERTILIZE",
              "FEED", "CARE", "COLLECT_FERTILIZER", "PICKUP", "PLACE")

MOVE_GAP_LIMIT = 0.10      # 佔比差，絕對值
RATIO_LIMIT = 0.10         # 我方 / 對方，低於這個算「少一個數量級」


def load_run(path):
    """吃 run 目錄或 result.json，回 `(name_a, name_b, totals_a, totals_b)`。"""
    path = Path(path)
    if path.is_dir():
        path = path / "result.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    name_a = (data.get("a") or {}).get("name", "A")
    name_b = (data.get("b") or {}).get("name", "B")
    totals = [{}, {}]
    games = 0
    for row in data["results"]:
        if row.get("error"):
            continue
        games += 1
        for k, side in (("actions_a", 0), ("actions_b", 1)):
            for op, count in (row[k].get("ops") or {}).items():
                totals[side][op] = totals[side].get(op, 0) + count
    return name_a, name_b, totals[0], totals[1], games


def rates(ops):
    total = sum(ops.values())
    return total, {op: c / total for op, c in ops.items()} if total else {}


def report(path):
    name_a, name_b, ops_a, ops_b, games = load_run(path)
    total_a, rate_a = rates(ops_a)
    total_b, rate_b = rates(ops_b)
    if not total_a or not total_b:
        print(f"{path}：沒有可用的動作紀錄", file=sys.stderr)
        return 1

    print(f"\n{Path(path).name}")
    print(f"  {name_a}  vs  {name_b}   {games} 局   "
          f"動作 {total_a:,} / {total_b:,}")

    move_a = sum(rate_a.get(op, 0.0) for op in MOVES)
    move_b = sum(rate_b.get(op, 0.0) for op in MOVES)

    print(f"\n  {'op':<22}{name_a[:10]:>11}{name_b[:10]:>11}{'比值':>9}")
    print(f"  {'MOVE 合計':<20}{move_a:>11.4f}{move_b:>11.4f}"
          f"{(move_a / move_b if move_b else float('nan')):>9.2f}")
    for op in sorted(set(ops_a) | set(ops_b),
                     key=lambda o: -rate_b.get(o, 0.0)):
        if op in MOVES:
            continue
        ra, rb = rate_a.get(op, 0.0), rate_b.get(op, 0.0)
        ratio = ra / rb if rb else float("inf") if ra else float("nan")
        print(f"  {op:<22}{ra:>11.4f}{rb:>11.4f}{ratio:>9.2f}")

    print()
    failures = []
    gap = move_a - move_b
    if abs(gap) > MOVE_GAP_LIMIT:
        failures.append(f"MOVE 佔比差 {gap:+.1%}（門檻 ±{MOVE_GAP_LIMIT:.0%}）")
    for op in PRODUCTION:
        ra, rb = rate_a.get(op, 0.0), rate_b.get(op, 0.0)
        if rb > 0 and ra / rb < RATIO_LIMIT:
            failures.append(
                f"{op} 只有對方的 {ra / rb:.1%}"
                f"（{ops_a.get(op, 0):,} vs {ops_b.get(op, 0):,}）")
    if failures:
        print("  ❌ 動作分布不像對方：")
        for line in failures:
            print(f"     - {line}")
    else:
        print("  ✅ 沒有踩到紅線（不代表比較強，強弱看 eval.runner 的勝率）")
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="eval.runner 的 run 目錄或 result.json")
    args = ap.parse_args(argv)

    # Windows 重導到檔案時 Python 會改用系統 locale（正體中文是 cp950），
    # ✅ ❌ 編不出來就整個 print 拋 UnicodeEncodeError（`eval/runner.py:800`）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):     # 不是真的 TextIOWrapper 就算了
            pass

    worst = 0
    for path in args.runs:
        worst = max(worst, report(path))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
