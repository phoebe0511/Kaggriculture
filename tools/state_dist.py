"""還在不在老師的資料分布裡？—— behavior cloning 的第二個 kill switch。

    python -m tools.state_dist temp/<run_dir> [<run_dir> ...]

## 為什麼要這支

`tools/action_dist.py` 量的是「動作像不像老師」。這支量的是**局面**像不像 ——
而 behavior cloning 真正會死的地方是後者：訓練時每一步的輸入都是老師走出來的
局面，上場時是自己前 t−1 步走出來的。一旦離開訓練資料的範圍，網路的輸出就
沒有任何資料背書，而且會自我強化。

## 為什麼要按天比

老師的**邊際**分布很寬 —— 60 局的現金 p5 只有 $23、雜草 p95 是 13、作物 p1 是 0。
單看邊際，我們的局面看起來都「在範圍內」。但那些低點全部發生在**開局**。
按天條件之後才看得出來（2026-08-20 `gen3_target` 實測）：

    day  動物(老師 p5/中位)  我們   現金(老師中位)   我們    百分位
      6         6 /  6         1          982        728     44.0%
     10        11 / 12         1        1,778        423      9.3%
     14        14 / 14         0       17,242      1,109      0.0%

**第一個跌破的是動物（day 6），那時候現金還完全正常。** 這種先後順序只有
按天比才看得到，而它直接指出要修的是市場那一半（動物是買的，不是 unit 決定的）。

## 紅線

動物 / 作物 / 現金三項，**第一次跌破老師 p5 的那天都要 ≥ day 14**（半個賽季）。

紅線是判定門檻，不是統計檢定。要比較兩個版本的強弱仍然要看 `eval.runner`
的勝率與信賴區間。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contracts as C  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "dataset"

#: 要比的欄位 -> (`SCALAR_FIELDS` 的名字, 還原成原始單位的乘數)
FIELDS = (
    ("動物", "n_animals", 20.0),
    ("作物", "n_crop_tiles", 75.0),
    ("現金", "money", 100_000.0),
)

#: 在這一天之前跌破老師的 p5 就算紅。半個賽季。
SURVIVE_UNTIL_DAY = 14


def teacher_by_day(dataset_dir=DEFAULT_DATASET):
    """回傳 `{欄位: {day: 該天所有樣本的陣列}}`。"""
    idx = {name: i for i, name in enumerate(C.SCALAR_FIELDS)}
    days, cols = [], {label: [] for label, _, _ in FIELDS}
    files = sorted(glob.glob(os.path.join(str(dataset_dir), "*.npz")))
    if not files:
        raise SystemExit(f"{dataset_dir} 裡沒有 npz —— 先跑 harness.build_dataset")
    for path in files:
        with np.load(path) as data:
            # 每 3 個取 1：同一天 24 個回合的盤面幾乎一樣，全取只是變慢
            scalar = data["board_scalar"][::3]
            days.append(data["board_step"][::3] // 24)
            for label, name, scale in FIELDS:
                cols[label].append(scalar[:, idx[name]] * scale)
    days = np.concatenate(days)
    out = {}
    for label, _, _ in FIELDS:
        values = np.concatenate(cols[label])
        out[label] = {d: values[days == d] for d in np.unique(days)}
    return out


def ours_by_day(run_dir):
    """從 `eval.runner` 的 log 讀我方每天 hour 12 的狀態。

    取 hour 12 是因為 hands 在每天 hour 0 是空的（`engine-notes.md` §1），
    hour 0 量到的東西不能跟老師的全天樣本比。
    """
    run_dir = Path(run_dir)
    logs = sorted(glob.glob(str(run_dir / "logs" / "*.jsonl")))
    if not logs:
        raise SystemExit(f"{run_dir}/logs 是空的 —— eval.runner 要帶 --log-level 2")

    name = json.loads((run_dir / "result.json").read_text(encoding="utf-8")) \
        .get("a", {}).get("name", "A")
    # 檔名是 `seed0000_{A 的名字}_vs_{B 的名字}[_a<現金>_b<現金>]`，所以
    # 「A 的名字排在 _vs_ 前面」的那些檔，A 一定是 **player 0**。
    mine = [p for p in logs if Path(p).stem.split("_vs_")[0].endswith(name)]
    if not mine:
        mine, expect = logs[:1], None       # 認不出來就只讀一個檔、不篩
    else:
        expect = 0
    per_day = {}
    seen_players = set()
    for path in mine:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            seen_players.add(row.get("player"))
            # 🩸 **一定要篩 player。** 2026-08-21 之前只有 `agents/gen0.py`
            #    會寫 log，一個檔裡剛好只有一邊，所以沒篩也對；
            #    `agents/gen2_model.py` 開始寫之後，同一個檔裡兩邊都有 ——
            #    不篩的話這張表是「我方和對手的平均」，而且不會報錯。
            if expect is not None and row.get("player") != expect:
                continue
            if row.get("hour") != 12:
                continue
            # 🩸 **作物那一欄要用實際種下去的格數**（`crops`），不是
            #    `agents/gen0.py` 記的 `budget["active_crop_count"]` —— 後者是
            #    規則式「打算種幾格」的計畫值。老師那一邊用的是
            #    `contracts.SCALAR_FIELDS` 的 `n_crop_tiles`（實際格數），
            #    拿計畫值去比是兩種東西。
            #    `crops` 是 2026-08-21 才加的，舊 log 沒有 -> 退回計畫值，
            #    但那些數字**不能跟新的比**。
            budget = row.get("budget") or {}
            crops = row.get("crops")
            if crops is None:
                crops = budget.get("active_crop_count", 0)
            per_day.setdefault(row["day"], []).append((
                sum((row.get("animals") or {}).values()),
                crops,
                row.get("cash", 0.0),
            ))
    if not per_day:
        # 🩸 **不能回空的**：`report()` 會印一張空表格然後蓋綠燈說
        #    「三項都撐過 day 14」—— 假的合格比錯的數字更糟。
        #    最常見的原因是這個 run 跑在 2026-08-21 之前，那時候
        #    `agents/gen2_model.py` 還不寫 log，檔案裡只有對手那一邊。
        raise SystemExit(
            f"{run_dir} 的 log 裡沒有 player {expect}（A = {name}）的資料，"
            f"有的是 player {sorted(p for p in seen_players if p is not None)}。\n"
            "2026-08-21 之前 agents/gen2_model.py 不寫 log —— 那些 run 的檔案裡"
            "只有規則式那一邊，量不到網路版。重跑一次 eval.runner 才有。")
    return {d: np.mean(v, axis=0) for d, v in per_day.items()}, name


def report(run_dir, teacher):
    ours, name = ours_by_day(run_dir)
    print(f"\n{Path(run_dir).name}   A = {name}")
    header = f"  {'day':>4}"
    for label, _, _ in FIELDS:
        header += f" | {label + ' p5':>9}{'中位':>8}{'我們':>9}{'百分位':>8}"
    print(header)

    first_below = {label: None for label, _, _ in FIELDS}
    for day in sorted(ours):
        line = f"  {day:>4}"
        for i, (label, _, _) in enumerate(FIELDS):
            values = teacher[label].get(day)
            if values is None or not len(values):
                line += f" | {'—':>34}"
                continue
            p5, mid = np.percentile(values, [5, 50])
            mine = ours[day][i]
            pct = float((values < mine).mean() * 100)
            flag = ""
            if mine < p5:
                flag = " *"
                if first_below[label] is None:
                    first_below[label] = day
            line += f" | {p5:>9,.0f}{mid:>8,.0f}{mine:>9,.0f}{pct:>7.1f}%{flag}"
        print(line)

    print("\n  第一次跌破老師 p5 的那天（* 標的那些）：")
    failures = []
    for label, _, _ in FIELDS:
        day = first_below[label]
        text = "沒跌破過" if day is None else f"day {day}"
        print(f"    {label:<6}{text}")
        if day is not None and day < SURVIVE_UNTIL_DAY:
            failures.append(f"{label} day {day} 就跌破老師 p5")

    if failures:
        print(f"\n  ❌ 沒撐到 day {SURVIVE_UNTIL_DAY}：")
        for text in failures:
            print(f"     - {text}")
    else:
        print(f"\n  ✅ 三項都撐過 day {SURVIVE_UNTIL_DAY}"
              "（不代表比較強，強弱看 eval.runner 的勝率）")
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="eval.runner 的 run 目錄")
    ap.add_argument("--data", default=str(DEFAULT_DATASET),
                    help="老師的訓練資料，用來算每天的分位數")
    args = ap.parse_args(argv)

    # Windows 重導到檔案時 Python 會改用系統 locale（`eval/runner.py:800`）
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    teacher = teacher_by_day(args.data)
    worst = 0
    for path in args.runs:
        worst = max(worst, report(path, teacher))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
