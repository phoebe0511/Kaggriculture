"""還在不在老師的資料分布裡？—— behavior cloning 的第二個 kill switch。

    python -m tools.state_dist temp/<run_dir> [<run_dir> ...]

## 為什麼要這支

`tools/action_dist.py` 量的是「動作像不像老師」。這支量的是**局面**像不像 ——
而 behavior cloning 真正會死的地方是後者：訓練時每一步的輸入都是老師走出來的
局面，上場時是自己前 t−1 步走出來的。一旦掉出訓練資料的範圍，網路的輸出就
沒有任何資料背書，而且會自我強化。

## 為什麼要按天比

老師的**邊際**分布很寬 —— 60 局的現金 p5 只有 $23、雜草 p95 是 13、作物 p1 是 0。
單看邊際，我們的局面看起來都「在範圍內」。但那些低點全部發生在**開局**。
按天條件之後才看得出來（2026-08-20 `gen3_target` 實測）：

    day  動物(老師 p5/中位)  我們   現金(老師中位)   我們    百分位
      6         6 /  6         1          982        728     44.0%
     10        11 / 12         1        1,778        423      9.3%
     14        14 / 14         0       17,242      1,109      0.0%

**第一個掉出去的是動物（day 6），那時候現金還完全正常。** 這種先後順序只有
按天比才看得到，而它直接指出要修的是市場那一半（動物是買的，不是 unit 決定的）。

## 紅線

動物 / 作物 / 現金三項，**掉出老師 p5 的第一天都要 ≥ day 14**（半個賽季）。

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

#: 撐到這一天之前掉出老師的 p5 就算紅。半個賽季。
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
    mine = [p for p in logs if Path(p).stem.split("_vs_")[0].endswith(name)]
    per_day = {}
    for path in mine or logs[:1]:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row.get("hour") != 12:
                continue
            budget = row.get("budget") or {}
            per_day.setdefault(row["day"], []).append((
                sum((row.get("animals") or {}).values()),
                budget.get("active_crop_count", 0),
                row.get("cash", 0.0),
            ))
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

    print("\n  掉出老師 p5 的第一天（* 標的那些）：")
    failures = []
    for label, _, _ in FIELDS:
        day = first_below[label]
        text = "沒掉出去" if day is None else f"day {day}"
        print(f"    {label:<6}{text}")
        if day is not None and day < SURVIVE_UNTIL_DAY:
            failures.append(f"{label} 在 day {day} 就掉出 p5")

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
