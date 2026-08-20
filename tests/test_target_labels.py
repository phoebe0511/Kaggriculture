"""v3 的 segment 標籤：`unit_target` / `unit_term_op` / `unit_term_qty`。

這組標籤是「這個 unit 要去哪一格、到了做什麼」，從 replay 往回推出來的
（`harness/build_dataset.segment_labels`）。推錯了**不會報錯**，只會訓練出一個
每一步都往錯的地方走的網路 —— 跟 2026-08-19 那次 observation/action 錯位一格
是同一類失敗（loss 照樣降，14.66% 的老師動作落在 mask 外才把它抓出來）。

所以這裡拿真的 replay 驗兩件事：

1. 沒有終點動作的比例（4.2%）—— 按天切錯或 unit slot 對錯會讓它爆掉
2. 照 `unit_target` 用 `gen0.step_toward` 重走，**能不能重現老師實際那一步**
   （93.02%）—— 這條同時擋住 +1 對齊、x/y 顛倒、target index 算錯
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

import contracts as C
from agents.gen0 import step_toward
from harness.build_dataset import build_episode, segment_labels

REPO_ROOT = Path(__file__).resolve().parents[1]
#: 需要含 observation 的原始 replay。`temp/` 被 gitignore，同伴的工作目錄不會有。
RAW_DIR = REPO_ROOT / "temp" / "episodes"

#: 實測基準（3 局 × 2 玩家、41,484 個 unit-turn，2026-08-20）。
DANGLING_RATE = 0.042
GREEDY_MATCH = 0.9302


def _episodes(limit=2):
    return sorted(glob.glob(str(RAW_DIR / "*.json")))[:limit]


@pytest.fixture(scope="module")
def arrays():
    files = _episodes()
    if not files:
        pytest.skip(f"{RAW_DIR} 沒有原始 replay（temp/ 被 gitignore）")
    out = []
    for path in files:
        built, _note = build_episode(path)
        if built is not None:
            out.append(built)
    if not out:
        pytest.skip("沒有引擎版本相符的 replay")
    return out


def test_dangling_rate_matches_the_measured_baseline(arrays):
    """走到一半就日終、沒有終點動作的比例。

    ⚠️ 這個數字**偏高**的典型原因是 segment 沒有按天切。引擎每天結算會
    `farm["hands"] = []`（`kaggriculture.py:880`），跨天串的話 hand index
    指的不是同一個人，鏈會在錯的地方斷掉。
    """
    dangling = total = 0
    for data in arrays:
        dangling += int((data["unit_target"] < 0).sum())
        total += len(data["unit_target"])

    assert total > 10_000
    rate = dangling / total
    assert 0.02 < rate < 0.08, \
        f"沒有終點的比例 {rate:.2%}，實測基準 {DANGLING_RATE:.1%}"


def test_target_labels_agree_with_the_teacher_movement(arrays):
    """照 target 標籤用 `step_toward` 重走，要重現老師實際送出的那一步。

    只看老師實際做 MOVE 的那些 unit-turn（`unit_op` 是 MOVE）。門檻 0.90，
    實測 0.9302 —— 差的那 7% 是先對齊 x 還先對齊 y 的順序差別。

    這條會紅的三種寫法：observation/action 沒有 +1、`target_index` 的 x/y
    顛倒、unit slot 對到別的 unit。三種都會讓比例掉到接近亂猜。
    """
    moves = {C.UNIT_OP_INDEX[(m, None)] for m in C.MOVES}

    matched = total = 0
    for data in arrays:
        board = C.BOARD_SIZE
        for op, target, (x, y) in zip(
                data["unit_op"], data["unit_target"], data["unit_pos"]):
            if int(op) not in moves or int(target) < 0:
                continue
            total += 1
            expected = step_toward((int(x), int(y)),
                                   C.target_xy(int(target), board))
            if C.UNIT_OP_INDEX[(expected[0], None)] == int(op):
                matched += 1

    assert total > 5_000, f"只有 {total} 個 MOVE，樣本太少"
    rate = matched / total
    assert rate > 0.90, \
        f"step_toward 只重現 {rate:.2%} 的老師 MOVE，實測基準 {GREEDY_MATCH:.2%}"


def test_terminal_action_is_at_the_target_cell(arrays):
    """非 MOVE 的 unit-turn，target 一定是它自己站的那一格。

    這是定義，不是統計 —— 有一筆不成立就是 `segment_labels` 的迴圈寫錯了。
    """
    moves = {C.UNIT_OP_INDEX[(m, None)] for m in C.MOVES}

    checked = 0
    for data in arrays:
        for op, target, term_op, (x, y) in zip(
                data["unit_op"], data["unit_target"], data["unit_term_op"],
                data["unit_pos"]):
            if int(op) in moves:
                continue
            checked += 1
            assert int(target) == C.target_index(int(x), int(y), C.BOARD_SIZE)
            assert int(term_op) == int(op)

    assert checked > 5_000


def _rows(*entries):
    """`(step, op_name, x, y)` -> `segment_labels` 吃的 row，slot 固定 0。"""
    return [(step, 0, C.UNIT_OP_INDEX[(op, None)], -1, x, y)
            for step, op, x, y in entries]


def test_segment_stops_at_the_day_boundary():
    """段落不可以跨天。

    跨天的話 unit 會被指去「昨天那個目標」，而昨天結束時所有 hand 都被清掉、
    主農夫也回到出生點（`kaggriculture.py:880`）。目標本身是個有效格子，
    所以**不會報錯**，只會學到往沒有意義的地方走。
    """
    labels = segment_labels(
        _rows((23, "WEST", 5, 5), (24, "WATER", 3, 5)), 24, C.BOARD_SIZE)

    assert labels[0] == (-1, -1, -1), "day 0 的 MOVE 不該指到 day 1 的動作"
    assert labels[1] == (C.target_index(3, 5, C.BOARD_SIZE),
                         C.UNIT_OP_INDEX[("WATER", None)], -1)


def test_segment_within_a_day_points_at_the_terminal_action():
    labels = segment_labels(
        _rows((10, "WEST", 5, 5), (11, "WEST", 4, 5), (12, "WATER", 3, 5)),
        24, C.BOARD_SIZE)

    expected = (C.target_index(3, 5, C.BOARD_SIZE),
                C.UNIT_OP_INDEX[("WATER", None)], -1)
    assert labels == [expected, expected, expected]


def test_unrecognised_action_breaks_the_chain():
    """step 不連續代表中間那一步的動作 `encode_unit_action` 認不得、被跳過了。

    寧可讓前面幾步變成沒有標籤，也不要讓它們指到一個隔著未知動作的終點。
    """
    labels = segment_labels(
        _rows((10, "WEST", 5, 5), (12, "WATER", 3, 5)), 24, C.BOARD_SIZE)

    assert labels[0] == (-1, -1, -1)
    assert labels[1][1] == C.UNIT_OP_INDEX[("WATER", None)]
