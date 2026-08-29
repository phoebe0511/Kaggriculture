"""`structure_tiles` 的象限分佈。

🩸 2026-08-29 量到：ladder 八個對手的建物分佈是 NW/NE/SW 約 6/6/2，
我們是 **1,280 局全部 12/0/0** —— `structure_tiles` 的兩層迴圈都是
`range(half)`，只掃 NW。每局 276 次 FEED 全要走回 NW，MOVE 佔 57.5%
（全場最高）。

這裡守三件事：預設值不能動到出貨行為、順序不能隨解鎖狀態改變、
分散程度要單調。
"""

from __future__ import annotations

import pytest

from agents.gen0 import quadrant_for, structure_tiles

BOARD = 10


def _old_impl(board, n):
    """2026-08-29 之前的實作，一字不改抄過來當對照。"""
    half = board // 2
    cx, cy = half - 1, half - 1
    ranked = sorted(
        (abs(x - cx) + abs(y - cy), y, x)
        for y in range(half)
        for x in range(half)
    )
    return tuple((x, y) for _d, y, x in ranked[:n])


@pytest.mark.parametrize("n", [1, 2, 6, 11, 12, 20, 25])
def test_default_is_byte_identical_to_the_old_implementation(n):
    """預設 spread=0 一定要跟舊實作逐項相同 —— 出貨版吃的就是這條路。"""
    assert structure_tiles(BOARD, n) == _old_impl(BOARD, n)
    assert structure_tiles(BOARD, n, 0.0) == _old_impl(BOARD, n)


def _counts(tiles):
    c = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
    for (x, y) in tiles:
        c[quadrant_for(x, y, BOARD)] += 1
    return c


def test_default_puts_everything_in_nw():
    """這就是要驗的那個現象本身。"""
    c = _counts(structure_tiles(BOARD, 12))
    assert (c["NW"], c["NE"], c["SW"]) == (12, 0, 0)


@pytest.mark.parametrize("spread", [0.25, 0.5, 0.67, 1.0])
def test_spread_moves_structures_out_of_nw(spread):
    c = _counts(structure_tiles(BOARD, 12, spread))
    assert c["NW"] < 12, "spread > 0 卻沒有任何建物離開 NW"
    assert c["NE"] + c["SW"] > 0
    assert c["SE"] == 0, "SE 從來不會解鎖，不該預留在那裡"
    assert sum(c.values()) == 12


def test_nw_share_is_monotone_in_spread():
    """劑量要單調，不然驗出來的東西沒辦法解讀。"""
    got = [_counts(structure_tiles(BOARD, 12, s))["NW"]
           for s in (0.0, 0.25, 0.5, 0.67, 1.0)]
    assert got == sorted(got, reverse=True), got


@pytest.mark.parametrize("spread", [0.0, 0.5, 1.0])
def test_order_is_stable_within_a_game(spread):
    """🩸 順序在一局之內必須固定 —— `animal_for` 靠索引決定哪一格養哪一種，
    而 `PLACE` / `FETCH` / `BUILD` 必須看到同一份配置。買地讓已蓋好的建物換
    索引的話會卡死，而且不會報錯。

    一局之內 `n_structures` 是固定的，而這個函式**只**吃 board / n / spread
    —— 收不到解鎖狀態，也就不可能隨買地而變。所以要驗的是純函式性質。

    （跨不同 `n` 的前綴性質**不成立**，spread>0 時 NW 的配額是 `n` 的比例。
    那個性質也不需要成立：一局之內 `n` 不會變。）
    """
    first = structure_tiles(BOARD, 12, spread)
    for _ in range(5):
        assert structure_tiles(BOARD, 12, spread) == first
    import inspect

    params = list(inspect.signature(structure_tiles).parameters)
    assert params == ["board", "n", "spread"], (
        f"多了參數 {params} —— 只要它收得到解鎖狀態，順序就可能中途改變")


def test_tiles_are_unique_and_on_the_board():
    for spread in (0.0, 0.5, 1.0):
        t = structure_tiles(BOARD, 12, spread)
        assert len(set(t)) == len(t), "有重複的格子"
        for (x, y) in t:
            assert 0 <= x < BOARD and 0 <= y < BOARD
