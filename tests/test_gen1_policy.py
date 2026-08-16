from __future__ import annotations

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

from agents.gen0 import (
    _needs_water,
    active_crop_tiles,
    quadrant_for,
    structure_tiles,
)


def _plant(crop, planted_day, *, consecutive=0, yield_units=1, fertilized=-1):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "watered_today": False,
        "consecutive_unwatered": consecutive,
        "yield_units": yield_units,
        "fertilized_until_day": fertilized,
    }


def test_on_demand_water_preserves_survival_and_one_time_yield_window():
    crop = "CARROT"
    cd = CROPS[crop]
    window_start = (cd["max_yield_day"] + 1) // 2

    new_seedling = _plant(crop, planted_day=5, consecutive=1)
    assert _needs_water(new_seedling, cd, day=5, on_demand=True)

    safe_outside_window = _plant(crop, planted_day=5, consecutive=0)
    assert not _needs_water(safe_outside_window, cd, day=5, on_demand=True)

    yield_window = _plant(crop, planted_day=5, consecutive=0, yield_units=1)
    assert _needs_water(
        yield_window, cd, day=5 + window_start, on_demand=True
    )

    full_yield = _plant(
        crop, planted_day=5, consecutive=0, yield_units=cd["max_yield"]
    )
    assert not _needs_water(full_yield, cd, day=5 + window_start, on_demand=True)


def test_active_tiles_are_balanced_across_unlocked_quadrants():
    board = 10
    unlocked = ["NW", "NE", "SW"]
    tiles = [
        [None if quadrant_for(x, y, board) in unlocked else "LOCKED"
         for x in range(board)]
        for y in range(board)
    ]
    structures = structure_tiles(board, 6)
    active = active_crop_tiles(
        tiles, board, structures, unlocked, max_tiles=39
    )

    assert len(active) == 39
    assert Counter(quadrant_for(x, y, board) for x, y in active) == {
        "NW": 13,
        "NE": 13,
        "SW": 13,
    }
    assert not set(structures) & active
