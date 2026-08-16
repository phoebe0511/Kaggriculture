from __future__ import annotations

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

from agents.gen0 import (
    _needs_water,
    active_crop_tiles,
    quadrant_for,
    species_for_structure,
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


def test_structure_fallback_follows_plan_quota_not_alphabet():
    """plan 指定的species放不進這個建物時，要退回**規劃器當下想要的**那一種。

    重要性：建物蓋下去就固定了，但 plan 會隨 shop 與市場變動。舊版是
    `for s in sorted(ANIMALS)`，而 sorted = COW < GOOSE < SHEEP，所以 PASTURE
    的 fallback 永遠是 COW —— 即使 plan 給 COW 的配額是 0、給 SHEEP 的是 3。
    實測 seed 41003 因此六個建物全變成牛：MILK 賣 156 個的收入比賣 108 個少
    $14,921，WOOL 收入整個消失（-$12,394）。
    """
    # 規劃器要 GOOSE×3 + SHEEP×3；GOOSE 進不了 PASTURE。
    plan = ("GOOSE", "GOOSE", "GOOSE", "SHEEP", "SHEEP", "SHEEP")
    assert species_for_structure("PASTURE", "GOOSE", plan) == "SHEEP"

    # 配額相符時原樣採用，不受 fallback 影響。
    assert species_for_structure("PASTURE", "COW", plan) == "COW"
    assert species_for_structure("COOP", "GOOSE", plan) == "GOOSE"

    # COW 配額較高時就該是 COW —— 這條確保修正不是「一律改挑 SHEEP」。
    cow_heavy = ("COW", "COW", "COW", "GOOSE", "GOOSE", "GOOSE")
    assert species_for_structure("PASTURE", "GOOSE", cow_heavy) == "COW"

    # plan 裡兩種都沒有（或沒給 plan）時退回字母序，保證可重現。
    assert species_for_structure("PASTURE", "GOOSE", ("GOOSE",)) == "COW"
    assert species_for_structure("PASTURE", None) == "COW"
