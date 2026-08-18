from __future__ import annotations

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

from agents.gen0 import (
    DEFAULT_PARAMS,
    _animal_housing_room,
    _crop_can_harvest,
    _inventory_qty,
    _needs_water,
    _pickup_quantities,
    _tasks,
    _wheat_reserve,
    active_crop_tiles,
    planned_crew,
    quadrant_for,
    species_for_structure,
    structure_tiles,
)
from agents.gen1 import DEFAULT_PARAMS as GEN1_PARAMS


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


def test_inventory_quantity_counts_units_not_carriers():
    inventories = [
        {"WHEAT": 83},
        {"WHEAT": 4, "GOOSE": 1},
        {},
    ]

    assert _inventory_qty(inventories, "WHEAT") == 87
    assert _inventory_qty(inventories, "GOOSE") == 1


def test_market_stock_accounts_for_same_turn_pickups():
    actions = [
        ["PICKUP", "WHEAT", 4],
        ["PICKUP", "FERTILIZER", 1],
        ["NORTH"],
        ["PASS"],
    ]

    assert _pickup_quantities(actions) == {"WHEAT": 4, "FERTILIZER": 1}


def test_ladder_wages_keep_land_crew_cap_through_final_day():
    farm = {
        "unlocked_quadrants": ["NW", "NE", "SW"],
        "tiles": [[None] * 10 for _ in range(10)],
    }

    assert planned_crew(
        farm,
        DEFAULT_PARAMS,
        {"farmHandCostMult": 1},
        {},
        days_left=1,
    ) == 12


def test_wheat_reserve_disappears_during_liquidation():
    assert _wheat_reserve(9, days_left=1, params=DEFAULT_PARAMS) == 0
    assert _wheat_reserve(9, days_left=2, params=DEFAULT_PARAMS) == 0
    assert _wheat_reserve(9, days_left=3, params=DEFAULT_PARAMS) == 18
    assert _wheat_reserve(9, days_left=10, params=DEFAULT_PARAMS) == 36


def test_animal_purchase_respects_committed_housing_capacity():
    alive = {"COW": 4, "SHEEP": 4, "GOOSE": 1}
    assert _animal_housing_room(DEFAULT_PARAMS, alive, {}, [{}]) == 0

    alive = {"COW": 2}
    shed = {"GOOSE": 1}
    inventories = [{"SHEEP": 2}, {}]
    assert _animal_housing_room(DEFAULT_PARAMS, alive, shed, inventories) == 4


def test_crop_must_have_time_to_reach_first_harvest():
    assert _crop_can_harvest("CARROT", days_left=3)
    assert not _crop_can_harvest("CARROT", days_left=2)
    assert not _crop_can_harvest("WHEAT", days_left=2)
    assert not _crop_can_harvest("TOMATO", days_left=8)
    assert not _crop_can_harvest("STRAWBERRY", days_left=10)


def test_gen1_reserves_twelve_animal_structures():
    assert GEN1_PARAMS["n_structures"] == 12


def test_reserved_structure_weed_is_dug_for_waiting_animal():
    board = 10
    tiles = [[None] * board for _ in range(board)]
    target = structure_tiles(board, 1)[0]
    tiles[target[1]][target[0]] = {"kind": "WEED"}
    params = dict(DEFAULT_PARAMS)
    params["basket"] = ("CARROT",)

    tasks = _tasks(
        tiles,
        day=5,
        board=board,
        params=params,
        struct_order=(target,),
        struct_plan={target: "SHEEP"},
        unit_inv=[{}],
        private={"shed": {"SHEEP": 1}},
        active_tiles=set(),
        days_left=10,
    )

    assert any(task[1:4] == ("DIG", target[0], target[1]) for task in tasks)
