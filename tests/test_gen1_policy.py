from __future__ import annotations

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS,
    MARKET_PARAMS,
)

from agents.gen0 import (
    DEFAULT_PARAMS,
    _add_internal_feed_demand,
    _animal_housing_room,
    _adaptive_crop_shares,
    _crop_can_harvest,
    _daytime_return_assignments,
    _effective_competitive_demand,
    _inventory_qty,
    _fertilize_worth_it,
    _late_crew_limit,
    _limit_early_coops,
    _assign,
    _needs_water,
    _pickup_quantities,
    _planned_sale_orders,
    _project_shed_after_unit_actions,
    _tasks,
    _wheat_reserve,
    active_crop_tiles,
    planned_crew,
    quadrant_for,
    species_for_structure,
    structure_tiles,
)
from agents.gen1 import DEFAULT_PARAMS as GEN1_PARAMS, _resolve_params


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


def test_roi_mode_still_waters_unfertilized_ongoing_production():
    crop = "TOMATO"
    cd = CROPS[crop]
    tile = _plant(crop, planted_day=0, consecutive=0, yield_units=0)

    assert not _needs_water(tile, cd, day=7, on_demand=True)
    assert _needs_water(
        tile,
        cd,
        day=7,
        on_demand=True,
        produce_without_fertilizer=True,
    )


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


def test_last_hour_never_creates_doomed_plant_tasks():
    tiles = [[None] * 4 for _ in range(4)]
    params = {
        "basket": ("WHEAT",),
        "use_fertilizer": False,
        "wheat_carry": 4,
        "avoid_last_hour_planting": True,
    }
    common = dict(
        tiles=tiles,
        day=5,
        board=4,
        params=params,
        struct_order=(),
        struct_plan={},
        unit_inv=[{}],
        private={"shed": {}},
        active_tiles={(0, 0)},
        days_left=20,
        turns_per_day=24,
    )

    before_last_hour = _tasks(**common, hour=22)
    last_hour = _tasks(**common, hour=23)

    assert [task[1] for task in before_last_hour] == ["PLANT"]
    assert all(task[1] != "PLANT" for task in last_hour)


def test_frozen_policies_keep_legacy_last_hour_planting_without_flag():
    tasks = _tasks(
        tiles=[[None]],
        day=5,
        board=1,
        params={
            "basket": ("WHEAT",),
            "use_fertilizer": False,
            "wheat_carry": 4,
        },
        struct_order=(),
        struct_plan={},
        unit_inv=[{}],
        private={"shed": {}},
        active_tiles={(0, 0)},
        days_left=20,
        hour=23,
        turns_per_day=24,
    )

    assert [task[1] for task in tasks] == ["PLANT"]


def test_same_turn_drop_is_visible_to_later_market_orders():
    farm = {"farmer": [4, 4], "hands": [[4, 5]]}
    private = {
        "shed": {"WHEAT": 2},
        "inventories": [
            {"MILK": 7, "WOOL": 4},
            {"STRAWBERRY": 3},
        ],
    }

    projected = _project_shed_after_unit_actions(
        farm,
        private,
        [["DROP"], ["PLACE", "STRAWBERRY", 2]],
        board=10,
        shed_capacity=100,
    )

    assert projected == {
        "WHEAT": 2,
        "MILK": 7,
        "WOOL": 4,
        "STRAWBERRY": 2,
    }
    assert private["shed"] == {"WHEAT": 2}


def test_daytime_return_moves_only_a_valuable_product_and_keeps_wheat():
    assigned = {}
    inventories = [{"WOOL": 4, "WHEAT": 9}]
    _daytime_return_assignments(
        assigned,
        {0},
        [(2, 4)],
        inventories,
        board=10,
        prices={"WOOL": 200, "WHEAT": 40},
        shed={},
        shed_capacity=100,
        hour=10,
        turns_per_day=24,
        params={"daytime_return_min_value": 600, "daytime_return_max_distance": 4},
    )

    assert assigned[0][0] == "DROP_PRODUCT"
    assert assigned[0][3] == ("WOOL", 4)
    assert inventories[0]["WHEAT"] == 9


def test_daytime_return_skips_low_value_or_too_late_inventory():
    for hour, value in ((10, 100), (23, 800)):
        assigned = {}
        _daytime_return_assignments(
            assigned,
            {0},
            [(0, 0)],
            [{"WOOL": 4}],
            board=10,
            prices={"WOOL": value / 4},
            shed={},
            shed_capacity=100,
            hour=hour,
            turns_per_day=24,
            params={"daytime_return_min_value": 600, "daytime_return_max_distance": 10},
        )
        assert assigned == {}


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


def test_optional_late_crew_schedule_uses_the_tightest_matching_cap():
    params = {"late_crew_caps": ((6, 11), (2, 10), (1, 8))}

    assert _late_crew_limit(12, 7, params) == 12
    assert _late_crew_limit(12, 6, params) == 11
    assert _late_crew_limit(12, 2, params) == 10
    assert _late_crew_limit(12, 1, params) == 8


def test_fertilizer_roi_only_spends_when_expected_bonus_beats_sale_value():
    crop = "STRAWBERRY"
    cd = CROPS[crop]
    tile = _plant(crop, planted_day=0, consecutive=0, yield_units=0)
    params = {
        "fertilizer_roi_margin": 1.0,
        "fertilize_one_time": False,
    }

    assert _fertilize_worth_it(
        tile,
        cd,
        day=9,
        prices={crop: 300, "FERTILIZER": 100},
        days_left=10,
        params=params,
    )
    assert not _fertilize_worth_it(
        tile,
        cd,
        day=9,
        prices={crop: 30, "FERTILIZER": 100},
        days_left=10,
        params=params,
    )

    wheat = _plant("WHEAT", planted_day=7, consecutive=0, yield_units=1)
    assert not _fertilize_worth_it(
        wheat,
        CROPS["WHEAT"],
        day=9,
        prices={"WHEAT": 100, "FERTILIZER": 20},
        days_left=10,
        params=params,
    )


def test_optional_global_assignment_minimizes_total_distance_within_tier():
    unit_pos = [(0, 0), (4, 0)]
    tasks = [
        (4, "HARVEST", 1, 0, None),
        (4, "HARVEST", 0, 2, None),
    ]
    params = {"basket": ("WHEAT",), "optimal_assignment": True}

    assigned, free = _assign(unit_pos, [{}, {}], tasks, {}, params)

    assert not free
    assert assigned[0][1:3] == (0, 2)
    assert assigned[1][1:3] == (1, 0)


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


def test_gen1_enables_round_two_efficiency_rules():
    assert GEN1_PARAMS["fertilizer_roi_margin"] == 1.0
    assert GEN1_PARAMS["fertilize_one_time"] is False
    assert GEN1_PARAMS["late_crew_caps"] == ((2, 10), (1, 8))
    assert GEN1_PARAMS["optimal_assignment"] is True


def test_frozen_full_params_replace_live_gen1_defaults():
    frozen = _resolve_params({
        "_replace_defaults": True,
        "max_quadrants": 3,
        "n_structures": 12,
    })

    assert frozen == {
        "_replace_defaults": True,
        "max_quadrants": 3,
        "n_structures": 12,
    }
    assert "optimal_assignment" not in frozen


def test_opponent_supply_keeps_a_minimum_competitive_market_share():
    params = {
        "opponent_supply_weight": 0.5,
        "animal_demand_share_floor": 0.4,
    }

    assert _effective_competitive_demand(25, 15, params) == 17.5
    assert _effective_competitive_demand(10, 30, params) == 4
    assert _effective_competitive_demand(10, 30, {}) == 0


def test_internal_feed_demand_bootstraps_wheat_before_animals_arrive():
    params = {
        "n_structures": 12,
        "feed_crop_bootstrap_share": 0.5,
        "feed_crop_demand_weight": 1.0,
    }

    assert _add_internal_feed_demand({"WHEAT": 7}, 2, params)["WHEAT"] == 13
    assert _add_internal_feed_demand({"WHEAT": 7}, 10, params)["WHEAT"] == 17
    assert _add_internal_feed_demand({"WHEAT": 7}, 10, {})["WHEAT"] == 7


def test_strawberry_share_only_expands_when_observed_demand_is_high():
    params = {
        "crop_share": {"STRAWBERRY": 0.4},
        "strawberry_high_demand": 13,
        "strawberry_high_share": 0.7,
    }

    assert _adaptive_crop_shares({"STRAWBERRY": 12}, params)["STRAWBERRY"] == 0.4
    assert _adaptive_crop_shares({"STRAWBERRY": 13}, params)["STRAWBERRY"] == 0.7


def test_early_coop_limit_leaves_irreversible_slots_uncommitted():
    plan = ("GOOSE",) * 6 + ("COW",) * 3 + ("SHEEP",) * 3
    params = {"early_coop_limit": 2, "early_coop_until_day": 15}

    limited = _limit_early_coops(plan, day=12, params=params)
    assert limited.count("GOOSE") == 2
    assert len(limited) == 8
    assert _limit_early_coops(plan, day=15, params=params) == plan


def test_planned_sales_keep_wheat_reserve_and_expose_same_turn_cash():
    inventory = {item: MARKET_PARAMS[item]["I0"] for item in MARKET_PARAMS}
    obs = {"market": {"inventory": inventory}}
    private = {
        "shed": {"WHEAT": 10, "STRAWBERRY": 5},
        "inventories": [{}],
    }
    params = {
        "sell_chunk": 40,
        "sell_price_frac": 0.55,
        "shed_force_sell": 0.75,
    }

    orders, revenue, sold = _planned_sale_orders(
        obs,
        private,
        params,
        shed_capacity=100,
        picked_from_shed={},
        wheat_reserve=6,
        liquidating=False,
    )

    assert ["SELL", "WHEAT", 4] in orders
    assert ["SELL", "STRAWBERRY", 5] in orders
    assert sold == 9
    assert revenue > 0


def test_liquidation_sells_all_products_returned_this_turn():
    inventory = {item: MARKET_PARAMS[item]["I0"] for item in MARKET_PARAMS}
    obs = {"market": {"inventory": inventory}}
    private = {"shed": {}, "inventories": [{"MILK": 47}]}
    params = {
        "sell_chunk": 40,
        "sell_price_frac": 0.55,
        "shed_force_sell": 0.75,
        "sell_same_turn_returns": True,
    }

    orders, _revenue, sold = _planned_sale_orders(
        obs,
        private,
        params,
        shed_capacity=100,
        picked_from_shed={},
        wheat_reserve=0,
        liquidating=True,
        projected_shed={"MILK": 47},
    )

    assert ["SELL", "MILK", 47] in orders
    assert sold == 47


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
