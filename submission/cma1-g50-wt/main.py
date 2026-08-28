"""Kaggle 出貨用的進入點 —— `cma1-g50-wt`。

`python -m serving.build_submission` 攤平之後 `agents/gen0.py` 變成同目錄的
`gen0.py`，`contracts.py` 也在旁邊。Kaggle 的載入器是
`exec(compile(raw, path, "exec"), {})` 加上 `sys.path.append(exec_dir)`，
所以 `__file__` 不存在、只有同目錄的檔案 import 得到。

⚠️ 2026-08-28 踩過：`contracts.py` 沒放進來，上榜每一回合都
`ModuleNotFoundError: No module named 'contracts'`，而本機 640 局驗收零錯誤
（`eval/runner.py` 在 repo root 底下跑，root 就在 sys.path 上）。
驗收一定要用 `python -m tools.submission_check submission/cma1-g50-wt`。
"""
from gen0 import act  # noqa: E402

#: 🩸 `_replace_defaults` 一定要在裡面 —— 驗收走的是 `eval.runner.build_agent`，
#: 它看到 spec 的 `frozen` 會自動補這一項（`eval/runner.py:135-137`），
#: 但 submission 不經過 build_agent，不內嵌的話兩邊行為會不一樣。
PARAMS = {   '_replace_defaults': True,
    'animal': 'GOOSE',
    'animal_demand_share_floor': 0.2504081850740226,
    'animal_reserve': 298,
    'animal_spend_frac': 0.7614473584950167,
    'avoid_last_hour_planting': True,
    'basket': ['MELON', 'CARROT', 'CARROT', 'WHEAT', 'WHEAT'],
    'buy_land': True,
    'cash_reserve_days': 0,
    'crop_oversupply': {'MELON': 4.0},
    'crop_share': {'STRAWBERRY': 0.4},
    'daytime_return_max_distance': 9,
    'daytime_return_min_value': 138,
    'dynamic_animals': True,
    'dynamic_basket': True,
    'early_coop_limit': 4,
    'early_coop_until_day': 21,
    'fallback_crop': 'WHEAT',
    'feed_crop_bootstrap_share': 0.0772708354663828,
    'feed_crop_demand_weight': 0.39930217486156255,
    'fertilize_one_time': False,
    'fertilizer_roi_margin': 1.2491323653413258,
    'hands_per_quadrant': 4,
    'hire_margin': 1.0497741916829,
    'hire_per_turn': 4,
    'land_first_min_day': 5,
    'land_min_days_left': 5,
    'late_crew_caps': [[2, 10], [1, 8]],
    'liquidate_days_left': 3,
    'market_aware_pricing': True,
    'max_animal_share': 0.7743249911752783,
    'max_crop_share': 0.2874364458062525,
    'max_hands': 11,
    'max_quadrants': 3,
    'n_structures': 12,
    'opponent_supply_weight': 0.35940589406834345,
    'optimal_assignment': True,
    'oversupply_factor': 2.8930691298383002,
    'priority_step_cost': 1,
    'seed_buffer_days': 2,
    'sell_before_spending': False,
    'sell_chunk': 20,
    'sell_price_frac': 0.44156676275478285,
    'sell_same_turn_returns': True,
    'shed_force_sell': 0.6517888746109459,
    'strawberry_high_demand': 19,
    'strawberry_high_share': 0.6399926861305701,
    'supply_lookahead_days': 3,
    'tiles_per_unit': 6,
    'use_fertilizer': True,
    'water_on_demand': True,
    'wheat_carry': 9,
    'wheat_reserve_days': 2,
    'whole_turn_assignment': True}


def agent(obs, config):
    return act(obs, config, PARAMS)
