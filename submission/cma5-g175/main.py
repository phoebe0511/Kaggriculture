"""Kaggle 出貨用的進入點 —— `cma5-g175`。

第五輪 CMA-ES 第 175 代（`gen 180` 那次 checkpoint）。同一輪、同一個暖啟動。

    train 18,330   holdout 7,198   現役6 -8,470   其他五支 -11,480

⚠️ **train 比 `cma5-g139` 高 6,706，但那個差讀不得。** 同一個對手換 seed 的
兩個量測都比 g139 低：holdout（40 個沒訓練過的 seed）-1,249、Dmitry 那一欄
（20 個 team seed）-2,515。train 是被優化的那個數字，40 個訓練 seed 上的
領先不代表別的地方也領先。

兩支在「其他五支」上差 2,638，那一欄 SE 約 2,588 —— 差距 1 個 SE，分不出高下。

兩支都比暖啟動點 `cma4-g134` 好（其他五支 +5,426 / +2,788；現役6 +9,238 / +6,621），
但整輪的進步集中在訓練對手身上：Dmitry Larko +25,783，其他五支 +2,788（1.1 SE）。

`serving.build_submission.copy_files` 攤平之後 `agents/gen0.py` 變成同目錄的
`gen0.py`，`contracts.py` 也在旁邊。Kaggle 的載入器是
`exec(compile(raw, path, "exec"), {})` 加上 `sys.path.append(exec_dir)`，
所以 `__file__` 不存在、只有同目錄的檔案 import 得到。

⚠️ 2026-08-28 踩過：`contracts.py` 沒放進來，上榜每一回合都
`ModuleNotFoundError: No module named 'contracts'`，而本機 640 局驗收零錯誤。
驗收一定要用 `python -m tools.submission_check submission/<名字>`。
"""
from gen0 import act  # noqa: E402

#: 🩸 `_replace_defaults` 一定要在裡面 —— 驗收走的是 `eval.runner.build_agent`，
#: 它看到 spec 的 `frozen` 會自動補這一項（`eval/runner.py:135-137`），
#: 但 submission 不經過 build_agent，不內嵌的話兩邊行為會不一樣。
PARAMS = {   '_replace_defaults': True,
    'animal': 'GOOSE',
    'animal_demand_share_floor': 0.5160864794651167,
    'animal_reserve': 638,
    'animal_spend_frac': 0.7645717622833793,
    'avoid_last_hour_planting': True,
    'basket': ['MELON', 'CARROT', 'CARROT', 'WHEAT', 'WHEAT'],
    'buy_land': True,
    'cash_reserve_days': 0,
    'crop_oversupply': {'MELON': 4.0},
    'crop_share': {   'CARROT': 0.9447241289793964,
                      'MELON': 0.22236324917599903,
                      'STRAWBERRY': 0.4017448012994426,
                      'TOMATO': 0.6236233411439649,
                      'WHEAT': 0.8513964572628612},
    'daytime_return_max_distance': 12,
    'daytime_return_min_value': 839,
    'dynamic_animals': True,
    'dynamic_basket': True,
    'early_coop_limit': 2,
    'early_coop_until_day': 18,
    'fallback_crop': 'WHEAT',
    'feed_crop_bootstrap_share': 0.08993029440619832,
    'feed_crop_demand_weight': 1.4694350041383464,
    'fertilize_one_time': False,
    'fertilizer_roi_margin': 3.995122090012244,
    'hands_per_quadrant': 4,
    'hire_margin': 2.3413304622699433,
    'hire_per_turn': 9,
    'land_first_min_day': 5,
    'land_min_days_left': 16,
    'late_crew_caps': [[2, 10], [1, 8]],
    'liquidate_days_left': 2,
    'market_aware_pricing': True,
    'max_animal_share': 0.990097404043162,
    'max_crop_share': 0.2874364458062525,
    'max_hands': 10,
    'max_quadrants': 3,
    'n_structures': 13,
    'opponent_supply_weight': 1.4978469716273253,
    'optimal_assignment': True,
    'oversupply_factor': 2.9072919545584255,
    'per_crop_lookahead': False,
    'priority_step_cost': 0.2963198895564026,
    'quadrant_zoning': True,
    'seed_backlog': 0.4009334881311669,
    'seed_buffer_days': 1,
    'sell_before_spending': False,
    'sell_chunk': 42,
    'sell_price_frac': 0.21223118348027434,
    'sell_same_turn_returns': True,
    'shed_force_sell': 0.9344164878932941,
    'strawberry_high_demand': 25,
    'strawberry_high_share': 0.730004019042534,
    'structure_spread': 0.8219444722692295,
    'supply_lookahead_days': 1,
    'tiles_per_unit': 7,
    'use_fertilizer': True,
    'water_on_demand': True,
    'wheat_carry': 8,
    'wheat_reserve_days': 1,
    'whole_turn_assignment': True,
    'zone_penalty': 1,
    'zone_planting_only': True}


def agent(obs, config):
    return act(obs, config, PARAMS)
