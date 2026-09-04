"""Kaggle 出貨用的進入點 —— `cma5-g139`。

第五輪 CMA-ES 第 139 代（`gen 140` 那次 checkpoint 打的就是這組）。
來源 `temp/cma/20260903-111413/gen_0140.pkl` 的 `es.best.x`，`best.f` 對得上
checkpoint 記的 train 11,623.55。維度 42，暖啟動 `cma4-g134`，sigma0 0.2，
train seed 0~39 對 `Dmitry Larko` 的開迴路重播。

    train 11,624   holdout 8,447   現役6 -5,853   其他五支 -8,842

「其他五支」= 現役六支排除訓練對手 Dmitry Larko 的平均。-8,842 是整輪 18 次
checkpoint 裡最好的一次。

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
    'animal_demand_share_floor': 0.4493948671147152,
    'animal_reserve': 471,
    'animal_spend_frac': 0.7889279195457863,
    'avoid_last_hour_planting': True,
    'basket': ['MELON', 'CARROT', 'CARROT', 'WHEAT', 'WHEAT'],
    'buy_land': True,
    'cash_reserve_days': 0,
    'crop_oversupply': {'MELON': 4.0},
    'crop_share': {   'CARROT': 0.877808940903245,
                      'MELON': 0.16051947834301322,
                      'STRAWBERRY': 0.38181455906985573,
                      'TOMATO': 0.7178154009635174,
                      'WHEAT': 0.8663801917941285},
    'daytime_return_max_distance': 11,
    'daytime_return_min_value': 710,
    'dynamic_animals': True,
    'dynamic_basket': True,
    'early_coop_limit': 2,
    'early_coop_until_day': 19,
    'fallback_crop': 'WHEAT',
    'feed_crop_bootstrap_share': 0.09998584259791128,
    'feed_crop_demand_weight': 1.3569087553783277,
    'fertilize_one_time': False,
    'fertilizer_roi_margin': 3.993553889109006,
    'hands_per_quadrant': 4,
    'hire_margin': 2.3485663830486794,
    'hire_per_turn': 9,
    'land_first_min_day': 5,
    'land_min_days_left': 17,
    'late_crew_caps': [[2, 10], [1, 8]],
    'liquidate_days_left': 2,
    'market_aware_pricing': True,
    'max_animal_share': 0.9671747119108621,
    'max_crop_share': 0.2874364458062525,
    'max_hands': 11,
    'max_quadrants': 3,
    'n_structures': 12,
    'opponent_supply_weight': 1.4659147401444492,
    'optimal_assignment': True,
    'oversupply_factor': 2.8608253640386625,
    'per_crop_lookahead': False,
    'priority_step_cost': 0.3983499234810248,
    'quadrant_zoning': True,
    'seed_backlog': 0.44990554056033544,
    'seed_buffer_days': 1,
    'sell_before_spending': False,
    'sell_chunk': 38,
    'sell_price_frac': 0.18220027540666425,
    'sell_same_turn_returns': True,
    'shed_force_sell': 0.9154961165403113,
    'strawberry_high_demand': 20,
    'strawberry_high_share': 0.7978505299275284,
    'structure_spread': 0.7959724188080938,
    'supply_lookahead_days': 2,
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
