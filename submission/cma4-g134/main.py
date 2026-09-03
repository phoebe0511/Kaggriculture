"""Kaggle 出貨用的進入點 —— `cma4-g134`。

第四輪 CMA-ES（42 維，暖啟動自 `cma2-g119-zone0`）第 134 代的參數。
來源：`temp/cma/20260902-134323/`，train 分數 9,175。

跟 `cma2-g119` 比，這一輪動最大的是**排班那一層**（前三輪完全沒搜過）：

    priority_step_cost  1.0 -> 0.044   優先序幾乎不算數，誰近派誰
    zone_penalty        0（分區機制搜過但不要）
    tiles_per_unit      7 -> 4         顧比較少的地，但顧得動

本機實測（8 個 seed，對 ladder-top-a）：day 12~26 的空地率從 13.5%->25.2%
的爬升變成平在 9.7~11.2%；作物格數從 50 掉到 41 變成穩在 51~53；
作物枯死變雜草 10.0 -> 3.6 株。

⚠️ **尚未對現役對手驗收。** 第四輪的 train / holdout / teams 三個指標，對手
全部是 2026-08-19 抓的開迴路重播（journal §78），而線上 ladder 已經洗牌。

`serving.build_submission.copy_files` 攤平之後 `agents/gen0.py` 變成同目錄的
`gen0.py`，`contracts.py` 也在旁邊。Kaggle 的載入器是
`exec(compile(raw, path, "exec"), {})` 加上 `sys.path.append(exec_dir)`，
所以 `__file__` 不存在、只有同目錄的檔案 import 得到。

⚠️ 2026-08-28 踩過：`contracts.py` 沒放進來，上榜每一回合都
`ModuleNotFoundError: No module named 'contracts'`，而本機 640 局驗收零錯誤。
驗收一定要用 `python -m tools.submission_check submission/cma4-g134`。
"""
from gen0 import act  # noqa: E402

#: 🩸 `_replace_defaults` 一定要在裡面 —— 驗收走的是 `eval.runner.build_agent`，
#: 它看到 spec 的 `frozen` 會自動補這一項（`eval/runner.py:135-137`），
#: 但 submission 不經過 build_agent，不內嵌的話兩邊行為會不一樣。
PARAMS = {   '_replace_defaults': True,
    'animal': 'GOOSE',
    'animal_demand_share_floor': 0.07230695534870103,
    'animal_reserve': 383,
    'animal_spend_frac': 0.5940550928178425,
    'avoid_last_hour_planting': True,
    'basket': ['MELON', 'CARROT', 'CARROT', 'WHEAT', 'WHEAT'],
    'buy_land': True,
    'cash_reserve_days': 0,
    'crop_oversupply': {'MELON': 4.0},
    'crop_share': {   'CARROT': 0.8298431035464646,
                      'MELON': 0.16046792404538412,
                      'STRAWBERRY': 0.2744660149004099,
                      'TOMATO': 0.22454324346914684,
                      'WHEAT': 0.7900037462489385},
    'daytime_return_max_distance': 8,
    'daytime_return_min_value': 201,
    'dynamic_animals': True,
    'dynamic_basket': True,
    'early_coop_limit': 3,
    'early_coop_until_day': 17,
    'fallback_crop': 'WHEAT',
    'feed_crop_bootstrap_share': 0.1895856991612453,
    'feed_crop_demand_weight': 1.4110223281702612,
    'fertilize_one_time': False,
    'fertilizer_roi_margin': 3.4281811245328164,
    'hands_per_quadrant': 4,
    'hire_margin': 1.8858150921717192,
    'hire_per_turn': 10,
    'land_first_min_day': 2,
    'land_min_days_left': 16,
    'late_crew_caps': [[2, 10], [1, 8]],
    'liquidate_days_left': 3,
    'market_aware_pricing': True,
    'max_animal_share': 0.751436461635602,
    'max_crop_share': 0.2874364458062525,
    'max_hands': 14,
    'max_quadrants': 3,
    'n_structures': 14,
    'opponent_supply_weight': 1.2530635601023774,
    'optimal_assignment': True,
    'oversupply_factor': 3.7169140174621194,
    'per_crop_lookahead': False,
    'priority_step_cost': 0.04434998764916002,
    'quadrant_zoning': True,
    'seed_backlog': 0.4737029259911827,
    'seed_buffer_days': 1,
    'sell_before_spending': False,
    'sell_chunk': 13,
    'sell_price_frac': 0.2638377488453688,
    'sell_same_turn_returns': True,
    'shed_force_sell': 0.9696111875237916,
    'strawberry_high_demand': 3,
    'strawberry_high_share': 0.6922756623291464,
    'structure_spread': 0.7547931443445703,
    'supply_lookahead_days': 2,
    'tiles_per_unit': 4,
    'use_fertilizer': True,
    'water_on_demand': True,
    'wheat_carry': 6,
    'wheat_reserve_days': 1,
    'whole_turn_assignment': True,
    'zone_penalty': 0,
    'zone_planting_only': True}


def agent(obs, config):
    return act(obs, config, PARAMS)
