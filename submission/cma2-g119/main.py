"""Kaggle 出貨用的進入點 —— `cma2-g119`。

第二輪 CMA-ES（40 維全搜、暖啟動自 `cma1-g50-wt`）第 119 代的參數。
來源：`temp/cma/20260901-174022/`，130 代收工，train 分數 5,593。

`serving.build_submission.copy_files` 攤平之後 `agents/gen0.py` 變成同目錄的
`gen0.py`，`contracts.py` 也在旁邊。Kaggle 的載入器是
`exec(compile(raw, path, "exec"), {})` 加上 `sys.path.append(exec_dir)`，
所以 `__file__` 不存在、只有同目錄的檔案 import 得到。

⚠️ 2026-08-28 踩過：`contracts.py` 沒放進來，上榜每一回合都
`ModuleNotFoundError: No module named 'contracts'`，而本機 640 局驗收零錯誤。
驗收一定要用 `python -m tools.submission_check submission/cma2-g119`。
"""
from gen0 import act  # noqa: E402

#: 🩸 `_replace_defaults` 一定要在裡面 —— 驗收走的是 `eval.runner.build_agent`，
#: 它看到 spec 的 `frozen` 會自動補這一項（`eval/runner.py:135-137`），
#: 但 submission 不經過 build_agent，不內嵌的話兩邊行為會不一樣。
PARAMS = {   '_replace_defaults': True,
    'animal': 'GOOSE',
    'animal_demand_share_floor': 0.6124880204588585,
    'animal_reserve': 252,
    'animal_spend_frac': 0.6608521709041358,
    'avoid_last_hour_planting': True,
    'basket': ['MELON', 'CARROT', 'CARROT', 'WHEAT', 'WHEAT'],
    'buy_land': True,
    'cash_reserve_days': 0,
    'crop_oversupply': {'MELON': 4.0},
    'crop_share': {   'CARROT': 0.5568841589210506,
                      'MELON': 0.18781541108479646,
                      'STRAWBERRY': 0.22816618418390922,
                      'TOMATO': 0.3811710671589828,
                      'WHEAT': 0.5340881973662174},
    'daytime_return_max_distance': 5,
    'daytime_return_min_value': 368,
    'dynamic_animals': True,
    'dynamic_basket': True,
    'early_coop_limit': 4,
    'early_coop_until_day': 26,
    'fallback_crop': 'WHEAT',
    'feed_crop_bootstrap_share': 0.10484629768886045,
    'feed_crop_demand_weight': 1.1239326487201444,
    'fertilize_one_time': False,
    'fertilizer_roi_margin': 2.6750492961232526,
    'hands_per_quadrant': 4,
    'hire_margin': 1.6171308102232422,
    'hire_per_turn': 10,
    'land_first_min_day': 1,
    'land_min_days_left': 7,
    'late_crew_caps': [[2, 10], [1, 8]],
    'liquidate_days_left': 4,
    'market_aware_pricing': True,
    'max_animal_share': 0.9645862930950265,
    'max_crop_share': 0.2874364458062525,
    'max_hands': 16,
    'max_quadrants': 3,
    'n_structures': 15,
    'opponent_supply_weight': 1.0754324362798628,
    'optimal_assignment': True,
    'oversupply_factor': 3.1516477907149345,
    'per_crop_lookahead': False,
    'priority_step_cost': 1,
    'seed_backlog': 0.2246778918283468,
    'seed_buffer_days': 3,
    'sell_before_spending': False,
    'sell_chunk': 20,
    'sell_price_frac': 0.2682777032967104,
    'sell_same_turn_returns': True,
    'shed_force_sell': 0.9353310142273317,
    'strawberry_high_demand': 0,
    'strawberry_high_share': 0.7046901338460827,
    'structure_spread': 0.7942282518897951,
    'supply_lookahead_days': 6,
    'tiles_per_unit': 7,
    'use_fertilizer': True,
    'water_on_demand': True,
    'wheat_carry': 9,
    'wheat_reserve_days': 1,
    'whole_turn_assignment': True}


def agent(obs, config):
    return act(obs, config, PARAMS)
