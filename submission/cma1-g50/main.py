"""Kaggle submission 入口 —— CMA-ES 調過參數的規則式（`cma1-g50`）。

## 這一版跟 `submission/main.py`（榜上那版）的差別

只有一件事：`act()` 多帶一組參數。程式碼完全是同一支 `agents/gen0.py`。

    參數來源  config/params/cma1-g50.json（51 項完整展開）
    CMA state model/cma1-g50.pkl（run temp/cma/20260827-164222 第 50 代）

## 2026-08-28 的實測（seed 2000~2039，跟 CMA 的 train / holdout 都不重疊）

    對 gen0 預設參數（會反應）    我方 81,641 -> 94,357   勝率 46.2% -> 95.0%
                                  雙方合計 163,282 -> 170,108（餅變大）
    對 8 支 ladder replay         現金比 57.9% -> 76.7%   勝率 3.0% -> 14.4%

🩸 **後面那組被灌水了。** 對 replay 的改善有 **81%** 來自壓低對手（對手現金
−21,109、我方只 +4,815），而 replay 是開迴路、買不到東西不會改買別的。
對會反應的對手只有 32% 來自壓低對手 —— 那一組才是可轉移的證據。

所以真實的期望改善介於「我方 +4,815」和「我方 +12,716」之間，
榜上的勝率提升應該比 3.0% -> 14.4% 小。

⚠️ 本機 ladder 是 1.32.7 + 本機預設 config。Kaggle notebook 量到是 1.29.3
且 config 不同，兩邊的分數不能互相換算。
"""

import os

# agents.gen0 在 import 時讀取此值，所以必須先設定。
os.environ["KAGGRI_LOG_LEVEL"] = "0"

from gen0 import act  # noqa: E402

#: CMA-ES 調出來的 51 項參數（`config/params/cma1-g50.json`）。
#: 🩸 `_replace_defaults` 一定要在裡面 —— 驗收走的是
#: `eval.runner.build_agent`，它看到 spec 的 `frozen` 會自動補這一項
#: （`eval/runner.py:135-137`）。submission 不經過那條路，不自己補的話
#: 這 51 項會跟 `gen0.DEFAULT_PARAMS` 合併：今天是 no-op，但之後 gen0
#: 多一個預設 key，上場的 agent 就跟量過的不是同一個了。
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
    'seed_buffer_days': 2,
    'sell_before_spending': False,
    'sell_chunk': 20,
    'sell_price_frac': 0.44156676275478285,
    'sell_same_turn_returns': True,
    'shed_force_sell': 0.6517888746109459,
    'strawberry_high_demand': 19,
    'strawberry_high_share': 0.6399926861305701,
    'supply_lookahead_days': 3,
    'tiles_per_unit': 3,
    'use_fertilizer': True,
    'water_on_demand': True,
    'wheat_carry': 9,
    'wheat_reserve_days': 2}


def agent(obs, config):
    return act(obs, config, PARAMS)


# 這裡**故意不呼叫** `serving.action_validation.assert_legal_action`。
#
# 那支驗證器的做法是拿引擎的**私有函式**（`_commit_unit`、`_apply_unit_action`、
# `_do_hire`、`_do_buy_land`）在深拷貝上重放一次動作，看狀態有沒有變 ——
# 沒變就代表引擎會靜默忽略它。開發時很有用，抓過 `["PLACE"]` 少帶參數
# 那個 bug（337 個回合空轉、12 隻鵝卡在倉庫，引擎一聲不吭）。
#
# 但放進 submission 是三重虧本：
#
#   1. 比賽當下抓到 bug 也不能改，**價值是零**；而抓到就拋錯會讓整局死掉、
#      拿 0 分 —— 本來只是那一個動作被忽略而已。
#   2. 時間：本機實測整局 2.6s -> 5.8s，單回合尖峰 15.7 -> 30.6 ms。
#      它每個 unit 做兩次 deepcopy，13 個 unit x 720 回合 = 18,720 次。
#   3. 私有 API 會隨版本改。2026-08-17 在 Kaggle notebook（1.29.3）實測：
#        1.29.3  _commit_unit(op, item, price, farm, private, market)
#        1.32.7  _commit_unit(..., market, shed_capacity=100)
#      多傳一個參數的結果是**每一回合都 TypeError**。ladder 用哪個版本
#      我們看不到，所以這個風險沒辦法靠「本機測過了」消掉。
#
# 驗證留在開發側：`tests/test_l0_smoke.py` 每回合都會呼叫它。

