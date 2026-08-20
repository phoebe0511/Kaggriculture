"""Kaggle submission 入口。

## 現在送出的是 v5 的網路版（`agents/gen4_demand.py`）

網路只決定**哪一格要做什麼**（`contracts.TASK_OPS` 的 11 個 op × 100 格）；
派誰去、怎麼走、市場下什麼單全部走 `agents/gen0.py` 的既有程式碼。

⚠️ **`model_market` 明寫 `False`。** `gen4_demand.act` 的預設是 `True`，
而那是**比較差**的那一臂 —— 2026-08-20 實測同一份權重、同樣 4 個 seed
對 ladder-top-a：

    model_market: False   $69,140     PASS 17.7%   期末作物 14.2/75.0
    model_market: True    $27,965     PASS 47.5%   期末作物  4.1/56.2

market head 的召回率只有 0.79，少下的訂單裡有種子和雇工，所以田是空的、
人是少的，unit 沒事幹就 PASS。市場那一半修好之前不要開。

⚠️ 權重是 `weights.npz`，跟這支攤平在同一個目錄（`serving/build_submission.py`）。
`ENCODER_VERSION` 對不上的話 `agents/gen2_model._policy()` 會直接 `SystemExit`
—— 那是刻意的，載到不同 schema 的權重只會無聲地打得很爛。
"""

import os

# agents.gen0（由 gen1 / gen4 共用核心）在 import 時讀取此值，所以必須先設定。
os.environ["KAGGRI_LOG_LEVEL"] = "0"

from agents.gen4_demand import act  # noqa: E402


def agent(obs, config):
    return act(obs, config, {"model_market": False})


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

