"""Kaggle submission 入口。"""

import os

# agents.gen0（由 gen1 共用核心）在 import 時讀取此值，所以必須先設定。
os.environ["KAGGRI_LOG_LEVEL"] = "0"

from gen1 import act  # noqa: E402


def agent(obs, config):
    return act(obs, config)


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

