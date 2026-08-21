"""Kaggle submission 入口。

## 現在送出的是規則式（`agents/gen0.py`）

**這是目前最強的版本**，榜上 7 萬+。網路那條路（`agents/gen2_model.py`）
2026-08-21 量到是規則式的 76%，還沒到可以換上去的程度。

> 換上去的條件寫在 `docs/memory/journal/2026-08-21.md`：
> ① 要先在 `eval.runner` 打贏門檻。在那之前榜上維持規則式。
> `tests/test_l0_smoke.py::test_submission_entry_survives_a_whole_game`
> 斷言 `reward > 50_000`，那條線就是守這件事的。

⚠️ **不要因為「網路版比較新」就換上去。** 2026-08-20 曾經把 main.py 換成
v5 的 `gen4_demand` + `model_market: False`，那個組合的市場其實還是
`gen0._market` —— 分數有一半是規則式的功勞，而它仍然比純規則式低 4.4%。
那條線已於 2026-08-21 刪除。
"""

import os

# agents.gen0 在 import 時讀取此值，所以必須先設定。
os.environ["KAGGRI_LOG_LEVEL"] = "0"

from agents.gen0 import act  # noqa: E402


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

