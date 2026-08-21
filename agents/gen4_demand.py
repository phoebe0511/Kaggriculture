"""第 4 代：網路只判斷「哪一格要做什麼」，配對和走路還給演算法。

## 為什麼從 gen3 改成這樣

gen3（v4）是**逐 unit** 輸出目標格：12 個 unit 各自猜自己要去哪。DAgger 三輪
之後 target 準確率停在 0.6301，而整個回合要全部放對的機率是 `0.63 ** 12 =
0.4%`。v3 把錯誤從時間軸（老師 73 步的開局腳本，`0.868 ** 73 = 0.003%`）搬掉了，
但它搬到了 unit 軸 —— 同一個乘法，換一根軸。

更糟的是那個準確率**隨資料量下降**：

    round 0  0.7580     round 1  0.6505     round 2  0.6301

資料愈多愈差 = 標籤有雜訊。雜訊來自 `gen0._assign` 的
`_minimum_cost_assignment`：兩個 unit 對兩個等距的任務，配對哪一種都是最佳解，
選哪一種由 Hungarian algorithm 的內部順序決定。**同一個盤面，不同標籤。**
那是學不起來的東西，加再多資料只會讓它更明顯。

## 切法：判斷給網路，配對給演算法

| | 誰做 | 為什麼 |
|---|---|---|
| 哪一格該澆水、值不值得施肥 | 網路 | 是盤面的函式，有唯一答案 |
| 誰去哪一格比較近 | `_minimum_cost_assignment` | 算得出最佳解，不會錯 |
| 走哪一步 | `gen0.step_toward` | 對 60 局 replay 命中 93.02% |

網路輸出 `[11, 100]` 的 multi-label demand map（`contracts.TASK_OPS`），
`gen0.demand_tile_tasks()` 把它補成完整任務清單（優先序、PLANT 的作物、
PLACE / BUILD 的species），之後**整條 gen0 的下游原封不動**：`_assign` 配對、
`_task_action` 走路、`_daytime_return_assignments` 回倉、`_market` 或網路下單。

一格猜錯只影響那一格 —— 不會像 v4 那樣讓整個回合的配對錯位。

## `FETCH_*` 不由網路決定

要幾趟 WHEAT 由 FEED 的格數決定、要幾趟 FERTILIZER 由 FERTILIZE 的格數決定、
要幾趟動物由 PLACE 的格數決定。那是導出量，`gen0._tasks()` 的補給那一段算得
比網路準。理由寫在 `contracts.TASK_OPS` 的註解。

## 兩個掛在 params 上的開關

- `demand_threshold`：logit 門檻，預設 0（sigmoid 0.5）。調低 = 做得多，
  調高 = 做得少。**不要拿它救分數** —— 它會同時改變 FETCH 的數量，
  兩邊一起動很難歸因。
- `model_market`：跟 gen3 同一個意思，開著就整包用網路的訂單。
  ⚠️ 兩邊的訂單**不可以混**，混的話 HIRE 和 BUY_LAND 會下兩次。
"""

from __future__ import annotations

import contracts as C
from agents.gen0 import act as gen0_act
from agents.gen0 import DEFAULT_PARAMS as RULE_DEFAULTS
from agents.gen2_model import _policy


def choose_demand(demand_logits, legal, threshold=0.0):
    """logits + 合法遮罩 -> `[N_TASK_OPS, N_TARGET_CELLS]` 的 bool。

    遮罩是硬的：不合法的格子一律不發任務。引擎對不合法的動作是**靜默忽略**，
    發出去只是白丟一個 unit-turn，而且 log 裡看起來像「有做事」。
    """
    return (demand_logits > threshold) & legal


def act(obs, config=None, params=None):
    resolved = dict(RULE_DEFAULTS)
    resolved.update(params or {})
    resolved.pop("_replace_defaults", None)
    model_market = bool(resolved.get("model_market", True))
    threshold = float(resolved.get("demand_threshold", 0.0))

    policy = _policy()
    spatial, scalar = C.encode(obs, config)
    positions, unit_features = C.encode_units(obs, config)
    (_op, _qty, _target, mk_present, mk_qty, _value, demand_logits) = policy(
        spatial, scalar, positions, unit_features)

    demand = choose_demand(
        demand_logits, C.legal_demand_mask(obs, config), threshold)
    base = gen0_act(obs, config, resolved, demand=demand)

    market = (C.decode_market_orders(mk_present, mk_qty, obs, config)
              if model_market else base["market"])
    return {"farmer": base["farmer"], "hands": base["hands"], "market": market}


def agent(obs, config):
    return act(obs, config)
