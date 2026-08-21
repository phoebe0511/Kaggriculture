"""第 3 代：網路只決定「去哪一格、到了做什麼」，走路交給 `gen0.step_toward`。

## 為什麼要改成這樣

`gen2_model` 每一步直接輸出動作。對老師局面的 unit-turn 準確率 0.9666
（2026-08-20 實測，驗證集 2 局 27,658 個 unit-turn，走的是這裡同一條路），
自己下場卻 0 勝 4 負、平均現金 8,002 對 122,114。

`tools/action_dist.py` 量得出壞在哪：

    MOVE                75.6%  vs  老師 51.8%
    FEED                70 次  vs  老師 1,252 次
    CARE                79     vs  1,240
    COLLECT_FERTILIZER  110    vs  1,224
    FERTILIZE           6      vs  312

四條斷線**全部是多步例行動作**：走到 shed → `PICKUP WHEAT` → 走到 pasture →
`FEED`。「這個 unit 要去哪」不在任何 observation 欄位裡，所以 gen2 每回合重新猜
目標，猜的一變就掉頭（方向反轉率 32.9%，老師 9.2%）。

v3 把那個看不見的變數變成顯式輸出：`target_logits` 說去哪一格，`op_logits` 說
到了做什麼（`contracts.ENCODER_VERSION = 3`）。

## 走路不學

`agents/gen0.step_toward`（貪婪先 x 後 y）重現老師實際那一步的比例是 **93.02%**
（60 局 replay、19,750 個 MOVE）。剩下 7% 是先走 x 還先走 y 的差別。

## v1 不存狀態

每回合重算 target。老師的 segment 平均只有 1.79 步、56.4% 只有 1 步，而盤面
每回合幾乎不變 —— 連續兩步的 target 預測本來就該一致。不存狀態就沒有
journal §7e 擔心的「推論時自己維護的狀態跟訓練分布不一致」。

若 `action_dist` 顯示反轉率仍高，`_STICKY` 那個掛點就是加黏性的地方。

## 沿用 gen2_capped 的兩條跨 unit 仲裁

單一 unit 的 mask 表達不了跨 unit 的累積約束，所以放在挑動作這一層：

- **PLANT 種子**：同回合對某作物的請求總數超過手上種子數 → 該作物的**所有**
  PLANT 全部變 PASS（`engine-notes.md` §10.9）。超額的往下一個候選走。
- **建物上限**：`BUILD_*` 是 unit 動作，`n_structures` 只走 gen0 的市場那一路。
  2026-08-20 實測 gen2_model 蓋到 25.2 個（老師 14.0）。

## 市場：`model_market` 決定走網路還是走 gen0

v4 之前市場一直沿用 `gen0._market`，而 2026-08-20 量到那是致命的：

| | 老師（2 局平均） | gen0 的市場 |
|---|---|---|
| 整季賣出 | $185,358 | $9,977 |
| BUY_PRODUCT WHEAT（飼料） | 553 個 | 13 個 |
| BUY_ANIMAL | COW 10 + SHEEP 4 | COW 3 |

按天比對狀態分布：**day 6 動物就掉出老師的 p5**（我們 1 隻、老師 p5 是 6 隻），
那時候現金還在第 44 百分位。動物是老師 34% 的收入、24% 的終點動作
（FEED / CARE / COLLECT_FERTILIZER）—— 沒有動物，unit 就真的沒事幹（PASS 36%）。

所以 v4 把市場也交給網路。`model_market: false` 保留舊行為當 A/B 對照組，
**同一份權重跑兩個 config**（`gen3_target.json` / `gen4_market.json`），
量得出市場那一半單獨值多少。

⚠️ 兩邊的訂單**不可以混**。開著時 gen0 那包整份丟掉 —— 混的話 `HIRE` 和
`BUY_LAND` 會下兩次。
"""

from __future__ import annotations

import numpy as np

import contracts as C
from agents.gen0 import act as gen0_act, step_toward
from agents.gen0 import DEFAULT_PARAMS as RULE_DEFAULTS
from agents.gen2_capped import BUILD_OPS, structure_count
from agents.gen2_model import _policy

def _choose_targets(target_logits, target_mask):
    """每個 unit 挑一個目標格。回傳 `[n]` 的 cell index。

    v1 不存狀態 —— 要加黏性的話就是在這裡：傳入上一回合的 `(slot -> cell)`，
    目標仍合法就沿用，不要重挑。
    """
    return np.where(target_mask, target_logits, -np.inf).argmax(axis=1)


def _choose_actions(op_logits, qty_logits, op_mask, targets, positions,
                    obs, board, build_cap):
    """已經有目標了，組出這一回合真正要送出的動作。

    沒到目標就走一步；到了就用 op head 的第一名（過 mask 之後），再套上
    PLANT 種子與建物上限這兩條跨 unit 仲裁。
    """
    scores = np.where(op_mask, op_logits, -np.inf)
    order = np.argsort(-scores, axis=1)

    seeds = dict(obs["private"]["seeds"])
    farm = obs["farms"][int(obs["player"])]
    build_budget = max(0, build_cap - structure_count(farm))

    chosen = []
    for i in range(len(positions)):
        pos = (int(positions[i][0]), int(positions[i][1]))
        if pos != C.target_xy(targets[i], board):
            chosen.append(step_toward(pos, C.target_xy(targets[i], board)))
            continue

        picked = None
        for op_index in order[i]:
            if not np.isfinite(scores[i, op_index]):
                break                             # 後面都是不合法的
            op, item = C.UNIT_OPS[op_index]
            if op == "PLANT":
                if seeds.get(item, 0) <= 0:
                    continue
                seeds[item] -= 1
            elif op in BUILD_OPS:
                if build_budget <= 0:
                    continue
                build_budget -= 1
            picked = op_index
            break
        if picked is None:
            picked = C.UNIT_OP_INDEX[("PASS", None)]
        chosen.append(C.decode_unit(picked, int(np.argmax(qty_logits[i]))))
    return chosen


def act(obs, config=None, params=None):
    resolved = dict(RULE_DEFAULTS)
    resolved.update(params or {})
    resolved.pop("_replace_defaults", None)
    model_market = bool(resolved.get("model_market", True))

    # gen0 仍然要跑：`model_market` 關掉時要用它的市場，開著時**整包丟掉**。
    # 不要把兩邊的訂單混在一起 —— 雇工和買地會變成下兩次。
    base = gen0_act(obs, config, resolved)

    policy = _policy()
    spatial, scalar = C.encode(obs, config)
    positions, unit_features = C.encode_units(obs, config)
    # ⚠️ v5 起多一個 demand head，這一版用不到 —— 它的比較對象就是 v5
    # （`agents/gen4_demand.py`），留著當 A/B 的舊臂。
    (op_logits, qty_logits, target_logits,
     mk_present, mk_qty, _value, _demand) = policy(
        spatial, scalar, positions, unit_features)

    board = len(obs["farms"][int(obs["player"])]["tiles"])
    targets = _choose_targets(target_logits, C.legal_target_mask(obs, config))
    units = _choose_actions(
        op_logits, qty_logits, C.legal_unit_mask(obs, config),
        targets, positions, obs, board, int(resolved.get("n_structures", 12)))

    market = (C.decode_market_orders(mk_present, mk_qty, obs, config)
              if model_market else base["market"])
    return {"farmer": units[0], "hands": units[1:], "market": market}


def agent(obs, config):
    return act(obs, config)
