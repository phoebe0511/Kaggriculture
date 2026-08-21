"""`gen2_model` + 建物數量上限。**單變數對照組，不是新一代。**

2026-08-20 實測 `gen2_model` 對 `ladder-top-a`（seed 1）：期末建物 **22 個**，
老師 14 個。原因是 `BUILD_PASTURE` / `BUILD_COOP` 是 unit 動作，由網路決定，
而 `config/opponents/gen2_model.json` 的 `n_structures` 只走 `gen0` 的市場那一路
—— `gen2_model` 把 gen0 的 unit 動作丟掉了，所以那個參數對建物數**完全無效**。

`contracts.legal_unit_mask` 也擋不住：它對 BUILD 的條件只有 `tile is None`
（`contracts.py:632`），因為那條的不變式是「寧可放寬，絕不擋掉老師做過的動作」，
數量上限是跨 unit、跨回合的累積約束，單一 unit 的 mask 表達不了。

所以跟 `PLANT` 的種子仲裁放同一層：在 `_choose` 裡數盤面上已經有幾個建物，
超過上限就往該 unit 的下一個候選動作走。

⚠️ **這一版只改這一件事。** 動物餓死、MOVE 佔比過高那些都沒動，目的是量出
「建物失控」單獨值多少錢。閉迴路發散的根因（老師的策略不是 observation 的
函數，見 `docs/memory/journal/2026-08-19.md` §7d）這裡沒有處理。
"""

from __future__ import annotations

import numpy as np

import contracts as C
from agents.gen0 import act as gen0_act
from agents.gen0 import DEFAULT_PARAMS as RULE_DEFAULTS
from agents.gen2_model import _policy

#: 兩種建物共用同一個上限 —— 引擎的建物就是佔掉一格可種的地。
BUILD_OPS = ("BUILD_COOP", "BUILD_PASTURE")


def structure_count(farm):
    """盤面上已經蓋好的建物數（含空的、含有動物的）。"""
    n = 0
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
                n += 1
    return n


def _choose(op_logits, qty_logits, mask, obs, build_budget):
    """挑動作：mask → PLANT 種子仲裁 → BUILD 數量上限。

    `build_budget` 是「這回合還能再蓋幾個」。同一回合可能有好幾個 unit 都想蓋，
    所以要邊配邊扣，不能各自看盤面上的數字。
    """
    scores = np.where(mask, op_logits, -np.inf)
    order = np.argsort(-scores, axis=1)

    seeds = dict(obs["private"]["seeds"])
    chosen = []
    for i in range(scores.shape[0]):
        picked = None
        for op_index in order[i]:
            if not np.isfinite(scores[i, op_index]):
                break                             # 後面都是不合法的
            op, item = C.UNIT_OPS[op_index]
            if op == "PLANT":
                # 種子全體共用，先搶先贏。硬送會讓該作物**所有** PLANT 變 PASS。
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

    base = gen0_act(obs, config, resolved)

    policy = _policy()
    spatial, scalar = C.encode(obs, config)
    positions, unit_features = C.encode_units(obs, config)
    op_logits, qty_logits, _target, _value = policy(
        spatial, scalar, positions, unit_features)
    mask = C.legal_unit_mask(obs, config)

    farm = obs["farms"][int(obs["player"])]
    # 上限用 gen0 市場那一路同一個參數，兩邊才會對同一個數字規劃。
    cap = int(resolved.get("n_structures", 12))
    budget = max(0, cap - structure_count(farm))

    units = _choose(op_logits, qty_logits, mask, obs, budget)
    return {"farmer": units[0], "hands": units[1:], "market": base["market"]}


def agent(obs, config):
    return act(obs, config)
