"""v5 的逐格需求：`contracts.TASK_OPS` / `legal_demand_mask` / `demand_tile_tasks`。

v5 把 unit 那一半切成兩段：網路判斷「哪一格要做什麼」，
`gen0._minimum_cost_assignment` 決定「誰去」。切口在
`gen0._tasks(demand=...)`，而切口是最容易無聲出錯的地方 —— 少一種 op、
座標 x/y 顛倒、`arg` 沒補上，三種都**不會報錯**，只會表現成分數莫名其妙變低。

所以這裡驗三件事，一局真的對局、每個回合都驗：

1. `_tasks()` 掃出來的每一筆都落在 `legal_demand_mask` 裡（mask 不變式）
2. demand map 餵回去，**任務清單一模一樣**（切口無失真）
3. demand map 餵回去，**送出的動作一模一樣**（下游沒被改到）

第 3 條是最重要的：它成立就代表 gen4 跟 gen1 之間只剩「網路的 F1」這一個變數。
"""

from __future__ import annotations

import numpy as np
import pytest
from kaggle_environments import make

import contracts as C
from agents.gen0 import act as gen1_act
from harness.rollout import demand_from_tasks, unpack_demand


#: 一局 720 回合太久（L0 要 < 60 秒），取前面這幾天。開局涵蓋 PLANT / WATER /
#: BUILD / PLACE / FETCH，中段才有 HARVEST / FEED / CARE，所以不能只取 day 0。
CHECK_STEPS = 200


@pytest.fixture(scope="module")
def turns():
    """跑一局 gen1，把前 `CHECK_STEPS` 個回合的 `(obs, config, action, plan)` 留下。"""
    captured = []

    def recorder(obs, config):
        action, plan = gen1_act(obs, config, None, return_plan=True)
        if len(captured) < CHECK_STEPS:
            captured.append((obs, config, action, plan))
        return action

    env = make("kaggriculture", configuration={"seed": 7}, debug=False)
    env.run([recorder, "starter"])
    assert len(captured) == CHECK_STEPS
    return captured


def test_every_expert_task_is_inside_the_legal_mask(turns):
    """mask 不變式：擋掉 expert 做過的事 = 網路連學都學不到，而且不會報錯。

    紅了就去放寬 `contracts.legal_demand_mask`，**不要改這條** ——
    `legal_unit_mask` 和 `legal_target_mask` 走的是同一條規矩。
    """
    seen = np.zeros(C.N_TASK_OPS, dtype=int)
    for obs, config, _action, plan in turns:
        demand = demand_from_tasks(plan["tasks"], plan["board"])
        legal = C.legal_demand_mask(obs, config)
        outside = demand & ~legal
        if outside.any():
            op_index, cell = (int(a[0]) for a in np.nonzero(outside))
            raise AssertionError(
                f"step {obs.get('step')}：{C.TASK_OPS[op_index]} @ "
                f"{C.target_xy(cell, plan['board'])} 被 mask 擋掉了")
        seen += demand.sum(axis=1)

    # 樣本要真的涵蓋到多種 op，不然這條測試等於沒驗
    assert (seen > 0).sum() >= 5, f"前 {CHECK_STEPS} 個回合只出現 {seen} 種需求"


def test_demand_map_round_trips_to_the_same_tasks(turns):
    """demand map 餵回 `_tasks()` -> 逐格任務的 `(op, x, y)` 集合要一樣。

    ⚠️ 比的是集合不是 list：`FETCH_*` 是導出的、可能有重複筆數，`arg` 也不在
    demand map 裡（`PLANT` 的作物、`PLACE` 的species都是查表補的）。所以這裡只
    比「哪一格要做什麼」—— 那正好就是 demand map 承載的全部資訊。
    """
    for obs, config, _action, plan in turns:
        board = plan["board"]
        demand = demand_from_tasks(plan["tasks"], board)
        _action2, plan2 = gen1_act(obs, config, None, return_plan=True,
                                   demand=demand)

        def cells(tasks):
            return {(op, x, y) for _pri, op, x, y, _arg in tasks
                    if op in C.TASK_OP_INDEX}

        assert cells(plan2["tasks"]) == cells(plan["tasks"]), \
            f"step {obs.get('step')} 的任務集合對不上"


def test_demand_map_round_trips_to_the_same_action(turns):
    """同一份 demand 餵回去，**送出的動作要一模一樣**。

    這條成立 = 切口沒有失真，gen4 跟 gen1 之間只剩「網路預測得準不準」。
    紅了先看 `demand_tile_tasks()` 的 `arg`：引擎對少了參數的動作
    （`["PLACE"]`、`["PLANT"]`）是**靜默忽略**，動作會變成 PASS。
    """
    for obs, config, action, plan in turns:
        demand = demand_from_tasks(plan["tasks"], plan["board"])
        replayed = gen1_act(obs, config, None, demand=demand)
        assert replayed["farmer"] == action["farmer"], \
            f"step {obs.get('step')} 的 farmer 動作不同"
        assert replayed["hands"] == action["hands"], \
            f"step {obs.get('step')} 的 hands 動作不同"
        assert replayed["market"] == action["market"], \
            f"step {obs.get('step')} 的市場訂單不同"


def test_packbits_round_trip():
    """`np.packbits` 的 padding bit 會多出 4 格，`count` 沒給就會整個錯位。"""
    rng = np.random.default_rng(0)
    demand = rng.random((C.N_TASK_OPS, C.N_TARGET_CELLS)) < 0.1
    packed = np.packbits(demand, axis=-1)

    assert packed.shape == (C.N_TASK_OPS, 13)
    assert (unpack_demand(packed) > 0).tolist() == demand.tolist()


def test_task_ops_cover_every_op_gen0_can_emit(turns):
    """`TASK_OPS` 少一種 op = 那種需求永遠傳不過切口，而且不會報錯。

    `FETCH_*` 是刻意不在裡面的（導出量，見 `contracts.TASK_OPS` 註解），
    `DROP` / `DROP_PRODUCT` 也不是 `_tasks()` 發的（那是
    `_daytime_return_assignments` 和最後一天的收尾）。
    """
    emitted = set()
    for _obs, _config, _action, plan in turns:
        emitted.update(op for _pri, op, _x, _y, _arg in plan["tasks"])

    missing = {op for op in emitted if not op.startswith("FETCH_")} - set(C.TASK_OPS)
    assert not missing, f"`_tasks()` 發得出 {sorted(missing)}，但 TASK_OPS 沒有"
