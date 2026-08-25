"""快照 → 套用候選 → 用 rollout policy 打到底 → 比期末現金。

## 為什麼可以這樣搜

引擎的 forward model 是**決定性的**：同一個快照 + 同一個 policy 跑幾次，期末現金
逐位元組相同（`tools/value_probe.py` 驗過，2026-08-25 在 seed 7 / day 12 又驗一次，
三次都是 92,214）。所以：

- **一個候選只要打一次**就是準確值，不用重複取樣
- 候選之間可以直接比大小，不需要統計檢定

這跟 08-25 量到的「CMA-ES 的 seed 抽樣誤差」是不同層次的東西 —— 那個是「換一組
參數之後跨 seed 的平均會不會比較好」，這裡是「**同一個 seed、同一個盤面**，換這一步
會不會比較好」。後者沒有噪音。

## 回溯的做法

`env.clone()` 要 13.75 ms，太貴。改成：

    deepcopy(env.state)  →  砍掉 env.steps 多出來的部分  →  status 設回 "ACTIVE"

（`env.done` 是唯讀 property，只能透過 `status` 反推。）
2026-08-25 在 day 12 的盤面實測 `deepcopy(env.state)` 是 **1.58 ms**。
⚠️ 那個成本會隨局面長大，季末更貴。

## rollout policy 用 gen0

2026-08-25 定案。理由：`gen0.act` 0.46 ms 對 `gen2_model.act` 3.54 ms，便宜 4 倍；
而且現在 gen0 比網路強（網路只有 gen1 的 93%），用它往下打估出來的候選價值更接近
「這步到底好不好」。

代價是**基準線也必須是 gen0 自己的動作** —— 用 gen0 的延續去評估網路的動作，
量到的是「這個動作對 gen0 好不好」，不是「對網路好不好」。
"""
from __future__ import annotations

import copy

#: 一季幾天。引擎預設 episodeSteps 720 / turnsPerDay 24。
DAYS = 30


def snapshot(env):
    """存檔。回傳的東西丟給 `restore()`。"""
    return copy.deepcopy(env.state), len(env.steps)


def restore(env, snap):
    """回到快照那一刻。

    🩸 三件事都要做，少一件就會安靜地錯：
    `env.state` 要**再 deepcopy 一次**（不然下一次 restore 拿到的是被改過的），
    `env.steps` 要砍掉多出來的，`status` 要設回 `ACTIVE`。
    """
    state, nsteps = snap
    env.state = copy.deepcopy(state)
    del env.steps[nsteps:]
    for s in env.state:
        s.status = "ACTIVE"


def final_cash(env, player=0):
    """期末現金。期末只算現金，留在 shed 的東西一毛都不算。"""
    return float(env.state[0].observation["farms"][player]["money"])


def play_to_end(env, act_us, act_them, player=0, days=DAYS):
    """從現在打到季末，回傳我方期末現金。`act_*` 吃 obs 回傳 action。"""
    while env.state[0].observation["day"] < days and not env.done:
        obs_us = env.state[player].observation
        obs_them = env.state[1 - player].observation
        actions = [None, None]
        actions[player] = act_us(obs_us)
        actions[1 - player] = act_them(obs_them)
        env.step(actions)
    return final_cash(env, player)


def evaluate(env, snap, action, act_them, rollout_act, player=0, days=DAYS):
    """把 `action` 套在快照那一步，之後用 `rollout_act` 打到底。

    回傳期末現金 —— 也就是 Q(s, action)，在 `rollout_act` 這個延續策略之下。
    """
    restore(env, snap)
    actions = [None, None]
    actions[player] = action
    actions[1 - player] = act_them(env.state[1 - player].observation)
    env.step(actions)
    return play_to_end(env, rollout_act, act_them, player, days)


def evaluate_all(env, snap, candidates, act_them, rollout_act, player=0,
                 days=DAYS, on_result=None):
    """一批候選，回傳 `[(label, 期末現金), ...]`，順序同輸入。

    `candidates` 是 `[(label, action), ...]`。
    """
    out = []
    for label, action in candidates:
        cash = evaluate(env, snap, action, act_them, rollout_act, player, days)
        out.append((label, cash))
        if on_result is not None:
            on_result(label, cash)
    return out


def search_action(env, snap, candidates, act_them, rollout_act, player=0,
                  days=DAYS, base_label=None):
    """搜出最好的候選，回傳 `(label, action, cash, 全部結果)`。

    🩸 `candidates` **必須含基準線**（gen0 自己的動作）。含了的話搜出來的結果
    在數學上不可能比不搜差 —— 那是這支唯一的正確性保證。`base_label` 有給的話
    會斷言它在裡面。
    """
    labels = [c[0] for c in candidates]
    if base_label is not None and base_label not in labels:
        raise ValueError(
            f"候選裡沒有基準線 {base_label!r} —— 搜出來的結果就沒有下界保證了")

    scored = evaluate_all(env, snap, candidates, act_them, rollout_act,
                          player, days)
    best_i = max(range(len(scored)), key=lambda i: scored[i][1])
    label, cash = scored[best_i]
    return label, candidates[best_i][1], cash, scored
