"""讓 `_PRI`（任務優先序）變成可搜尋的參數，**不動 `agents/gen0.py`**。

    "entry": "tools.gen0_pri:act"

## 為什麼要有這支

`agents/gen0.py:306-316` 的 `_PRI` 是模組層級常數，`PLANT` 是全表最低的 9。
2026-08-28 實測（`tools/tasks_probe.py`，3 局 2,157 個決策回合）：

    每回合想做 33.46 件任務，只指派得到 7.91 件
    PLANT 想種 5.39 格、只種到 0.75 格 —— 73.8% 的回合有 PLANT 被擠掉

結果是作物格停在 33 格（對手 recursion 是 63），期末現金差 38,085。
`_PRI` 不在 CMA 的 34 個維度裡，所以那一輪從頭到尾碰不到這件事。

`agents/` 是 B 的（`docs/CLAUDE.md:102`），所以先在這裡用 wrapper 試。
**確認有效再去問 B 把它折進 `gen0.py`。**

## 怎麼做的

`act()` 前把 `agents.gen0._PRI` 換成參數指定的那張表，回來再換回去。

🩸 **這是行程層級的全域狀態。** 同一個行程同時跑兩個不同優先序的 agent 會
互相污染。`env.run` 是循序的（一次只有一個 agent 在算），`eval/runner.py`
的平行度在 process 層，所以現況安全 —— 但**不要**把它搬進 thread pool。

參數用 `pri_<動作名>` 命名，沒給的沿用 `gen0._PRI` 的原值：

    {"pri_PLANT": 2, "pri_WATER": 3}
"""
from __future__ import annotations

import agents.gen0 as _G

#: 參數名字首。`pri_PLANT` -> `_PRI["PLANT"]`
PREFIX = "pri_"


def priority_overrides(params):
    """從 params 抽出 `pri_*`，回傳 {動作名: 優先序}。只含真的有給的。"""
    out = {}
    for key, value in params.items():
        if not key.startswith(PREFIX):
            continue
        name = key[len(PREFIX):]
        if name in _G._PRI:
            out[name] = int(round(float(value)))
    return out


def act(obs, config, params=None):
    """`agents.gen0.act` 加上兩個開關：

    - `pri_<動作名>`：覆寫 `_PRI` 的那一項優先序
    - `global_plant_assignment: True`：讓 PLANT 那一層也走全域最短配對

    兩個都沒給就原樣轉呼叫，零開銷。
    """
    p = params or {}
    over = priority_overrides(p)
    global_plant = bool(p.get("global_plant_assignment", False))
    if not over and not global_plant and not p.get("dense_active_tiles"):
        return _G.act(obs, config, params)

    dense = bool(p.get("dense_active_tiles", False))
    saved_pri = dict(_G._PRI) if over else None
    saved_assign = _G._assign if global_plant else None
    saved_active = _G.active_crop_tiles if dense else None
    if over:
        _G._PRI.update(over)
    if global_plant:
        _G._assign = _assign_with_global_plant()
    if dense:
        _G.active_crop_tiles = _dense_active_crop_tiles
    try:
        return _G.act(obs, config, params)
    finally:
        if saved_pri is not None:
            _G._PRI.clear()
            _G._PRI.update(saved_pri)
        if saved_assign is not None:
            _G._assign = saved_assign
        if saved_active is not None:
            _G.active_crop_tiles = saved_active


# --------------------------------------------------------------------------
# 讓 PLANT 也走全域最短配對
# --------------------------------------------------------------------------
#
# `agents/gen0.py:1531-1535` 的全域最短配對有一個排除條件：
#
#     and all(task[1] != "PLANT" for task in tier)
#
# 理由寫在它上面的註解 —— PLANT 有「同一作物的請求數不能超過種子數」這個
# **跨任務**限制（引擎是原子驗證：超過的話該作物所有 PLANT 全變 PASS，
# `gen0.py:1576-1579`），逐筆計數才擋得住，所以含 PLANT 的那一層整層退回
# 「近的先搶 unit」的貪婪法。
#
# 但 PLANT 自己就是完整一層（`_PRI["PLANT"] = 9`，全表最低且獨佔一級），
# 所以**唯一沒享受到路徑最佳化的，正好是被擠掉最多的那個任務**。
# 2026-08-28 實測（`tools/tasks_probe.py`，3 局 2,157 個決策回合）：
#
#     走路佔 unit-回合 57.2%（對手 ReCurSiON 43.4%）
#     每回合想做 33.46 件任務，只指派得到 7.91 件
#     PLANT 想種 5.39 格、只種到 0.75 格 —— 73.8% 的回合有 PLANT 被擠掉
#
# 這裡改成**先裁再配**：先把 PLANT 裁到種子數以內（原子驗證就過得了），
# 再讓整層走全域最短配對。
#
# 🩸 不改 `agents/gen0.py`（B 的檔案，`docs/CLAUDE.md:102`）。原始碼用
# `inspect.getsource` 取出來做字串替換再 exec —— B 改了那個函式的話這裡會
# **在 import 時就 assert 失敗**，不會安靜地跑舊行為。

import inspect as _inspect

from agents.gen0 import crop_for as _crop_for

#: 要替換掉的原始片段。對不上就代表 `agents/gen0.py` 動過了。
_EXCLUDE_SRC = """        if (
            params.get("optimal_assignment", False)
            and tier
            and all(task[1] != "PLANT" for task in tier)
            and free
        ):
            units = sorted(free)
            real_tasks = list(tier)"""

_INCLUDE_SRC = """        if (
            params.get("optimal_assignment", False)
            and tier
            and free
        ):
            units = sorted(free)
            real_tasks = _limit_plant_by_seeds(
                tier, basket, seeds, free, assignment_cost)"""


def _limit_plant_by_seeds(tier, basket, seeds, free, cost_fn):
    """全域配對之前先把 PLANT 裁到種子數以內。

    引擎的原子驗證是「同一回合某作物的 PLANT 請求數超過種子數，該作物的
    **所有** PLANT 全部變 PASS」。原版靠逐筆計數擋，代價是整層不能走全域配對。
    這裡先裁掉多的，留下離最近閒置 unit 最近的那幾格，再交給配對演算法。

    非 PLANT 的任務原封不動。`free` 空的時候呼叫端不會走到這裡。
    """
    plants, others = [], []
    for task in tier:
        (plants if task[1] == "PLANT" else others).append(task)
    if not plants:
        return list(tier)
    plants.sort(key=lambda t: (
        min((cost_fn(u, t) for u in free), default=0), t[3], t[2]))
    kept, used = [], {}
    for task in plants:
        crop = _crop_for(task[2], task[3], basket)
        if used.get(crop, 0) >= seeds.get(crop, 0):
            continue
        used[crop] = used.get(crop, 0) + 1
        kept.append(task)
    return others + kept


def _build_assign_with_global_plant():
    src = _inspect.getsource(_G._assign)
    if _EXCLUDE_SRC not in src:
        raise RuntimeError(
            "agents/gen0.py 的 _assign 改過了 —— tools/gen0_pri.py 的字串替換"
            "對不上，請重新對照 gen0.py:1531 附近再更新 _EXCLUDE_SRC。")
    ns = dict(_G.__dict__)
    ns["_limit_plant_by_seeds"] = _limit_plant_by_seeds
    exec(compile(src.replace(_EXCLUDE_SRC, _INCLUDE_SRC),   # noqa: S102
                 "<gen0_pri._assign>", "exec"), ns)
    return ns["_assign"]


#: 只在真的被用到時才建，import 這個模組本身不該有副作用。
_ASSIGN_GLOBAL_PLANT = None


def _assign_with_global_plant():
    global _ASSIGN_GLOBAL_PLANT
    if _ASSIGN_GLOBAL_PLANT is None:
        _ASSIGN_GLOBAL_PLANT = _build_assign_with_global_plant()
    return _ASSIGN_GLOBAL_PLANT


# --------------------------------------------------------------------------
# 密集配置：先填滿最近的象限，而不是跨象限 round-robin
# --------------------------------------------------------------------------
#
# `agents/gen0.py:384-392` 的 `active_crop_tiles` 是跨象限輪流取格：
#
#     # round-robin，而不是先塞滿 NW 再輪到 NE/SW/SE。
#
# 註解說理由是「避免新增象限永遠被 y/x 掃描順序餓死」—— 那是為了**公平**，
# 不是為了效率。實測的代價（2026-08-28）：
#
#     33 格 active 攤在 3 個象限、75 格的範圍上，密度 44%
#     unit 走過的格子有 39% 是空的
#     走路吃掉 57.2% 的 unit-回合（對手 ReCurSiON 43.4%）
#     每回合 33.46 件任務只做得到 7.91 件
#     PLANT（優先序最低）被擠掉 73.8%
#
# 而且這是自我維持的：稀疏 -> 走路多 -> 做得少 -> PLANT 被擠 -> 更稀疏。
# 所以「單純加產能」的改動全部讓現金變差（§10 那張表 10 組全負）。
#
# 這裡改成**先填滿離 shed 最近的象限**：同樣的格數集中在小範圍內，
# 總量不變、密度提高。
#
# 🩸 這是**假設**，不是已知的改善。round-robin 的公平性論點可能有它的道理
#    （例如新象限的地會一直長雜草）。要看實測。

def _dense_active_crop_tiles(tiles, board, struct_order, unlocked_quadrants,
                             max_tiles=None):
    """`active_crop_tiles` 的密集版：先填滿最近的象限再開下一個。

    象限內的排序沿用原版（離 shed 內角最近優先，同距離固定 y/x），
    只把「跨象限輪流」換成「一個象限填滿再換」。象限之間的順序也按內角
    離 shed 的距離排，所以先填的一定是最近的那個。
    """
    struct_set = set(struct_order)
    half = board // 2
    groups = {q: [] for q in unlocked_quadrants}
    for y in range(board):
        for x in range(board):
            if tiles[y][x] == "LOCKED" or (x, y) in struct_set:
                continue
            q = _G.quadrant_for(x, y, board)
            groups.setdefault(q, []).append((x, y))

    def inner(q):
        return (half - 1 if q.endswith("W") else half,
                half - 1 if q.startswith("N") else half)

    for q, coords in groups.items():
        ix, iy = inner(q)
        coords.sort(key=lambda p: (abs(p[0] - ix) + abs(p[1] - iy), p[1], p[0]))

    total = sum(len(c) for c in groups.values())
    limit = total if max_tiles is None else max(0, min(total, int(max_tiles)))

    # 象限順序：先解鎖的優先（`unlocked_quadrants` 本身就是解鎖順序），
    # 這樣新解鎖的象限不會把已經在耕作的那一塊擠掉。
    selected = set()
    for q in unlocked_quadrants:
        for coord in groups.get(q, []):
            if len(selected) >= limit:
                return selected
            selected.add(coord)
    return selected
