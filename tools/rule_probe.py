"""植物鏈的 game rule：用 scripted agent 一條一條問引擎，不從原始碼推。

    python -m tools.rule_probe

## 為什麼不直接讀原始碼就好

2026-09-18 早上的教訓（journal §128.1.1）：我從引擎原始碼推出「這回合 BUY_SEED
買的種子這回合種不了」，跑 `tools/seed_timing.py` 才發現還有第二件事 —— 引擎的
PLANT 原子驗證會把該作物這回合的 **全部** PLANT 換成 PASS，不是只擋多出來的。
推論會漏東西。所以每一條要寫進 `legal_unit_mask` 的規則，都先在這裡問過引擎。

## 作法

scripted agent + 手動 `env.step`，並且包住引擎的 `_apply_unit_action` ——
在**單一次呼叫**的前後各抓一份狀態，所以「這個 unit 的這個動作引擎到底有沒有
執行」是確定的，不用從回合前後的盤面猜（那正是舊版 `op_waste` 錯的地方，§130.1）。

🩸 FERTILIZE 要特判：引擎行 478 的 `_inv_take` 在行 481 的 `max(...)` 之前，
同一天重複施肥「肥料扣掉了但沒有延長」—— inventory 有變，用「有沒有任何改變」
會誤判成生效。這個 op 只認 `fertilized_until_day` 有沒有變大。

⚠️ 玩家 1 在所有情境裡都只送 PASS，所以這支工具用「op 不是 PASS」來認我方的
呼叫，沒有像 `tools/op_waste.py` 那樣去包 `interpreter` 分辨玩家。加情境時如果
讓玩家 1 動起來，這個假設就不成立了。

## 初始狀態（引擎行 140-176，`tools/seed_timing.py` 驗過）

    private["seeds"] 全部 0        farmer 生在 (4,4)
    NW 象限已解鎖、其餘 LOCKED      (4,4) 是空的已解鎖 tile
    startingMoney 3000             turnsPerDay 24

`_shed_access_tiles` 是 (4,4) (5,4) (4,5) (5,5)，其中只有 (4,4) 屬於 NW，
其餘三格一開始是 LOCKED —— 但 hands 站得上去，而且 shed 操作在 LOCKED guard
之前處理（`engine-notes.md` §2）。
"""
from __future__ import annotations
import copy
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
TPD = 24        # turnsPerDay


# --------------------------------------------------------------------------
# 引擎掛鉤
# --------------------------------------------------------------------------
class Probe:
    """包住 `_apply_unit_action`，記下每一次呼叫引擎有沒有真的執行。"""

    def __init__(self):
        from kaggle_environments.envs.kaggriculture import kaggriculture as K
        self.K = K
        self.orig = K._apply_unit_action
        self.step = 0
        self.log = []          # (step, day, idx, op, arg, pos, changed, note)

    def __enter__(self):
        self.K._apply_unit_action = self._wrapper
        return self

    def __exit__(self, *exc):
        self.K._apply_unit_action = self.orig
        return False

    def _wrapper(self, farm, private, idx, action, board_size, day,
                 turns_per_day, shed_capacity=100):
        op = action[0] if isinstance(action, list) and action else None
        if op is None or op == "PASS":
            return self.orig(farm, private, idx, action, board_size, day,
                             turns_per_day, shed_capacity)

        arg = action[1] if len(action) > 1 else None
        pos = (tuple(farm["farmer"]) if idx == 0
               else tuple(farm["hands"][idx - 1]))
        x, y = pos
        invs = private["inventories"]
        b_tile = copy.deepcopy(farm["tiles"][y][x])
        b_inv = dict(invs[idx]) if idx < len(invs) else {}
        b_shed = dict(private["shed"])
        b_seeds = dict(private["seeds"])

        self.orig(farm, private, idx, action, board_size, day,
                  turns_per_day, shed_capacity)

        a_tile = farm["tiles"][y][x]
        a_inv = dict(invs[idx]) if idx < len(invs) else {}
        note = ""
        if op in ("NORTH", "SOUTH", "EAST", "WEST"):
            # 🩸 移動改的是 unit 的位置，不是它腳下那個 tile —— 用「tile 有沒有變」
            # 判會一律得到「否」。移動只是情境的前置，另外判。
            a_pos = (tuple(farm["farmer"]) if idx == 0
                     else tuple(farm["hands"][idx - 1]))
            self.log.append((self.step, day, idx, op, arg, pos,
                             a_pos != pos, f"{pos} -> {a_pos}"))
            return
        if op == "FERTILIZE":
            bd = b_tile if isinstance(b_tile, dict) else {}
            ad = a_tile if isinstance(a_tile, dict) else {}
            b_until = bd.get("fertilized_until_day", -1)
            a_until = ad.get("fertilized_until_day", -1)
            changed = a_until > b_until
            burned = b_inv.get("FERTILIZER", 0) - a_inv.get("FERTILIZER", 0)
            note = f"until {b_until}->{a_until}  肥料扣掉 {burned}"
        else:
            changed = (a_tile != b_tile or a_inv != b_inv
                       or dict(private["shed"]) != b_shed
                       or dict(private["seeds"]) != b_seeds)
            note = _tile_note(op, b_tile, a_tile)

        self.log.append((self.step, day, idx, op, arg, pos, changed, note))


def _tile_note(op, before, after):
    """印出那個 op 關心的欄位前後值。"""
    bd = before if isinstance(before, dict) else {}
    ad = after if isinstance(after, dict) else {}
    if op == "WATER":
        return (f"watered_today {bd.get('watered_today')}->{ad.get('watered_today')}"
                f"  yield {bd.get('yield_units')}->{ad.get('yield_units')}")
    if op == "HARVEST":
        return (f"yield {bd.get('yield_units')}->{ad.get('yield_units')}"
                f"  tile {_kind(before)}->{_kind(after)}")
    if op == "PLANT":
        return f"tile {_kind(before)}->{_kind(after)}"
    if op == "DIG":
        return f"tile {_kind(before)}->{_kind(after)}"
    if op in ("BUILD_COOP", "BUILD_PASTURE", "PLACE"):
        return f"tile {_kind(before)}->{_kind(after)}"
    return f"tile {_kind(before)}->{_kind(after)}"


def _kind(tile):
    if tile is None:
        return "空地"
    if tile == "LOCKED":
        return "LOCKED"
    if isinstance(tile, dict):
        k = tile.get("kind")
        if tile.get("animal"):
            return f"{k}+{tile['animal']}"
        if k == "PLANT":
            return f"{tile.get('crop')}(種於 day {tile.get('planted_day')})"
        return str(k)
    return str(tile)


# --------------------------------------------------------------------------
# 跑一個情境
# --------------------------------------------------------------------------
def run(name, script, n_steps, default=None, seed=900_000):
    """`script` 是 {step -> 玩家 0 的 action}，其餘回合用 `default`。

    `default` 預設是 farmer WATER —— 多日情境要每天澆水，作物才不會因為
    `consecutive_unwatered >= 2` 變成雜草（引擎 783-784）。已經澆過的回合
    再澆是靜默 no-op，不影響測試。
    """
    from eval.runner import _quiet_make
    if default is None:
        default = {"farmer": ["WATER"], "hands": [], "market": []}

    env = _quiet_make()("kaggriculture", configuration={"seed": seed}, debug=True)
    env.reset(2)
    with Probe() as p:
        for t in range(n_steps):
            p.step = t
            env.step([script.get(t, default), dict(PASS)])
    return name, p.log, env


def report(name, log, watch, expect=None):
    """只印 `watch` 裡的那些 step。`expect` 是 {(step, idx): 期望生效}。"""
    print(f"\n{name}")
    print(f"  {'step':>5}{'day':>5}{'unit':>6}  {'動作':<22}"
          f"{'引擎執行':>9}  說明")
    ok = True
    for step, day, idx, op, arg, pos, changed, note in log:
        if step not in watch:
            continue
        who = "farmer" if idx == 0 else f"hand{idx - 1}"
        act = op + (f" {arg}" if arg else "")
        flag = "是" if changed else "否"
        mark = ""
        if expect is not None and (step, idx) in expect:
            want = expect[(step, idx)]
            if want != changed:
                mark = "   <- 跟預期不符"
                ok = False
        print(f"  {step:>5}{day:>5}{who:>6}  {act:<22}{flag:>9}  {note}{mark}")
    return ok


# --------------------------------------------------------------------------
# 情境
# --------------------------------------------------------------------------
def s_water():
    """WATER 是每回合還是每日？"""
    # step0 買種子雇人 -> step1 種下、hand 走過來 -> step2 兩個都澆（同回合）
    # -> step3 farmer 再澆（同天不同回合）-> day1 的 step25 再澆（隔天）
    script = {
        0: {"farmer": ["PASS"], "hands": [],
            "market": [["HIRE"], ["BUY_SEED", "WHEAT", 1]]},
        1: {"farmer": ["PLANT", "WHEAT"], "hands": [["WEST"]], "market": []},
        2: {"farmer": ["WATER"], "hands": [["WATER"]], "market": []},
        3: {"farmer": ["WATER"], "hands": [["PASS"]], "market": []},
        25: {"farmer": ["WATER"], "hands": [], "market": []},
    }
    name, log, _ = run("S1  WATER：同回合 / 同天不同回合 / 隔天", script, 27,
                       default=dict(PASS))
    return report(name, log, {1, 2, 3, 25}, expect={
        (1, 1): True,     # hand 從 (5,4) 走到 (4,4)
        (1, 0): True,     # PLANT 成功
        (2, 0): True,     # 第一次澆
        (2, 1): False,    # 同回合第二個 -> 預期 no-op
        (3, 0): False,    # 同天不同回合 -> 預期 no-op
        (25, 0): True,    # 隔天 -> 預期生效
    })


def s_fertilize():
    """FERTILIZE 是每日旗標還是 day+2 的三天窗口？第二次會不會照樣扣肥料？"""
    # farmer 每天早上被重設到 (4,4)、inventory 每晚自動存回 shed，
    # 所以每一天都要重新 PICKUP。(4,4) 本身就是 shed 旁。
    water = {"farmer": ["WATER"], "hands": [], "market": []}
    pick = {"farmer": ["PICKUP", "FERTILIZER", 1], "hands": [], "market": []}
    fert = {"farmer": ["FERTILIZE"], "hands": [], "market": []}
    script = {
        0: {"farmer": ["PASS"], "hands": [],
            "market": [["HIRE"], ["BUY_SEED", "WHEAT", 1],
                       ["BUY_PRODUCT", "FERTILIZER", 8]]},
        1: {"farmer": ["PICKUP", "FERTILIZER", 4],
            "hands": [["PICKUP", "FERTILIZER", 4]], "market": []},
        2: {"farmer": ["PLANT", "WHEAT"], "hands": [["WEST"]], "market": []},
        3: {"farmer": ["FERTILIZE"], "hands": [["FERTILIZE"]], "market": []},
        4: fert,          # 同天不同回合
        5: water,
        # day 1 / 2 / 3：重新拿肥料再施
        25: pick, 26: fert, 27: water,
        49: pick, 50: fert, 51: water,
        73: pick, 74: fert, 75: water,
    }
    name, log, _ = run("S2  FERTILIZE：同回合 / 同天不同回合 / 隔 1、2、3 天",
                       script, 77, default=water)
    return report(name, log, {3, 4, 26, 50, 74}, expect={
        (3, 0): True,     # 第一次
        (3, 1): False,    # 同回合第二個
        (4, 0): False,    # 同天不同回合
        (26, 0): True,    # 隔 1 天
        (50, 0): True,    # 隔 2 天
        (74, 0): True,    # 隔 3 天
    })


def s_harvest_wheat():
    """WHEAT 的 HARVEST 成熟度門檻在哪一天？（first_yield_day = 2）"""
    water = {"farmer": ["WATER"], "hands": [], "market": []}
    harv = {"farmer": ["HARVEST"], "hands": [], "market": []}
    script = {
        0: {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
        1: {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
        2: harv,          # age 0
        3: water,
        25: harv,         # age 1
        26: water,
        49: harv,         # age 2
    }
    name, log, _ = run("S3  HARVEST WHEAT：age 0 / 1 / 2（first_yield_day = 2）",
                       script, 51, default=water)
    return report(name, log, {1, 2, 25, 49}, expect={
        (1, 0): True,
        (2, 0): False,    # age 0 -> 預期引擎拒收
        (25, 0): False,   # age 1 -> 預期引擎拒收
        (49, 0): True,    # age 2 -> 預期收得到
    })


def s_harvest_melon():
    """MELON 的門檻（first_yield_day = 10，但加產窗口從 age 6 就開了）。"""
    water = {"farmer": ["WATER"], "hands": [], "market": []}
    harv = {"farmer": ["HARVEST"], "hands": [], "market": []}
    script = {
        0: {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 1]]},
        1: {"farmer": ["PLANT", "MELON"], "hands": [], "market": []},
        2: harv,                       # age 0
        6 * TPD + 2: harv,             # age 6：窗口內但未到 first_yield_day
        9 * TPD + 2: harv,             # age 9
        10 * TPD + 2: harv,            # age 10
    }
    name, log, _ = run("S4  HARVEST MELON：age 0 / 6 / 9 / 10"
                       "（first_yield_day = 10，加產窗口 age 6 就開）",
                       script, 10 * TPD + 4, default=water)
    return report(name, log, {1, 2, 6 * TPD + 2, 9 * TPD + 2, 10 * TPD + 2},
                  expect={
                      (1, 0): True,
                      (2, 0): False,
                      (6 * TPD + 2, 0): False,
                      (9 * TPD + 2, 0): False,
                      (10 * TPD + 2, 0): True,
                  })


def s_dig():
    """DIG 挖得動什麼？剛種下的作物 / 空建物 / 有動物的建物。"""
    script = {
        0: {"farmer": ["PASS"], "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1], ["BUY_ANIMAL", "GOOSE", 1]]},
        1: {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
        2: {"farmer": ["DIG"], "hands": [], "market": []},        # 剛種下的作物
        3: {"farmer": ["BUILD_COOP"], "hands": [], "market": []},
        4: {"farmer": ["DIG"], "hands": [], "market": []},        # 空建物
        5: {"farmer": ["BUILD_COOP"], "hands": [], "market": []},
        6: {"farmer": ["PICKUP", "GOOSE", 1], "hands": [], "market": []},
        7: {"farmer": ["PLACE", "GOOSE", 1], "hands": [], "market": []},
        8: {"farmer": ["DIG"], "hands": [], "market": []},        # 有動物的建物
    }
    name, log, _ = run("S5  DIG：剛種下的作物 / 空建物 / 有動物的建物",
                       script, 10, default=dict(PASS))
    return report(name, log, {1, 2, 3, 4, 5, 6, 7, 8}, expect={
        (2, 0): True,     # 剛種下就挖得掉 —— 引擎完全接受（是策略問題不是規則）
        (4, 0): True,     # 空建物挖得掉
        (8, 0): False,    # 有動物的建物挖不掉
    })


def s_dig_weed():
    """作物連續兩天沒澆 -> 變雜草；雜草挖得掉。"""
    script = {
        0: {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
        1: {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
        # 之後完全不澆水，預設就是 PASS
        2 * TPD + 1: {"farmer": ["DIG"], "hands": [], "market": []},
    }
    name, log, _ = run("S6  兩天沒澆水變雜草，雜草挖得掉",
                       script, 2 * TPD + 3, default=dict(PASS))
    return report(name, log, {1, 2 * TPD + 1}, expect={
        (1, 0): True,
        (2 * TPD + 1, 0): True,
    })


def s_plant():
    """PLANT：空地 / 已有作物。（種子那兩條在 tools/seed_timing.py 已經驗過）"""
    script = {
        0: {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 2]]},
        1: {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},   # 空地
        2: {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},   # 已有作物
    }
    name, log, _ = run("S7  PLANT：空地 / 已有作物", script, 4, default=dict(PASS))
    return report(name, log, {1, 2}, expect={
        (1, 0): True,
        (2, 0): False,
    })


def main():
    results = [
        s_water(),
        s_fertilize(),
        s_harvest_wheat(),
        s_harvest_melon(),
        s_dig(),
        s_dig_weed(),
        s_plant(),
    ]
    print("\n" + "=" * 72)
    bad = results.count(False)
    if bad:
        print(f"🩸 有 {bad} 個情境跟預期不符 —— 我對規則的理解是錯的，")
        print("   不要照現在的預期去改 legal_unit_mask，先把上面那幾列看懂。")
        return 1
    print("全部情境都跟預期相符。這張表就是 legal_unit_mask 的依據。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
