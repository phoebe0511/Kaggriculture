"""同一回合 `BUY_SEED` 買的種子，這一回合種得了嗎？直接問引擎。

    python -m tools.seed_timing

背景：`contracts.turn_guard_state`（contracts.py:1080）的 docstring 寫
「引擎**先**處理市場訂單」，所以它把這回合的 `BUY_SEED` 加進可用種子。但引擎
`interpreter` 讀起來是 unit 動作（行 935-938）在前、`_process_market`（行 941）
在後。兩者只有一個是對的，這支腳本直接量。

測試環境沒有混淆（引擎行 140-176）：
    private["seeds"] 全部 0        行 172
    farmer 生在 (4,4)              行 159-164（NW 的第一個 shed-access 格）
    NW 象限已解鎖、其餘 LOCKED     行 157-158
    startingMoney 3000             kaggriculture.json

四個情境：
    T1  同回合   step0: market BUY_SEED WHEAT 1 + farmer PLANT WHEAT
    T2  隔回合   step0: 只買；step1: 才種                      （控制組）
    T3  原子驗證 2 個 unit 站不同空地都 PLANT，只有 1 顆種子
                 引擎行 919-931 說「該作物全部變 PASS」-> 預期長出 0 株而不是 1 株
    T4  同上但買 2 顆                                          （控制組）
"""
from __future__ import annotations
import os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _obs(env):
    """player 0 的 observation。"""
    return env.state[0].observation


def _snapshot(env):
    o = _obs(env)
    farm = o["farms"][0]
    priv = o["private"]
    tiles = farm["tiles"]
    plants = {(x, y): t.get("crop")
              for y, row in enumerate(tiles) for x, t in enumerate(row)
              if isinstance(t, dict) and t.get("kind") == "PLANT"}
    return {
        "money": farm["money"],
        "seeds": {k: v for k, v in priv["seeds"].items() if v},
        "farmer": tuple(farm["farmer"]),
        "hands": [tuple(h) for h in farm["hands"]],
        "plants": plants,
    }


def _line(tag, s):
    seeds = str(s["seeds"] or {})
    print(f"    {tag:<26} seeds={seeds:<16} "
          f"farmer={s['farmer']} hands={s['hands']} "
          f"作物={s['plants'] or '無'}")


def run(name, script, n_steps):
    """`script` 是 {step -> player0 的 action}，其餘回合 PASS。"""
    from eval.runner import _quiet_make
    env = _quiet_make()("kaggriculture", configuration={"seed": 900_000}, debug=True)
    env.reset(2)
    print(f"\n{name}")
    _line("step 0 之前", _snapshot(env))
    for t in range(n_steps):
        act = script.get(t, PASS)
        env.step([act, dict(PASS)])
        _line(f"送出 {_fmt(act)}", _snapshot(env))
    return _snapshot(env)


def _fmt(act):
    parts = ["F:" + "/".join(map(str, act["farmer"]))]
    for i, h in enumerate(act.get("hands") or []):
        parts.append(f"H{i}:" + "/".join(map(str, h)))
    for m in act.get("market") or []:
        parts.append("M:" + "/".join(map(str, m)))
    return " ".join(parts)


def main():
    # ---- T1 同回合買 + 種 -------------------------------------------------
    t1 = run("T1  同回合 BUY_SEED + PLANT", {
        0: {"farmer": ["PLANT", "WHEAT"], "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]]},
    }, 2)

    # ---- T2 控制組：先買，下一回合種 --------------------------------------
    t2 = run("T2  控制組：step0 只買、step1 才種", {
        0: {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
        1: {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
    }, 3)

    # ---- T3 原子驗證：2 個 unit、1 顆種子 ---------------------------------
    # step0 HIRE + 買 1 顆；step1 兩個 unit 各自走到不同的 NW 空地；step2 都種。
    t3 = run("T3  原子驗證：2 個 unit 都 PLANT，只有 1 顆種子", {
        0: {"farmer": ["PASS"], "hands": [],
            "market": [["HIRE"], ["BUY_SEED", "WHEAT", 1]]},
        1: {"farmer": ["WEST"], "hands": [["WEST"]], "market": []},
        2: {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]], "market": []},
    }, 4)

    # ---- T4 控制組：同樣兩個 unit，買 2 顆 --------------------------------
    t4 = run("T4  控制組：同樣 2 個 unit，買 2 顆種子", {
        0: {"farmer": ["PASS"], "hands": [],
            "market": [["HIRE"], ["BUY_SEED", "WHEAT", 2]]},
        1: {"farmer": ["WEST"], "hands": [["WEST"]], "market": []},
        2: {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]], "market": []},
    }, 4)

    print("\n" + "=" * 72)
    print(f"T1 同回合買+種        長出 {len(t1['plants'])} 株   "
          f"（0 = 市場在 unit 動作之後）")
    print(f"T2 隔回合種（控制）   長出 {len(t2['plants'])} 株   （1 = 種子確實買到了）")
    print(f"T3 2 unit / 1 顆種子  長出 {len(t3['plants'])} 株   "
          f"（0 = 原子驗證把該作物全部擋掉；1 = 只擋多出來的）")
    print(f"T4 2 unit / 2 顆（控制）長出 {len(t4['plants'])} 株")
    return 0


if __name__ == "__main__":
    sys.exit(main())
