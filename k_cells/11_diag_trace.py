# === CELL 11: 現金接近 0 的時候，找出斷在哪一環 ===
# 用途：cell 10 跑完是 DONE 但現金只有個位數 —— agent 有跑、沒拋錯，
# 所以問題不是載入，是**送出的動作被引擎靜默忽略**（引擎對不合法的動作
# 不拋錯，直接 return）。
#
# 這格不改 agent、不設 log level、不碰引擎私有函式。它做兩件事：
#   1. 從 `env.steps[t][0].observation` 讀引擎自己記的狀態（每天一列）
#   2. 從 `env.steps[t][0].action` 讀 agent 那回合送出什麼
# 兩邊對照就知道是哪一類動作沒生效：
#   送出 PLANT 但已種格數永遠 0     -> PLANT 被忽略（座標/參數格式對不上）
#   有種但倉庫永遠空                 -> 收成沒進倉
#   倉庫有貨、送出 SELL 但現金不動   -> SELL 被忽略
#   送出 HIRE 但人手不增加           -> HIRE 被忽略（價格或參數對不上）
#
# 另外會 dump 一格 tile 的原始 dict 和 obs 的 key 列表 —— 引擎換版本後
# 欄位改名的話，這裡一眼就看得到。
import json
import os
from collections import Counter

MAIN = "/kaggle/working/main.py"
SEED = 41001

if not os.path.isfile(MAIN):
    print(f"{MAIN} 不存在 —— 先跑 cell 7。跳過。")
else:
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _saved = os.dup(1)
    os.dup2(_devnull, 1)                  # open_spiel 匯入時會噴幾百行遊戲清單
    try:
        from kaggle_environments import make
    finally:
        os.dup2(_saved, 1)
        os.close(_saved)
        os.close(_devnull)

    def g(obj, key, default=None):
        """steps 裡的元素有時是 dict、有時是 Struct，兩種都要能讀。"""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    env = make("kaggriculture", configuration={"seed": SEED}, debug=True)
    env.run([MAIN, "starter"])

    me = env.state[0]
    print(f"seed {SEED}   現金 {me.reward:,.0f}   {me.status}   {len(env.steps)} steps")

    # --- 1. 引擎給的 schema（版本不同時最先看這裡）-----------------------------
    obs0 = g(env.steps[0][0], "observation")
    print("\n--- obs 的 top-level key ---")
    print(sorted(obs0.keys()))

    player = obs0["player"]
    farm0 = obs0["farms"][player]
    print("--- farm 的 key ---")
    print(sorted(farm0.keys()))
    print("--- private 的 key ---")
    print(sorted(obs0["private"].keys()))

    # day 0 開局時每一格都還是字串（"LOCKED" / 空地），要往後找才有 dict。
    sample = None
    for step in env.steps:
        o = g(step[0], "observation")
        if not o or "farms" not in o:
            continue
        for row in o["farms"][o["player"]]["tiles"]:
            for tile in row:
                if isinstance(tile, dict):
                    sample = tile
                    break
            if sample:
                break
        if sample:
            break
    print("--- 一格 tile 的原始內容（第一個非字串的格子）---")
    print(json.dumps(sample, ensure_ascii=False, default=str))

    # --- 1b. 這一局的 configuration -----------------------------------------------
    # 引擎的收費看的是 configuration，不是模組常數：
    #   hire_mult = get(env.configuration, "farmHandCostMult", FARM_HAND_COST_MULT)
    # agent 也是從這裡讀價格（gen0.py 的 `hire_cost_for`），所以這一包要印出來。
    # `marketParams` 也在裡面，價格結構被覆寫的話同樣看得到。
    print("\n--- env.configuration ---")
    print(json.dumps(dict(env.configuration), indent=2, ensure_ascii=False,
                     default=str, sort_keys=True))

    # --- 1c. 這個引擎實際收多少錢 -------------------------------------------------
    # `_hire_cost(n)` 用的是**模組預設**倍率，configuration 的 `farmHandCostMult`
    # 可能不同 —— 兩個都印，對不上就代表 agent 不能用 `_hire_cost(n)` 當價格。
    # 私有 API 只在這格用，submission 不碰 —— 改名了這格會直接 ImportError。
    print("\n--- 引擎的價格常數 ---")
    try:
        from kaggle_environments.envs.kaggriculture.kaggriculture import (
            FARM_HAND_COST_MULT, _hire_cost,
        )
        cfg_mult = env.configuration.get("farmHandCostMult")
        print(f"FARM_HAND_COST_MULT（模組預設）= {FARM_HAND_COST_MULT}   "
              f"farmHandCostMult（configuration）= {cfg_mult}")
        if cfg_mult is not None and cfg_mult != FARM_HAND_COST_MULT:
            print("  !! 兩者不同 —— 實際收費以 configuration 為準")
        print("_hire_cost(0..11) =", [_hire_cost(i) for i in range(12)])
        print("  雇滿 10 人一天 =", sum(_hire_cost(i) for i in range(10)))
    except Exception as exc:                     # noqa: BLE001
        print(f"  讀不到：{type(exc).__name__}: {exc}")

    # --- 1d. 整包規則常數 ----------------------------------------------------------
    # 同 seed、同 configuration、同動作，本機和 Kaggle 的結果還是不同 ——
    # 所以引擎本身有差。一次把所有規則常數印出來跟本機 diff，不要一項一項猜。
    # 已知相同的：MARKET_PARAMS 的 base（9 項）、CROPS 的 seed（5 項）。
    # 還沒比過的：T / I0 / glut 曲線、產量參數、動物、SHOPS、地價。
    print("\n--- 引擎規則常數（整包，拿去跟本機 diff）---")
    for name in ("MARKET_I0", "MARKET_PARAMS", "CROPS", "ANIMALS",
                 "SHOPS", "PRODUCTS", "LAND_PRICES", "LAND_ORDER"):
        try:
            mod = __import__(
                "kaggle_environments.envs.kaggriculture.kaggriculture",
                fromlist=[name])
            print(f"{name} = "
                  + json.dumps(getattr(mod, name), ensure_ascii=False,
                               default=str, sort_keys=True))
        except Exception as exc:                 # noqa: BLE001
            print(f"{name}: 讀不到 —— {type(exc).__name__}: {exc}")

    try:
        import importlib.metadata as _md
        print("kaggle-environments ==", _md.version("kaggle-environments"))
    except Exception as exc:                     # noqa: BLE001
        print("版本讀不到：", exc)

    # --- 2. 逐日狀態 + 當日送出的動作 ------------------------------------------
    def classify(tiles):
        """回傳 (可用格, 已種, 有動物, 蓋好但空著, 雜草)。

        官方 tutorial 列的 tile 型態：`None`（空地）、`"LOCKED"`（沒買的象限）、
        或 dict —— kind 是 PLANT / WEED / COOP / PASTURE。
        WEED 是「連續兩天沒澆水」變成的，要 DIG 過才能再用，所以要單獨數：
        它佔著 open 的名額但完全不產出，混在一起看不出田荒了多少。
        """
        openn = planted = animal = estruct = weed = 0
        for row in tiles:
            for tile in row:
                if tile == "LOCKED":
                    continue
                openn += 1
                if not isinstance(tile, dict):
                    continue
                if "animal" in tile:
                    animal += 1
                elif tile.get("kind") == "WEED":
                    weed += 1
                elif tile.get("kind") in ("COOP", "PASTURE"):
                    estruct += 1
                elif tile.get("crop"):
                    planted += 1
        return openn, planted, animal, estruct, weed

    def ops_of(action):
        """把一回合的 action 拆成 op 名稱的計數。形狀不合就記成 ?。"""
        unit_ops, market_ops = Counter(), Counter()
        if not isinstance(action, dict):
            return unit_ops, market_ops
        units = [action.get("farmer")] + list(action.get("hands") or [])
        for a in units:
            if isinstance(a, (list, tuple)) and a:
                unit_ops[str(a[0])] += 1
            elif a is not None:
                unit_ops["?"] += 1
        for o in (action.get("market") or []):
            if isinstance(o, (list, tuple)) and o:
                market_ops[str(o[0])] += 1
            else:
                market_ops["?"] += 1
        return unit_ops, market_ops

    daily = {}                      # day -> 最後一筆狀態
    day_ops = {}                    # day -> (unit Counter, market Counter)
    total_unit, total_market = Counter(), Counter()

    for step in env.steps:
        obs = g(step[0], "observation")
        if not obs or "day" not in obs:
            continue
        day = obs["day"]
        farm = obs["farms"][obs["player"]]
        priv = obs["private"]
        openn, planted, animal, estruct, weed = classify(farm["tiles"])
        daily[day] = {
            "cash": farm["money"],
            "crew": len(farm["hands"]),
            "quad": len(farm["unlocked_quadrants"]),
            "open": openn,
            "plant": planted,
            "weed": weed,
            "animal": animal,
            "estruct": estruct,
            "seeds": sum(priv["seeds"].values()),
            "shed": sum(priv["shed"].values()),
            "carry": sum(sum(inv.values()) for inv in priv["inventories"]),
        }
        u, m = ops_of(g(step[0], "action"))
        du, dm = day_ops.setdefault(day, (Counter(), Counter()))
        du.update(u)
        dm.update(m)
        total_unit.update(u)
        total_market.update(m)

    # 每天的現金變化除以人手數 —— 沒有任何採購落地的日子（seeds/shed/carry
    # 都是 0）就是純工資，這個比值直接給出「一個 hand 一天實收多少」。
    print("\n--- 逐日狀態（引擎記的）---")
    print("day   cash    dcash crew quad open plant weed animal estr "
          "seeds shed carry  $/hand")
    prev = None
    for day in sorted(daily):
        d = daily[day]
        dcash = "" if prev is None else f"{d['cash'] - prev:+,.0f}"
        per = ""
        if (prev is not None and d["crew"] > 0 and d["cash"] < prev
                and d["seeds"] == 0 and d["shed"] == 0 and d["carry"] == 0):
            per = f"{(prev - d['cash']) / d['crew']:.1f}"
        print(f"{day:3d} {d['cash']:8,.0f} {dcash:>8s} {d['crew']:4d} {d['quad']:4d} "
              f"{d['open']:4d} {d['plant']:5d} {d['weed']:4d} {d['animal']:6d} "
              f"{d['estruct']:4d} {d['seeds']:5d} {d['shed']:4d} {d['carry']:5d} "
              f"{per:>7s}")
        prev = d["cash"]

    print("\n--- agent 送出的動作總計 ---")
    print("unit  :", dict(total_unit.most_common()))
    print("market:", dict(total_market.most_common()))

    print("\n--- 前 3 回合的原始 action ---")
    for step in env.steps[:3]:
        print(" ", json.dumps(g(step[0], "action"), ensure_ascii=False, default=str)[:400])

    # --- 3. 自動判斷斷在哪 ------------------------------------------------------
    days = sorted(daily)
    peak = {k: max(daily[d][k] for d in days) for k in
            ("plant", "weed", "animal", "shed", "carry", "crew", "quad")}
    cash0, cashN = daily[days[0]]["cash"], daily[days[-1]]["cash"]

    cash_peak = max(daily[d]["cash"] for d in days)

    # A. 有沒有哪一類動作被引擎靜默丟掉（送出了，但狀態沒有對應變化）。
    #    現金類的要看**整局最高點**，不能看期末 —— 賣得掉但錢又花光的話，
    #    期末一樣很低，用期末比會誤判成「SELL 沒生效」。
    ignored = []
    if total_unit.get("PLANT", 0) > 0 and peak["plant"] == 0:
        ignored.append(f"送出 {total_unit['PLANT']} 次 PLANT，但沒有任何一格變成已種")
    if peak["plant"] > 0 and peak["carry"] == 0 and peak["shed"] == 0:
        ignored.append("有種下去但倉庫和身上永遠是空的 -> 收成那一環沒生效")
    if (peak["shed"] > 0 or peak["carry"] > 0) and total_market.get("SELL", 0) > 0 \
            and cash_peak <= cash0:
        ignored.append(f"倉庫有貨、送出 {total_market['SELL']} 筆 SELL，"
                       f"現金整局沒有任何一天高過期初")
    if total_market.get("HIRE", 0) > 0 and peak["crew"] <= daily[days[0]]["crew"]:
        ignored.append(f"送出 {total_market['HIRE']} 次 HIRE，人手數沒增加")
    if total_market.get("BUY_LAND", 0) > 0 and peak["quad"] <= daily[days[0]]["quad"]:
        ignored.append(f"送出 {total_market['BUY_LAND']} 次 BUY_LAND，象限沒增加")
    if "?" in total_unit or "?" in total_market:
        ignored.append("有動作的形狀不是 [op, ...] -> action 格式跟這個引擎版本對不上")

    # B. 動作都生效，但產能起不來。把「閒置」和「缺種子」分開講。
    idle = total_unit.get("PASS", 0)
    n_unit_ops = sum(total_unit.values())
    dry_days = sum(1 for d in days if daily[d]["seeds"] == 0)
    broke_days = sum(1 for d in days if daily[d]["cash"] < 100)

    print("\n--- 判斷 ---")
    print(f"現金 期初 {cash0:,.0f}  最高 {cash_peak:,.0f}  期末 {cashN:,.0f}")
    print(f"峰值 {peak}")
    if n_unit_ops:
        print(f"unit 閒置 {idle}/{n_unit_ops} = {idle / n_unit_ops:.1%}")
    print(f"種子存量為 0 的天數 {dry_days}/{len(days)}   "
          f"現金 < 100 的天數 {broke_days}/{len(days)}")

    if ignored:
        print("\n動作被引擎丟掉：")
        for v in ignored:
            print(f"!! {v}")
    else:
        print("\n沒有任何一類動作被丟掉 —— 動作格式跟這個引擎相容。")

    if dry_days > len(days) * 0.5:
        print(f"!! 一半以上的日子沒有種子 -> PLANT 排不出任務。"
              f"BUY_SEED 只送出 {total_market.get('BUY_SEED', 0)} 次，"
              f"而 BUY_SEED 的數量卡在 spendable = money - floor（gen0.py:1210）")
    if peak["weed"] > peak["plant"]:
        print(f"!! 雜草峰值 {peak['weed']} 高於已種峰值 {peak['plant']} -> "
              f"種下去照顧不過來（連兩天沒澆水就變雜草）。DIG 只送出 "
              f"{total_unit.get('DIG', 0)} 次，那些格子等於報廢")
    if broke_days > len(days) * 0.5 and total_market.get("HIRE", 0) > len(days) * 5:
        print(f"!! 一半以上的日子現金 < 100，同時送出 {total_market['HIRE']} 次 HIRE。"
              f"HIRE 是唯一不檢查 spendable 的採購（gen0.py:1255-1257）—— "
              f"收入一進來就被轉成工資，種子永遠排在後面")
