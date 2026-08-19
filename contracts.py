"""訓練端與比賽端共用的編碼介面。

`architecture.md` §4 定義的三件事：把 observation 轉成數字、標出哪些動作合法、
把網路輸出轉回引擎吃的動作格式。**訓練時和比賽時必須用完全一樣的轉法**，
所以這是唯一兩邊共用的檔案。

## schema 與 ENCODER_VERSION

`SPATIAL_CHANNELS` / `SCALAR_FIELDS` / `UNIT_FEATURES` / `UNIT_OPS` 這四份清單
就是 schema —— 「陣列的第 k 個位置代表什麼」的對照表。

訓練好的權重綁死在位置上。把 `scalar[0]=現金` 和 `scalar[1]=天數` 對調，
舊權重載進去**不會報錯**（形狀一樣），只是把天數當現金用，然後分數莫名其妙
掉下來。所以規則是：

    ✅ 只能往後加      ❌ 不准調換順序      ❌ 不准刪除

每次加 → `ENCODER_VERSION += 1`；存權重時把版本寫進檔案；載入時版本不符
直接拋錯，不要試圖相容。

（注意：repo 裡「凍結」這個詞另外用在 `config/opponents/ref-vN.json` 那些
量尺快照上，跟這裡的 schema 版本是兩回事。）

## 為什麼動作是「每個 unit 一個」而不是「每格一個」

實測 60 局 replay：**11.6% 的 unit-turn 是「兩個以上 unit 站同一格、但動作不同」**
（每格 unit 數分布 `{1: 226933, 2: 16709, 3: 1353, 4: 168, 5: 36, 6: 2}`）。

純 per-cell 的 `policy[n_ops, H, W]` 在那 11.6% 上只能給一個答案。
但同格的 unit 是分得出來的 —— **每個 unit 有自己的 inventory**
（`engine-notes.md` §7），而 `FEED` 是從 unit 自己的 inventory 扣 WHEAT。
手上有飼料的可以餵，空手的得先回倉庫拿。

所以 `encode_units()` 另外給每個 unit 一份特徵向量。網路的用法是：
CNN 整盤跑一次 → 在每個 unit 的格子取出該格的特徵 → 接上 unit 自己的特徵
→ 小 MLP 出 44 個分數。一回合 1 次 CNN + n 次便宜的 head。

## v1 不含市場訂單

`action["market"]` 這一路維持 `agents/gen0.py` 的規則式。理由與代價寫在
`docs/memory/journal/2026-08-19.md`：市場是「選一個大小不定的集合、每筆帶數量、
而且逐筆相依」，跟 unit 的「44 選 1」是完全不同的問題；混在一起做的話，
做不出來時分不清是哪一邊的問題。

**已知代價**：早上的逐件收入歸因顯示 WHEAT(−10,122) + FERTILIZER(−6,273)
= 總差距 $60,553 的 27% 屬於市場行為。沿用規則式市場的 v1，上限就是補掉
約 73%。抽訓練資料時市場標籤照樣存下來，要加第二個 head 不用重跑。

## 規則常數一律 import，不做鏡像

`CLAUDE.md` 硬規則。抄一份會有「抄錯」和「上游改版沒跟上」兩個風險，
兩者都不會報錯。而且本機引擎原始碼被人手改過（`LAND_ORDER` 曾被改成
`["NE"]`），所以連順序都要動態走。

⚠️ 但 **schema 的順序不能跟著引擎的 dict 順序跑** —— 引擎哪天調換
`CROPS` 的鍵順序，我們的 channel 就會無聲位移。所以下面的順序是寫死的，
另外用 `_assert_covers_engine()` 確認集合一致：引擎新增品項時**故意**炸掉，
逼人升 `ENCODER_VERSION`。
"""

from __future__ import annotations

import numpy as np

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    FARMER_MOVES,
    LAND_ORDER,
    MARKET_I0,
    MARKET_PARAMS,
    MAX_SHOP_INSTANCES,
    PRODUCTS,
    SHOPS,
)

#: v1 -> v2（2026-08-19）：`SCALAR_FIELDS` 補上 9 個全盤計數。
#: v1 訓練出來的網路驗證準確率 93.96%，實戰卻 0 勝 12 負 —— 它蓋到 24 個建物
#: 停不下來、16 隻買回來的動物只放進去 3 隻。原因是 v1 沒給「已經蓋了幾個」
#: 這種計數，而那正是那些罕見但決定勝負的動作的觸發條件。
ENCODER_VERSION = 2


# --------------------------------------------------------------------------
# 順序（寫死，只能往後加）
# --------------------------------------------------------------------------

#: 移動方向。順序跟 `FARMER_MOVES` 的鍵一致，但寫死以免引擎調換。
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")

#: 作物。順序即 channel 順序。
CROP_ORDER = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")

#: 動物。
ANIMAL_ORDER = ("GOOSE", "COW", "SHEEP")

#: 產品（可買賣、可放進 shed 的東西）。
PRODUCT_ORDER = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                 "EGG", "MILK", "WOOL", "FERTILIZER")

#: shop 種類。`unlocked_shops` 是抽取有放回的，所以要記各開幾間。
SHOP_ORDER = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
              "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")

#: 象限。順序跟 `LAND_ORDER` 無關 —— 那是購買順序，這是 channel 順序。
QUADRANT_ORDER = ("NW", "NE", "SW", "SE")


def _assert_covers_engine():
    """schema 的集合要跟引擎一致。不一致就是引擎改版了，必須升 ENCODER_VERSION。

    比對的是**集合**不是順序 —— 順序由我們寫死，引擎調換不影響我們；
    但引擎新增／移除品項就會讓 encoder 漏資訊或編出不存在的東西。
    """
    mismatches = []
    for name, ours, theirs in (
        ("MOVES", set(MOVES), set(FARMER_MOVES)),
        ("CROP_ORDER", set(CROP_ORDER), set(CROPS)),
        ("ANIMAL_ORDER", set(ANIMAL_ORDER), set(ANIMALS)),
        ("PRODUCT_ORDER", set(PRODUCT_ORDER), set(PRODUCTS)),
        ("SHOP_ORDER", set(SHOP_ORDER), set(SHOPS)),
    ):
        if ours != theirs:
            mismatches.append(
                f"{name}: 多了 {sorted(ours - theirs)}，少了 {sorted(theirs - ours)}")
    if mismatches:
        raise AssertionError(
            "contracts.py 的 schema 跟引擎對不上，必須升 ENCODER_VERSION：\n  "
            + "\n  ".join(mismatches))


_assert_covers_engine()


# --------------------------------------------------------------------------
# 動作空間
# --------------------------------------------------------------------------

#: 不帶參數的 unit op。順序寫死。
NO_ARG_OPS = (
    "NORTH", "SOUTH", "EAST", "WEST",
    "PASS",
    "WATER", "FERTILIZE", "HARVEST", "DIG",
    "BUILD_COOP", "BUILD_PASTURE",
    "DROP", "FEED", "CARE", "COLLECT_FERTILIZER",
)

#: `PICKUP` / `PLACE` 的對象：產品 + 動物。
CARRY_ITEMS = PRODUCT_ORDER + ANIMAL_ORDER

#: 完整動作詞彙。每個元素是 `(op, item_or_None)`。
#:
#: 44 種 = 15 個無參數 + 5 個 PLANT + 12 個 PICKUP + 12 個 PLACE。
#: 60 局 replay 只用到 32 種（例如沒有人 `PICKUP TOMATO`），但**完整文法都列進來** ——
#: 多 12 個輸出格子幾乎不花成本，而少列的話，換個老師就要升版重訓。
UNIT_OPS = (
    tuple((op, None) for op in NO_ARG_OPS)
    + tuple(("PLANT", crop) for crop in CROP_ORDER)
    + tuple(("PICKUP", item) for item in CARRY_ITEMS)
    + tuple(("PLACE", item) for item in CARRY_ITEMS)
)

N_UNIT_OPS = len(UNIT_OPS)

#: `(op, item) -> 索引`，抽訓練標籤時用。
UNIT_OP_INDEX = {op: i for i, op in enumerate(UNIT_OPS)}

#: `PICKUP` / `PLACE` 的數量。replay 實測值域是 1~12
#: （`PICKUP` 最常見 6，`PLACE` 最常見 1）。其他 op 用不到這個 head。
QTY_CHOICES = tuple(range(1, 13))
N_QTY = len(QTY_CHOICES)


# --------------------------------------------------------------------------
# 空間通道
# --------------------------------------------------------------------------

def _spatial_channel_names():
    names = [
        "unlocked",          # 這格已解鎖（不是 "LOCKED"）
        "empty",             # 已解鎖且空的
        "weed",
    ]
    names += [f"quadrant_{q}" for q in QUADRANT_ORDER]
    names += [
        "shed_adjacent",     # 站這裡能 DROP / PICKUP / PLACE
        "shed_distance",     # 曼哈頓距離 / board
    ]
    names += [f"crop_{c}" for c in CROP_ORDER]
    names += [
        "crop_age",              # (day - planted_day) / max_yield_day
        "crop_yield_frac",       # yield_units / max_yield
        "crop_unwatered",        # consecutive_unwatered / 2（>=2 變雜草）
        "crop_watered_today",    # 今天澆過了沒 —— 直接決定要不要再派人去
        "crop_fertilized_days",  # (fertilized_until_day - day) / 2
        "crop_ongoing",          # TOMATO / STRAWBERRY
        "crop_lifespan_left",    # (max_lifespan_step - step) / 24，-1 表示不適用
        "coop",
        "pasture",
    ]
    names += [f"animal_{a}" for a in ANIMAL_ORDER]
    names += [
        "animal_yield_frac",     # yield_units / max_held
        "animal_unfed",          # consecutive_unfed / 2（>=2 逃走）
        "animal_fed_today",
        "animal_cared_today",
        "animal_care_bonus",
        "animal_fertilizer_ready",
        "farmer_here",
        "hand_count",            # 這格有幾個 hand / 12
        # 對手的盤面。`farms` 是公開的（`engine-notes.md` §10.12），
        # 但對手的 shed / inventory 看不到，所以只有格子層級。
        "opp_unlocked",
        "opp_crop",
        "opp_structure",
        "opp_animal",
    ]
    return tuple(names)


SPATIAL_CHANNELS = _spatial_channel_names()
N_SPATIAL = len(SPATIAL_CHANNELS)


# --------------------------------------------------------------------------
# 全域純量
# --------------------------------------------------------------------------

def _scalar_field_names():
    names = [
        "money", "opp_money", "money_ratio",
        "day", "hour", "days_left", "step_frac",
        "n_hands", "hires_today", "next_hire_cost",
        "n_quadrants", "opp_n_quadrants",
        "shed_used_frac",
        # --- v2 新增：全盤計數 ------------------------------------------
        # 為什麼要明確給：老師的 BUILD_PASTURE 是「蓋到 14 個就停」，那是計數
        # 判斷。CNN 用 3×3 卷積要精確數出全盤有幾個建物很難，v1 沒給這些數字，
        # 結果網路蓋到 24 個停不下來、整季只養活 1 隻動物（市場買了 16 隻）。
        #
        # ⚠️ 而準確率指標看不出這件事：`PLACE COW` 只佔 0.13%，全錯也只掉
        # 0.13 個百分點。v1 的驗證準確率 93.96%，實戰 0 勝 12 負。
        # `gen0._log` 早就在記 `animals` / `empty_structs`，是我漏了。
        "n_structures", "n_structures_empty", "n_animals",
        "n_crop_tiles", "n_weeds", "n_empty_tiles",
        "opp_n_structures", "opp_n_animals", "opp_n_crop_tiles",
    ]
    names += [f"seed_{c}" for c in CROP_ORDER]
    names += [f"shed_{p}" for p in PRODUCT_ORDER]
    # shed 也放動物。「買了的動物躺在倉庫沒人去放」是已知的失敗模式
    # （journal 08-16：8 隻鵝只放進去 7 次），這個訊號要看得到。
    names += [f"shed_{a}" for a in ANIMAL_ORDER]
    names += [f"carry_{p}" for p in PRODUCT_ORDER]      # 全部 unit 的 inventory 加總
    names += [f"price_{p}" for p in PRODUCT_ORDER]      # / base
    names += [f"market_inv_{p}" for p in PRODUCT_ORDER]  # (inv - I0) / I0
    names += [f"shop_{s}" for s in SHOP_ORDER]          # 各開了幾間 / MAX_SHOP_INSTANCES
    names += ["n_shops"]
    return tuple(names)


SCALAR_FIELDS = _scalar_field_names()
N_SCALAR = len(SCALAR_FIELDS)


# --------------------------------------------------------------------------
# 每個 unit 的特徵
# --------------------------------------------------------------------------

def _unit_feature_names():
    names = ["is_farmer", "unit_index", "x", "y", "shed_distance", "carry_total"]
    names += [f"carry_{p}" for p in PRODUCT_ORDER]
    names += [f"carry_{a}" for a in ANIMAL_ORDER]
    return tuple(names)


UNIT_FEATURES = _unit_feature_names()
N_UNIT_FEATURES = len(UNIT_FEATURES)


# --------------------------------------------------------------------------
# 索引表（一次建好，encode 每回合要用）
# --------------------------------------------------------------------------

_SP = {name: i for i, name in enumerate(SPATIAL_CHANNELS)}
_SC = {name: i for i, name in enumerate(SCALAR_FIELDS)}
_UF = {name: i for i, name in enumerate(UNIT_FEATURES)}


def _shed_tiles(board):
    """能 DROP / PICKUP / PLACE 的四格（`engine-notes.md` §2）。"""
    half = board // 2
    return ((half - 1, half - 1), (half, half - 1),
            (half - 1, half), (half, half))


def _quadrant_of(x, y, board):
    half = board // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _shed_distance(x, y, board):
    return min(abs(x - sx) + abs(y - sy) for sx, sy in _shed_tiles(board))


def _cfg(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# --------------------------------------------------------------------------
# encode
# --------------------------------------------------------------------------

def encode(obs, config=None):
    """局面 → `(spatial [C,H,W] float32, scalar [D] float32)`。

    正規化的原則是把每個欄位壓到大致 0~1（或 -1~1）。用的除數寫在各行註解裡，
    改除數等於改 schema —— 舊權重會讀到不同尺度的輸入，一樣要升
    `ENCODER_VERSION`。
    """
    player = int(obs["player"])
    farm = obs["farms"][player]
    tiles = farm["tiles"]
    board = len(tiles)
    day = int(obs["day"])
    step = int(obs.get("step", day * int(_cfg(config, "turnsPerDay", 24)) + int(obs["hour"])))

    spatial = np.zeros((N_SPATIAL, board, board), dtype=np.float32)
    sheds = set(_shed_tiles(board))

    for y in range(board):
        for x in range(board):
            tile = tiles[y][x]
            spatial[_SP[f"quadrant_{_quadrant_of(x, y, board)}"], y, x] = 1.0
            spatial[_SP["shed_distance"], y, x] = _shed_distance(x, y, board) / board
            if (x, y) in sheds:
                spatial[_SP["shed_adjacent"], y, x] = 1.0

            if tile == "LOCKED":
                continue
            spatial[_SP["unlocked"], y, x] = 1.0

            if tile is None:
                spatial[_SP["empty"], y, x] = 1.0
                continue

            kind = tile.get("kind")
            if kind == "WEED":
                spatial[_SP["weed"], y, x] = 1.0
            elif kind == "PLANT":
                crop = tile.get("crop")
                if crop in CROP_ORDER:
                    spatial[_SP[f"crop_{crop}"], y, x] = 1.0
                    spec = CROPS[crop]
                    spatial[_SP["crop_age"], y, x] = (
                        (day - tile.get("planted_day", day))
                        / max(1, spec["max_yield_day"]))
                    spatial[_SP["crop_yield_frac"], y, x] = (
                        tile.get("yield_units", 0) / max(1, spec["max_yield"]))
                    spatial[_SP["crop_ongoing"], y, x] = 1.0 if spec["ongoing"] else 0.0
                spatial[_SP["crop_unwatered"], y, x] = (
                    tile.get("consecutive_unwatered", 0) / 2.0)
                spatial[_SP["crop_watered_today"], y, x] = (
                    1.0 if tile.get("watered_today") else 0.0)
                # 施肥涵蓋 day / day+1 / day+2 共 3 天（`engine-notes.md` §5）
                spatial[_SP["crop_fertilized_days"], y, x] = max(
                    0.0, (tile.get("fertilized_until_day", -1) - day) / 2.0)
                lifespan = tile.get("max_lifespan_step", -1)
                # -1 = ongoing 作物還沒進入枯萎期，用 -1 明確區分「還沒設定」
                spatial[_SP["crop_lifespan_left"], y, x] = (
                    -1.0 if lifespan is None or lifespan < 0
                    else (lifespan - step) / 24.0)
            elif kind in ("COOP", "PASTURE"):
                spatial[_SP["coop" if kind == "COOP" else "pasture"], y, x] = 1.0
                animal = tile.get("animal")
                if animal in ANIMAL_ORDER:
                    spatial[_SP[f"animal_{animal}"], y, x] = 1.0
                    spatial[_SP["animal_yield_frac"], y, x] = (
                        tile.get("yield_units", 0) / max(1, ANIMALS[animal]["max_held"]))
                    spatial[_SP["animal_unfed"], y, x] = (
                        tile.get("consecutive_unfed", 0) / 2.0)
                    spatial[_SP["animal_fed_today"], y, x] = (
                        1.0 if tile.get("fed_today") else 0.0)
                    spatial[_SP["animal_cared_today"], y, x] = (
                        1.0 if tile.get("cared_today") else 0.0)
                    # care bonus 會累積到首次產出，實測看過 3
                    spatial[_SP["animal_care_bonus"], y, x] = (
                        tile.get("pending_care_bonus", 0) / 4.0)
                    spatial[_SP["animal_fertilizer_ready"], y, x] = (
                        1.0 if tile.get("fertilizer_available") else 0.0)

    fx, fy = farm["farmer"][0], farm["farmer"][1]
    spatial[_SP["farmer_here"], fy, fx] = 1.0
    for hx, hy in farm["hands"]:
        spatial[_SP["hand_count"], hy, hx] += 1.0 / 12.0

    # 對手：`farms` 是公開的，但 shed / inventory 看不到（`engine-notes.md` §10.12）
    opp_farm = obs["farms"][1 - player]
    for y in range(board):
        for x in range(board):
            tile = opp_farm["tiles"][y][x]
            if tile == "LOCKED":
                continue
            spatial[_SP["opp_unlocked"], y, x] = 1.0
            if isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "PLANT":
                    spatial[_SP["opp_crop"], y, x] = 1.0
                elif kind in ("COOP", "PASTURE"):
                    spatial[_SP["opp_structure"], y, x] = 1.0
                    if tile.get("animal"):
                        spatial[_SP["opp_animal"], y, x] = 1.0

    return spatial, _encode_scalar(obs, config, farm, opp_farm, board, day, step)


def _encode_scalar(obs, config, farm, opp_farm, board, day, step):
    private = obs["private"]
    market = obs["market"]
    scalar = np.zeros(N_SCALAR, dtype=np.float32)

    money = float(farm["money"])
    opp_money = float(opp_farm["money"])
    scalar[_SC["money"]] = money / 100_000.0          # 頂端期末約 $140k
    scalar[_SC["opp_money"]] = opp_money / 100_000.0
    scalar[_SC["money_ratio"]] = money / max(1.0, money + opp_money)

    turns_per_day = int(_cfg(config, "turnsPerDay", 24))
    episode_steps = int(_cfg(config, "episodeSteps", 720))
    total_days = max(1, episode_steps // turns_per_day)
    scalar[_SC["day"]] = day / total_days
    scalar[_SC["hour"]] = int(obs["hour"]) / turns_per_day
    scalar[_SC["days_left"]] = (total_days - day) / total_days
    scalar[_SC["step_frac"]] = step / max(1, episode_steps)

    n_hands = len(farm["hands"])
    scalar[_SC["n_hands"]] = n_hands / 12.0
    hires_today = int(farm.get("hires_today", 0))
    scalar[_SC["hires_today"]] = hires_today / 12.0
    # 雇工成本 = mult × fib(當天已雇人數)（`engine-notes.md` §7）
    scalar[_SC["next_hire_cost"]] = (
        _fib(hires_today + 1) * int(_cfg(config, "farmHandCostMult", 1)) / 100.0)

    scalar[_SC["n_quadrants"]] = len(farm["unlocked_quadrants"]) / (len(LAND_ORDER) + 1)
    scalar[_SC["opp_n_quadrants"]] = (
        len(opp_farm["unlocked_quadrants"]) / (len(LAND_ORDER) + 1))

    shed = private["shed"]
    capacity = int(_cfg(config, "shedCapacity", 100))
    scalar[_SC["shed_used_frac"]] = sum(shed.values()) / max(1, capacity)

    for crop in CROP_ORDER:
        scalar[_SC[f"seed_{crop}"]] = private["seeds"].get(crop, 0) / 20.0
    for item in PRODUCT_ORDER:
        scalar[_SC[f"shed_{item}"]] = shed.get(item, 0) / max(1, capacity)
    for animal in ANIMAL_ORDER:
        scalar[_SC[f"shed_{animal}"]] = shed.get(animal, 0) / 10.0

    carried = {}
    for inv in private["inventories"]:
        for item, n in inv.items():
            carried[item] = carried.get(item, 0) + n
    for item in PRODUCT_ORDER:
        scalar[_SC[f"carry_{item}"]] = carried.get(item, 0) / max(1, capacity)

    for item in PRODUCT_ORDER:
        base = MARKET_PARAMS[item]["base"]
        scalar[_SC[f"price_{item}"]] = market["prices"].get(item, base) / base
        # 庫存實測在 I0 ± 500 之間動，除以 500 讓它落在大約 ±1
        scalar[_SC[f"market_inv_{item}"]] = (
            market["inventory"].get(item, MARKET_I0) - MARKET_I0) / 500.0

    # --- v2：全盤計數。除數取「大概的上限」，讓數值落在 0~1 附近 ---------
    def _counts(f):
        structures = empty_structures = animals = crops = weeds = empty = 0
        for row in f["tiles"]:
            for tile in row:
                if tile == "LOCKED":
                    continue
                if tile is None:
                    empty += 1
                elif tile.get("kind") == "PLANT":
                    crops += 1
                elif tile.get("kind") == "WEED":
                    weeds += 1
                elif tile.get("kind") in ("COOP", "PASTURE"):
                    structures += 1
                    if tile.get("animal"):
                        animals += 1
                    else:
                        empty_structures += 1
        return structures, empty_structures, animals, crops, weeds, empty

    mine = _counts(farm)
    theirs = _counts(opp_farm)
    scalar[_SC["n_structures"]] = mine[0] / 20.0
    scalar[_SC["n_structures_empty"]] = mine[1] / 20.0
    scalar[_SC["n_animals"]] = mine[2] / 20.0
    scalar[_SC["n_crop_tiles"]] = mine[3] / 75.0
    scalar[_SC["n_weeds"]] = mine[4] / 20.0
    scalar[_SC["n_empty_tiles"]] = mine[5] / 75.0
    scalar[_SC["opp_n_structures"]] = theirs[0] / 20.0
    scalar[_SC["opp_n_animals"]] = theirs[2] / 20.0
    scalar[_SC["opp_n_crop_tiles"]] = theirs[3] / 75.0

    shops = obs["town"].get("unlocked_shops", [])
    for shop in SHOP_ORDER:
        scalar[_SC[f"shop_{shop}"]] = shops.count(shop) / MAX_SHOP_INSTANCES
    scalar[_SC["n_shops"]] = len(shops) / MAX_SHOP_INSTANCES
    return scalar


def _fib(n):
    """引擎的雇工價格序列 1,1,2,3,5,8...（`engine-notes.md` §7）。"""
    a, b = 1, 1
    for _ in range(max(0, n - 1)):
        a, b = b, a + b
    return a


# --------------------------------------------------------------------------
# 每個 unit
# --------------------------------------------------------------------------

def encode_units(obs, config=None):
    """回傳 `(positions [n,2] int16, features [n,U] float32)`。

    順序就是引擎的 unit 順序：索引 0 是主農夫，1 以後是 hands，
    跟 `action["hands"]` 的順序一致（`kaggriculture.py:937`）。
    """
    player = int(obs["player"])
    farm = obs["farms"][player]
    board = len(farm["tiles"])
    inventories = obs["private"]["inventories"]
    capacity = int(_cfg(config, "shedCapacity", 100))

    positions = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
    n = len(positions)
    pos = np.zeros((n, 2), dtype=np.int16)
    feats = np.zeros((n, N_UNIT_FEATURES), dtype=np.float32)

    for i, (x, y) in enumerate(positions):
        pos[i] = (x, y)
        inv = inventories[i] if i < len(inventories) else {}
        feats[i, _UF["is_farmer"]] = 1.0 if i == 0 else 0.0
        feats[i, _UF["unit_index"]] = i / 12.0
        feats[i, _UF["x"]] = x / board
        feats[i, _UF["y"]] = y / board
        feats[i, _UF["shed_distance"]] = _shed_distance(x, y, board) / board
        feats[i, _UF["carry_total"]] = sum(inv.values()) / max(1, capacity)
        for item in PRODUCT_ORDER:
            feats[i, _UF[f"carry_{item}"]] = inv.get(item, 0) / max(1, capacity)
        for animal in ANIMAL_ORDER:
            feats[i, _UF[f"carry_{animal}"]] = inv.get(animal, 0) / 4.0

    return pos, feats


# --------------------------------------------------------------------------
# 合法性
# --------------------------------------------------------------------------

def legal_unit_mask(obs, config=None):
    """回傳 `[n_units, N_UNIT_OPS]` 的 bool 陣列。

    ## 不變式：寧可放寬，絕不擋掉老師做過的動作

    擋掉的話網路連學都學不到那個動作，而且 loss 會出現無限大；放寬只是讓
    引擎靜默忽略、浪費那個 unit 一個回合。所以下面的條件都取寬鬆的一側。

    `tests/test_contracts.py::test_mask_never_blocks_teacher_action` 拿 60 局
    replay 的每一個 unit-turn 驗這條 —— 有任何一筆被擋掉就是這裡寫錯了。

    ## 這裡**沒有**檢查的事

    - 錢夠不夠（那是 market 的事，unit 動作不花錢）
    - shed 會不會滿（`DROP` 超過容量的部分是被丟棄，不是動作失敗）
    - 對手同回合的行為
    - PLANT 的原子驗證（同回合總需求超過種子數 → 該作物全部變 PASS，
      `engine-notes.md` §10.9）。那是**跨 unit** 的約束，單一 unit 的 mask
      表達不了，留給 decode 之後的整批檢查。
    """
    player = int(obs["player"])
    farm = obs["farms"][player]
    tiles = farm["tiles"]
    board = len(tiles)
    private = obs["private"]
    shed = private["shed"]
    seeds = private["seeds"]
    inventories = private["inventories"]
    sheds = set(_shed_tiles(board))

    positions = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
    mask = np.zeros((len(positions), N_UNIT_OPS), dtype=bool)

    for i, (x, y) in enumerate(positions):
        inv = inventories[i] if i < len(inventories) else {}
        tile = tiles[y][x]
        locked = tile == "LOCKED"
        kind = tile.get("kind") if isinstance(tile, dict) else None
        near_shed = (x, y) in sheds

        for j, (op, item) in enumerate(UNIT_OPS):
            if op in MOVES:
                dx, dy = FARMER_MOVES[op]
                mask[i, j] = 0 <= x + dx < board and 0 <= y + dy < board
            elif op == "PASS":
                mask[i, j] = True
            # --- shed 操作：引擎在 LOCKED guard 之前處理（engine-notes §2）---
            elif op == "DROP":
                mask[i, j] = near_shed and bool(inv)
            elif op == "PICKUP":
                mask[i, j] = near_shed and shed.get(item, 0) > 0
            elif op == "PLACE":
                # 兩用：shed 旁存入，或把動物放進對應的空建物
                housing = (
                    item in ANIMAL_ORDER
                    and kind == ANIMALS[item]["structure"]
                    and not (isinstance(tile, dict) and tile.get("animal"))
                    and inv.get(item, 0) > 0
                )
                mask[i, j] = (near_shed and inv.get(item, 0) > 0) or housing
            # --- 以下都要站在自己已解鎖的格子上 ---
            elif locked:
                mask[i, j] = False
            elif op == "PLANT":
                mask[i, j] = tile is None and seeds.get(item, 0) > 0
            elif op in ("BUILD_COOP", "BUILD_PASTURE"):
                mask[i, j] = tile is None
            elif op == "WATER":
                mask[i, j] = kind == "PLANT"
            elif op == "HARVEST":
                # ⚠️ 不只作物 —— 動物的 EGG / MILK / WOOL 也是 HARVEST 收的。
                # 第一版只寫了 PLANT，replay 驗證時有 222 筆老師的 HARVEST 站在
                # PASTURE 上被擋掉，才發現漏了這條。
                mask[i, j] = kind == "PLANT" or bool(
                    isinstance(tile, dict) and tile.get("animal"))
            elif op == "FERTILIZE":
                mask[i, j] = kind == "PLANT" and inv.get("FERTILIZER", 0) > 0
            elif op == "DIG":
                # 作物、雜草、空建物都能挖；有動物的建物不行（engine-notes §10.7）
                mask[i, j] = kind in ("PLANT", "WEED") or (
                    kind in ("COOP", "PASTURE")
                    and not (isinstance(tile, dict) and tile.get("animal")))
            elif op in ("CARE", "COLLECT_FERTILIZER"):
                mask[i, j] = bool(isinstance(tile, dict) and tile.get("animal"))
            elif op == "FEED":
                mask[i, j] = bool(
                    isinstance(tile, dict) and tile.get("animal")
                    and inv.get("WHEAT", 0) > 0)

    return mask


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------

def decode_unit(op_index, qty_index=None):
    """`(動作編號, 數量編號) -> 引擎吃的 unit 動作 list`。

    ⚠️ 帶參數的 op 一定要把參數補上。引擎對 `["PLACE"]`（少了物品名）的處理是
    `if len(action) < 2: return` —— **靜默忽略**。gen0 第一版就是漏了這個，
    整季 337 個回合白丟（`gen0._task_action` 的註解）。
    """
    op, item = UNIT_OPS[int(op_index)]
    if item is None:
        return [op]
    if op == "PLANT":
        return [op, item]
    # PICKUP / PLACE 帶數量。沒給就用 1 —— replay 裡 PLACE 最常見的就是 1。
    qty = 1 if qty_index is None else QTY_CHOICES[int(qty_index)]
    return [op, item, qty]


def encode_unit_action(action):
    """引擎的 unit 動作 list -> `(op_index, qty_index or None)`。抽訓練標籤用。

    認不得的動作回 `None`，呼叫端自己決定要跳過還是拋錯 —— 靜默當成 PASS
    會讓標籤髒掉而且查不出來。
    """
    if not isinstance(action, (list, tuple)) or not action:
        return None
    op = action[0]
    item = action[1] if len(action) > 1 else None
    key = (op, item if (op in ("PLANT", "PICKUP", "PLACE")) else None)
    index = UNIT_OP_INDEX.get(key)
    if index is None:
        return None
    if op in ("PICKUP", "PLACE") and len(action) > 2:
        try:
            qty = int(action[2])
        except (TypeError, ValueError):
            return None
        # replay 值域是 1~12；超出就夾到邊界，不要丟掉整筆樣本
        qty = min(max(qty, QTY_CHOICES[0]), QTY_CHOICES[-1])
        return index, QTY_CHOICES.index(qty)
    return index, None


__all__ = [
    "ENCODER_VERSION",
    "encode", "encode_units", "legal_unit_mask",
    "decode_unit", "encode_unit_action",
    "SPATIAL_CHANNELS", "N_SPATIAL",
    "SCALAR_FIELDS", "N_SCALAR",
    "UNIT_FEATURES", "N_UNIT_FEATURES",
    "UNIT_OPS", "N_UNIT_OPS", "UNIT_OP_INDEX",
    "QTY_CHOICES", "N_QTY",
    "CROP_ORDER", "ANIMAL_ORDER", "PRODUCT_ORDER", "SHOP_ORDER", "QUADRANT_ORDER",
]
