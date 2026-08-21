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
#:
#: v2 -> v3（2026-08-20）：**輸入 schema 一個 bit 都沒動**，升版的是
#: `op_logits` 的語意 —— 從「這一步做什麼」改成「**這一段的終點做什麼**」，
#: 另外多一個 target head 指出終點在哪一格。
#:
#: 為什麼語意變更也要升版：v2 權重載進 v3 的 serving **不會報錯**，只會把
#: 「終點動作」當「立即動作」用，然後分數莫名其妙掉下來 —— 正是版本號要擋的
#: 那種無聲失敗。
#:
#: v2 的實測（`tools/action_dist.py`，gen2_model vs ladder-top-a 4 局）：對老師
#: 局面的 unit-turn 準確率 0.9666，自己下場卻 MOVE 佔 75.6%（老師 51.8%）、
#: FEED 只做了 70 次（老師 1,252 次）。壞的是閉迴路，不是每一步的準確率。
#:
#: v3 -> v4（2026-08-20）：市場那一半也交給網路（`MARKET_OPS` 搬進來、加兩個
#: market head），`legal_unit_mask` 補上「同一天重複做」的五條。**輸入 schema
#: 仍然一個 bit 都沒動**，升版是因為 v3 的權重沒有 market head，載進 v4 的
#: serving 會炸在缺 key 上。
#:
#: v3 的實測：unit 動作已經像老師了（target 準確率 0.8716、MOVE 36.9%），但
#: 期末現金只有 2,405 對 140,210。按天比對狀態分布發現 **day 6 動物就掉出老師
#: 的 p5**（我們 1 隻、老師 p5 是 6 隻），而動物是市場買的不是 unit 決定的 ——
#: 整季 gen0 只買了 3 隻 COW、13 個 WHEAT，老師是 14 隻動物、553 個 WHEAT。
#: 「只模仿一半的 policy」就是這個意思。
#:
#: v4 -> v5（2026-08-20）：**unit 的輸出從「每個 unit 去哪一格」改成「每一格
#: 要做什麼」**（`TASK_OPS` + `legal_demand_mask` + net 的 demand head）。
#: 輸入 schema 仍然一個 bit 都沒動。
#:
#: 為什麼要改：v4 的 target head 是逐 unit 的，12 個 unit 各自獨立猜格子，
#: 而 DAgger 三輪之後 target 準確率停在 0.6301 —— `0.63 ** 12 = 0.4%`，
#: 也就是「整個回合 12 個 unit 全部放對」幾乎不會發生。錯誤從時間軸（v3 修掉的
#: 那條）搬到了 unit 軸，仍然是乘法。
#:
#: 而且 0.6301 這個數字是**資料愈多愈低**（0.7580 -> 0.6505 -> 0.6301）。原因是
#: 標籤本身有雜訊：`agents/gen0._assign` 用 `_minimum_cost_assignment` 做全域
#: 最短配對，兩個 unit 對兩個等距任務的配對是任意的（Hungarian 的內部順序決定），
#: 盤面相同、標籤不同 —— 那是學不起來的東西。
#:
#: v5 把這件事切開：
#:   - **判斷**（哪一格該澆水、值不值得施肥）是盤面的函式，有唯一答案 -> 網路
#:   - **配對**（哪個 unit 去哪一格比較近）算得出最佳解 -> 還給
#:     `_minimum_cost_assignment`
#: demand map 是 `[N_TASK_OPS, N_TARGET_CELLS]` 的 multi-label，逐格獨立，
#: 一格猜錯只影響那一格，不會像 v4 那樣讓整個回合的配對錯位。
ENCODER_VERSION = 5


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

#: v3 的 target head：unit 這一段要走去哪一格。`board ** 2`，60 局 replay
#: 全部是 10×10。寫死是為了讓換板子大小**炸掉**而不是無聲位移 ——
#: `target_index` / `target_xy` 都會 assert。
BOARD_SIZE = 10
N_TARGET_CELLS = BOARD_SIZE * BOARD_SIZE


def target_index(x, y, board):
    """`(x, y) -> 0..N_TARGET_CELLS-1`。row-major，跟 `tiles[y][x]` 同一個順序。"""
    if board * board != N_TARGET_CELLS:
        raise AssertionError(
            f"板子是 {board}×{board}，target head 是 {N_TARGET_CELLS} 格 —— "
            "換板子大小要升 ENCODER_VERSION 並重訓，不要硬轉")
    return int(y) * board + int(x)


def target_xy(index, board):
    """`target_index` 的反向。"""
    if board * board != N_TARGET_CELLS:
        raise AssertionError(
            f"板子是 {board}×{board}，target head 是 {N_TARGET_CELLS} 格")
    index = int(index)
    return index % board, index // board


#: v5 的 demand head：**這一格要做什麼**，跟哪個 unit 去做無關。
#:
#: 順序寫死，跟 `UNIT_OPS` 一樣是 schema 的一部分。這 11 個就是
#: `agents/gen0._tasks()` 會掃出來的逐格任務。
#:
#: **`FETCH_*` 不在裡面** —— 那三個是導出的：要幾趟 WHEAT 由 FEED 的格數決定、
#: 要幾趟 FERTILIZER 由 FERTILIZE 的格數決定、要幾趟動物由 PLACE 的格數決定。
#: 它們的目標格永遠是 shed 那四格，也不是「哪一格該做什麼」這個問題的答案。
#: 讓網路預測導出量只會多一個對不上的機會，所以留給 `gen0` 算。
#:
#: **`arg` 也不在裡面**：`PLANT` 種什麼由 `crop_for(x, y, basket)` 決定（同一格
#: 永遠同一種），`PLACE` / `BUILD_*` 放哪一species由 `_house_animals()` 決定
#: （看手上真的有什麼）。兩個都是查表，不是判斷。
TASK_OPS = (
    "WATER", "HARVEST", "FERTILIZE", "DIG", "PLANT",
    "FEED", "CARE", "COLLECT_FERTILIZER",
    "PLACE", "BUILD_COOP", "BUILD_PASTURE",
)
N_TASK_OPS = len(TASK_OPS)
TASK_OP_INDEX = {op: i for i, op in enumerate(TASK_OPS)}


# --------------------------------------------------------------------------
# 市場動作（v4）
# --------------------------------------------------------------------------

#: 市場動作詞彙。順序寫死，跟 `UNIT_OPS` 同樣是 schema 的一部分。
#:
#: v4 之前這份表在 `harness/build_dataset.py`（只有訓練端要用）。現在比賽端
#: 也要靠它把網路輸出轉回訂單，所以搬進來 —— **順序完全沒動**。
MARKET_OPS = (
    ("HIRE", None),
    ("BUY_LAND", None),
    *(("SELL", item) for item in PRODUCT_ORDER),
    # 引擎只讓 WHEAT / FERTILIZER 用 BUY_PRODUCT（`engine-notes.md` §1）
    ("BUY_PRODUCT", "WHEAT"),
    ("BUY_PRODUCT", "FERTILIZER"),
    *(("BUY_SEED", crop) for crop in CROP_ORDER),
    *(("BUY_ANIMAL", animal) for animal in ANIMAL_ORDER),
)
N_MARKET_OPS = len(MARKET_OPS)
MARKET_OP_INDEX = {key: i for i, key in enumerate(MARKET_OPS)}

#: 市場訂單的數量桶。實測 60 局：91.8% 的數量 ≤ 12、99.9% ≤ 32、最大 79
#: （`SELL WHEAT`）。小數量逐一給桶，大的用粗桶 —— 賣 48 跟賣 52 的差別
#: 遠小於賣 4 跟賣 8。
MARKET_QTY_BUCKETS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      16, 20, 24, 32, 48, 80)
N_MARKET_QTY = len(MARKET_QTY_BUCKETS)

def market_qty_bucket(qty):
    """數量 -> 桶編號。取**不超過**實際值的最大桶。

    往下取而不是取最近的：賣少一點只是少賺，賣爆會把價格砸到地板
    （`engine-notes.md` §6：MELON 的 `above_func = sq / 3.60`）。
    """
    qty = int(qty)
    best = 0
    for i, edge in enumerate(MARKET_QTY_BUCKETS):
        if edge <= qty:
            best = i
    return best


def market_qty_value(bucket):
    """桶編號 -> 實際數量。"""
    return MARKET_QTY_BUCKETS[int(bucket)]


def encode_market_action(order):
    """引擎的市場訂單 -> `(op_index, qty)`。認不得回 `None`。

    `["HIRE"]` 這種沒帶數量的算 1 個。
    """
    if not isinstance(order, (list, tuple)) or not order:
        return None
    key = (order[0], order[1] if len(order) > 1 else None)
    index = MARKET_OP_INDEX.get(key)
    if index is None:
        # `["HIRE", something]` 之類 —— 用不帶參數的形式再試一次
        index = MARKET_OP_INDEX.get((order[0], None))
    if index is None:
        return None
    try:
        qty = int(order[2]) if len(order) > 2 else 1
    except (TypeError, ValueError):
        return None
    return index, max(1, qty)


def decode_market(op_index, qty):
    """`(op_index, 數量) -> 引擎吃的訂單 list`。

    ⚠️ 帶參數的 op 一定要把參數補上 —— 引擎對格式不對的訂單是**靜默忽略**，
    跟 `decode_unit` 同一個坑。
    """
    op, item = MARKET_OPS[int(op_index)]
    qty = max(1, int(qty))
    if op in ("HIRE", "BUY_LAND"):
        return [op]                      # 這兩個一筆就是一個，不帶數量
    return [op, item, qty]


def legal_market_mask(obs, config=None):
    """回傳 `[N_MARKET_OPS]` 的 bool 陣列。

    不變式跟 `legal_unit_mask` 同一條：**寧可放寬，絕不擋掉老師下過的訂單。**
    `tests/test_market_labels.py` 拿 replay 的每一筆訂單驗，門檻同樣是 0.5%。

    ## 🩸 為什麼「錢夠不夠」「shed 有沒有貨」都不在這裡

    第一版把這兩條寫進 mask，結果擋掉老師 **23.1%** 的訂單（3,484 筆裡 804 筆，
    `SELL FERTILIZER` 406 筆、`SELL WHEAT` 234 筆）。

    因為**老師自己就會送出送不成的訂單** —— journal 2026-08-19 §5 早就量過：
    頂端玩家整季送出約 1,460 筆 `FERTILIZER SELL`，`market.inventory` 顯示實際
    只成交約 449 個，其餘因為 shed 沒貨被引擎中止。那是「有多少賣多少」的寫法，
    不是 bug。

    所以那些條件搬到 `clamp_market_qty`：數量夾到 0 的訂單在
    `decode_market_orders` 會被丟掉，執行結果跟擋掉一樣，但 mask 保持誠實。

    這裡只留**結構上不可能**的那一條：象限買完了就不能再買地。
    """
    farm = obs["farms"][int(obs["player"])]
    mask = np.ones(N_MARKET_OPS, dtype=bool)
    if len(farm["unlocked_quadrants"]) >= len(LAND_ORDER) + 1:
        mask[MARKET_OP_INDEX[("BUY_LAND", None)]] = False
    return mask


def clamp_market_qty(op_index, qty, obs):
    """把賣單的數量夾到 shed 實有量。夾到 0 的訂單呼叫端會丟掉。

    ## 為什麼**不**夾「現金買不買得起」

    引擎是**逐筆結算**的，而 `MARKET_OPS` 的順序讓 `SELL`（index 2 起）排在
    `BUY_*`（index 11 起）前面 —— 賣單先進帳，後面的買單才有錢。用「送出當下
    的現金」去夾買單，會夾掉本來會成功的訂單。買不起的訂單引擎自己會中止，
    那是無害的。

    ⚠️ 這裡也**不**夾「賣太多會砸盤」。網路一次倒 79 個 MELON 出去是策略問題
    不是合法性問題（`engine-notes.md` §6：MELON 的 `above_func = sq / 3.60`）。
    要擋要另外加價格門檻，先量 `tools/revenue.py` 的均價有沒有崩。
    """
    op, item = MARKET_OPS[int(op_index)]
    qty = max(1, int(qty))
    farm = obs["farms"][int(obs["player"])]
    if op == "SELL":
        return min(qty, obs["private"]["shed"].get(item, 0))
    if op == "BUY_LAND":
        return 1                          # 老師 42 次買地全部是一回合一塊
    if op == "HIRE":
        # 雇工價是**當天第 n 個**的 fib，所以「還能雇幾個」算得出來，
        # 跟其他採購不一樣（那些的單價是固定的，見上面）。
        return min(qty, _affordable_hires(
            float(farm["money"]), int(farm.get("hires_today", 0)), obs))
    return qty


def _affordable_hires(money, hires_today, obs, mult=1):
    """現金還雇得起幾個人。價格是當天第 n 個的 fib（`engine-notes.md` §7）。"""
    n = 0
    while n < 24:                         # 一天 24 回合，再多也沒意義
        cost = _fib(hires_today + n + 1) * mult
        if cost > money:
            break
        money -= cost
        n += 1
    return n


def decode_market_orders(present_logits, qty_logits, obs, config=None,
                         threshold=0.0):
    """網路的兩個 market head -> 引擎吃的訂單清單。

    `present_logits` 是 `[N_MARKET_OPS]` 的 logit（訓練用 BCE，所以門檻對應
    sigmoid 0.5 就是 logit 0）；`qty_logits` 是 `[N_MARKET_OPS, N_MARKET_QTY]`。

    `threshold` 可以是純量，也可以是 `[N_MARKET_OPS]` —— **逐 op 分開設**。
    理由是兩邊的錯誤成本差很多：

    - 少買一次種子 -> 整片田空著好幾天（2026-08-21 實測期末作物 3.6 對 14.0）
    - 多買一顆種子 -> 花 $10~$100
    - 賣錯東西 -> 直接虧價差

    而且 `BUY_SEED` 的正例只佔 3.2%，AUC 有 0.965~0.992（排序幾乎完美）但
    sigmoid 0.5 只召回得到 0.25~0.38 —— 那是類別不平衡造成的門檻問題，
    不是沒學會。齊頭式降門檻沒用（實測 -1.0 只讓現金 18,194 -> 19,138），
    因為它連 `SELL` 一起放寬。

    訂單順序照 `MARKET_OPS` 的順序送出。**引擎是逐筆結算的**，所以順序會影響
    「錢花完了後面就失敗」—— 賣單排在買單前面（`SELL` 在 `MARKET_OPS` 裡
    index 2 起，`BUY_*` 在 11 起），剛好是先收錢再花錢。
    """
    mask = legal_market_mask(obs, config)
    th = np.broadcast_to(np.asarray(threshold, dtype=np.float64),
                         (N_MARKET_OPS,))
    orders = []
    for j in range(N_MARKET_OPS):
        if not mask[j] or present_logits[j] <= th[j]:
            continue
        qty = market_qty_value(int(np.argmax(qty_logits[j])))
        qty = clamp_market_qty(j, qty, obs)
        if qty <= 0:
            continue
        op = MARKET_OPS[j][0]
        if op in ("HIRE", "BUY_LAND"):
            # 🩸 這兩個不帶數量，要送 n 筆。實測老師 **57.6% 的雇工回合一次送
            # 5 筆 HIRE**（最多 9 筆）—— hands 每天被清空，開工那一刻要一次
            # 補滿。只送 1 筆的話每天上午都在慢慢補人，等於少掉半天的勞力。
            orders.extend([decode_market(j, 1)] * qty)
        else:
            orders.append(decode_market(j, qty))
    return orders


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

    ## 同一天重複做的動作**要**擋（2026-08-20 補）

    引擎對「今天已經澆過的格子再澆一次」是靜默 no-op。第一版沒擋，結果
    `gen3_target` 送出的 852 次 `WATER` 裡有 **501 次（58.8%）** 落在已經澆過的
    格子上，有效澆水只剩 351 次/局（老師 928 次）。作物連續兩天沒澆就變雜草，
    整季作物格數卡在 6~20（老師約 60）。

    加這五條**不違反上面的不變式** —— 拿 3 局 replay 的 13,466 個動作驗過，
    老師的白費率是：

        WATER 0.00%（5,569 筆）   FEED 0.00%（1,777）   CARE 0.00%（1,867）
        COLLECT_FERTILIZER 0.00%（1,868）   HARVEST 0.08%（2,385 筆裡 2 筆）

    它們是**漏掉**，不是刻意留給網路學的。（量過網路學不學得會：老師的局面上
    白費率 1.02%，自己下場 9.2% —— 學到八成，但閉迴路放大 14 倍。）

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
                # 「今天已經澆過了」也要擋 —— 見下面「同一天重複做」那一段。
                mask[i, j] = kind == "PLANT" and not tile.get("watered_today")
            elif op == "HARVEST":
                # ⚠️ 不只作物 —— 動物的 EGG / MILK / WOOL 也是 HARVEST 收的。
                # 第一版只寫了 PLANT，replay 驗證時有 222 筆老師的 HARVEST 站在
                # PASTURE 上被擋掉，才發現漏了這條。
                mask[i, j] = bool(
                    (kind == "PLANT" or (isinstance(tile, dict) and tile.get("animal")))
                    and tile.get("yield_units", 0) > 0)
            elif op == "FERTILIZE":
                mask[i, j] = kind == "PLANT" and inv.get("FERTILIZER", 0) > 0
            elif op == "DIG":
                # 作物、雜草、空建物都能挖；有動物的建物不行（engine-notes §10.7）
                mask[i, j] = kind in ("PLANT", "WEED") or (
                    kind in ("COOP", "PASTURE")
                    and not (isinstance(tile, dict) and tile.get("animal")))
            elif op == "CARE":
                mask[i, j] = bool(
                    isinstance(tile, dict) and tile.get("animal")
                    and not tile.get("cared_today"))
            elif op == "COLLECT_FERTILIZER":
                mask[i, j] = bool(
                    isinstance(tile, dict) and tile.get("animal")
                    and tile.get("fertilizer_available"))
            elif op == "FEED":
                mask[i, j] = bool(
                    isinstance(tile, dict) and tile.get("animal")
                    and not tile.get("fed_today")
                    and inv.get("WHEAT", 0) > 0)

    return mask


def legal_target_mask(obs, config=None):
    """v3 的 target head 遮罩，回傳 `[n_units, N_TARGET_CELLS]` 的 bool 陣列。

    不變式跟 `legal_unit_mask` 同一條：**寧可放寬，絕不擋掉老師去過的格子。**
    所以只擋一種格子 —— `LOCKED` 而且不是 shed 那四格。

    - `LOCKED` 的格子**走得進去**（引擎只擋「對這格做事」，不擋移動，
      見 `agents/gen0._task_action` 的註解），但沒有理由派 unit 去那裡
    - shed 那四格是例外：引擎的 `DROP` / `PICKUP` / `PLACE` 在 `LOCKED` guard
      **之前**處理（`engine-notes.md` §2），所以象限還沒買下來也能在那裡取放

    ⚠️ 這裡**不看** unit 的 inventory 或格子上有什麼。「到了那裡能不能做那件事」
    是到達之後用 `legal_unit_mask` 驗的，不是出發前。出發前就擋掉的話，
    「先走過去、到了再說」這種老師常做的事會學不到。

    每個 unit 拿到的遮罩其實一樣（只看盤面不看 unit），但仍然回傳
    `[n_units, ...]`，讓呼叫端跟 `legal_unit_mask` 用同一個形狀。
    """
    player = int(obs["player"])
    farm = obs["farms"][player]
    tiles = farm["tiles"]
    board = len(tiles)
    sheds = set(_shed_tiles(board))

    row = np.zeros(N_TARGET_CELLS, dtype=bool)
    for y in range(board):
        for x in range(board):
            row[target_index(x, y, board)] = (
                tiles[y][x] != "LOCKED" or (x, y) in sheds)

    n_units = 1 + len(farm["hands"])
    return np.broadcast_to(row, (n_units, N_TARGET_CELLS)).copy()


def legal_demand_mask(obs, config=None):
    """v5 的 demand head 遮罩，回傳 `[N_TASK_OPS, N_TARGET_CELLS]` 的 bool。

    「這一格**有可能**需要做這件事嗎」——只看格子的狀態。

    ## 跟 `legal_unit_mask` 的差別：這裡不看 inventory

    `legal_unit_mask` 的 `FEED` 要求該 unit 手上有 WHEAT、`FERTILIZE` 要求有
    FERTILIZER、`PLANT` 要求有種子。**demand 不看這些** —— 「那頭牛今天還沒吃」
    是格子的事實，手上有沒有飼料是物流問題，答案是派人去 shed 拿
    （`FETCH_WHEAT`），不是把需求抹掉。出發前就擋掉的話，整條補給鏈就沒有
    觸發條件了。

    ## 不變式跟前兩個 mask 同一條：寧可放寬

    `agents/gen0._tasks()` 掃出來的每一筆任務都必須落在這個 mask 裡面 ——
    `harness/rollout.Recorder.record()` 每個回合都會驗，不合就直接拋錯。
    擋掉的話那筆需求連學都學不到，而且不會有人告訴你。

    所以這裡取的都是寬鬆側：`_tasks` 的 `FERTILIZE` 要過
    `_fertilize_worth_it()`（看價格、看天數），mask 只要求「這格是作物」；
    `DIG` 只在 WEED 上發，mask 連作物和空建物都放行（引擎允許）。
    **值不值得做是網路要學的判斷，能不能做才是 mask 的事。**

    `LOCKED` 全擋 —— 引擎的 `DROP` / `PICKUP` / `PLACE`（存貨那一路）雖然在
    LOCKED guard 之前處理，但那三個都不在 `TASK_OPS` 裡；`TASK_OPS` 的
    `PLACE` 是「把動物放進建物」，那需要格子上先有建物，LOCKED 的格子沒有。
    """
    player = int(obs["player"])
    tiles = obs["farms"][player]["tiles"]
    board = len(tiles)

    mask = np.zeros((N_TASK_OPS, N_TARGET_CELLS), dtype=bool)
    for y in range(board):
        for x in range(board):
            tile = tiles[y][x]
            if tile == "LOCKED":
                continue
            cell = target_index(x, y, board)
            if tile is None:
                mask[TASK_OP_INDEX["PLANT"], cell] = True
                mask[TASK_OP_INDEX["BUILD_COOP"], cell] = True
                mask[TASK_OP_INDEX["BUILD_PASTURE"], cell] = True
                continue
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            animal = bool(tile.get("animal"))
            yield_units = tile.get("yield_units", 0)

            if kind == "PLANT":
                mask[TASK_OP_INDEX["WATER"], cell] = not tile.get("watered_today")
                mask[TASK_OP_INDEX["FERTILIZE"], cell] = True
            if kind in ("PLANT", "WEED") or (
                    kind in ("COOP", "PASTURE") and not animal):
                # 有動物的建物挖不掉（engine-notes §10.7）
                mask[TASK_OP_INDEX["DIG"], cell] = True
            if kind in ("COOP", "PASTURE") and not animal:
                mask[TASK_OP_INDEX["PLACE"], cell] = True
            if (kind == "PLANT" or animal) and yield_units > 0:
                mask[TASK_OP_INDEX["HARVEST"], cell] = True
            if animal:
                mask[TASK_OP_INDEX["FEED"], cell] = not tile.get("fed_today")
                mask[TASK_OP_INDEX["CARE"], cell] = not tile.get("cared_today")
                mask[TASK_OP_INDEX["COLLECT_FERTILIZER"], cell] = bool(
                    tile.get("fertilizer_available"))
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
    "encode", "encode_units", "legal_unit_mask", "legal_target_mask",
    "decode_unit", "encode_unit_action",
    "BOARD_SIZE", "N_TARGET_CELLS", "target_index", "target_xy",
    "MARKET_OPS", "N_MARKET_OPS", "MARKET_OP_INDEX",
    "MARKET_QTY_BUCKETS", "N_MARKET_QTY",
    "market_qty_bucket", "market_qty_value",
    "encode_market_action", "decode_market", "decode_market_orders",
    "legal_market_mask", "clamp_market_qty",
    "SPATIAL_CHANNELS", "N_SPATIAL",
    "SCALAR_FIELDS", "N_SCALAR",
    "UNIT_FEATURES", "N_UNIT_FEATURES",
    "UNIT_OPS", "N_UNIT_OPS", "UNIT_OP_INDEX",
    "QTY_CHOICES", "N_QTY",
    "CROP_ORDER", "ANIMAL_ORDER", "PRODUCT_ORDER", "SHOP_ORDER", "QUADRANT_ORDER",
]
