"""第 0 代規則式 AI。

目標不是聰明，是**正確**（見 `docs/division-of-labor.md` T20）：

- 不送出非法動作 —— 引擎遇到非法動作是靜默忽略，錯了不會有人告訴你
- 完全不用亂數 —— 同一個 seed 每次跑出來的期末現金必須一模一樣，
  這是 `docs/rules.md` §2.2 那條 `assert final_cash == baseline[seed]` 的前提
- 決策依據可以從 log 看出來

規則常數直接 import 引擎，不做鏡像（理由見 `docs/architecture.md`）。

## 策略

每格有固定的**用途**（作物 / 動物），每回合掃出全盤待辦任務，
按優先序指派給距離最近的閒置 unit。

三件事情決定了設計：

1. **多作物分散賣壓。** 第一版只種 CARROT，30 天只賺 $437 ——
   每天往市場倒 60+ 根，`above_func = sqrt`，累積約 900 個就打到 $1 地板，
   城鎮每天只消耗十幾個。所以每格固定種不同作物，並且市價太低就囤著不賣。

2. **物流是硬約束。** `FEED` 消耗 inventory 裡的 WHEAT、`FERTILIZE` 消耗
   FERTILIZER、`PLACE` 消耗 inventory 裡的動物 —— 都不是從 shed 直接扣。
   unit 得先站到 shed 相鄰格 `PICKUP`。好在每天早上所有 unit 都已經在那四格
   （農夫重設到 (4,4)，hands spawn 在 shed 相鄰格），補給不用繞路。

3. **動物會餓死。** 連續 2 天沒 `FEED` 就跑掉，$300~$500 直接消失，
   所以 FEED 排在所有任務的最前面。
"""

import functools
import os

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_I0,
    MARKET_PARAMS,
    PRODUCTS,
    SHOPS,
    market_price,
)
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    _hire_cost as hire_cost,   # 私有 API，改名了就在這裡修一處
)

#: 0 安靜 / 1 每日 / 2 每回合 / 3 決策細節。submission 必須是 0（`docs/rules.md` §3）。
LOG_LEVEL = int(os.environ.get("KAGGRI_LOG_LEVEL", "3"))

#: log 寫到哪。沒設就寫 stderr —— 但 kaggle_environments 會把 agent 的 stderr
#: 攔進 StringIO 並截斷到每步 10,000 字元（`agent.py` 184-205），`debug=False`
#: 時整包丟掉。寫檔可以繞過這兩件事，多行程跑的時候也不會混在一起。
#: `eval/runner.py` 會自動幫每一局設好路徑。
LOG_FILE = os.environ.get("KAGGRI_LOG_FILE")

#: 引擎一回合最多處理這麼多筆市場訂單，多的靜默丟棄（kaggriculture.json）。
MAX_MARKET_ORDERS = 10

DEFAULT_SHED_CAPACITY = 100

DEFAULT_PARAMS = {
    # --- 土地用途 ---
    # 每格種什麼：用 (x, y) 決定，讓賣壓分散在多種商品上。
    # MELON 單價最高但完全沒有 shop 需求（只有 town center 每天買 1 個），
    # 配額必須壓低，不然一樣砸到地板。
    "basket": ("MELON", "CARROT", "CARROT", "WHEAT", "WHEAT"),
    # 開了的話忽略上面的 basket，改成依這一局實際開出來的 shop 動態決定。
    # shop 是抽取有放回的，每局需求結構差很多（實測 CARROT 的日消耗量在不同
    # seed 之間從 1.0 到 18.0），固定的 basket 在需求低的局裡就是砸自己的盤。
    "dynamic_basket": True,
    # 動物同理。跟 dynamic_basket 分開，才量得出兩邊各自的貢獻。
    "dynamic_animals": True,
    # 需求分配完之後，剩下的格子種這個。WHEAT 的 glut 曲線最平
    # （賣 400 個單價還有 $20），而且餵動物本來就要。
    "fallback_crop": "WHEAT",
    # 單一作物最多佔多少格。防止貪婪配置全押同一種 —— 全 MELON 的話
    # 前 10 天零收入（first_yield_day=10），工資先把現金燒光。
    "max_crop_share": 0.4,
    # 同上，單一動物species最多佔多少個建物。
    "max_animal_share": 0.5,
    # 估作物/動物價值時，從「當下的市場庫存」起算還是從中性的 MARKET_I0 起算。
    # 開著的話市場缺貨（價格高於 base）就會被視為機會，反之亦然。
    "market_aware_pricing": True,
    # 估價時往前看幾天的在途產量（自己的田 + **對手的田**）。
    # 0 = 只看當下庫存。市場共用，對手種一樣的東西時實際崩盤速度是兩倍。
    "supply_lookahead_days": 6,
    # 留幾格蓋動物建物。挑離 shed 最近的格子 —— 動物每天要餵、要收、要收肥，
    # 是全場最耗回合的東西，走路成本要壓到最低。
    "n_structures": 6,
    # 養什麼。GOOSE：interval=1 每天產、EGG 的 glut 曲線是 log（`above_target=0.20`），
    # 賣 400 個單價還有 $40，是唯一能長期大量出貨的動物產品。
    "animal": "GOOSE",

    # --- 人力 ---
    # 整局的人數上限。
    "max_hands": 12,
    # 但一個象限（25 格）最多只補到 6 個。人力跟土地綁在一起 —— 買了第二塊地
    # 才補更多，不然多出來的 hand 沒事做，工資還照付（fib 成本，第 13 個 $233）。
    # 實際上限 = min(max_hands, hands_per_quadrant × 已解鎖象限數)
    "hands_per_quadrant": 5,
    "hire_per_turn": 4,          # 每回合最多幾筆 HIRE，要留 order 額度給買賣

    # --- 物流 ---
    "wheat_carry": 4,            # 一趟補給拿幾個 WHEAT
    "wheat_reserve_days": 4,     # shed 至少留幾天份的飼料，不賣掉
    "use_fertilizer": True,      # 收到的肥料拿去施肥，還是純賣錢

    # --- 市場 ---
    # 種子存幾天份。目標量 = 該作物每天要種幾顆 × 這個天數，不是固定顆數 ——
    # 固定 6 顆的話光 STRAWBERRY($100) + MELON($80) 開局就要 $1,080。
    "seed_buffer_days": 2,
    # 現金底線 = 維持現有規模的日常開銷（雇工 + 飼料 + 種子）× 這個天數。
    # 低於底線就不買動物、不買地。
    "cash_reserve_days": 3,
    "sell_chunk": 40,
    "sell_price_frac": 0.55,     # 市價低於 base 的這個比例就不賣，等城鎮消耗拉回來
    "shed_force_sell": 0.75,     # shed 用量超過這個比例就無視價格照賣（溢出會被丟掉）
    # 剩幾天就無視價格清倉。期末只算現金，留在 shed 的東西一毛都不算。
    "liquidate_days_left": 2,
    "animal_reserve": 800,       # 買動物後要留多少現金

    # --- 擴張 ---
    # 土地本身不產出，是 farmer / hand 站上去做事才產出 —— 產能上限是 unit-turns 不是格數。
    # 所以買地前要先看人力：買下去之後的格數不能超過 (unit 數 × tiles_per_unit)。
    # 一個象限 25 格，所以 9 個 unit、tiles_per_unit=3 → 容量 27，剛好夠買第二塊。
    "buy_land": True,
    # 最早第幾天買地。day 0 買會同時少 $1,000、又讓要照顧的格數從 25 變 50，
    # 撐不到 day 3 的第一次 CARROT 收成（實測 0 勝 20 負、差 $17,402）。
    "land_first_min_day": 3,
    # 最多解鎖幾個象限（含一開始就有的 NW）。3 = 買 NE 和 SW，不買 SE。
    # 不設上限的話會三塊全買，100 格配 12 個 hand 根本顧不動 ——
    # 實測 0 勝 16 負、$9,828 vs $50,694。
    "max_quadrants": 3,
    # 剩餘天數少於這個就不買了。新的 25 格要有時間生產才回得了本 ——
    # day 23 花 $4,000 買 SE，剩 7 天連一輪 MELON（11 天）都跑不完。
    "land_min_days_left": 12,
}

# 任務優先序，數字小的先做。
# 「去 shed 拿東西」也是正式任務（FETCH_*），排在對應的使用任務前面 ——
# 第一版把補給當成「閒置 unit 的填充動作」，結果 unit 永遠不閒置，
# 買來的鵝整季躺在 shed 裡（8 隻只放進去 7 次，空雞舍長期 2~3 個）。
_PRI = {
    "FETCH_WHEAT": 0,           # 沒飼料就餵不了
    "FEED": 0,                  # 連續 2 天沒餵，動物跑掉，$300~$500 蒸發
    "WATER": 1,                 # 連續 2 天沒澆，作物變雜草
    "HARVEST": 2,               # max_held / max_yield 滿了就停產
    "COLLECT_FERTILIZER": 3,    # 免費資源，錯過就是當天沒了
    "CARE": 4,                  # 產出加成
    "FETCH_ANIMAL": 5,          # 躺在 shed 裡的動物是 $300~$500 的死資本
    "PLACE": 5,                 # 把動物放進空建物
    "BUILD": 6,                 # 蓋建物（免費，只花一回合）
    "FETCH_FERTILIZER": 7,
    "FERTILIZE": 7,
    "DIG": 8,                   # 清雜草
    "PLANT": 9,
}

#: 這些任務需要 unit 的 inventory 裡有東西，沒有的話指派了也是白做。
_NEEDS_ITEM = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}


# --------------------------------------------------------------------------
# 土地用途（純函式，同一格永遠同一種用途 —— 保證可重現）
# --------------------------------------------------------------------------

def shed_tiles(board):
    """shed 相鄰的四格，NWSE 順序。只有第一格在初始解鎖的 NW 象限裡。"""
    half = board // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def structure_tiles(board, n):
    """留給動物的格子：NW 象限裡離 shed 最近的 n 格，由近到遠排序。

    回傳有序 tuple —— 順序要固定，`animal_for` 靠索引決定哪一格養哪一種。
    """
    half = board // 2
    cx, cy = half - 1, half - 1
    ranked = sorted(
        (abs(x - cx) + abs(y - cy), y, x)
        for y in range(half)
        for x in range(half)
    )
    return tuple((x, y) for _d, y, x in ranked[:n])


def crop_for(x, y, basket):
    return basket[(y * 7 + x) % len(basket)]


def quadrant_for(x, y, board):
    """回傳座標所屬象限；跟引擎的 `_quadrant_of` 保持同一個切法。"""
    half = board // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def active_crop_tiles(tiles, board, struct_order, unlocked_quadrants, max_tiles=None):
    """挑出這一版策略實際打算維護的作物格。

    土地解鎖不等於人力足以管理。設定 `max_tiles` 時，從各已解鎖象限的內側
    （離 shed 最近處）輪流取格，避免新增象限永遠被 y/x 掃描順序餓死。
    已經種下但後來不在 active set 的作物仍會照顧到收成；只是收完不再補種。
    """
    struct_set = set(struct_order)
    groups = {q: [] for q in unlocked_quadrants}
    half = board // 2
    for y in range(board):
        for x in range(board):
            if tiles[y][x] == "LOCKED" or (x, y) in struct_set:
                continue
            q = quadrant_for(x, y, board)
            groups.setdefault(q, []).append((x, y))

    # 各象限都以靠近中央 shed 的格子優先；同距離固定 y/x，保證可重現。
    for q, coords in groups.items():
        inner_x = half - 1 if q.endswith("W") else half
        inner_y = half - 1 if q.startswith("N") else half
        coords.sort(key=lambda p: (abs(p[0] - inner_x) + abs(p[1] - inner_y), p[1], p[0]))

    total = sum(len(coords) for coords in groups.values())
    limit = total if max_tiles is None else max(0, min(total, int(max_tiles)))
    selected = set()
    # round-robin，而不是先塞滿 NW 再輪到 NE/SW/SE。
    queues = {q: list(coords) for q, coords in groups.items()}
    while len(selected) < limit:
        progressed = False
        for q in unlocked_quadrants:
            queue = queues.get(q) or []
            if queue and len(selected) < limit:
                selected.add(queue.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


# --------------------------------------------------------------------------
# 依當局 shop 決定種什麼
# --------------------------------------------------------------------------

def crop_cycle(crop):
    """回傳 (一輪佔用幾天, 一輪產幾個)。每天澆水、不施肥。

    - one-time：`yield_units` 從 1 起算，在窗口 [ws, max_yield_day] 內每次澆水 +1。
      收成還要過 `first_yield_day` 的門檻。佔用天數 = 收成當天 + 1。
    - ongoing：總共產 `max_yield` 次，佔用到最後一次產出。

    實測對得上（也對得上引擎 README 的 "Yield / tile / day"）：
    WHEAT 4/5 / CARROT 3/4 / MELON 6/11 / TOMATO 4/12 / STRAWBERRY 4/17
    """
    cd = CROPS[crop]
    if cd["ongoing"]:
        days = cd["first_yield_day"] + (cd["max_yield"] - 1) * cd["interval"] + 1
        return days, cd["max_yield"]
    ws = (cd["max_yield_day"] + 1) // 2
    full_at = ws + cd["max_yield"] - 2          # 產量長滿的 age
    harvest_age = max(cd["first_yield_day"], min(cd["max_yield_day"], full_at))
    units = min(cd["max_yield"], 1 + max(0, harvest_age - ws + 1))
    return harvest_age + 1, units


def yield_per_tile_day(crop, days_left):
    """一格這種作物，從現在到賽季結束平均每天產幾個。

    ⚠️ 不能用穩態速率。賽季只有 30 天，跑不完一輪就是 **0 產出** ——
    STRAWBERRY 一輪要 17 天，day 13 之後種下去完全沒收成。
    第一版用穩態算，STRAWBERRY 看起來 $28.2/格/天（排第二），
    結果從 day 3 起吃掉 19 格裡的 17 格，比固定的 basket 差 $15,188。

    照剩餘天數算完整輪數之後：

        MELON $100 > CARROT $24.5 > WHEAT $20 > STRAWBERRY $16 = TOMATO $16

    而且到賽季末，長週期作物會自動歸零、不再被分配到格子。
    """
    if days_left <= 0:
        return 0.0
    days, units = crop_cycle(crop)
    cycles = days_left // days
    return (cycles * units) / days_left if cycles else 0.0


def town_demand(obs, config):
    """城鎮每天吃掉每項產品幾個。

    shop 每 `townShopSellInterval` 回合吃一次它需求清單上的每項各 1 個
    （單一產品的店 ×2），town center 每 `townCenterSellInterval` 回合吃
    每項非 FERTILIZER 產品各 1 個。

    `unlocked_shops` 會有重複值 —— shop 是抽取有放回的，每個 instance 獨立消費。
    """
    cfg = config or {}
    turns_per_day = int(cfg.get("turnsPerDay", 24))
    shop_ticks = turns_per_day / max(1, int(cfg.get("townShopSellInterval", 4)))
    center_ticks = turns_per_day / max(1, int(cfg.get("townCenterSellInterval", 24)))

    demand = {item: 0.0 for item in PRODUCTS}
    for shop in obs["town"].get("unlocked_shops", []):
        products = SHOPS[shop]
        mult = 2 if len(products) == 1 else 1
        for item in products:
            demand[item] += shop_ticks * mult
    for item in PRODUCTS:
        if item != "FERTILIZER":
            demand[item] += center_ticks
    return demand


@functools.lru_cache(maxsize=16384)
def _avg_sell_price(crop, inv0, excess):
    """從**當下庫存** `inv0` 起，再倒 `excess` 個進市場的平均成交價。

    逐一單位重新報價，所以要取平均而不是看最後一筆。用 8 點取樣近似積分 ——
    直接用「一半庫存時的價格」當平均會高估，MELON 的曲線是 `sq`，
    早期版本就是因此把 19 格全配給 MELON。
    """
    if excess <= 0:
        return market_price(crop, inv0)
    return sum(market_price(crop, inv0 + int(excess * k / 8))
               for k in range(1, 9)) / 8.0


def incoming_supply(obs, days):
    """未來 `days` 天內，**雙方**的田和動物會產出多少（每個產品幾個）。

    市場是共用的，價格由兩邊產量相加決定。`obs["farms"]` 是 `shared: true`，
    對手每一格種什麼、種幾天了、動物放多久了全部看得到（看不到的只有他的
    shed、inventory 和種子），所以他未來幾天要倒多少貨是算得出來的。

    不算這個的話會低估供給：模型只看自己會不會超過城鎮消耗，對手種一樣的東西
    時實際崩盤速度是估計的兩倍。實測 MELON 期末 $7（base 250）就是這樣來的。
    """
    day = obs["day"]
    out = {item: 0.0 for item in PRODUCTS}
    for farm in obs["farms"]:
        for row in farm["tiles"]:
            for t in row:
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT":
                    crop = t["crop"]
                    cd = CROPS[crop]
                    age = day - t["planted_day"]
                    if cd["ongoing"]:
                        # 每 interval 天產一次，數落在窗口內的次數
                        for a in range(age, age + days + 1):
                            if (a >= cd["first_yield_day"]
                                    and (a - cd["first_yield_day"]) % cd["interval"] == 0):
                                out[crop] += 1
                    else:
                        harvest_age, units = crop_cycle(crop)
                        if age <= harvest_age <= age + days:
                            out[crop] += units
                elif "animal" in t:
                    a = ANIMALS[t["animal"]]
                    age = day - t["placed_day"]
                    active = days - max(0, a["first_yield_day"] - age)
                    if active > 0:
                        out[a["product"]] += active * (1 + a["interval"]) / a["interval"]
    return out


def _inv_key(obs, items, market_aware=True, bucket=25, lookahead=0):
    """把當下的市場庫存分桶當 cache key。

    ⚠️ 必須用**當下庫存**而不是 `MARKET_I0` 起算。城鎮一直在消耗，實測期末
    價格幾乎全部高於 base（STRAWBERRY base $120 → $307、WOOL $200 → $251），
    代表市場長期缺貨。用 base 估價值的話，模型會把漲到 $307 的草莓當成 $120，
    永遠不去補那些最值錢的缺口。
    """
    if not market_aware:
        return tuple((i, MARKET_I0) for i in items)
    inv = obs["market"]["inventory"]
    soon = incoming_supply(obs, lookahead) if lookahead > 0 else {}
    return tuple((i, int((inv[i] + soon.get(i, 0.0)) / bucket) * bucket)
                 for i in items)


@functools.lru_cache(maxsize=8192)
def _plan_basket(days_left, demand_key, inv_key, n_crop_tiles, fallback, max_share):
    """逐格分配：每次把下一格給「邊際價值最高」的作物。

    `demand_key` 是 (作物, 該作物的日消耗量) 的 tuple，來自這一局實際開出來的
    shop。純函式 + lru_cache —— 同一天同一組 shop 只算一次。

    ## 為什麼不是「配格子去滿足需求」

    第一版是「算出滿足需求要幾格，按價值高低分配」。錯在**效率低的作物需要更多
    格子才填得滿需求，反而搶走配額** —— STRAWBERRY 一格每天只產 0.24 個，
    要滿足 7 個/天的需求得配 30 格，於是它把 19 格全吃光。

    這一版改成算每一格的邊際價值：產量會超過城鎮消耗多少，超出的部分推高市場
    庫存、壓低成交價，直接用引擎的 `market_price` 去查。結果是

    - WHEAT / EGG 這種 `above_func = log` 的，超產也幾乎不掉價 → 一直有價值
    - STRAWBERRY / MELON 這種掉得凶的，配幾格之後邊際價值就掉到底 → 自動收手
    """
    demand = dict(demand_key)
    inv0 = dict(inv_key)
    ypd = {c: yield_per_tile_day(c, days_left) for c in CROPS}
    alloc = {c: 0 for c in CROPS}

    # 單一作物的佔比上限。沒有這條的話貪婪配置會把所有格子押在同一種上：
    # 全 STRAWBERRY（效率低、要很多格才填滿需求）或全 MELON（單價高，
    # 但 first_yield_day = 10，前 10 天零收入，工資先把現金燒光，期末只剩 $3,400）。
    cap = max(1, int(n_crop_tiles * max_share))

    def marginal(crop):
        """再給這個作物一格，那一格每天值多少錢。"""
        if ypd[crop] <= 0:          # 剩下的天數跑不完一輪，種了也是白種
            return 0.0
        if alloc[crop] >= cap:
            return 0.0
        produced = (alloc[crop] + 1) * ypd[crop] * days_left
        absorbed = demand.get(crop, 0.0) * days_left
        excess = max(0.0, produced - absorbed)
        return ypd[crop] * _avg_sell_price(crop, inv0[crop], int(excess))

    for _ in range(n_crop_tiles):
        best = max(CROPS, key=lambda c: (marginal(c), c))
        if marginal(best) <= 0:
            break
        alloc[best] += 1

    left = n_crop_tiles - sum(alloc.values())
    if left > 0:                    # 賽季尾聲全部跑不完一輪，挑週期最短的
        alloc[min(CROPS, key=lambda c: crop_cycle(c)[0])] += left

    basket = []
    for crop in sorted(CROPS, key=lambda c: (-alloc[c], c)):
        basket.extend([crop] * alloc[crop])
    return tuple(basket) if basket else (fallback,)


def animal_rate(species, days_left):
    """一隻這種動物，從放下去到賽季結束平均每天產幾個（每天餵 + 照顧）。

    產出規則（`_daily_refresh_animals`）：產出日拿 `1 + pending_care_bonus`，
    bonus 是「上次產出之後每天餵+照顧各累積 1」，所以每次產出 = `1 + interval` 個。

        GOOSE  interval 1 → 每天 2 個
        COW    interval 2 → 每 2 天 3 個
        SHEEP  interval 3 → 每 3 天 4 個

    首次產出在 `first_yield_day`，之前是零。跟作物一樣，賽季只剩幾天的時候
    買動物是純虧 —— GOOSE 要 4 天才回本第一顆蛋，SHEEP 要 6 天。
    """
    a = ANIMALS[species]
    if days_left <= a["first_yield_day"]:
        return 0.0
    events = (days_left - a["first_yield_day"]) // a["interval"] + 1
    return events * (1 + a["interval"]) / days_left


@functools.lru_cache(maxsize=4096)
def _plan_animals(days_left, demand_key, inv_key, n_structures, max_share):
    """哪一格養哪一種。跟作物同一套：逐格給邊際價值最高的species。

    動物的需求結構跟作物一樣受 shop 抽籤影響，而且更極端 ——
    YARN_STORE 和 PET_CAFE 是單一產品店，吃 2 倍。實測 WOOL 的日消耗量
    在不同 seed 之間從 1.0 到 23.3。
    """
    demand = dict(demand_key)
    inv0 = dict(inv_key)
    rate = {s: animal_rate(s, days_left) for s in ANIMALS}
    alloc = {s: 0 for s in ANIMALS}
    cap = max(1, int(n_structures * max_share))

    def marginal(species):
        if rate[species] <= 0 or alloc[species] >= cap:
            return 0.0
        product = ANIMALS[species]["product"]
        produced = (alloc[species] + 1) * rate[species] * days_left
        absorbed = demand.get(product, 0.0) * days_left
        excess = max(0.0, produced - absorbed)
        gross = rate[species] * _avg_sell_price(product, inv0[product], int(excess))
        # 飼料成本：每天 1 個 WHEAT。用 base 價估，實際自己種會更便宜。
        return gross - MARKET_PARAMS["WHEAT"]["base"]

    for _ in range(n_structures):
        best = max(ANIMALS, key=lambda s: (marginal(s), s))
        if marginal(best) <= 0:
            break
        alloc[best] += 1

    plan = []
    for species in sorted(ANIMALS, key=lambda s: (-alloc[s], s)):
        plan.extend([species] * alloc[species])
    return tuple(plan)


def animal_for(x, y, struct_order, plan):
    """這一格養哪一種。`struct_order` 是 `structure_tiles` 的固定順序。"""
    if not plan:
        return None
    try:
        idx = struct_order.index((x, y))
    except ValueError:
        return None
    return plan[idx] if idx < len(plan) else None


def species_for_structure(kind, plan_species):
    """空建物該放哪一種。

    plan 會隨 shop 變動，但建物蓋下去就固定了 —— COOP 放不了 COW。
    所以先看計畫的species合不合，不合就退回任何結構相符的。
    """
    if plan_species and ANIMALS[plan_species]["structure"] == kind:
        return plan_species
    for s in sorted(ANIMALS):
        if ANIMALS[s]["structure"] == kind:
            return s
    return None


def _days_left(obs, config):
    cfg = config or {}
    turns_per_day = max(1, int(cfg.get("turnsPerDay", 24)))
    total_days = int(cfg.get("episodeSteps", 720)) // turns_per_day
    return total_days - obs["day"]


def dynamic_basket(obs, config, params, n_crop_tiles):
    """依這一局實際開出來的 shop 決定每格種什麼。

    shop 是抽取有放回的，所以每局的需求結構差很多 —— 實測 CARROT 的日消耗量
    在不同 seed 之間從 1.0 到 18.0（seed 2 一間 PET_CAFE 和 FARMERS_MARKET
    都沒抽到）。固定的 basket 在那種局裡就是拿胡蘿蔔砸自己的盤。
    """
    demand = town_demand(obs, config)
    demand_key = tuple(sorted((c, round(demand[c], 3)) for c in CROPS))
    return _plan_basket(_days_left(obs, config), demand_key,
                        _inv_key(obs, sorted(CROPS), params["market_aware_pricing"],
                                 lookahead=params["supply_lookahead_days"]),
                        n_crop_tiles,
                        params["fallback_crop"], params["max_crop_share"])


def dynamic_animals(obs, config, params, n_structures):
    """依這一局實際開出來的 shop 決定每格養什麼。"""
    demand = town_demand(obs, config)
    products = sorted({ANIMALS[s]["product"] for s in ANIMALS})
    demand_key = tuple((p, round(demand[p], 3)) for p in products)
    return _plan_animals(_days_left(obs, config), demand_key,
                         _inv_key(obs, products, params["market_aware_pricing"],
                                  lookahead=params["supply_lookahead_days"]),
                         n_structures,
                         params["max_animal_share"])


# --------------------------------------------------------------------------
# 任務掃描
# --------------------------------------------------------------------------

def _fertilize_worth_it(tile, cd, day):
    """施肥只在「加成真的會生效的窗口」內才划算。

    肥效涵蓋 day / day+1 / day+2 三天（`fertilized_until_day = day + 2`）。
    - one-time 作物：加成在澆水時結算，窗口是 age ∈ [(max_yield_day+1)//2, max_yield_day]
    - ongoing 作物：加成在每日產出時結算，且**當天必須有澆水**
    """
    if tile.get("fertilized_until_day", -1) >= day:
        return False
    age = day - tile["planted_day"]
    if cd["ongoing"]:
        return age >= cd["first_yield_day"] - 1
    window_start = (cd["max_yield_day"] + 1) // 2
    return window_start - 1 <= age <= cd["max_yield_day"]


def _ongoing_produces_tonight(tile, cd, day):
    """ongoing 作物是否會在今天 EOD 結算下一次產出。"""
    next_age = day + 1 - tile["planted_day"]
    if next_age < cd["first_yield_day"]:
        return False
    offset = next_age - cd["first_yield_day"]
    if offset % cd["interval"] != 0:
        return False
    return offset // cd["interval"] + 1 <= cd["max_yield"]


def _needs_water(tile, cd, day, on_demand=False):
    """今天是否必須澆水。

    legacy 模式維持 Gen0 原行為。on-demand 模式只在以下情況澆：新種當天／
    再不澆今晚會成 WEED／one-time 增產窗口尚未滿產／ongoing 今晚有施肥產出。
    引擎的新苗 `consecutive_unwatered=1`，所以種下當天自然會命中第二條。
    """
    if tile["watered_today"]:
        return False
    age = day - tile["planted_day"]
    if not on_demand:
        return cd["ongoing"] or age <= cd["max_yield_day"]
    if tile.get("consecutive_unwatered", 1) >= 1:
        return True
    if not cd["ongoing"]:
        window_start = (cd["max_yield_day"] + 1) // 2
        return (
            window_start <= age <= cd["max_yield_day"]
            and tile.get("yield_units", 0) < cd["max_yield"]
        )
    return (
        tile.get("fertilized_until_day", -1) >= day
        and _ongoing_produces_tonight(tile, cd, day)
    )


def _tasks(
    tiles,
    day,
    board,
    params,
    struct_order,
    animal_plan,
    unit_inv,
    private,
    active_tiles=None,
):
    """掃出全盤待辦任務，回傳 [(優先序, op, x, y, arg)]。

    `arg`：FETCH_* 是 `(item, 數量)`，BUILD_* / PLACE 是要養的species，其餘 None。
    順序只取決於 (優先序, y, x)，不依賴 dict 迭代順序 —— 保證可重現。
    """
    out = []
    struct_set = set(struct_order)
    n_feed = n_fert = 0
    place_want = {}          # species -> 有幾個空建物在等它

    for y in range(board):
        for x in range(board):
            tile = tiles[y][x]
            if tile == "LOCKED":
                continue

            if tile is None:
                if (x, y) in struct_set:
                    species = animal_for(x, y, struct_order, animal_plan)
                    if species:
                        build_op = ("BUILD_COOP"
                                    if ANIMALS[species]["structure"] == "COOP"
                                    else "BUILD_PASTURE")
                        out.append((_PRI["BUILD"], build_op, x, y, species))
                elif active_tiles is None or (x, y) in active_tiles:
                    out.append((_PRI["PLANT"], "PLANT", x, y, None))
                continue

            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")

            if kind == "WEED":
                if active_tiles is None or (x, y) in active_tiles:
                    out.append((_PRI["DIG"], "DIG", x, y, None))

            elif kind == "PLANT":
                cd = CROPS[tile["crop"]]
                age = day - tile["planted_day"]
                if _needs_water(
                    tile, cd, day, on_demand=params.get("water_on_demand", False)
                ):
                    out.append((_PRI["WATER"], "WATER", x, y, None))
                elif tile["yield_units"] > 0 and age >= cd["first_yield_day"]:
                    out.append((_PRI["HARVEST"], "HARVEST", x, y, None))
                if params["use_fertilizer"] and _fertilize_worth_it(tile, cd, day):
                    out.append((_PRI["FERTILIZE"], "FERTILIZE", x, y, None))
                    n_fert += 1

            elif "animal" in tile:
                if not tile["fed_today"]:
                    out.append((_PRI["FEED"], "FEED", x, y, None))
                    n_feed += 1
                if tile["yield_units"] > 0:
                    out.append((_PRI["HARVEST"], "HARVEST", x, y, None))
                if tile["fertilizer_available"]:
                    out.append((_PRI["COLLECT_FERTILIZER"], "COLLECT_FERTILIZER", x, y, None))
                if not tile["cared_today"]:
                    out.append((_PRI["CARE"], "CARE", x, y, None))

            elif kind in ("COOP", "PASTURE"):
                # 空建物：等一隻動物。tile 有 "animal" key 才代表住了東西。
                # 建物蓋下去就固定了，COOP 放不了 COW —— 所以要挑結構相符的。
                species = species_for_structure(
                    kind, animal_for(x, y, struct_order, animal_plan))
                if species:
                    out.append((_PRI["PLACE"], "PLACE", x, y, species))
                    place_want[species] = place_want.get(species, 0) + 1

    # --- 補給任務 ---
    # 每筆 FETCH 讓一個 unit 去 shed 拿一趟。目標格用 shed 相鄰四格的第一格
    # 當距離代表值，實際走位在 `_task_action` 裡挑最近的那一格。
    sx, sy = shed_tiles(board)[0]
    shed = private["shed"]

    def _short(wanted, item):
        carrying = sum(1 for inv in unit_inv if inv.get(item, 0) > 0)
        return max(0, wanted - carrying)

    # 每種動物各自算：shed 裡有幾隻、有幾個相符的空建物在等。
    for species, wanted in sorted(place_want.items()):
        for _ in range(min(_short(wanted, species), shed.get(species, 0))):
            out.append((_PRI["FETCH_ANIMAL"], "FETCH_ANIMAL", sx, sy, (species, 1)))

    carry = params["wheat_carry"]
    have_wheat = shed.get("WHEAT", 0)
    need_feeders = _short(n_feed, "WHEAT")
    while need_feeders > 0 and have_wheat > 0:
        take = min(carry, have_wheat)
        out.append((_PRI["FETCH_WHEAT"], "FETCH_WHEAT", sx, sy, ("WHEAT", take)))
        have_wheat -= take
        need_feeders -= take

    if params["use_fertilizer"]:
        for _ in range(min(_short(n_fert, "FERTILIZER"), shed.get("FERTILIZER", 0))):
            out.append((_PRI["FETCH_FERTILIZER"], "FETCH_FERTILIZER", sx, sy, ("FERTILIZER", 1)))

    out.sort(key=lambda t: (t[0], t[3], t[2]))
    return out


# --------------------------------------------------------------------------
# 指派
# --------------------------------------------------------------------------

def _requirement(op, arg):
    """這個 op 需要 unit 的 inventory 裡有什麼。沒有就不能指派。

    `PLACE` 要的是那一格計畫養的species，寫在任務的 `arg` 裡 —— 每格可能不同。
    """
    if op == "PLACE":
        return arg
    return _NEEDS_ITEM.get(op)


def _assign(
    unit_pos,
    unit_inv,
    tasks,
    seeds,
    params,
    board=None,
    unlocked_quadrants=None,
):
    """任務 → unit。

    先按優先序分層，**每一層裡面先做離 unit 最近的**。第一版是按 (y, x) 掃描順序，
    結果一個遠處的任務會先搶走 unit，走路佔掉 74% 的回合。
    """
    assigned = {}
    free = set(range(len(unit_pos)))
    planted = {}
    basket = params["basket"]
    zoning = bool(params.get("quadrant_zoning", False) and board and unlocked_quadrants)
    quadrants = list(unlocked_quadrants or ())
    zone_penalty = int(params.get("zone_penalty", board or 0))

    # unit 0（farmer）固定從 NW 開始，其餘依 index 輪流分區。hands 每天也是
    # NW/NE/SW/SE 內角輪流出生，index 分區不需要跨回合狀態，且完全可重現。
    home = {
        u: quadrants[u % len(quadrants)]
        for u in range(len(unit_pos))
    } if zoning else {}

    def dist(u, tx, ty):
        return abs(unit_pos[u][0] - tx) + abs(unit_pos[u][1] - ty)

    def assignment_cost(u, task):
        _pri, _op, tx, ty, _arg = task
        cost = dist(u, tx, ty)
        zone_this_op = (
            not params.get("zone_planting_only", False)
            or _op in ("PLANT", "DIG")
        )
        if zoning and zone_this_op and home[u] != quadrant_for(tx, ty, board):
            cost += zone_penalty
        return cost

    tier = []
    current_pri = None

    def flush():
        """把同一優先序的任務按「離最近的閒置 unit 多遠」排序後再指派。"""
        tier.sort(key=lambda t: (
            min((assignment_cost(u, t) for u in free), default=0), t[3], t[2]
        ))
        for _pri, op, tx, ty, arg in tier:
            if not free:
                return
            if op == "PLANT":
                crop = crop_for(tx, ty, basket)
                # 引擎有原子驗證：同一回合某作物的 PLANT 請求總數超過種子數，
                # 該作物的**所有** PLANT 會全部變成 PASS。
                if planted.get(crop, 0) >= seeds.get(crop, 0):
                    continue
                planted[crop] = planted.get(crop, 0) + 1
            need = _requirement(op, arg)
            cands = [u for u in free if need is None or unit_inv[u].get(need, 0) > 0]
            if not cands:
                continue
            best = min(cands, key=lambda u: (assignment_cost(u, (_pri, op, tx, ty, arg)), u))
            free.discard(best)
            assigned[best] = (op, tx, ty, arg)
        tier.clear()

    for task in tasks:
        if not free:
            break
        if task[0] != current_pri:
            flush()
            current_pri = task[0]
        tier.append(task)
    flush()

    return assigned, free


# --------------------------------------------------------------------------
# 動作組裝
# --------------------------------------------------------------------------

def _step_toward(pos, target):
    x, y = pos
    tx, ty = target
    if tx > x:
        return ["EAST"]
    if tx < x:
        return ["WEST"]
    if ty > y:
        return ["SOUTH"]
    return ["NORTH"]


def _task_action(pos, target, op, arg, params, board):
    """組出 unit 的動作。

    ⚠️ 帶參數的 op 一定要把參數補上。引擎對 `["PLACE"]`（少了物品名）
    的處理是 `if len(action) < 2: return` —— **靜默忽略**，不會有任何錯誤。
    第一版就是漏了這個，整季 337 個回合白丟、6 個雞舍全空、shed 囤了 12 隻鵝。
    """
    if op.startswith("FETCH_"):
        # 走到 shed 相鄰四格的任何一格都能 PICKUP，挑最近的。
        # 走路不用避開 LOCKED —— 引擎允許走進去，只有「對這格做事」會被擋。
        spots = shed_tiles(board)
        if tuple(pos) in {tuple(s) for s in spots}:
            item, qty = arg
            return ["PICKUP", item, qty]
        nearest = min(spots, key=lambda s: abs(s[0] - pos[0]) + abs(s[1] - pos[1]))
        return _step_toward(pos, nearest)

    if tuple(pos) != tuple(target):
        return _step_toward(pos, target)
    if op == "PLANT":
        return ["PLANT", crop_for(target[0], target[1], params["basket"])]
    if op == "PLACE":
        return ["PLACE", arg]
    return [op]


# --------------------------------------------------------------------------
# 市場
# --------------------------------------------------------------------------

def _sell_qty(item, inv, avail, floor_price):
    """從庫存 `inv` 開始賣，賣到第幾個時單價會掉破 `floor_price`。

    價格對庫存單調遞減，所以二分搜尋。回傳可以賣的數量（0 ~ avail）。
    """
    if avail <= 0 or market_price(item, inv) < floor_price:
        return 0
    lo, hi = 0, avail
    while lo < hi:
        mid = (lo + hi + 1) // 2
        # 賣掉 mid 個之後，最後那一個的成交價
        if market_price(item, inv + mid - 1) >= floor_price:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _seed_burn_per_day(basket, n_crop_tiles):
    """維持所有格子都有東西種，每天要花多少錢買種子。

    一格每 `cycle` 天要重種一次，所以日耗 = 格數 / cycle。各作物差很多：
    CARROT 4 天一輪、種子 $20 → 每格每天 $5；STRAWBERRY 17 天一輪、$100 →
    每格每天 $5.9；MELON 11 天、$80 → $7.3。
    """
    if not basket or n_crop_tiles <= 0:
        return 0.0
    total = 0.0
    for crop in set(basket):
        share = basket.count(crop) / len(basket)
        days = crop_cycle(crop)[0]
        total += n_crop_tiles * share / days * CROPS[crop]["seed"]
    return total


def _market(
    obs,
    config,
    farm,
    private,
    params,
    shed_capacity,
    tasks,
    board,
    active_crop_count=None,
):
    orders = []
    shed = private["shed"]
    # 市場在 unit 動作之後執行；這裡用回合開始時的容量做保守上限。
    # PICKUP 可能在同回合多騰出一些空間，但絕不能反過來送出裝不下的訂單，
    # 否則引擎只成交前幾個、剩下的靜默中止。
    shed_room = max(0, shed_capacity - sum(shed.values()))
    prices = obs["market"]["prices"]
    money = farm["money"]

    # --- 現金底線 -----------------------------------------------------------
    # 底線 = **維持現有規模的日常開銷 × cash_reserve_days**。
    #
    # 第一版只算「明天的雇工 + 飼料」，day 0 一隻動物都還沒放（`n_animals = 0`）、
    # 只有 5 個 hand（fib 總和 $12），所以底線是 $12 —— 等於沒有。log 顯示它
    # 精準地把 $3,000 花到只剩 $12，day 1 hour 2 歸零，然後 day 2~10 現金 $0、
    # crew 0、雜草 8~11，整整 10 天空轉。
    #
    # 漏掉的是**種子的續買**：`seed_buffer: 6` 對五種作物各補到 6 顆，
    # 光 STRAWBERRY（$100）和 MELON（$80）開局就是 $1,080，而且收成後還會一直補。
    cap = min(params["max_hands"],
              params["hands_per_quadrant"] * len(farm["unlocked_quadrants"]))
    n_animals = sum(1 for t in tasks if t[1] in ("FEED", "CARE"))
    n_crop_tiles = (
        max(0, int(active_crop_count))
        if active_crop_count is not None
        else max(0, sum(1 for row in farm["tiles"] for t in row
                        if t != "LOCKED") - params["n_structures"])
    )
    seed_rate = _seed_burn_per_day(params["basket"], n_crop_tiles)

    daily_ops = (
        sum(hire_cost(i) for i in range(cap))          # 每天重雇一次
        + n_animals * prices["WHEAT"]                  # 每隻動物每天 1 個 WHEAT
        + seed_rate                                    # 維持所有格子有東西種
    )
    floor = daily_ops * params["cash_reserve_days"]
    spendable = max(0.0, money - floor)

    # 1) 買地。價格 1000/2000/4000，順序由引擎的 `LAND_ORDER` 決定。
    #
    #    **排在所有採購最前面**，連雇工都在它後面。它是唯一「錯過就沒了」的
    #    支出 —— 種子和動物下個回合再買價格一樣，但土地買得越晚，多出來的
    #    25 格能生產的天數就越少。
    #    四個條件：夠晚（有收入了）、夠早（新地來得及生產）、沒買滿、付得起。
    n_owned = len(farm["unlocked_quadrants"])
    n_extra = n_owned - 1
    next_is_fourth = n_owned == 3
    crop_count = sum(
        1 for row in farm["tiles"] for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "PLANT"
    )
    plant_backlog = sum(1 for task in tasks if task[1] == "PLANT")
    active_utilization = min(
        1.0,
        crop_count / max(1, n_crop_tiles),
    )
    fourth_ready = True
    if next_is_fourth and params.get("adaptive_fourth_land", False):
        fourth_ready = (
            obs["day"] >= params.get("fourth_min_day", params["land_first_min_day"])
            and active_utilization >= params.get("fourth_min_utilization", 0.8)
            and plant_backlog <= params.get("fourth_max_plant_backlog", 5)
        )
    if (
        params["buy_land"]
        and n_extra < len(LAND_ORDER)
        and n_owned < params["max_quadrants"]
        and fourth_ready
        and obs["day"] >= params["land_first_min_day"]
        and _days_left(obs, config) >= params["land_min_days_left"]
        and spendable >= LAND_PRICES[n_extra]
    ):
        orders.append(["BUY_LAND"])
        spendable -= LAND_PRICES[n_extra]

    # 2) 雇工。hands 每天結束會全部消失，所以每天重新雇。
    #    成本是 fib(當天已雇人數)：1, 1, 2, 3, 5, 8, 13, 21...
    #
    #    分批雇：一回合最多送 hire_per_turn 筆，不夠就下個回合繼續。
    #    `hire_per_turn` 存在是因為市場訂單一回合上限 10 筆，
    #    全部拿去雇工就沒額度買種子、買動物、賣貨了。
    want = cap - len(farm["hands"])
    for _ in range(max(0, min(want, params["hire_per_turn"]))):
        orders.append(["HIRE"])

    # 3) 補種子。種子不佔 shed，PLANT 直接從 private["seeds"] 扣。
    #    只買買得起的數量 —— 引擎對付不起的訂單是靜默失敗，全額送出的話
    #    會變成每回合重試、把現金刮到 0。
    #    存量目標照**種植速率**算，不是每種都固定 6 顆。長週期的作物種得慢，
    #    囤 6 顆 STRAWBERRY（$600）是把現金鎖死在倉庫裡好幾天。
    basket = params["basket"]
    for crop in sorted(set(basket)):
        share = basket.count(crop) / len(basket)
        per_day = n_crop_tiles * share / crop_cycle(crop)[0]
        target = max(1, int(per_day * params["seed_buffer_days"] + 0.5))
        need = target - private["seeds"].get(crop, 0)
        if need <= 0:
            continue
        price = CROPS[crop]["seed"]
        n = min(need, int(spendable // price))
        if n > 0:
            orders.append(["BUY_SEED", crop, n])
            spendable -= n * price

    # 4) 補飼料。斷糧 = 動物 2 天後跑掉，$300~$500 蒸發，所以排在買動物前面。
    #    WHEAT 的 glut 曲線最平（賣 400 個單價還有 $20），買回來也不會太貴。
    reserve = n_animals * params["wheat_reserve_days"]
    if reserve and shed.get("WHEAT", 0) < reserve // 2:
        need = reserve - shed.get("WHEAT", 0)
        n = min(need, shed_room, int(spendable // max(1, prices["WHEAT"])))
        if n > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", n])
            spendable -= n * prices["WHEAT"]
            shed_room -= n

    # 5) 買動物。最後才買 —— 一隻 $300~$500，是最容易把現金一次抽乾的支出。
    want = {}
    for t in tasks:
        if t[1] == "PLACE":
            want[t[4]] = want.get(t[4], 0) + 1
    budget = max(0.0, spendable - params["animal_reserve"])
    for species in sorted(want):
        cost = ANIMALS[species]["cost"]
        n_buy = min(
            want[species] - shed.get(species, 0),
            shed_room,
            int(budget // cost),
        )
        if n_buy > 0:
            orders.append(["BUY_ANIMAL", species, n_buy])
            budget -= n_buy * cost
            shed_room -= n_buy

    # 6) 賣貨。**賣到價格掉到門檻為止**，不是固定賣 40 個。
    #
    #    逐一單位重新報價，所以一次倒 40 個進去，後面幾個的成交價會比第一個
    #    低很多。曲線陡的（MELON 是 `sq`）賣 40 個價格砍半，平的（WHEAT 是
    #    `log`）賣 400 個才掉 $5 —— 用同一個數字對兩者都不對。
    #
    #    shed 快滿時無視門檻照賣：每日結算溢出的部分會被直接丟掉。
    #    兩種情況無視價格門檻照賣：
    #
    #    a) **今晚會裝不下**。每日結算把所有 unit 的 inventory 倒進 shed，
    #       裝不下的部分是 `del inv[item]` —— 直接消失。所以要看的是
    #       「shed 現有量 + 大家手上拿著的量」會不會超過 100，
    #       不是 shed 現在幾成滿。
    #    b) **賽季最後幾天**。`reward = farm["money"]`，留在 shed 裡的東西
    #       一毛都不算，價格再爛也比作廢好。
    inv = obs["market"]["inventory"]
    carried = sum(sum(i.values()) for i in private["inventories"])
    tonight = sum(shed.values()) + carried
    forced = (tonight > shed_capacity
              or sum(shed.values()) >= params["shed_force_sell"] * shed_capacity
              or _days_left(obs, config) <= params["liquidate_days_left"])
    for item in PRODUCTS:
        n = shed.get(item, 0)
        if item == "WHEAT":
            n -= reserve          # 飼料不賣
        n = min(n, params["sell_chunk"])
        if n <= 0:
            continue
        if forced:
            orders.append(["SELL", item, n])
            continue
        floor_price = params["sell_price_frac"] * MARKET_PARAMS[item]["base"]
        qty = _sell_qty(item, inv[item], n, floor_price)
        if qty > 0:
            orders.append(["SELL", item, qty])

    # 給 log 用：買地買不成的時候要分得出是「錢不夠」還是「條件沒過」。
    _market.last_budget = {
        "money": round(money),
        "floor": round(floor),
        "spendable": round(spendable),
        "daily_ops": round(daily_ops),
        "seed_rate": round(seed_rate),
        "hire_cap": cap,
        "shed_room": shed_room,
        "active_crop_count": n_crop_tiles,
        "active_utilization": round(active_utilization, 3),
        "plant_backlog": plant_backlog,
        "fourth_ready": fourth_ready,
    }
    return orders[:MAX_MARKET_ORDERS]


# --------------------------------------------------------------------------
# 進入點
# --------------------------------------------------------------------------

def act(obs, config=None, params=None):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    shed_capacity = DEFAULT_SHED_CAPACITY
    if config is not None:
        shed_capacity = int(config.get("shedCapacity", DEFAULT_SHED_CAPACITY))

    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]
    tiles = farm["tiles"]
    board = len(tiles)
    day = obs["day"]

    struct_order = structure_tiles(board, p["n_structures"])

    tiles_per_unit = p.get("tiles_per_unit")
    active_tiles = None
    if tiles_per_unit is not None:
        planned_hands = min(
            p["max_hands"],
            p["hands_per_quadrant"] * len(farm["unlocked_quadrants"]),
        )
        active_tiles = active_crop_tiles(
            tiles,
            board,
            struct_order,
            farm["unlocked_quadrants"],
            max_tiles=(1 + planned_hands) * float(tiles_per_unit),
        )
    active_crop_count = (
        len(active_tiles)
        if active_tiles is not None
        else sum(
            1 for y in range(board) for x in range(board)
            if tiles[y][x] != "LOCKED" and (x, y) not in set(struct_order)
        )
    )

    # basket 和動物配置每回合重算：shop 每 3 天才開一間，需求結構是會變的。
    # 已經種下 / 蓋好的格子不受影響，只有下一次 PLANT / BUILD 才會用到新的分配。
    if p["dynamic_basket"]:
        p["basket"] = dynamic_basket(obs, config, p, active_crop_count)

    if p["dynamic_animals"]:
        animal_plan = dynamic_animals(obs, config, p, len(struct_order))
    else:
        animal_plan = (p["animal"],) * len(struct_order)

    unit_pos = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
    inventories = private["inventories"]
    unit_inv = [dict(inventories[i]) if i < len(inventories) else {}
                for i in range(len(unit_pos))]

    tasks = _tasks(
        tiles, day, board, p, struct_order, animal_plan,
        unit_inv, private, active_tiles=active_tiles,
    )
    assigned, _idle = _assign(
        unit_pos,
        unit_inv,
        tasks,
        private["seeds"],
        p,
        board=board,
        unlocked_quadrants=farm["unlocked_quadrants"],
    )

    unit_actions = []
    for i, pos in enumerate(unit_pos):
        if i in assigned:
            op, tx, ty, arg = assigned[i]
            unit_actions.append(_task_action(pos, (tx, ty), op, arg, p, board))
        else:
            unit_actions.append(["PASS"])

    action = {
        "farmer": unit_actions[0],
        "hands": unit_actions[1:],
        "market": _market(
            obs,
            config,
            farm,
            private,
            p,
            shed_capacity,
            tasks,
            board,
            active_crop_count=active_crop_count,
        ),
    }

    if LOG_LEVEL >= 2:
        _log(obs, farm, private, tasks, assigned, action)
    return action


_LOG_FH = None
_LOG_FH_PATH = None


def _log_sink():
    """回傳要寫進去的 file object。

    `KAGGRI_LOG_FILE` 每一局會換路徑（`eval/runner.py` 設的），所以 handle 要
    在路徑變了的時候重開。開著不關、每筆 flush —— 多行程各寫各的檔，不會互相蓋。
    """
    global _LOG_FH, _LOG_FH_PATH
    path = os.environ.get("KAGGRI_LOG_FILE", LOG_FILE)
    if not path:
        import sys
        return sys.stderr
    if path != _LOG_FH_PATH:
        if _LOG_FH is not None:
            _LOG_FH.close()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _LOG_FH = open(path, "a", encoding="utf-8")
        _LOG_FH_PATH = path
    return _LOG_FH


def _log(obs, farm, private, tasks, assigned, action):
    import json

    counts = {}
    for task in tasks:
        counts[task[1]] = counts.get(task[1], 0) + 1
    rec = {
        # 多局的 log 合併成一個檔之後，靠 tag 分辨是哪一局哪一邊。
        # 引擎會把 seed 從 observation 抹掉，所以只能由 runner 從外面塞進來。
        "tag": os.environ.get("KAGGRI_LOG_TAG", ""),
        "player": obs["player"],
        "t": obs["day"] * 24 + obs["hour"],
        "day": obs["day"],
        "hour": obs["hour"],
        "cash": farm["money"],
        "budget": getattr(_market, "last_budget", None),
        "crew": len(farm["hands"]),
        # 已解鎖的象限。不能從 action 的 BUY_LAND 反推 —— 那是「送出訂單」，
        # 錢不夠的話引擎會靜默 return，成功和失敗看起來一樣。
        "quadrants": list(farm["unlocked_quadrants"]),
        "tiles_open": sum(1 for row in farm["tiles"] for t in row if t != "LOCKED"),
        "prices": dict(obs["market"]["prices"]),
        "seeds": {k: v for k, v in private["seeds"].items() if v},
        "shed": {k: v for k, v in private["shed"].items() if v},
        "carry": [{k: v for k, v in inv.items() if v} for inv in private["inventories"]],
        "task_counts": counts,
        "top5": [[t[1], [t[2], t[3]], t[0]] for t in tasks[:5]],
        "assigned": {str(i): list(j) for i, j in sorted(assigned.items())},
        "action": action,
    }
    fh = _log_sink()
    fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    fh.flush()


def agent(obs, config):
    """kaggle_environments 的進入點。"""
    return act(obs, config)
