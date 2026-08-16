# 遊戲引擎規則筆記（T00）

原始完整稽核來源：`kaggle-environments 1.32.6`
`site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`（1073 行）
+ `kaggriculture.json` + 核心的 `core.py` / `agent.py`。

**2026-08-16 更新**：專案與 L0 已固定使用 `kaggle-environments 1.32.7`。新版把
CARROT、TOMATO、EGG 的稀缺側價格函式改為 `hinge`；§6 的表格已更新。其餘舊行號
仍指 1.32.6 原始碼，待 T00 第二人複核時一併重標。

**標記規則**（見 `CLAUDE.md`）：
`VERIFIED` = 有原始碼行號或實測支撐；`UNVERIFIED` = 未經證實；`UNKNOWN` = 不知道。
標 UNVERIFIED 時要另外註明數值是怎麼來的（例如從 replay 反推）。
本文所有數字都是 VERIFIED，行號指 `kaggriculture.py`，另有標註者除外。

> ⚠️ 這份是**一個人讀出來的**。`division-of-labor.md` T00 要求兩人各自獨立讀完再對答案。
> 這份可以當作對答案的基準，但不能取代第二個人自己讀一遍。

---

## 1. 一回合的結構

### 動作格式（`kaggriculture.json` 第 131 行）

```python
{
  "farmer": [op, *args],           # 主農夫，1 個動作
  "hands":  [[op, *args], ...],    # 每個雇工 1 個動作
  "market": [[op, *args], ...],    # 市場訂單，最多 10 筆
}
```

**unit 動作**（farmer / hands 共用同一組，行 299-517）

| op | 參數 | 作用 |
|---|---|---|
| `NORTH` `SOUTH` `EAST` `WEST` | — | 移動 1 格 |
| `PASS` | — | 什麼都不做 |
| `PLANT` | crop | 種下（消耗 1 種子） |
| `WATER` | — | 澆水 |
| `FERTILIZE` | — | 施肥（消耗 inventory 裡 1 個 FERTILIZER） |
| `HARVEST` | — | 收成到該 unit 的 **inventory** |
| `DIG` | — | 清除作物 / 雜草 / 空的 COOP·PASTURE |
| `BUILD_COOP` `BUILD_PASTURE` | — | 蓋建物（**免費**） |
| `PLACE` | item [n] | 放動物到建物，或在 shed 旁存入指定數量 |
| `PICKUP` | item [n] | 從 shed 取出 |
| `DROP` | — | 在 shed 旁把 inventory **全部**存入 |
| `FEED` | — | 餵動物（消耗 inventory 裡 1 個 WHEAT） |
| `CARE` | — | 照顧動物 |
| `COLLECT_FERTILIZER` | — | 收集肥料 |

**市場動作**（行 618-636、行 559-568）

| op | 參數 | 作用 |
|---|---|---|
| `BUY_SEED` | crop, n | 買種子，進 `private["seeds"]` |
| `BUY_PRODUCT` | item, n | **只有 WHEAT / FERTILIZER 能買**（行 585） |
| `BUY_ANIMAL` | animal, n | 買動物，進 shed |
| `SELL` | item, n | 從 shed 賣出 |
| `HIRE` | — | 雇一個工（atomic，無參數） |
| `BUY_LAND` | — | 買下一塊地（atomic，順序固定） |

> **`unknowns.md` #1 解答**：BUY / SELL / HIRE 走 `action["market"]` 這條獨立通道，
> 跟 unit 動作完全分開。一回合最多 `maxMarketOrdersPerTurn = 10` 筆，超過的**靜默丟棄**
> （`kaggriculture.json` 第 23 行）。

**非法動作一律靜默 no-op**（行 300 docstring）—— 送出前必須自己過 `legal_mask`。

### 引擎每回合的處理順序（`interpreter`，行 881-952）

```
1. 雙方所有 unit 動作            行 900-926
   （先做 PLANT 原子驗證，見 §7）
2. _process_market             行 928   市場撮合
3. _town_consume               行 929   城鎮消耗 + 刷新價格
4. _decay_plants               行 930-931
5. if (step+1) % 24 == 0:
       _end_of_day             行 932-933
```

### 每日結算（`_end_of_day`，行 847-878）

每個玩家依序：

```
1. _daily_refresh_plants      行 862   澆水判定 → 枯萎判定 → ongoing 作物產出
2. _daily_refresh_animals     行 863   餵食判定 → 逃走判定 → 產出 → care bonus
3. _spawn_weeds               行 864   每個空的已解鎖格 0.5% 長雜草
4. _drop_inventories_to_shed  行 865   所有 unit 的 inventory 自動存進 shed，超過 100 的丟棄
5. farmer 位置重設為 (4,4)     行 866
6. hands 全部清空              行 867
7. hires_today 歸零            行 868
```

然後（不分玩家，共用）：每 `townShopUnlockInterval = 3` 天解鎖一間 shop（行 873-878）。

---

## 2. 網格與土地（`unknowns.md` #2 解答）

- `boardSize = 10` → 10×10 = 100 格，切成四個 5×5 quadrant（行 114-116）
- 象限判定：`half = 5`；`"N" if y < 5 else "S"` + `"W" if x < 5 else "E"`（y 向下增長）
- **初始只有 NW 解鎖 = 25 格可用**（行 144-145，實測確認）
- 買地順序**固定**：`NE → SW → SE`，價格 `1000 → 2000 → 4000`（行 83-84）
- 一個「區塊」= 一個 quadrant = 25 格；上限 = 4 個 quadrant = 100 格

### Shed（倉庫）

- **shed 不是 tile**，永遠不出現在 `tiles` 陣列裡（`tiles` 的值只有 `None` / `"LOCKED"` / dict）
- 「shed 相鄰」= 站在中央四格之一：**(4,4) (5,4) (4,5) (5,5)**（行 119-126，NWSE 順序）
- 這四格裡只有 (4,4) 屬於 NW，**其餘三格一開始是 LOCKED**
- 因此引擎特別讓 **LOCKED 格可以走進去**，且 `DROP`/`PICKUP`/`PLACE` 在 LOCKED guard 之前處理（行 315-317、326-329）
- 農夫初始位置 = (4,4)（行 148-153，實測確認）

### tile 的可能值

| 值 | 意義 |
|---|---|
| `None` | 空的已解鎖格 |
| `"LOCKED"` | 未購買 |
| `{"kind": "WEED"}` | 雜草，要 DIG |
| `{"kind": "PLANT", "crop": ..., ...}` | 作物 |
| `{"kind": "COOP"}` / `{"kind": "PASTURE"}` | 空建物 |
| `{"kind": "COOP", "animal": "GOOSE", ...}` | 有動物的建物 |

---

## 3. 作物（`unknowns.md` #5 解答）

`CROPS`，行 11-17：

| crop | 種子 | first_yield_day | max_yield_day | interval | max_yield | ongoing |
|---|---|---|---|---|---|---|
| WHEAT | 10 | 2 | 4 | 0 | 6 | ✗ |
| CARROT | 20 | 2 | 3 | 0 | 4 | ✗ |
| TOMATO | 50 | 8 | 8 | 1 | 4 | ✓ |
| STRAWBERRY | 100 | 10 | 10 | 2 | 4 | ✓ |
| MELON | 80 | 10 | 12 | 0 | 6 | ✗ |

### One-time 作物（WHEAT / CARROT / MELON）

產量**靠澆水累積**，在 `WATER` 當下結算（行 418-431）：

```python
window_start = (max_yield_day + 1) // 2
if window_start <= (day - planted_day) <= max_yield_day:
    yield_units += 2 if 施肥中 else 1        # 上限 max_yield
```

種下時 `yield_units = 1`（行 210）。**窗口外澆水不加產量**，但仍然防枯萎。

實測結果（每天澆水，在 `max_yield_day` 當天收成）：

| crop | 加產窗口（age） | 無肥 | 全程施肥 |
|---|---|---|---|
| WHEAT | 2~4（3 天） | **4** | **6**（capped） |
| CARROT | 2~3（2 天） | **3** | **4**（capped） |
| MELON | 6~12（7 天） | **6**（capped） | 6 |

`HARVEST` 後 tile 變回 `None`（行 454-455），要重種。

### Ongoing 作物（TOMATO / STRAWBERRY）

- **澆水不加產量**（行 425 明確排除 ongoing），澆水只防枯萎
- 產出在每日結算（行 776-789）：
  `days_since_first = next_day - planted_day - first_yield_day`，
  `>= 0` 且 `% interval == 0` 時 `yield_units += 1`（當天有澆水且施肥中則 `+2`）
- 累計 `production_count` 次數達 `max_yield` 後，設 `max_lifespan_step` 開始枯萎（行 788-789）
- 所以 TOMATO 最多產 4 次（day 8,9,10,11），STRAWBERRY 最多 4 次（day 10,12,14,16）

### 枯萎與衰減

- **`consecutive_unwatered >= 2` → 變 WEED**（行 770-771）
- 新種的作物 `consecutive_unwatered = 1`（行 209）
  → **種下當天一定要澆水**，否則當晚結算變 2，隔天就是雜草（實測確認）
- One-time 作物 `max_lifespan_step = (planted_day + max_yield_day + 1) * 24`（行 211）
  到期後每 2 步 `yield_units -= 1`，歸零變 WEED（行 739-753）

---

## 4. 動物（`unknowns.md` #9 解答）

`ANIMALS`，行 19-23：

| animal | 售價 | 建物 | first_yield_day | interval | max_held | 產物 |
|---|---|---|---|---|---|---|
| GOOSE | 300 | COOP | 4 | 1 | 4 | EGG |
| COW | 400 | PASTURE | 8 | 2 | 6 | MILK |
| SHEEP | 500 | PASTURE | 6 | 3 | 6 | WOOL |

每日結算（行 792-820）：

- `FEED` 從該 unit 的 **inventory** 扣 1 個 WHEAT（行 497）—— 不是從 shed 直接扣
- 有餵 → `consecutive_unfed = 0`；沒餵 → `+1`
- **`consecutive_unfed >= 2` → 動物逃走**，tile 變回空建物（行 804-806）
- 新放的動物 `consecutive_unfed = 0`（行 223）→ 第一天不餵也活著（實測確認）
- 產出：`days_since_first = next_day - placed_day - first_yield_day`，
  `>= 0` 且 `% interval == 0` → `yield_units += 1 + bonus`，上限 `max_held`
- `bonus` 只有**當天有餵**才會用掉 `pending_care_bonus`（行 812-813）
- 當天 `fed_today and cared_today` → `pending_care_bonus += 1`（行 816-817）
- **`fertilizer_available = True` 每晚無條件設**（行 818）—— 只要動物還活著，
  每天都能 `COLLECT_FERTILIZER` 拿 1 個肥料，跟有沒有 CARE 無關

> **實測發現**：care bonus 會在首次產出前一直累積。
> GOOSE 每天餵+照顧，day 0~3 累積 bonus 3，day 3 結算首次產出 `1 + 3 = 4`
> → 直接到 `max_held = 4`。`max_held` 是**未收成產物**的上限，收成後才能繼續累積。

---

## 5. 肥料

- **來源 1**：動物，每天 `COLLECT_FERTILIZER` 拿 1 個
- **來源 2**：市場 `BUY_PRODUCT FERTILIZER`（base $100）
- **用法**：站在作物上 `FERTILIZE`，消耗 inventory 裡 1 個
  → `fertilized_until_day = day + 2`，涵蓋 **day / day+1 / day+2 共 3 天**（行 467-468）
- **效果**：one-time 作物澆水時 `+2`（行 429）；ongoing 作物每日產出 `+2`，
  但**必須當天有澆水**（行 786）
- 也可以直接賣

---

## 6. 市場（`unknowns.md` #4 解答）

### 價格公式（行 179-193）

```python
price(inv) = base + amp * f_below(I0 - inv)     if inv <  I0    # 稀缺
           = base - amp * f_above(inv - I0)     if inv >= I0    # 供過於求
amp = target * base / f(T)
f ∈ {linear, sq, sqrt, log(=ln(1+x)), log10}
最後：max(PRICE_FLOOR=1, int(round(price)))
```

所有產品 `I0 = 10000`，初始庫存 = `I0` → 初始價格 = `base`（行 165-172）。

`MARKET_PARAMS`（行 41-51）：

| item | base | T | below_func | below_target | above_func | above_target |
|---|---|---|---|---|---|---|
| WHEAT | 25 | 400 | sqrt | 0.80 | log | 0.20 |
| CARROT | 35 | 450 | hinge | 1.00 | sqrt | 0.70 |
| TOMATO | 60 | 200 | hinge | 0.40 | sqrt | 0.60 |
| STRAWBERRY | 120 | 100 | sqrt | 0.70 | linear | **1.60** |
| MELON | 250 | 300 | log | 0.20 | sq | **3.60** |
| EGG | 50 | 332 | hinge | 0.40 | log | 0.20 |
| MILK | 160 | 122 | sqrt | 0.60 | linear | **1.60** |
| WOOL | 200 | 105 | log | 0.20 | sq | **3.20** |
| FERTILIZER | 100 | 200 | linear | 0.40 | linear | 0.40 |

### 實測：從 I0 連續賣出的邊際單價

| item | #1 | #10 | #50 | #100 | #200 | #400 |
|---|---|---|---|---|---|---|
| WHEAT | 25 | 23 | 22 | 21 | 21 | 20 |
| CARROT | 35 | 32 | 27 | 24 | 19 | 12 |
| TOMATO | 60 | 52 | 42 | 35 | 24 | 9 |
| STRAWBERRY | 120 | 103 | **26** | **1** | 1 | 1 |
| MELON | 250 | 249 | 226 | 152 | **1** | 1 |
| EGG | 50 | 46 | 43 | 42 | 41 | 40 |
| MILK | 160 | 141 | **57** | **1** | 1 | 1 |
| WOOL | 200 | 195 | **61** | **1** | 1 | 1 |
| FERTILIZER | 100 | 98 | 90 | 80 | 60 | 20 |

累計收入：

| item | 賣10 | 賣50 | 賣100 | 賣200 |
|---|---|---|---|---|
| WHEAT | 237 | 1,127 | 2,193 | 4,293 |
| CARROT | 328 | 1,482 | 2,738 | 4,832 |
| TOMATO | 550 | 2,411 | 4,318 | 7,221 |
| STRAWBERRY | 1,113 | 3,648 | **3,847** | 3,947 |
| MELON | 2,498 | 12,098 | 21,721 | **26,527** |
| EGG | 474 | 2,244 | 4,371 | 8,510 |
| MILK | 1,506 | 5,430 | **6,205** | 6,305 |
| WOOL | 1,983 | 7,655 | **7,969** | 8,069 |
| FERTILIZER | 991 | 4,755 | 9,010 | 16,020 |

**WHEAT 和 EGG 的價格幾乎不動**（`above_func = log`，`above_target = 0.20`）
—— 可以無限量傾銷。**STRAWBERRY / MILK / WOOL 大約 60~80 個就打到 $1**。

### 撮合機制（行 570-615）

- **逐一個商品撮合**：每賣 1 個就用當下庫存重新報價
- **兩人 lockstep**：同一個商品的每一件上，雙方看到相同的 pre-commit 庫存，同價成交後才更新（行 599-605）
- `BUY_PRODUCT` 報價用 `inventory - 1`（行 588），讓 buy/sell round-trip 淨值為 0
- **價格 == $1 時賣出不增加市場庫存**（行 645-647）→ 地板會自我限制，
  但賣掉的東西也拿不回來
- 錢不夠 / shed 滿了 → 該筆訂單中止（行 609-610）

### 城鎮需求（行 715-736）

- **Shop**：每 `townShopSellInterval = 4` 回合一次 → **一天 6 次**
  每個 shop instance 對它的每個產品各消耗 1（單一產品的 shop 消耗 2 倍）
  → 一個 shop instance 對一項產品的日需求 = **6 個**（單一產品 shop = 12 個）
- **Town center**：每 `townCenterSellInterval = 24` 回合一次 → **一天 1 次**，
  每個非 FERTILIZER 產品各 1 個。整季固定不變（1.32.6 移除了舊版的 day 10/20 加乘）
- Shop 每 3 天解鎖一間，**抽取有放回**，總數上限 8（行 105、873-878）
  → 可能開出三間 BAKERY、一間 YARN_STORE 都沒有

`SHOPS`（行 90-99）：

| shop | 需求產品 |
|---|---|
| BAKERY | EGG, WHEAT |
| PIZZA_SHOP | MILK, TOMATO, WHEAT |
| BRUNCH_SPOT | EGG, WHEAT, STRAWBERRY |
| YARN_STORE | WOOL（單一 → 2 倍） |
| ICE_CREAM_SHOP | STRAWBERRY, MILK, WHEAT |
| PET_CAFE | CARROT（單一 → 2 倍） |
| SMOOTHIE_SHOP | STRAWBERRY, MILK |
| FARMERS_MARKET | WHEAT, CARROT, TOMATO, STRAWBERRY |

> 🔍 **MELON 和 FERTILIZER 不在任何 shop 的需求清單裡。**
> MELON 只有 town center 每天買 1 個，FERTILIZER 連 town center 都不買
> （`TOWN_CENTER_PRODUCTS` 排除 FERTILIZER，行 101）。
> 意思是**這兩項賣掉之後庫存幾乎不會恢復，價格不會回升**。
> MELON 又是 `above_func = sq` + `above_target = 3.60`（全場最凶），
> 對照上表：賣到第 200 個就是 $1。

---

## 7. 雇工（`unknowns.md` #7 解答）

- `HIRE` 是 market 動作，atomic、無參數（行 563-565）
- 成本 = `farmHandCostMult × fib(當天已雇人數)`，fib 從 1,1,2,3,5,8,13...（行 677-686）
- 預設 `farmHandCostMult = 1`

實測成本序列：

```
第 1~12 個：1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144
前 10 個累計：$143
前 20 個累計：$17,710
```

- **每天結束所有 hands 全部消失**（行 867），`hires_today` 歸零（行 868）→ **按日雇用**
- Hand spawn 在 shed 相鄰四格中人最少的那一格，NWSE 破平手（行 520-528）
- **程式碼裡找不到人數上限** —— 只受金錢和 fib 成長限制
- Hand 能做**全部** unit 動作，跟主農夫完全一樣（行 922-926 走同一個 `_apply_unit_action`）
- 每個 hand 有自己獨立的 inventory（行 279-283）

> 前 10 個 hand 只要 $143，起始資金 $3000。**雇工便宜到不合理，很可能是主要策略軸線。**
> 這條需要在第 0 代 AI 上實測驗證。

---

## 8. 時限（`unknowns.md` #3 解答）

| 參數 | 值 | 來源 |
|---|---|---|
| `actTimeout` | **1 秒**/回合 | `kaggriculture.json` 第 9 行 |
| `remainingOverageTime` | **60 秒**/局 | `kaggriculture.json` 第 128 行 |
| `runTimeout` | 1200 秒/局 | `core.py` 第 326 行 |
| `episodeSteps` | 720 | `kaggriculture.json` 第 8 行 |

時間單位是**秒**：`core.py` 第 631 行用 `perf_counter()` 的 duration 減 `actTimeout`；
`helpers.py` 第 293 行的 docstring 明講 `remainingOverageTime` 是 "banked time (seconds)"。

機制（`core.py` 631-632、`agent.py` 220-222）：

```python
overage_consumed = max(0, duration - actTimeout)     # 每回合超出 1 秒的部分
remainingOverageTime -= overage_consumed             # 從 60 秒銀行扣
if duration - actTimeout > remainingOverageTime:     # 銀行不夠
    → DeadlineExceeded → status TIMEOUT → reward = None
```

**對 anytime 搜尋的意義**：每回合預算 1 秒，全局可透支 60 秒。
720 回合平均只能多用 0.083 秒/回合。實務上 budget 要設在 1 秒以下並留 margin。
**TIMEOUT 的 reward 是 `None`，不是 0** —— 整局作廢。

---

`hinge` 在 `x <= T` 時等同 `x/T`；超過 T 後增加二次項
`8 × max(0, x/T - 1)²`，讓真正短缺時價格快速上升。

## 9. Rating（已確認）

引擎只提供 `reward = 期末現金`；排行榜 rating 由 Kaggle 平台計算。官方 Evaluation
已確認：rating 只看勝／負／和，現金差額不影響 rating 變化；最終排行榜會對累積
episodes 跑 Bradley–Terry tournament。

---

## 10. 會咬人的細節（實作時要特別注意）

1. **農夫每天早上被重設到 (4,4)**（行 866）—— 排程演算法不能假設跨日位置連續。
2. **hands 每天全部消失**（行 867），inventory 會先存進 shed（行 865 在 867 之前）。
3. **收成品每天結算自動入庫**（行 865）—— 不需要手動搬回 shed，
   除非當天就要賣，或會超過 `shedCapacity = 100`（**超過的部分直接丟棄**，行 830-844）。
   `docs/README.md` 已依原始碼修正這一點。
4. **種下當天沒澆水 → 隔天變雜草**（行 209 + 770，實測確認）。
5. **ongoing 作物澆水不加產量**，只防枯萎（行 425）。
6. **`BUILD_COOP` / `BUILD_PASTURE` 免費**（行 480-490，沒有任何扣錢邏輯），只花一個回合。
7. **`DIG` 不能移除有動物的建物**（行 474-476）。
8. **種子不佔 shed**，存在 `private["seeds"]`，`PLANT` 直接消耗（行 354-355、行 664）。
9. **PLANT 原子驗證**（行 907-920）：同一回合所有 unit對某作物的 PLANT 請求總數
   超過手上種子數 → 該作物的**所有** PLANT 請求全部變成 `PASS`。
10. **`FEED` 從 inventory 扣 WHEAT，不是從 shed 扣** —— 餵動物前要先 `PICKUP WHEAT`。
11. **reward 在 `step >= episodeSteps - 2`（即 step 718）就寫入**（行 947-950）。
12. **對手的 shed 和 inventory 看不到**（`kaggriculture.json` 第 93 行）：
    `farms` 是公開的（tiles、money、農夫位置、已解鎖象限、hires_today），
    `private`（shed / inventories / seeds）**只有自己看得到**。
    → 對手的錢看得到，但庫存看不到。
