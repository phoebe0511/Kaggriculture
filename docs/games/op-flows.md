# Op 流程速查

每個物件要下哪些 op、依什麼順序。規則細節與行號在 `engine-notes.md`。

動作格式：

```python
{
  "farmer": [op, *args],           # 主農夫，1 個動作
  "hands":  [[op, *args], ...],    # 每個雇工各 1 個動作
  "market": [[op, *args], ...],    # 市場訂單，一回合最多 10 筆
}
```

`market` 是遠端操作，不用人走過去。`farmer` / `hands` 一回合只能做一件事。
**非法動作靜默 no-op**，送出前要自己過 `legal_mask`。

shed 相鄰格 = **(4,4) (5,4) (4,5) (5,5)**。農夫每天早上重設到 (4,4)，hands 每天清空。

---

## 作物

```
market: ["BUY_SEED", crop, n]        # 種子進 private["seeds"]，不佔 shed

[移動到空的已解鎖格]
["PLANT", crop]                      # 消耗 1 顆種子

── 每天 ──
["FERTILIZE"]                        # 選用，消耗 inventory 裡 1 個 FERTILIZER，效期 3 天
["WATER"]                            # 一天限一次

["HARVEST"]                          # 產物進該 unit 的 inventory

market: ["SELL", crop, n]            # 從 shed 扣，不是從 inventory 扣
```

### 哪幾天要下 `WATER`

| | one-time（WHEAT / CARROT / MELON） | ongoing（TOMATO / STRAWBERRY） |
|---|---|---|
| `WATER` 的作用 | 窗口內澆水才加產量 | **不加產量**，純防雜草 |
| 加產窗口 | `age ∈ [(max_yield_day+1)//2, max_yield_day]` | — |
| 窗口外 | 隔天澆一次就夠（防雜草） | 隔天澆一次就夠 |
| `FERTILIZE` | 窗口內澆水時 +2 而非 +1 | 產出日 +2，但**該天必須有澆水** |
| `HARVEST` | 一次，收完 tile 變空 | 多次 |

各作物窗口與產量（實測，每天澆水）：

| crop | 種子 | 加產窗口 age | 可收成 age | 無肥 | 全程施肥 |
|---|---|---|---|---|---|
| WHEAT | 10 | 2~4 | ≥2 | 4 | 6 |
| CARROT | 20 | 2~3 | ≥2 | 3 | 4 |
| MELON | 80 | 6~12 | ≥10 | 6 | 6 |

ongoing 作物的產出日（實測，收成後立刻清空）：

| crop | 種子 | 產出日 | 無肥 | 每日施肥 |
|---|---|---|---|---|
| TOMATO | 50 | day 8, 9, 10, 11 | 各 1，共 4 | 各 2，共 8 |
| STRAWBERRY | 100 | day 10, 12, 14, 16 | 各 1，共 4 | 各 2，共 8 |

### 三種死法

- **種下當天沒 `WATER`** → 隔天變 WEED（新苗的 `consecutive_unwatered` 從 1 起算）
- **連續 2 天沒 `WATER`** → 變 WEED
- **過了 `max_lifespan_step` 沒收** → 每 2 步掉 1 個產量，歸零變 WEED

`HARVEST` 有 age 門檻：`age < first_yield_day` 時下 `HARVEST` 拿到 0 個
（實測：MELON 在 day 9 收成拿到 0）。

---

## 動物

### 一次性建置

```
market: ["BUY_ANIMAL", animal, 1]    # 進 shed

[移動到空的已解鎖格]
["BUILD_COOP"]  或  ["BUILD_PASTURE"]   # 免費，佔 1 回合

[移動到 shed 相鄰格]
["PICKUP", animal, 1]                # shed → inventory

[移動到那格空建物]
["PLACE", animal]                    # 放進去，開始計時
```

### 每天

```
market: ["BUY_PRODUCT", "WHEAT", n]  # 或自己種

[移動到 shed 相鄰格]
["PICKUP", "WHEAT", n]               # ★ FEED 從 inventory 扣，不是從 shed 扣

[移動到動物格]
["FEED"]                             # 消耗 inventory 裡 1 個 WHEAT
["CARE"]                             # 選用，累積 bonus
["HARVEST"]                          # 產物進該 unit 的 inventory
["COLLECT_FERTILIZER"]               # 每天 1 個

market: ["SELL", product, n]
```

| animal | 售價 | 建物 | 首次產出 | 之後間隔 | max_held | 產物 |
|---|---|---|---|---|---|---|
| GOOSE | 300 | COOP | 第 4 天早上 | 每天 | 4 | EGG |
| COW | 400 | PASTURE | 第 8 天早上 | 每 2 天 | 6 | MILK |
| SHEEP | 500 | PASTURE | 第 6 天早上 | 每 3 天 | 6 | WOOL |

- `max_held` 是**未收成產物**的上限，塞滿就停產 → 要定期 `HARVEST`
- **連續 2 天沒 `FEED` → 動物跑掉**，建物留著。剛放下的第一天不餵不會死
- `CARE` 的 bonus 只在**當天有餵**的產出日才用掉；首次產出前會一路累積
  → 每天餵+照顧的話，首次產出直接到 `max_held`
- `COLLECT_FERTILIZER` 跟 `CARE` 無關，動物活著每晚就會補上
- `DIG` **不能**移除有動物的建物

---

## 搬運與入庫

```
[移動到 shed 相鄰格]
["PICKUP", item, n]      # shed → inventory
["DROP"]                 # inventory 全部 → shed
["PLACE", item, n]       # inventory 裡指定 n 個 → shed
```

三種入庫方式的差別（`shedCapacity = 100`）：

| | 數量 | 裝不下時 |
|---|---|---|
| `DROP` | inventory 全部 | **溢出的丟掉** |
| `PLACE item n` | 指定 n 個 | 留在 inventory |
| 每日結算自動入庫 | 全部 | **溢出的丟掉** |

**收成品不需要手動搬** —— 每日結算會自動把所有 unit 的 inventory 存進 shed。
只有「當天就要賣」或「會超過 100」時才需要走一趟。

種子不佔 shed，存在 `private["seeds"]`，`PLANT` 直接消耗。

---

## 市場

```
market: ["SELL",        item,   n]   # 從 shed 賣
market: ["BUY_SEED",    crop,   n]   # → private["seeds"]
market: ["BUY_PRODUCT", item,   n]   # 只有 WHEAT / FERTILIZER 能買，→ shed
market: ["BUY_ANIMAL",  animal, n]   # → shed
```

一回合最多 10 筆，超過的靜默丟棄。錢不夠或 shed 滿了 → 該筆訂單中止。

**逐一個商品重新報價**，賣越多單價越低：

```
price(第 n 個) = base ± amp × f(n-1)        最低 $1
amp = above_target × base / f(T)
```

實測邊際單價：

| item | #1 | #10 | #50 | #100 | #200 | #400 |
|---|---|---|---|---|---|---|
| WHEAT | 25 | 23 | 22 | 21 | 21 | 20 |
| CARROT | 35 | 32 | 27 | 24 | 19 | 12 |
| TOMATO | 60 | 52 | 42 | 35 | 24 | 9 |
| STRAWBERRY | 120 | 103 | 26 | 1 | 1 | 1 |
| MELON | 250 | 249 | 226 | 152 | 1 | 1 |
| EGG | 50 | 46 | 43 | 42 | 41 | 40 |
| MILK | 160 | 141 | 57 | 1 | 1 | 1 |
| WOOL | 200 | 195 | 61 | 1 | 1 | 1 |
| FERTILIZER | 100 | 98 | 90 | 80 | 60 | 20 |

- **WHEAT / EGG 幾乎壓不下去**（`above_func = log`）→ 可以傾銷
- **STRAWBERRY 線性遞減 $1.92/個，第 63 個就到 $1**
- 價格 $1 時賣出**不再增加市場庫存** → 地板自我限制，但賣掉的拿不回來
- 價格靠城鎮消耗回升：每間 shop 對它需求清單上的每項產品**每天消耗 6 個**（單一產品店 12 個），
  town center 每天每項各 1 個
- **MELON 不在任何 shop 的需求清單**（只有 town center 每天 1 個），
  **FERTILIZER 連 town center 都不買** → 這兩項賣掉後價格幾乎回不來

---

## 雇工

```
market: ["HIRE"]                     # 無參數，一次雇一個
```

- 當天第 n 個花 `fib(n)` 元：`1, 1, 2, 3, 5, 8, 13, 21, 34, 55...`
  → 前 10 個總共 **$143**（起始資金 $3000）
- spawn 在 shed 相鄰四格中人最少的那格
- **每天結束全部消失**，隔天要重雇
- 能做**全部** unit 動作，跟主農夫一樣，各自有獨立 inventory
- 程式碼裡找不到人數上限

---

## 買地

```
market: ["BUY_LAND"]                 # 無參數，順序 hardcoded
```

初始只有 NW 25 格。順序 **NE ($1000) → SW ($2000) → SE ($4000)**，全開 $7000。
