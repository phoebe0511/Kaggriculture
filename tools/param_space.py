"""CMA-ES 的搜尋空間宣告 + 編解碼 —— 連續向量 <-> `gen0.DEFAULT_PARAMS` 的 51 項。

    python -m tools.param_space          # 印出維度、凍結清單、round-trip 檢查

    from tools.param_space import KEYS, encode, decode, x0
    params = decode(x)                   # 完整 51 個 key，可直接當 spec 的 params

## 設計

**搜尋空間是資料不是程式。** 要把某個 key 移進/移出搜尋，改 `SEARCH_SPACE`
一行就好，不用動任何邏輯。維度 = `len(SEARCH_SPACE)`。

CMA-ES 在**正規化空間** `[0, 1]^d` 裡搜，所以 `bounds` 固定 `[0, 1]`、`sigma0`
對所有維度同一個值。真實界線由 `SEARCH_SPACE` 的 `(lo, hi)` 決定。

🩸 **`KEYS` 的順序就是向量順序，不要重排。** 重排會讓已經存下來的 pickle 和
每代的 JSONL 記錄對不上，而且不會報錯。

## 為什麼 bool / tuple / str / dict 全部凍結（2026-08-25 的決定）

- **3 個動態開關**（`dynamic_basket` / `dynamic_animals` / `market_aware_pricing`）
  關掉會讓 `basket` / `animal` 兩個參數重新生效 —— 搜尋空間會自己變形。
  而且 `dynamic_basket: True` 時 `params["basket"]` 在 `gen0.py:2326` 就被整個
  覆寫，凍結它零損失。
- **3 個已經量過的**：`optimal_assignment`（+$8,686）、`avoid_last_hour_planting`
  （+$4.2k）、`sell_before_spending`（−$1,064）。數字來自 `agents/gen0.py` 的參數
  註解，不是這裡量的。
- **5 個沒定論的**（`use_fertilizer` / `buy_land` / `fertilize_one_time` /
  `water_on_demand` / `sell_same_turn_returns`）留到 Stage 3 逐個翻轉。放進
  CMA-ES 要用閾值化，會產生平台區、拿不到方向資訊。
- `crop_oversupply` 是逐作物覆寫的 dict，還沒展開。

## dict 型參數怎麼展開（2026-09-01）

`crop_share` 是 `{作物: 上限}` 的覆寫表，整包塞不進連續空間。做法是在
`SEARCH_SPACE` 用帶點的 key —— `"crop_share.WHEAT"` —— 一個作物一個維度，
`encode` / `decode` 認得那個點。**`agents/gen0.py` 一行都不用改**：decode 出來
還是同一個 dict，只是五種作物全都列出來了。

🩸 五種全列出來之後 `max_crop_share` 在 `_greedy_basket` 就再也讀不到
（`share.get(c, max_share)` 永遠命中），所以它**從 `SEARCH_SPACE` 移出去** ——
留著會是一個對目標函數完全沒有影響的維度，CMA-ES 得花代數才學得會忽略它。
它仍然留在 `DEFAULT_PARAMS`（`_adaptive_crop_shares` 和舊的凍結 config 還讀它）。

「沒列到的作物走哪個值」由 `DICT_FALLBACK` 宣告，那是 `agents/gen0.py` 既有的
語意，不是這裡新訂的規則 —— 所以 `encode(舊 config)` 會落在跟它行為相同的點上。

## 界線怎麼訂

**不套統一公式。** 每一項在 `SEARCH_SPACE` 裡都有一行理由。有硬上界的一律從引擎
`import` 出來動態算（`docs/CLAUDE.md`：規則常數一律 import 引擎、不做鏡像；本機
引擎原始碼被人手改過，常數不能當永久事實）。

🩸 **`decode()` 只保證「合法」，不保證「合理」**（型別對、在界內）。荒謬的組合由
目標函數自己淘汰 —— 不要在這裡加啟發式修正，那等於偷偷限制搜尋空間而且不會報錯。
"""
from __future__ import annotations

import json
from pathlib import Path

from tools._quiet import silenced

with silenced():          # open_spiel 掃遊戲清單會噴 336 行到 stderr
    from kaggle_environments.envs.kaggriculture import kaggriculture as _engine

from agents.gen0 import DEFAULT_PARAMS

# --------------------------------------------------------------------------
# 引擎常數（動態讀，不鏡像）
# --------------------------------------------------------------------------

_SCHEMA = json.loads(
    Path(_engine.__file__).with_name("kaggriculture.json").read_text(encoding="utf-8")
)


def _cfg(name):
    """引擎 configuration 的預設值。有些欄位是純量，有些是帶 default 的 schema。"""
    raw = _SCHEMA["configuration"][name]
    return raw.get("default") if isinstance(raw, dict) else raw


#: 一季幾天。實測 720 / 24 = 30。
DAYS = _cfg("episodeSteps") // _cfg("turnsPerDay")
#: 象限總數 = 一開始就有的 NW + 買得到的。實測 LAND_ORDER = ['NE','SW','SE'] -> 4。
QUADRANTS = 1 + len(_engine.LAND_ORDER)
#: 每回合的 order 額度。`hire_per_turn` 的註解說 HIRE 跟買賣共用這個額度。
MAX_ORDERS = _cfg("maxMarketOrdersPerTurn")
#: 一個象限幾格。boardSize 10 = 四個 5x5。
TILES_PER_QUADRANT = (_cfg("boardSize") // 2) ** 2
#: 起始現金，當 `animal_reserve` 的上界。
STARTING_MONEY = _cfg("startingMoney")

# --------------------------------------------------------------------------
# 搜尋空間
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# dict 型參數的展開
# --------------------------------------------------------------------------

#: 展開維度「沒列到的項目」實際會走的那個全域參數。
#: 🩸 這兩條是 `agents/gen0.py` 既有的語意（`share.get(c, max_share)`、
#: `over_by_crop.get(c, params_oversupply)`），照抄過來的，不是這裡新訂的。
#: 抄錯的話 `encode(舊 config)` 會落在跟那份 config 行為不同的點上，
#: 而暖啟動整個前提就是「起點等於現在出貨的那組參數」。
DICT_FALLBACK = {
    "crop_share": "max_crop_share",
    "crop_oversupply": "oversupply_factor",
}


def split_key(key):
    """`"crop_share.WHEAT"` -> `("crop_share", "WHEAT")`；平的 key -> `(key, None)`。"""
    base, _dot, sub = key.partition(".")
    return base, (sub or None)


def read_key(src, key):
    """一份 params 在這個維度上現在的值。展開維度沒列到就走 `DICT_FALLBACK`。"""
    base, sub = split_key(key)
    if sub is None:
        return src[base]
    table = src.get(base) or {}
    if sub in table:
        return table[sub]
    return src[DICT_FALLBACK[base]]


def write_key(dst, key, value):
    """就地寫回。展開維度會**換一個新的 dict**，不會動到 `DEFAULT_PARAMS`
    裡的那份 —— `decode` 只做淺拷貝，直接改巢狀 dict 會污染全域預設值。"""
    base, sub = split_key(key)
    if sub is None:
        dst[base] = value
        return
    table = dict(dst.get(base) or {})
    table[sub] = value
    dst[base] = table


#: key -> (lo, hi, kind)。kind 是 "int" 或 "float"。順序即向量順序。
SEARCH_SPACE = {
    # --- 作物與市場配置 ---
    # 逐作物的格數上限（`crop_share` 展開）。0.2 已接近「五種平均分」，
    # 1.0 = 允許全押一種。取代了原本的全域 `max_crop_share`，理由見 docstring。
    #
    # 🩸 **下界 0.0 不等於「完全不種」**：`cap = max(1, int(格數 × share))`
    # 有一格的地板，所以 0.0 的作物仍然配得到 1 格。真正的差別在補位那段 ——
    # `pool = [c for c in CROPS if share.get(c, max_share) > 0]`，share=0 的
    # 作物不會被拿去填賽季尾聲跑不完一輪的空格（2026-09-01 量到：CARROT
    # 設 0 之後配額掉到 1 格，但補位那行仍把剩下的 43 格全倒給它）。
    #
    # 為什麼要逐作物：五種的超產曲線差三個數量級，同一個上限對它們的意義
    # 完全不同（`above_func`/`above_target`）——
    #   WHEAT      log/0.2     賣 400 個單價還有 $20，幾乎壓不垮
    #   CARROT     sqrt/0.7
    #   TOMATO     sqrt/0.6    base $60、first_yield_day 8
    #   STRAWBERRY linear/1.6  2026-09-01 量到超產 +50 個就從 $120 掉到 $1
    #   MELON      sq/3.6      base $250 最高，但城鎮日需求固定 1.0
    "crop_share.WHEAT": (0.0, 1.0, "float"),
    "crop_share.CARROT": (0.0, 1.0, "float"),
    "crop_share.TOMATO": (0.0, 1.0, "float"),
    "crop_share.STRAWBERRY": (0.0, 1.0, "float"),
    "crop_share.MELON": (0.0, 1.0, "float"),
    # 配額最多是「填滿城鎮日需求所需格數」的幾倍。1.0 = 剛好填滿、不超產。
    "oversupply_factor": (0.5, 6.0, "float"),
    # 單一動物 species 佔多少建物。同 max_crop_share。
    "max_animal_share": (0.1, 1.0, "float"),
    # 估價往前看幾天的在途產量。0 = 只看當下庫存（參數註解明說 0 有意義）。
    # 上界取半季 —— 看超過半季的在途量沒有意義，季末會清倉。
    "supply_lookahead_days": (0, DAYS // 2, "int"),

    # --- 人力 ---
    # 人數上限。4 = farmer + 3，再低連一個象限都顧不動；上界 24 = 現值兩倍，
    # 雇用成本是 Fibonacci，再高必然虧。
    "max_hands": (4, 24, "int"),
    # 一個象限配幾個人。25 格 / tiles_per_unit 上界 8 -> 約 3~4 人就飽和，
    # 上界給 12 留餘裕。
    "hands_per_quadrant": (2, 12, "int"),
    # 邊際門檻鬆緊。1.0 = 淨產值打平就雇；低於 0.5 等於不看價格。
    "hire_margin": (0.5, 3.0, "float"),
    # 每回合最多幾筆 HIRE。上界 = 每回合 order 額度（參數註解說兩者共用）。
    "hire_per_turn": (1, MAX_ORDERS, "int"),

    # --- 物流 ---
    # 一趟補給拿幾個 WHEAT。上界 12 是一個 unit 合理的攜帶量。
    "wheat_carry": (1, 12, "int"),
    # shed 留幾天份飼料不賣。0 = 全賣。上界取一季的三分之一。
    "wheat_reserve_days": (0, DAYS // 3, "int"),

    # --- 現金與買賣 ---
    # 種子存幾天份。0 會變成完全不備種子、直接讓 policy 失效，所以下界 1。
    "seed_buffer_days": (1, 8, "int"),
    # 一次性缺口要補幾成。0 = 只看流量（舊行為），1.0 = 現在等著種的每一格
    # 都先把種子買齊。上界 1.5 留一點餘裕給「這一批種下去之前又空出幾格」，
    # 再高就是把現金鎖在倉庫裡。
    #
    # 🩸 單獨開這個實測**更差**（2026-08-28，0.5/1.0/1.5 三版都輸）。放進
    # 這裡不是因為它單獨有效，而是因為它跟 `seed_buffer_days` /
    # `cash_reserve_days` / `animal_spend_frac` 搶的是同一筆開局現金 ——
    # 那三個是 CMA-ES 在 `seed_backlog=0` 的前提下調出來的。
    "seed_backlog": (0.0, 1.5, "float"),
    # 現金底線 = 日常開銷 x 這個天數。0 = 不留底線。
    "cash_reserve_days": (0, 8, "int"),
    # 單一品項每筆賣多少。上界 120 已超過 shedCapacity(100)，等於「不設限」。
    "sell_chunk": (5, 120, "int"),
    # 市價低於 base 的這個比例就不賣。0 = 永遠賣；1.2 = 只在超過 base 才賣。
    "sell_price_frac": (0.0, 1.2, "float"),
    # shed 用量超過這個比例就無視價格照賣。低於 0.3 等於一直在賤賣。
    "shed_force_sell": (0.3, 1.0, "float"),
    # 剩幾天無視價格清倉。上界取一季的三分之一。
    "liquidate_days_left": (1, DAYS // 3, "int"),
    # 買動物後要留多少現金。上界 = startingMoney，再高等於永不買動物。
    "animal_reserve": (0, STARTING_MONEY, "int"),
    # 一天最多把現金的多少比例拿去買動物。
    "animal_spend_frac": (0.0, 1.0, "float"),

    # --- 擴張 ---
    # 最早第幾天買地。0 = day 0 就買（08-24 實測單獨改成 0 是 0/60 全輸，但那是
    # 單參數變動；聯合最佳化下不預先排除）。上界取半季。
    "land_first_min_day": (0, DAYS // 2, "int"),
    # 最多解鎖幾個象限（含一開始就有的）。硬上界 = 引擎的象限總數。
    "max_quadrants": (1, QUADRANTS, "int"),
    # 剩餘天數少於這個就不買地。0 = 到最後一天都還買。
    "land_min_days_left": (0, DAYS, "int"),
    # 蓋幾個建物。25 格的象限放 20 個建物已經很擠。
    "n_structures": (4, 20, "int"),
    # 建物散到幾個象限。0 = 全部在 NW（現在出貨的行為），1 = 攤平到 NW/NE/SW。
    #
    # 🩸 這是三個量到的瓶頸裡唯一還沒被單獨證偽的方向：ladder 八個對手都是
    # 約 6/6/2，我們 1,280 局全部 12/0/0，MOVE 佔動作 57.5%（全場最高）。
    # 但它跟 `n_structures` / `max_quadrants` / `land_first_min_day` 直接耦合
    # —— 攤出去的格子在還沒買下來的象限上，蓋不了。單獨調沒有意義。
    "structure_spread": (0.0, 1.0, "float"),

    # --- 需求分配 ---
    # 對手產能折算多少。0 = 完全不看對手，1.0 = 全額計入，1.5 允許「高估對手」。
    "opponent_supply_weight": (0.0, 1.5, "float"),
    # 自己至少要規劃城鎮需求的多少比例。
    "animal_demand_share_floor": (0.0, 1.0, "float"),
    # 先替預計動物數的多少比例準備 WHEAT。1.5 = 超額準備。
    "feed_crop_bootstrap_share": (0.0, 1.5, "float"),
    # 飼料需求在作物規劃裡的權重。
    "feed_crop_demand_weight": (0.0, 2.0, "float"),
    # 幾個吃草莓的商店才放寬草莓上限。上界 40 在 8 個 shop 實例的上限之上，
    # 等於「永不觸發」。
    "strawberry_high_demand": (0, 40, "int"),
    # 觸發後的草莓上限。
    "strawberry_high_share": (0.1, 1.0, "float"),

    # --- 前期限制 ---
    # 前半季最多先鎖幾格 COOP。0 = 不限制。
    "early_coop_limit": (0, 8, "int"),
    # 這個限制持續到第幾天。0 = 等於關掉。
    "early_coop_until_day": (0, DAYS, "int"),

    # --- 施肥 ---
    # ongoing 作物未來三天的額外產品價值要達到肥料售價的幾倍才施。
    # 🩸 不能是 None —— `gen0.py:1047` 用 `is not None` 當總開關，None 會關掉整條
    # 施肥 ROI 路徑，那是質變不是量變，不屬於連續搜尋。
    "fertilizer_roi_margin": (0.2, 4.0, "float"),

    # --- 白天回倉 ---
    # 攜帶價值至少多少才白天回倉。0 = 一律回倉。
    "daytime_return_min_value": (0, 1500, "int"),
    # 距 shed 幾步以內才回。0 = 等於關掉；12 約是 10x10 盤面的半個對角線。
    "daytime_return_max_distance": (0, 12, "int"),

    # --- 產能 ---
    # active tiles 上限 = (1 + planned_hands) x 這個值。08-16 實測 4 最好，
    # 4.5 / 5 / 無上限的勝率是 42.5% / 12.5% / 17.5%。
    "tiles_per_unit": (2, 8, "int"),
}

#: 向量順序。
KEYS = tuple(SEARCH_SPACE)
#: 維度。
DIM = len(KEYS)
#: 被搜到的 `DEFAULT_PARAMS` key（展開維度算它的 base，一個 base 只算一次）。
SEARCHED_BASES = tuple(dict.fromkeys(split_key(k)[0] for k in KEYS))
#: 不進搜尋的 key（值取自 `DEFAULT_PARAMS`）。
FROZEN_KEYS = tuple(k for k in DEFAULT_PARAMS if k not in SEARCHED_BASES)

#: 引擎的作物清單。展開維度要蓋滿它，少一種就是安靜地讓那個作物走不到的
#: `max_crop_share`（見 `_check_space`）。
_CROPS = tuple(_engine.CROPS)


def _check_space():
    """import 就驗：key 要存在、型別要對得上、現值要在界內、界線要有寬度。

    這些全部是「寫錯 SEARCH_SPACE」才會發生的事，愈早炸愈好 —— 不然會安靜地
    搜一個錯的空間。
    """
    expanded = {}
    for key, spec in SEARCH_SPACE.items():
        lo, hi, kind = spec
        base, sub = split_key(key)
        if base not in DEFAULT_PARAMS:
            raise ValueError(f"SEARCH_SPACE 有 {key!r}，但 DEFAULT_PARAMS 沒有")
        if sub is not None:
            if not isinstance(DEFAULT_PARAMS[base], dict):
                raise TypeError(
                    f"{key!r} 用了展開語法，但 DEFAULT_PARAMS[{base!r}] 是 "
                    f"{type(DEFAULT_PARAMS[base]).__name__} 不是 dict")
            if base not in DICT_FALLBACK:
                raise ValueError(f"{base!r} 要展開，但 DICT_FALLBACK 沒有宣告"
                                 "「沒列到的項目走哪個全域參數」")
            expanded.setdefault(base, []).append(sub)
        value = read_key(DEFAULT_PARAMS, key)
        if isinstance(value, bool):
            raise TypeError(f"{key!r} 是 bool，不能放進連續搜尋空間")
        if kind == "int" and not isinstance(value, int):
            raise TypeError(
                f"{key!r} 宣告成 int，DEFAULT_PARAMS 是 {type(value).__name__}")
        if kind == "float" and not isinstance(value, (int, float)):
            raise TypeError(
                f"{key!r} 宣告成 float，DEFAULT_PARAMS 是 {type(value).__name__}")
        if kind not in ("int", "float"):
            raise ValueError(f"{key!r} 的 kind {kind!r} 不是 int / float")
        if lo >= hi:
            raise ValueError(f"{key!r} 的界線 lo={lo} 不小於 hi={hi}")
        if not lo <= value <= hi:
            raise ValueError(f"{key!r} 的現值 {value} 不在界線 [{lo}, {hi}] 內")

    # 🩸 展開了就要蓋滿。只展開一部分的話，沒展開的那幾種會繼續走
    # `DICT_FALLBACK` 指到的全域參數 —— 而那個參數多半已經被移出搜尋空間，
    # 等於把它們釘死在 `DEFAULT_PARAMS` 的值上，安靜地少搜幾個方向。
    for base, subs in expanded.items():
        if base == "crop_share" and set(subs) != set(_CROPS):
            raise ValueError(
                f"crop_share 展開了 {sorted(subs)}，引擎的作物是 "
                f"{sorted(_CROPS)} —— 要嘛全展開、要嘛全不展開")
        if len(set(subs)) != len(subs):
            raise ValueError(f"{base} 有重複的展開項：{subs}")


_check_space()


# --------------------------------------------------------------------------
# 編解碼
# --------------------------------------------------------------------------

#: 聚焦重搜用的維度子集（`tools/param_search.py --only`）。
#:
#: 🩸 **收錄標準只有一條：跟這一輪新加的維度搶同一個稀缺資源，或直接夾住它。**
#: 不是「感覺重要」—— 其餘 26 維已經在 `cma1-g50-wt` 那一輪（108 代）聯合搜過，
#: 從那個點重搜整個空間要 55 小時 × 1.27，聚焦是 30 小時。
#:
#: `s65`（2026-09-01，理由見 `docs/memory/journal/2026-09-01.md` §65）：
#:
#:     新的 7 維    crop_share × 5、structure_spread、seed_backlog
#:     開局現金     seed_backlog 跟 animal_spend_frac / seed_buffer_days 搶
#:                  day 0 的 $3,000（對手 day 0 種 19 格、我們 5 格）
#:     unit-turns   structure_spread 省下來的走路要有人接手 ——
#:                  max_hands / hands_per_quadrant / tiles_per_unit。
#:                  🩸 兩個人數上限都要進：`land_cap = min(max_hands,
#:                  hands_per_quadrant × 已解鎖象限)`，出貨值 11 / 4 / 3 象限
#:                  -> 前期 4 人由 `hands_per_quadrant` 夾、後期 11 人由
#:                  `max_hands` 夾，凍任何一個都會鎖死半局的人數
#:     作物上限     crop_share 之外還夾得住配額的兩個：oversupply_factor
#:                  （城鎮吸收上限）、n_structures（建物吃掉的是同一批格子）
#:
#: 刻意**不**收：`cash_reserve_days`（出貨值 0，已經全關）、`animal_reserve`
#: （跟 `animal_spend_frac` 管同一筆錢，收一個就夠）、`land_first_min_day`
#: （買地時機是另一條軸，這一輪不動）。
SUBSETS = {
    "s65": (
        "crop_share.WHEAT", "crop_share.CARROT", "crop_share.TOMATO",
        "crop_share.STRAWBERRY", "crop_share.MELON",
        "structure_spread", "seed_backlog",
        "seed_buffer_days", "animal_spend_frac",
        "max_hands", "hands_per_quadrant", "tiles_per_unit",
        "oversupply_factor", "n_structures",
    ),
}

for _name, _subset in SUBSETS.items():
    _unknown = [k for k in _subset if k not in SEARCH_SPACE]
    if _unknown:
        raise ValueError(f"SUBSETS[{_name!r}] 有不在 SEARCH_SPACE 裡的 key：{_unknown}")
    if len(set(_subset)) != len(_subset):
        raise ValueError(f"SUBSETS[{_name!r}] 有重複的 key")


def resolve_subset(spec):
    """`--only` 的值 -> `KEYS` 的子集（照 `KEYS` 的順序，不照使用者打的順序）。

    收 `SUBSETS` 裡的名字，或逗號分隔的 key 清單。
    """
    if spec in SUBSETS:
        want = set(SUBSETS[spec])
    else:
        want = {k.strip() for k in spec.split(",") if k.strip()}
        unknown = sorted(want - set(SEARCH_SPACE))
        if unknown:
            raise ValueError(
                f"--only 有不在 SEARCH_SPACE 裡的 key：{unknown}"
                f"（可用的子集名字：{sorted(SUBSETS)}）")
    if not want:
        raise ValueError("--only 是空的")
    return tuple(k for k in KEYS if k in want)


def _clip01(t):
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


def encode(params=None):
    """params -> 正規化向量（list of float，長度 `DIM`，每項在 [0, 1]）。

    沒給 params 就用 `DEFAULT_PARAMS`；缺的 key fall back 到預設值。
    """
    src = dict(DEFAULT_PARAMS)
    src.update(params or {})
    x = []
    for key in KEYS:
        lo, hi, _kind = SEARCH_SPACE[key]
        x.append(_clip01((float(read_key(src, key)) - lo) / (hi - lo)))
    return x


def decode(x, base=None):
    """正規化向量 -> **完整**的 params dict。

    - int 用 `int(v + 0.5)` 取整，**不是 `round()`** —— 那是 banker's rounding，
      `round(0.5)` 給 0、`round(1.5)` 給 2，在界線邊緣會偏一格。
    - 界外的 x 直接夾回 [0, 1] 不報錯（CMA-ES 的 bounds 已經在管，這是第二道）。
    - 帶 `_replace_defaults: True`：這份 params 是完整快照，不該跟「今天的」
      `DEFAULT_PARAMS` 合併。否則之後有人加一個預設開關，這組參數會在完全沒有
      被修改的情況下偷偷變成另一個 agent。

    `base` = 沒被搜到的那些 key 從哪裡取值，預設 `DEFAULT_PARAMS`。

    🩸 **暖啟動一定要給 `base`。** `config/params/*.json` 有 `DEFAULT_PARAMS`
    **沒有**的 key —— 2026-09-01 實測 `cma1-g50-wt.json` 就有兩個：
    `whole_turn_assignment: True` 和 `priority_step_cost: 1`（`agents/gen0.py`
    用 `params.get()` 讀，所以不在預設表裡也能生效；不放進預設表是因為那會改掉
    `gen1` / `ref-v11` 這些對手的行為）。不給 base 的話 `decode` 從
    `DEFAULT_PARAMS` 重建，這兩個直接消失 —— 等於整輪 CMA-ES 在搜一個**沒有**
    那個 +7pp 演算法的 agent，而且分數只會低一截、不會報錯。
    短局自我對戰實測：4,910 -> 4,756。
    """
    if len(x) != DIM:
        raise ValueError(f"向量長度 {len(x)} != DIM {DIM}")
    params = dict(DEFAULT_PARAMS)
    if base:
        params.update(base)
    params.pop("_replace_defaults", None)
    for key, t in zip(KEYS, x):
        lo, hi, kind = SEARCH_SPACE[key]
        value = lo + _clip01(float(t)) * (hi - lo)
        if kind == "int":
            value = max(lo, min(hi, int(value + 0.5)))
        write_key(params, key, value)
    params["_replace_defaults"] = True
    return params


def x0():
    """CMA-ES 的起點 —— `DEFAULT_PARAMS`（已經人手調過）在正規化空間的位置。"""
    return encode()


def to_json_params(params):
    """把 `decode()` 的產出變成寫得進 config JSON 的形狀。

    tuple -> list（JSON 沒有 tuple），並拿掉 `_replace_defaults` —— config JSON
    靠 `"frozen"` 讓 `eval.runner.build_agent` 自己補那個旗標
    （`eval/runner.py:135-137`），JSON 裡重複寫只會多一個對不上的來源。
    """
    def conv(value):
        if isinstance(value, tuple):
            return [conv(item) for item in value]
        if isinstance(value, dict):
            return {key: conv(item) for key, item in value.items()}
        return value

    return {k: conv(v) for k, v in params.items() if k != "_replace_defaults"}


if __name__ == "__main__":
    n_int = sum(1 for k in KEYS if SEARCH_SPACE[k][2] == "int")
    n_float = DIM - n_int
    print(f"維度 {DIM}（{n_int} int + {n_float} float）")
    print(f"凍結 {len(FROZEN_KEYS)} 個：{', '.join(FROZEN_KEYS)}")
    print(f"引擎常數：DAYS={DAYS} QUADRANTS={QUADRANTS} MAX_ORDERS={MAX_ORDERS} "
          f"TILES_PER_QUADRANT={TILES_PER_QUADRANT} STARTING_MONEY={STARTING_MONEY}")

    round_trip = decode(x0())
    drift = [(k, read_key(DEFAULT_PARAMS, k), read_key(round_trip, k))
             for k in KEYS
             if read_key(DEFAULT_PARAMS, k) != read_key(round_trip, k)]
    print(f"round-trip 漂掉的 key：{drift if drift else '（無）'}")

    lo_params, hi_params = decode([0.0] * DIM), decode([1.0] * DIM)
    print("\n界線（現值 -> [lo, hi]）：")
    for key in KEYS:
        _lo, _hi, kind = SEARCH_SPACE[key]
        print(f"  {kind:5} {key:28} {read_key(DEFAULT_PARAMS, key)!r:>8} -> "
              f"[{read_key(lo_params, key)!r}, {read_key(hi_params, key)!r}]")
