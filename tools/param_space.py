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
- `crop_share` / `crop_oversupply` 是逐作物覆寫的 dict，展開才搜得動，第一輪不做。

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

#: key -> (lo, hi, kind)。kind 是 "int" 或 "float"。順序即向量順序。
SEARCH_SPACE = {
    # --- 作物與市場配置 ---
    # 單一作物佔比上限。0.2 已接近「五種平均分」，1.0 = 允許全押一種。
    "max_crop_share": (0.1, 1.0, "float"),
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
#: 不進搜尋的 key（值取自 `DEFAULT_PARAMS`）。
FROZEN_KEYS = tuple(k for k in DEFAULT_PARAMS if k not in SEARCH_SPACE)


def _check_space():
    """import 就驗：key 要存在、型別要對得上、現值要在界內、界線要有寬度。

    這些全部是「寫錯 SEARCH_SPACE」才會發生的事，愈早炸愈好 —— 不然會安靜地
    搜一個錯的空間。
    """
    for key, spec in SEARCH_SPACE.items():
        lo, hi, kind = spec
        if key not in DEFAULT_PARAMS:
            raise ValueError(f"SEARCH_SPACE 有 {key!r}，但 DEFAULT_PARAMS 沒有")
        value = DEFAULT_PARAMS[key]
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


_check_space()


# --------------------------------------------------------------------------
# 編解碼
# --------------------------------------------------------------------------

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
        x.append(_clip01((float(src[key]) - lo) / (hi - lo)))
    return x


def decode(x):
    """正規化向量 -> **完整 51 個 key** 的 params dict。

    - int 用 `int(v + 0.5)` 取整，**不是 `round()`** —— 那是 banker's rounding，
      `round(0.5)` 給 0、`round(1.5)` 給 2，在界線邊緣會偏一格。
    - 界外的 x 直接夾回 [0, 1] 不報錯（CMA-ES 的 bounds 已經在管，這是第二道）。
    - 帶 `_replace_defaults: True`：這份 params 是完整快照，不該跟「今天的」
      `DEFAULT_PARAMS` 合併。否則之後有人加一個預設開關，這組參數會在完全沒有
      被修改的情況下偷偷變成另一個 agent。
    """
    if len(x) != DIM:
        raise ValueError(f"向量長度 {len(x)} != DIM {DIM}")
    params = dict(DEFAULT_PARAMS)
    for key, t in zip(KEYS, x):
        lo, hi, kind = SEARCH_SPACE[key]
        value = lo + _clip01(float(t)) * (hi - lo)
        if kind == "int":
            value = max(lo, min(hi, int(value + 0.5)))
        params[key] = value
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
    drift = [(k, DEFAULT_PARAMS[k], round_trip[k])
             for k in KEYS if DEFAULT_PARAMS[k] != round_trip[k]]
    print(f"round-trip 漂掉的 key：{drift if drift else '（無）'}")

    lo_params, hi_params = decode([0.0] * DIM), decode([1.0] * DIM)
    print("\n界線（現值 -> [lo, hi]）：")
    for key in KEYS:
        lo, hi, kind = SEARCH_SPACE[key]
        print(f"  {kind:5} {key:28} {DEFAULT_PARAMS[key]!r:>8} -> "
              f"[{lo_params[key]!r}, {hi_params[key]!r}]")
