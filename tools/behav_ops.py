"""共用的動作詞彙與解析，供 Kaggle episode 與本機對局用同一套特徵。

動作格式（兩邊相同）：
    {"farmer": ["WEST"], "hands": [["WATER"], ...], "market": [["SELL","WOOL",3], ...]}
    PICKUP / PLACE  =  [op, item, qty]，qty 省略時視為 1
    PLANT           =  [op, crop]
    SELL / BUY_*    =  [op, item, qty]（HIRE / BUY_LAND 無參數）
"""
from __future__ import annotations

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
UNIT_OPS = ("NORTH", "SOUTH", "EAST", "WEST", "PASS", "WATER", "HARVEST",
            "COLLECT_FERTILIZER", "CARE", "FEED", "PLANT", "PICKUP", "PLACE",
            "FERTILIZE", "DIG", "DROP", "BUILD_PASTURE", "BUILD_COOP")
MARKET_OPS = ("HIRE", "BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT", "SELL", "BUY_LAND")
#: 需要另外累計數量的（count 之外再記 qty 總和）
QTY_OPS = ("PICKUP", "PLACE", "SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL")

PRODUCTIVE = {"PLANT", "WATER", "HARVEST", "DIG", "FERTILIZE", "FEED", "CARE",
              "COLLECT_FERTILIZER", "BUILD_COOP", "BUILD_PASTURE", "PLACE", "PICKUP"}

#: 特徵向量的欄位順序。count 在前，qty 在後。
FIELDS = (tuple(UNIT_OPS) + tuple(MARKET_OPS)
          + tuple(f"{o}_qty" for o in QTY_OPS))
IDX = {k: i for i, k in enumerate(FIELDS)}
N_FIELD = len(FIELDS)


def tally(action, out):
    """把一個 player 的一步動作累加進 `out`（長度 N_FIELD 的可索引序列）。"""
    if not isinstance(action, dict):
        return
    for ua in [action.get("farmer"), *(action.get("hands") or [])]:
        if not isinstance(ua, list) or not ua:
            continue
        op = ua[0]
        i = IDX.get(op)
        if i is not None:
            out[i] += 1
        if op in ("PICKUP", "PLACE"):
            q = ua[2] if len(ua) > 2 and isinstance(ua[2], (int, float)) else 1
            out[IDX[op + "_qty"]] += q
    for m in (action.get("market") or []):
        if not isinstance(m, list) or not m:
            continue
        op = m[0]
        i = IDX.get(op)
        if i is not None:
            out[i] += 1
        if op in ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"):
            q = m[2] if len(m) > 2 and isinstance(m[2], (int, float)) else 1
            out[IDX[op + "_qty"]] += q
