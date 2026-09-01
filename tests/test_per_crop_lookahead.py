"""逐作物的供給推算天數（`per_crop_lookahead`）。

🩸 2026-09-01 量到的病因：`_inv_key` 用一個全域 `supply_lookahead_days` 去推算
「賣出時的市場庫存」，`cma1-g50-wt` 是 3 天。但 MELON 的 `first_yield_day`
是 10 天 —— 決定它成交價的是 10 天後的庫存，不是 3 天後的。

MELON 的 glut 曲線是 `sq`（`above_target=3.6`）：從 $250 起倒 150 個掉到 $25。
ladder 頂端的 replay（6 局）雙方各種 12 格 MELON、步 ~284 同時收成，價格
271 -> 76；那 24 格從步 48 起就在 `obs["farms"]` 裡看得到。

這裡守三件事：預設不能動到出貨行為、horizon 要取自引擎的 `CROPS`、
打開之後要真的看得到 3 天窗口漏掉的那批。
"""

from __future__ import annotations

import pytest

from agents.gen0 import _inv_key, _item_lookahead, incoming_supply
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

BUCKET = 25


def _obs(day, melon_tiles=0, wheat_tiles=0, planted_day=0, inv=10_000):
    """兩座農場，第二座（對手）種 `melon_tiles` 格 MELON。"""

    def farm(n_melon, n_wheat):
        tiles = [[None] * 10 for _ in range(10)]
        put = [("MELON", n_melon), ("WHEAT", n_wheat)]
        i = 0
        for crop, n in put:
            for _ in range(n):
                tiles[i // 10][i % 10] = {
                    "kind": "PLANT", "crop": crop, "planted_day": planted_day,
                }
                i += 1
        return {"tiles": tiles}

    return {
        "day": day,
        "player": 0,
        "market": {"inventory": {c: inv for c in CROPS}},
        "farms": [farm(0, wheat_tiles), farm(melon_tiles, 0)],
    }


def _old_inv_key(obs, items, lookahead):
    """改動前的實作，一字不改抄過來當對照。"""
    inv = obs["market"]["inventory"]
    soon = incoming_supply(obs, lookahead) if lookahead > 0 else {}
    return tuple((i, int((inv[i] + soon.get(i, 0.0)) / BUCKET) * BUCKET)
                 for i in items)


@pytest.mark.parametrize("lookahead", [0, 3, 6, 12])
@pytest.mark.parametrize("day", [0, 5, 12, 25])
def test_default_is_identical_to_the_old_implementation(day, lookahead):
    """`per_crop=False` 一定要跟舊實作逐項相同 —— 出貨版吃的就是這條路。"""
    obs = _obs(day, melon_tiles=12, wheat_tiles=8, planted_day=0)
    items = sorted(CROPS)
    assert (_inv_key(obs, items, lookahead=lookahead)
            == _old_inv_key(obs, items, lookahead))
    assert (_inv_key(obs, items, lookahead=lookahead, per_crop=False)
            == _old_inv_key(obs, items, lookahead))


def test_market_aware_off_is_untouched():
    obs = _obs(5, melon_tiles=12)
    for pc in (False, True):
        assert (_inv_key(obs, sorted(CROPS), market_aware=False, per_crop=pc)
                == _inv_key(obs, sorted(CROPS), market_aware=False))


def test_horizon_comes_from_the_engine_not_a_hardcoded_table():
    for crop, cd in CROPS.items():
        assert _item_lookahead(crop, 3) == cd["first_yield_day"]


def test_non_crop_products_fall_back_to_the_global_value():
    for item in ("MILK", "EGG", "WOOL"):
        assert _item_lookahead(item, 3) == 3
        assert _item_lookahead(item, 9) == 9


def test_melon_wave_is_invisible_at_3_days_and_visible_per_crop():
    """對手 12 格 MELON、第 0 天種，我方第 5 天要決定種什麼。

    MELON 的 `crop_cycle` 收成 age 是 11 天。3 天窗口（5..8）看不到，
    自己的 `first_yield_day`＝10 天窗口（5..15）看得到。
    """
    obs = _obs(day=5, melon_tiles=12, planted_day=0)
    items = sorted(CROPS)
    old = dict(_inv_key(obs, items, lookahead=3))
    new = dict(_inv_key(obs, items, lookahead=3, per_crop=True))
    assert old["MELON"] == 10_000, "3 天窗口不該看到第 11 天的收成"
    assert new["MELON"] > old["MELON"], "逐作物窗口要看得到那批 MELON"
    assert new["WHEAT"] == old["WHEAT"], "WHEAT 的窗口沒變，不該動"


def test_projected_inventory_actually_lowers_the_melon_price():
    """看得到那批貨，估出來的價才會低 —— 這才是整個修正的目的。"""
    from agents.gen0 import _avg_sell_price

    obs = _obs(day=5, melon_tiles=12, planted_day=0)
    items = sorted(CROPS)
    old = dict(_inv_key(obs, items, lookahead=3))
    new = dict(_inv_key(obs, items, lookahead=3, per_crop=True))
    assert (_avg_sell_price("MELON", new["MELON"], 0)
            < _avg_sell_price("MELON", old["MELON"], 0))


def test_more_opponent_melon_means_lower_projected_price():
    """單調：對手種愈多，我們估的 MELON 價愈低。"""
    prices = []
    for n in (0, 4, 8, 12, 20):
        obs = _obs(day=5, melon_tiles=n, planted_day=0)
        key = dict(_inv_key(obs, sorted(CROPS), lookahead=3, per_crop=True))
        prices.append(key["MELON"])
    assert prices == sorted(prices), prices
    assert prices[0] < prices[-1]
