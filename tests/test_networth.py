"""Φ = 非現金資產（`model/networth.py`，§83.3）。

這裡釘的是**跟引擎對得起來**：淨變現價值要等於真的下 SELL 單收到的錢，
期末的 Φ 要能歸零，來不及的東西要減記。算錯不會報錯，只會讓 PPO 學到
一個不存在的目標。
"""
from __future__ import annotations

import numpy as np
import pytest

from model.networth import (ANIMALS, CROPS, LAND_PRICES, asset_value,
                            net_realisable_value)
from tools._quiet import silenced

with silenced():
    from kaggle_environments.envs.kaggriculture import kaggriculture as K


def _obs(day=0, hour=0, player=0, tiles=None, shed=None, seeds=None,
         inventories=None, quadrants=("NW",), money=3000.0, market_inv=None):
    """最小可用的 observation。沒給的欄位就是引擎的初始狀態。"""
    farm = K._new_farm(10, money)
    if tiles is not None:
        farm["tiles"] = tiles
    farm["unlocked_quadrants"] = list(quadrants)
    private = K._new_private()
    if shed:
        private["shed"].update(shed)
    if seeds:
        private["seeds"].update(seeds)
    if inventories is not None:
        private["inventories"] = inventories
    market = K._new_market()
    if market_inv:
        market["inventory"].update(market_inv)
    return {"player": player, "farms": [farm, K._new_farm(10, money)],
            "private": private, "market": market, "day": day, "hour": hour}


def _empty_tiles():
    return [[None for _ in range(10)] for _ in range(10)]


# ------------------------------------------------------------- 淨變現價值

def test_nrv_equals_what_the_engine_actually_pays():
    """🩸 這是整個 Φ 最容易錯的地方：賣一個庫存加一，牌價乘數量是錯的。"""
    for item, n in (("STRAWBERRY", 100), ("MELON", 50), ("WHEAT", 40)):
        market = K._new_market()
        farm = K._new_farm(10, 0.0)
        private = K._new_private()
        private["shed"][item] = n
        want = net_realisable_value({item: n}, market["inventory"])
        for _ in range(n):
            price = K.market_price(item, market["inventory"][item])
            assert K._commit_unit("SELL", item, price, farm, private, market)
        assert farm["money"] == pytest.approx(want)


def test_nrv_is_well_below_the_sticker_price_for_a_thin_market():
    """§83.5：草莓賣 100 個只實收牌價的三成。記牌價會獎勵囤貨。"""
    market = K._new_market()
    sticker = 100 * market["prices"]["STRAWBERRY"]
    assert net_realisable_value({"STRAWBERRY": 100},
                                market["inventory"]) < 0.40 * sticker


def test_nrv_ignores_things_that_are_not_products():
    market = K._new_market()
    assert net_realisable_value({"GOOSE": 3}, market["inventory"]) == 0.0


# --------------------------------------------------------------- 終端條件

def test_land_and_animals_are_worth_nothing_on_the_last_day():
    """Ng 的定理要 Φ(終端)=0。土地和動物按剩餘天數折舊，第 29 天自己歸零。"""
    tiles = _empty_tiles()
    tiles[0][0] = K._new_animal("COW", 0)
    obs = _obs(day=29, tiles=tiles, quadrants=("NW", "NE", "SW"))
    d = asset_value(obs, detail=True)
    assert d["land"] == 0.0
    assert d["animals"] == 0.0


def test_land_bought_early_is_nearly_neutral_and_late_is_nearly_a_write_off():
    early = asset_value(_obs(day=5, quadrants=("NW", "NE")), detail=True)
    late = asset_value(_obs(day=25, quadrants=("NW", "NE")), detail=True)
    assert early["land"] == pytest.approx(LAND_PRICES[0] * 24 / 29)
    assert late["land"] == pytest.approx(LAND_PRICES[0] * 4 / 29)


def test_seeds_are_written_down_once_they_cannot_reach_a_harvest():
    """種子只能種不能賣。種下去到能收要 first_yield_day 天。"""
    seeds = {"MELON": 5}                       # first_yield_day = 10
    assert asset_value(_obs(day=18, seeds=seeds),
                       detail=True)["seeds"] == 5 * CROPS["MELON"]["seed"]
    assert asset_value(_obs(day=20, seeds=seeds), detail=True)["seeds"] == 0.0


def test_a_plant_that_cannot_ripen_in_time_carries_no_cost_basis():
    tiles = _empty_tiles()
    tiles[0][0] = K._new_plant("MELON", 20, 24)     # 20 + 10 = 30 > 29
    assert asset_value(_obs(day=20, tiles=tiles), detail=True)["wip"] == 0.0
    tiles[0][0] = K._new_plant("MELON", 19, 24)     # 19 + 10 = 29，剛好趕上
    assert asset_value(_obs(day=19, tiles=tiles),
                       detail=True)["wip"] == CROPS["MELON"]["seed"]


def test_a_decaying_plant_loses_its_cost_basis():
    """過了 max_lifespan_step 的格子每 2 步掉一個 yield_units，成本收不回來。"""
    tiles = _empty_tiles()
    tile = K._new_plant("WHEAT", 0, 24)
    tiles[0][0] = tile
    alive = asset_value(_obs(day=4, hour=0, tiles=tiles), detail=True)
    assert alive["wip"] == CROPS["WHEAT"]["seed"]
    # max_lifespan_step = (0 + 4 + 1) * 24 = 120 -> day 5 hour 0
    assert tile["max_lifespan_step"] == 120
    dead = asset_value(_obs(day=5, hour=0, tiles=tiles), detail=True)
    assert dead["wip"] == 0.0


# --------------------------------------------------------------- 認列時點

def test_strict_and_produce_differ_exactly_on_an_immature_crop():
    """🩸 `_new_plant` 一種下去 yield_units 就是 1（非持續型），但引擎的
    HARVEST 要等到 first_yield_day。§83.4 的 A/B 就是這一格。"""
    tiles = _empty_tiles()
    tiles[0][0] = K._new_plant("WHEAT", 0, 24)      # first_yield_day = 2
    strict = asset_value(_obs(day=0, tiles=tiles), detail=True)
    produce = asset_value(_obs(day=0, tiles=tiles), recognise="produce",
                          detail=True)
    assert strict["goods"] == 0.0
    assert produce["goods"] > 0.0
    assert produce["units"] == {"WHEAT": 1}
    # 到得了 first_yield_day 之後兩邊一致。
    ripe = _obs(day=2, tiles=tiles)
    assert asset_value(ripe, detail=True)["goods"] == asset_value(
        ripe, recognise="produce", detail=True)["goods"]


def test_recognise_rejects_unknown_values():
    with pytest.raises(ValueError):
        asset_value(_obs(), recognise="whatever")


# ------------------------------------------------------------------- 科目

def test_structures_and_hands_are_worth_zero():
    """引擎蓋 COOP / PASTURE 不收錢，人手每晚解雇 —— 工資是費用不是資產。"""
    tiles = _empty_tiles()
    tiles[0][0] = {"kind": "COOP"}
    tiles[0][1] = {"kind": "PASTURE"}
    tiles[0][2] = {"kind": "WEED"}
    obs = _obs(day=10, tiles=tiles)
    obs["farms"][0]["hands"] = [[1, 1], [2, 2], [3, 3]]
    assert asset_value(obs) == 0.0


def test_an_animal_on_a_tile_carries_both_its_cost_and_its_output():
    tiles = _empty_tiles()
    tile = K._new_animal("GOOSE", 0)
    tile["yield_units"] = 3
    tiles[0][0] = tile
    d = asset_value(_obs(day=10, tiles=tiles), detail=True)
    assert d["animals"] == pytest.approx(ANIMALS["GOOSE"]["cost"] * 19 / 29)
    assert d["units"] == {"EGG": 3}
    assert d["goods"] > 0


def test_animals_in_the_shed_are_animals_not_goods():
    d = asset_value(_obs(day=0, shed={"COW": 2}), detail=True)
    assert d["animals"] == pytest.approx(2 * ANIMALS["COW"]["cost"])
    assert d["goods"] == 0.0


def test_stock_in_hand_counts_the_same_as_stock_in_the_shed():
    in_shed = asset_value(_obs(day=10, shed={"MELON": 4}))
    in_hand = asset_value(_obs(day=10, inventories=[{"MELON": 4}]))
    assert in_shed == pytest.approx(in_hand)


def test_goods_are_priced_as_one_pooled_sale():
    """田裡的 + 倉庫的要一起沿曲線走，不然 40 個草莓會被高估。"""
    tiles = _empty_tiles()
    tile = K._new_plant("STRAWBERRY", 0, 24)
    tile["yield_units"] = 4
    tiles[0][0] = tile
    both = asset_value(_obs(day=10, tiles=tiles, shed={"STRAWBERRY": 4}),
                       detail=True)
    assert both["units"] == {"STRAWBERRY": 8}
    market = K._new_market()
    assert both["goods"] == pytest.approx(
        net_realisable_value({"STRAWBERRY": 8}, market["inventory"]))


def test_fertilizer_waiting_on_an_animal_is_not_an_asset_yet():
    """要花一個 unit-turn 去 COLLECT_FERTILIZER，那一步才是它的 credit。"""
    tiles = _empty_tiles()
    tile = K._new_animal("GOOSE", 0)
    tile["fertilizer_available"] = True
    tiles[0][0] = tile
    d = asset_value(_obs(day=10, tiles=tiles), detail=True)
    assert d["units"] == {}


# --------------------------------------------------------- 真的跑一局

def test_phi_starts_at_zero_and_stays_finite_over_a_real_episode():
    from eval.runner import build_agent, load_spec
    with silenced():
        from kaggle_environments import make
        env = make("kaggriculture",
                   configuration={"seed": 4242, "episodeSteps": 240},
                   debug=False)
        spec = load_spec("config/params/cma5-g175.json")
        env.run([build_agent(spec), build_agent(spec)])
    vals = [asset_value(st[0]["observation"], episode_steps=240)
            for st in env.steps[:-1]]
    assert vals[0] == 0.0             # 開局身上只有現金
    assert np.isfinite(vals).all()
    assert min(vals) >= 0.0
    assert max(vals) > 1000.0         # 真的有在買東西
