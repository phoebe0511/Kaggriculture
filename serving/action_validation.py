"""在送進引擎前，把 Kaggriculture 的靜默 no-op 轉成明確錯誤。

驗證器直接呼叫 1.32.7 引擎的單位動作實作，在 observation 的深拷貝上模擬
farmer / hands；市場訂單則依同一套 pricing 和 commit 函式逐筆模擬。它不嘗試
預測對手同回合的市場訂單，但會抓出由目前狀態即可確定的非法動作、資源不足、
倉庫滿載與參數格式錯誤。
"""

from __future__ import annotations

import copy
import inspect
import os
import sys

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    PRODUCTS,
    _apply_unit_action,
    _commit_unit,
    _do_buy_land,
    _do_hire,
    _refresh_prices,
    market_price,
)


class IllegalAction(AssertionError):
    """Agent 產生引擎會靜默忽略的動作。"""


# --- 引擎版本差異的適配 ------------------------------------------------------
# 這個模組呼叫的是引擎的**私有函式**，簽名會隨版本改。實測：
#
#   1.29.3   _commit_unit(op, item, price, farm, private, market)
#   1.32.7   _commit_unit(op, item, price, farm, private, market, shed_capacity=100)
#
# 2026-08-17 在 Kaggle notebook（1.29.3）實測，多傳一個參數的後果是
# **每一回合都拋 TypeError**：
#   TypeError: _commit_unit() takes 6 positional arguments but 7 were given
#
# 而 ladder 用哪個版本我們看不到。驗證器是除錯工具，它自己跟引擎對不上時
# 賠掉整局是最差的結果 —— 所以按實際簽名決定要不要傳。
_COMMIT_TAKES_CAPACITY = (
    "shed_capacity" in inspect.signature(_commit_unit).parameters)
_APPLY_TAKES_CAPACITY = (
    "shed_capacity" in inspect.signature(_apply_unit_action).parameters)


def _commit(op, item, price, farm, private, market, shed_capacity):
    if _COMMIT_TAKES_CAPACITY:
        return _commit_unit(op, item, price, farm, private, market, shed_capacity)
    return _commit_unit(op, item, price, farm, private, market)


def _apply(farm, private, idx, action, board_size, day, turns_per_day,
           shed_capacity):
    if _APPLY_TAKES_CAPACITY:
        return _apply_unit_action(farm, private, idx, action, board_size, day,
                                  turns_per_day, shed_capacity)
    return _apply_unit_action(farm, private, idx, action, board_size, day,
                              turns_per_day)


_warned = set()


def _warn_once(key, message):
    if key in _warned:
        return
    _warned.add(key)
    print(f"[action_validation] {message}", file=sys.stderr)


def _cfg(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def assert_observation_invariants(obs):
    """檢查公開與私有狀態中不應為負數的欄位。"""
    for pid, farm in enumerate(obs["farms"]):
        if farm["money"] < 0:
            raise IllegalAction(f"player {pid} money < 0: {farm['money']}")
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and tile.get("yield_units", 0) < 0:
                    raise IllegalAction(
                        f"player {pid} tile {(x, y)} yield_units < 0: {tile}"
                    )

    private = obs["private"]
    groups = [
        ("shed", private["shed"]),
        ("seeds", private["seeds"]),
    ]
    groups.extend(
        (f"inventories[{i}]", inv)
        for i, inv in enumerate(private["inventories"])
    )
    for label, values in groups:
        for item, amount in values.items():
            if amount < 0:
                raise IllegalAction(f"{label}.{item} < 0: {amount}")


def _validate_units(obs, config, action, farm, private):
    farmer_action = action.get("farmer")
    hands_actions = action.get("hands")
    if not isinstance(farmer_action, list) or not farmer_action:
        raise IllegalAction(f"farmer 動作格式錯誤: {farmer_action!r}")
    if not isinstance(hands_actions, list):
        raise IllegalAction(f"hands 必須是 list: {hands_actions!r}")
    if len(hands_actions) != len(farm["hands"]):
        raise IllegalAction(
            f"hands 動作數 {len(hands_actions)} != 雇工數 {len(farm['hands'])}"
        )

    unit_actions = [farmer_action, *hands_actions]
    plant_demand = {}
    for unit_action in unit_actions:
        if (
            isinstance(unit_action, list)
            and len(unit_action) >= 2
            and unit_action[0] == "PLANT"
        ):
            crop = unit_action[1]
            plant_demand[crop] = plant_demand.get(crop, 0) + 1
    for crop, count in plant_demand.items():
        if crop not in CROPS:
            raise IllegalAction(f"未知作物: {crop!r}")
        available = private["seeds"].get(crop, 0)
        if count > available:
            raise IllegalAction(
                f"PLANT {crop} 同回合要求 {count} 顆，但只有 {available} 顆"
            )

    board_size = len(farm["tiles"])
    day = int(obs["day"])
    turns_per_day = int(_cfg(config, "turnsPerDay", 24))
    shed_capacity = int(_cfg(config, "shedCapacity", 100))

    for idx, unit_action in enumerate(unit_actions):
        if not isinstance(unit_action, list) or not unit_action:
            raise IllegalAction(f"unit {idx} 動作格式錯誤: {unit_action!r}")
        before_farm = copy.deepcopy(farm)
        before_private = copy.deepcopy(private)
        try:
            _apply(
                farm,
                private,
                idx,
                unit_action,
                board_size,
                day,
                turns_per_day,
                shed_capacity,
            )
        except Exception as exc:
            raise IllegalAction(
                f"unit {idx} 動作 {unit_action!r} 無法解析: {exc}"
            ) from exc
        if unit_action[0] != "PASS" and farm == before_farm and private == before_private:
            raise IllegalAction(f"unit {idx} 動作會被靜默忽略: {unit_action!r}")


def _unit_price(op, item, market):
    params = market.get("params")
    if op == "SELL" and item in PRODUCTS:
        return market_price(item, market["inventory"][item], params)
    if op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
        return market_price(item, market["inventory"][item] - 1, params)
    if op == "BUY_SEED" and item in CROPS:
        return CROPS[item]["seed"]
    if op == "BUY_ANIMAL" and item in ANIMALS:
        return ANIMALS[item]["cost"]
    return None


def _validate_market(obs, config, action, farm, private):
    orders = action.get("market")
    if not isinstance(orders, list):
        raise IllegalAction(f"market 必須是 list: {orders!r}")
    max_orders = int(_cfg(config, "maxMarketOrdersPerTurn", 10))
    if len(orders) > max_orders:
        raise IllegalAction(f"市場訂單 {len(orders)} 筆，超過上限 {max_orders}")

    board_size = len(farm["tiles"])
    hire_mult = int(_cfg(config, "farmHandCostMult", 1))
    shed_capacity = int(_cfg(config, "shedCapacity", 100))
    market = copy.deepcopy(obs["market"])

    for order_idx, order in enumerate(orders):
        if not isinstance(order, list) or not order:
            raise IllegalAction(f"market[{order_idx}] 格式錯誤: {order!r}")
        op = order[0]
        if op == "HIRE":
            if len(order) != 1:
                raise IllegalAction(f"HIRE 不接受參數: {order!r}")
            before = (copy.deepcopy(farm), copy.deepcopy(private))
            _do_hire(farm, private, board_size, hire_mult)
            if farm == before[0] and private == before[1]:
                raise IllegalAction(f"market[{order_idx}] HIRE 資金不足")
            continue
        if op == "BUY_LAND":
            if len(order) != 1:
                raise IllegalAction(f"BUY_LAND 不接受參數: {order!r}")
            if len(farm["unlocked_quadrants"]) - 1 >= len(LAND_ORDER):
                raise IllegalAction(f"market[{order_idx}] 已無土地可買")
            before = copy.deepcopy(farm)
            _do_buy_land(farm, board_size)
            if farm == before:
                raise IllegalAction(f"market[{order_idx}] BUY_LAND 資金不足")
            continue

        if op not in ("SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"):
            raise IllegalAction(f"market[{order_idx}] 未知 op: {op!r}")
        if len(order) != 3:
            raise IllegalAction(f"market[{order_idx}] 需要 [op, item, n]: {order!r}")
        item = order[1]
        try:
            quantity = int(order[2])
        except (TypeError, ValueError) as exc:
            raise IllegalAction(f"market[{order_idx}] 數量不是整數: {order!r}") from exc
        if quantity <= 0:
            raise IllegalAction(f"market[{order_idx}] 數量必須 > 0: {order!r}")

        for unit_idx in range(quantity):
            price = _unit_price(op, item, market)
            if price is None:
                raise IllegalAction(f"market[{order_idx}] 不支援的商品: {order!r}")
            if not _commit(
                op,
                item,
                price,
                farm,
                private,
                market,
                shed_capacity,
            ):
                raise IllegalAction(
                    f"market[{order_idx}] 第 {unit_idx + 1}/{quantity} 個無法成交: "
                    f"{order!r}"
                )
        _refresh_prices(market)


def assert_legal_action(obs, config, action):
    """若 action 在目前 observation 下會產生確定的 no-op，立即拋錯。"""
    assert_observation_invariants(obs)
    if not isinstance(action, dict):
        raise IllegalAction(f"action 必須是 dict: {action!r}")
    missing = {"farmer", "hands", "market"} - set(action)
    if missing:
        raise IllegalAction(f"action 缺少欄位: {sorted(missing)}")
    extra = set(action) - {"farmer", "hands", "market"}
    if extra:
        raise IllegalAction(f"action 有未知欄位: {sorted(extra)}")

    player = int(obs["player"])
    farm = copy.deepcopy(obs["farms"][player])
    private = copy.deepcopy(obs["private"])
    _validate_units(obs, config, action, farm, private)
    _validate_market(obs, config, action, farm, private)

