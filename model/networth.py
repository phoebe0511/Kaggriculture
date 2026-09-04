"""把一個 observation 換算成非現金資產的金額 —— PPO reward 的 Φ（§83.3）。

用途只有一個：`harness/ppo_rollout.py` 每一步算一次，餵給
`model.ppo.step_rewards()` 當 potential-based shaping 的 Φ：

    r_t = [Δ我方現金 − Δ對手現金] + γΦ(s_{t+1}) − Φ(s_t)

## 為什麼要有它

原本的 reward 是每步的現金增量。整局加總沒有偏誤（γ=1 時 telescoping
剛好等於期末現金差），但**單步的方向是反的**：買種子、雇人、買地、
買動物，每一筆正確的投資在當下都記成扣分，要等十天賣掉才回來 ——
`lam` 得開到 0.998 才傳得回去（§56）。Φ 把那筆錢改記成資產，
投資的當下就是 0。

## Φ 只算自己

`obs["private"]`（倉庫、手上的東西、種子）**只有自己的**，對手的看不到。
potential-based shaping 對任何只讀自己狀態的 Φ 都保證最優策略不變
（Ng et al. 1999），所以零和留在現金項，Φ 只算我方。

## 終端條件是 rollout 補的，不是這裡

Ng 的定理要 Φ(終端) = 0。土地和動物按剩餘天數折舊，第 29 天自動歸零；
**成品沒有** —— 賣不掉的庫存期末一分不值（引擎的期末 reward 只看
`farm["money"]`，kaggriculture.py:961）。所以 `ppo_rollout` 收尾時
**直接送 Φ_T = 0**，最後一步一次認列全部未變現庫存的損失。
那不是懲罰，那是事實。

## 沒有實作：空地的選擇權價值

§83.3 的表列過「空地數 × 種子均價」。§81.2 的因果測試之後拿掉了 ——
把中期空地砍半，六支對手的 margin 只動 −267（SE 2,363），量不到。
放進 Φ 等於把已經被推翻的假設再餵一次給 agent。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._quiet import silenced                      # noqa: E402

# 🩸 import kaggle_environments 會掃遊戲清單並印到 stderr（fd 層級）。
with silenced():
    from kaggle_environments.envs.kaggriculture import kaggriculture as K

CROPS = K.CROPS
ANIMALS = K.ANIMALS
PRODUCTS = set(K.PRODUCTS)
LAND_PRICES = K.LAND_PRICES

#: 認列時點（§83.4）。差別只在田裡的 `yield_units` 什麼時候算成品。
#:
#: `strict`  跟引擎的 HARVEST 條件一致（kaggriculture.py:449）：
#:           `day - planted_day >= first_yield_day` 才認。收不下來的東西
#:           不算資產。代價：哈密瓜/草莓的信用鏈還是 10 天（240 步）。
#: `produce` 澆水長出來就認。信用鏈最短，代價是對還不能收的單位提前認列
#:           —— 小麥/胡蘿蔔種下去的當下 `yield_units` 就是 1
#:           （`_new_plant`，kaggriculture.py:222），等於獎勵「種了不收」。
RECOGNISE = ("strict", "produce")


def net_realisable_value(counts, market_inventory, params=None):
    """一批成品沿價格曲線積分的淨變現價值。`counts` 是 {item: n}。

    🩸 **不能用牌價乘數量。** 引擎賣一個庫存加一（`_commit_unit`，
    kaggriculture.py:657），所以第 n 個單位的價錢是 `market_price(item,
    inv + n - 1)`。實測（§83.5）賣 100 個草莓只實收牌價的 32%，
    第 100 個只賣 1 塊。記牌價會**獎勵囤貨、懲罰賣出**。

    ⚠️ 這是保守估計的相反邊：實際上分幾天賣、城鎮每 4 和 24 步會消耗庫存
    把價格推回去，所以真的分批賣會賣得比這裡算的多。
    """
    total = 0.0
    for item, n in counts.items():
        n = int(n)
        if n <= 0 or item not in PRODUCTS:
            continue
        inv = market_inventory[item]
        for i in range(n):
            total += K.market_price(item, inv + i, params)
    return total


def asset_value(obs, *, recognise="strict", episode_steps=720,
                turns_per_day=24, detail=False):
    """`obs`（自己這一方的觀測）-> 非現金資產金額。現金不算在內。

    入帳規則按會計科目（§83.3）：

        原料（種子）      成本，來不及種到能收的減記 0
        在製品（田裡）    成本（種子價），來不及收成或已經在枯萎的減記 0
        製成品            淨變現價值（沿價格曲線積分）
        生物資產（動物）  買價 × 剩餘天數比例
        固定資產（土地）  買價 × 剩餘天數比例
        建物              0（引擎蓋起來不收錢）
        人手              0（每晚解雇，工資是費用）

    `detail=True` 回傳分科目的 dict（診斷用），否則回傳一個 float。
    """
    if recognise not in RECOGNISE:
        raise ValueError(f"recognise 只能是 {RECOGNISE}，收到 {recognise!r}")

    day = int(obs["day"])
    hour = int(obs["hour"])
    step = day * turns_per_day + hour
    last_day = episode_steps // turns_per_day - 1     # 預設 29
    days_left = max(0, last_day - day)
    ratio = days_left / last_day if last_day > 0 else 0.0

    farm = obs["farms"][int(obs["player"])]
    private = obs["private"]
    market = obs["market"]
    market_inv = market["inventory"]
    params = market.get("params")

    goods = {}          # 待變現的成品，最後一次沿曲線積分
    seeds_v = 0.0
    wip = 0.0
    animals_v = 0.0

    def _hold(item, n):
        if item in ANIMALS:
            # 沒放上建物的動物：一樣按買價折舊，它自己不生產。
            nonlocal animals_v
            animals_v += n * ANIMALS[item]["cost"] * ratio
        else:
            goods[item] = goods.get(item, 0) + n

    # 原料：種子只能種，不能賣回市場（引擎沒有 SELL_SEED）。種下去到收成
    # 要 first_yield_day 天，來不及就是 0。
    for crop, n in private["seeds"].items():
        if n > 0 and days_left >= CROPS[crop]["first_yield_day"]:
            seeds_v += n * CROPS[crop]["seed"]

    for item, n in private["shed"].items():
        if n > 0:
            _hold(item, n)
    for inv in private["inventories"]:
        for item, n in inv.items():
            if n > 0:
                _hold(item, n)

    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue                       # None（空地）/ "LOCKED"
            if tile.get("kind") == "PLANT":
                cd = CROPS[tile["crop"]]
                # 在製品的成本：還收得到才留著。max_lifespan_step >= 0 而且
                # 已經到了 = 這格在掉 yield_units（`_decay_plants`），成本
                # 認不回來了。
                mls = tile.get("max_lifespan_step", -1)
                decaying = 0 <= mls <= step
                in_time = tile["planted_day"] + cd["first_yield_day"] <= last_day
                if in_time and not decaying:
                    wip += cd["seed"]
                units = tile.get("yield_units", 0)
                if units > 0 and (
                        recognise == "produce"
                        or day - tile["planted_day"] >= cd["first_yield_day"]):
                    goods[tile["crop"]] = goods.get(tile["crop"], 0) + units
            elif "animal" in tile:
                a = ANIMALS[tile["animal"]]
                animals_v += a["cost"] * ratio
                units = tile.get("yield_units", 0)
                if units > 0:
                    goods[a["product"]] = goods.get(a["product"], 0) + units
                # `fertilizer_available` 不算 —— 它要花一個 unit-turn 去
                # COLLECT_FERTILIZER 才變成手上的東西，那一步的 +肥料價
                # 就是它的 credit（§83.6 的分錄）。
            # WEED / 空的 COOP / 空的 PASTURE：0

    n_extra = max(0, len(farm["unlocked_quadrants"]) - 1)
    land = sum(LAND_PRICES[:n_extra]) * ratio
    goods_v = net_realisable_value(goods, market_inv, params)

    if detail:
        return {"seeds": seeds_v, "wip": wip, "goods": goods_v,
                "animals": animals_v, "land": land, "units": dict(goods),
                "total": seeds_v + wip + goods_v + animals_v + land}
    return seeds_v + wip + goods_v + animals_v + land
