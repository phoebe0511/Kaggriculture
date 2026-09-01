"""`_plan_basket` 尾聲補位不能繞過 `crop_share`。

🩸 2026-09-01 量到：`crop_share` 設 CARROT=0 之後配額確實掉到 1 格，但
day 21 之後 STRAWBERRY / MELON 的 `ypd` 變 0 被排除，配額只用掉 20 格，
剩下 43 格被 `alloc[min(CROPS, key=cycle)] += left` 全部倒給週期最短的
CARROT（4 天，WHEAT 5 天）。basket 變成 CARROT 44、day 29 是 54 格全胡蘿蔔。
所以「不要種這個」在舊實作裡講不出來。

這裡守兩件事：預設設定下逐項不變（沒有作物的 share 是 0），
以及 share=0 的作物不能被補位塞進來。
"""

from __future__ import annotations

import pytest

from agents.gen0 import DEFAULT_PARAMS, _plan_basket, crop_cycle
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

MAX_SHARE = 0.2874364458062525          # cma1-g50-wt 的 max_crop_share
OVERSUPPLY = 2.8930691298383002
DEMAND = tuple(sorted({"WHEAT": 12.0, "CARROT": 6.0, "MELON": 3.0,
                       "STRAWBERRY": 20.0, "TOMATO": 4.0}.items()))
INV = tuple((c, 10_000) for c in sorted(CROPS))


def basket(days_left, shares, n=63):
    return _plan_basket(days_left, DEMAND, INV, n, "WHEAT", MAX_SHARE,
                        OVERSUPPLY, (("MELON", 4.0),),
                        tuple(sorted(shares.items())), 0, 720)


@pytest.mark.parametrize("days_left", [30, 20, 12, 9, 5, 3, 1])
def test_default_shares_still_fill_with_the_shortest_cycle_crop(days_left):
    """預設 `crop_share` 只列 STRAWBERRY，其餘走 max_crop_share（> 0），
    所以候選是全部作物、補位仍然挑週期最短的 —— 出貨版吃的就是這條路。"""
    shortest = min(CROPS, key=lambda c: crop_cycle(c)[0])
    assert shortest == "CARROT"          # 前提：4 天，WHEAT 是 5 天
    b = basket(days_left, dict(DEFAULT_PARAMS["crop_share"]))
    assert len(b) in (63, 1)             # 63 格填滿，或 fallback
    if len(b) == 63 and days_left <= 9:
        # 尾聲只有短週期作物排得進去，補位會集中在 CARROT
        assert b.count(shortest) > 0


@pytest.mark.parametrize("days_left", [20, 12, 9, 5, 3])
def test_share_zero_crops_are_never_used_to_fill(days_left):
    shares = {"STRAWBERRY": 0.4, "CARROT": 0.0, "TOMATO": 0.0}
    b = basket(days_left, shares)
    # cap = max(1, int(n * 0.0)) = 1，所以最多各 1 格，不能被補位灌爆
    assert b.count("CARROT") <= 1, dict((c, b.count(c)) for c in set(b))
    assert b.count("TOMATO") <= 1, dict((c, b.count(c)) for c in set(b))


def test_late_season_fills_with_wheat_when_carrot_is_off():
    """day 21（days_left=9）：舊實作給 CARROT 44，修好之後應該是 WHEAT。"""
    b = basket(9, {"STRAWBERRY": 0.4, "CARROT": 0.0, "TOMATO": 0.0})
    counts = {c: b.count(c) for c in set(b)}
    assert counts.get("WHEAT", 0) > 40, counts
    assert counts.get("CARROT", 0) <= 1, counts


def test_all_shares_zero_falls_back_instead_of_crashing():
    """全部設 0 也不能炸 —— `pool` 空的時候退回全部作物。"""
    b = basket(9, {c: 0.0 for c in CROPS})
    assert len(b) >= 1
