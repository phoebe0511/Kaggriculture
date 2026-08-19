"""第 1 代規則式 AI：以 unit 產能管理 active tiles，並按需澆水。

Gen0 保持為 ref-v2 的凍結量尺；Gen1 只疊加已通過 30-seed 消融與 60 局
paired 對戰的參數。額外 `params` 仍可覆寫，供四地與後續 sweep 使用。
"""

from __future__ import annotations

from agents.gen0 import act as gen0_act


DEFAULT_PARAMS = {
    "max_quadrants": 3,
    # 94144753 的敗局中，第 10 隻動物買回後沒有可用建物，整季躺在 shed。
    # 12 格對舊 9 格直接互打 30 個 paired seeds（60 局）為 52 勝 8 負，
    # 平均現金 +$7,001；對 starter 的平均現金也從 $96,500 升到 $103,561。
    # 多出的格子仍由 dynamic_animals 按商店需求分配，不固定押單一物種。
    # 第二輪在新分派器上重掃 13/14/15 格，各 24 局都由 12 格明顯勝出
    # （平均 +$6,264 / +$11,997 / +$12,771），所以仍維持 12。
    "n_structures": 12,
    # 不能因為對手先養了某物種就把整個市場讓出去。只折算一半對手產能，
    # 並確保自己至少規劃城鎮需求的 40%。
    "opponent_supply_weight": 0.5,
    "animal_demand_share_floor": 0.4,
    # 作物規劃除了城鎮需求，也先替預計動物數的一半準備 WHEAT；動物進場後
    # 按實際數量提高，避免一面低價賣作物、一面高價買回大量飼料。
    "feed_crop_bootstrap_share": 0.5,
    "feed_crop_demand_weight": 1.0,
    # 兩間以上會消耗草莓的商店出現時，才把草莓上限由 40% 放到 70%。
    "strawberry_high_demand": 13,
    "strawberry_high_share": 0.7,
    # 前半季商店資訊不足，最多先鎖兩格 COOP；其餘位置等更多商店公開。
    "early_coop_limit": 2,
    "early_coop_until_day": 15,
    # 保留可切換的「先賣再消費」流程，但 30-seed 消融平均少 $1,064，且土地
    # 仍是 day 9 才買，正式版維持原訂單順序。提早擴張要另做精準條件，不全域搬單。
    "sell_before_spending": False,
    # 肥料本身可以賣錢：只在 ongoing 作物未來三天的額外產品價值至少等於
    # 肥料售價時才施用；one-time 通常只提早滿產、不增加整輪總量，預設不施。
    # 對 ref-v7 的 60 局為 46 勝 14 負，平均現金 +$3,311。
    "fertilizer_roi_margin": 1.0,
    "fertilize_one_time": False,
    # 最後兩天不再維持完整 12 人：剩 2 天上限 10，最後一天上限 8。
    # 從剩 6 天就降到 11 的版本反而少賺；保守版與 ROI 合併後再多約 $150。
    "late_crew_caps": ((2, 10), (1, 8)),
    # 同一優先序的任務做全域最短配對，不用固定象限。直接對打舊的逐筆貪婪
    # 分派器 60 局為 58 勝 2 負，平均現金 +$8,686，買地時點完全相同。
    "optimal_assignment": True,
    # 閒置 unit 若攜帶至少 $300 的非 WHEAT 成品，且距 shed 不超過 4 步，
    # 白天就以指定品項 PLACE 回倉，不必等日終自動卸貨。兩批獨立 paired seeds：
    #   seeds 0-19：29 勝 11 負，平均現金 +$1,946
    #   seeds 1000-1029：43 勝 17 負，平均現金 +$2,185
    # 使用指定品項而非 DROP，避免把飼料、肥料或待安置動物一起清掉。
    "daytime_return_min_value": 300,
    "daytime_return_max_distance": 4,
    # 引擎在新苗建立時就記 1 次未澆水；若於 hour 23 才播種，當晚結算會
    # 直接累加至 2 並變成 WEED。對 ref-v9 的兩批獨立 paired seeds 合計
    # 92 勝 8 負，平均現金約 +$4.2k；tetsuya 敗局的 seed 正反兩局也全勝。
    "avoid_last_hour_planting": True,
    # active tiles 上限 = (1 + planned_hands) × 這個值。12 個 hand + farmer
    # → 4 給出 52 格（三地可種 69 格）。
    #
    # 3 → 4 的證據（2026-08-16，對 ref-v3 各 40 局 paired，兩組獨立種子）：
    #   seeds 0-19     32 勝 8 負 = 80.0%   CI [65%, 90%]
    #   seeds 1000-19  29 勝 11 負 = 72.5%  CI [57%, 84%]
    # 作物覆蓋率在 t=4 就飽和（56.4%），再往上只多買到走路：
    #   t=4.5 / 5 / 無上限的勝率是 42.5% / 12.5% / 17.5%。
    "tiles_per_unit": 4,
    "water_on_demand": True,
}


def _resolve_params(params=None):
    overrides = dict(params or {})
    replace = bool(overrides.get("_replace_defaults", False))
    resolved = {} if replace else dict(DEFAULT_PARAMS)
    resolved.update(overrides)
    return resolved


def act(obs, config=None, params=None):
    resolved = _resolve_params(params)
    return gen0_act(obs, config, resolved)


def agent(obs, config):
    return act(obs, config)
