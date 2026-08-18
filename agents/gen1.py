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
    "n_structures": 12,
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


def act(obs, config=None, params=None):
    resolved = dict(DEFAULT_PARAMS)
    if params:
        resolved.update(params)
    return gen0_act(obs, config, resolved)


def agent(obs, config):
    return act(obs, config)
