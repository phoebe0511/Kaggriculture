"""第 1 代規則式 AI：以 unit 產能管理 active tiles，並按需澆水。

Gen0 保持為 ref-v2 的凍結量尺；Gen1 只疊加已通過 30-seed 消融與 60 局
paired 對戰的參數。額外 `params` 仍可覆寫，供四地與後續 sweep 使用。
"""

from __future__ import annotations

from agents.gen0 import act as gen0_act


DEFAULT_PARAMS = {
    "max_quadrants": 3,
    "tiles_per_unit": 3,
    "water_on_demand": True,
}


def act(obs, config=None, params=None):
    resolved = dict(DEFAULT_PARAMS)
    if params:
        resolved.update(params)
    return gen0_act(obs, config, resolved)


def agent(obs, config):
    return act(obs, config)
