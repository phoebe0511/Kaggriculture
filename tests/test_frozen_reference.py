from __future__ import annotations

import json
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    LAND_ORDER,
    MARKET_PARAMS,
)

from agents.gen0 import DEFAULT_PARAMS as GEN0_PARAMS
from agents.gen1 import DEFAULT_PARAMS as GEN1_PARAMS


REPO_ROOT = Path(__file__).resolve().parents[1]


def _json_value(value):
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def test_ref_v2_expands_every_gen0_default():
    spec = json.loads(
        (REPO_ROOT / "config/opponents/ref-v2.json").read_text(encoding="utf-8")
    )
    assert spec["engine_version"] == "1.32.7"
    # 若未來 DEFAULT_PARAMS 新增欄位，這裡應該失敗並要求建立 ref-v3；
    # 不可以為了讓測試通過而修改已凍結的 ref-v2。
    assert spec["params"] == _json_value(GEN0_PARAMS)


def test_ref_v3_expands_every_gen1_default():
    spec = json.loads(
        (REPO_ROOT / "config/opponents/ref-v3.json").read_text(encoding="utf-8")
    )
    assert spec["engine_version"] == "1.32.7"
    expected = dict(GEN0_PARAMS)
    expected.update(GEN1_PARAMS)
    assert spec["params"] == _json_value(expected)


def test_engine_rule_fingerprint():
    assert LAND_ORDER == ["NE", "SW", "SE"]
    expected = {
        "CARROT": ("hinge", 1.0),
        "TOMATO": ("hinge", 0.4),
        "EGG": ("hinge", 0.4),
    }
    actual = {
        item: (MARKET_PARAMS[item]["below_func"], MARKET_PARAMS[item]["below_target"])
        for item in expected
    }
    assert actual == expected
