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


def test_ref_v3_stays_frozen_at_its_original_params():
    """ref-v3 是 2026-08-16 凍結的 Gen1 三地量尺（tiles_per_unit=3）。

    tiles_per_unit 升到 4 之後，「展開現行預設」的角色交給 ref-v4；ref-v3 保留
    成歷史量尺，參數永遠不准動 —— 改了的話 08-16 那批 sweep 結果就沒得比。
    """
    spec = json.loads(
        (REPO_ROOT / "config/opponents/ref-v3.json").read_text(encoding="utf-8")
    )
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["tiles_per_unit"] == 3
    assert spec["params"]["max_quadrants"] == 3
    assert spec["params"]["water_on_demand"] is True


def test_ref_v4_expands_every_gen1_default():
    spec = json.loads(
        (REPO_ROOT / "config/opponents/ref-v4.json").read_text(encoding="utf-8")
    )
    assert spec["engine_version"] == "1.32.7"
    # DEFAULT_PARAMS 再變動時這裡應該失敗並要求建立 ref-v5；
    # 不可以為了讓測試通過而修改已凍結的 ref-v4。
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
