from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    LAND_ORDER,
    MARKET_PARAMS,
)

from agents.gen0 import DEFAULT_PARAMS as GEN0_PARAMS
from agents.gen1 import DEFAULT_PARAMS as GEN1_PARAMS


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value):
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def test_ref_v2_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v2.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "92854b5e8e2768e7e8863019937bf93c9e10fe9845dd0dbbfb771362a7d5726c"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["n_structures"] == 6
    assert "hire_margin" not in spec["params"]


def test_ref_v3_stays_frozen_at_its_original_params():
    """ref-v3 是 2026-08-16 凍結的 Gen1 三地量尺（tiles_per_unit=3）。

    tiles_per_unit 升到 4 之後，「展開現行預設」的角色交給 ref-v4；ref-v3 保留
    成歷史量尺，參數永遠不准動 —— 改了的話 08-16 那批 sweep 結果就沒得比。
    """
    path = REPO_ROOT / "config/opponents/ref-v3.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "6035a293e740407388ec496627dc8f31cdfd7b904d0bb903f58820d6d7b3fc8f"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["tiles_per_unit"] == 3
    assert spec["params"]["max_quadrants"] == 3
    assert spec["params"]["water_on_demand"] is True


def test_ref_v4_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v4.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "bd9a11f82ee3ea26327dc9255b6d5acd81d1eaea82de81562cf3190c41958b41"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["n_structures"] == 6
    assert spec["params"]["tiles_per_unit"] == 4
    assert "hire_margin" not in spec["params"]


def test_ref_v5_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v5.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "40173b7129f738d8f4c7fa1c4bdd7dc82cc956e3a2ef7fa7b6118ff49854b4eb"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["n_structures"] == 9


def test_ref_v6_expands_every_gen1_default():
    spec = json.loads(
        (REPO_ROOT / "config/opponents/ref-v6.json").read_text(encoding="utf-8")
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
