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
    """凍結檔的內容指紋，**行尾正規化成 LF**。

    ⚠️ 不要拿掉 `.replace(b"\\r\\n", b"\\n")`。`core.autocrlf=true` 的
    Windows checkout 會把 repo 裡的 LF 換成 CRLF（實測 `ref-v7.json`
    工作目錄 2100 bytes / 70 個 CRLF，git blob 2030 bytes / 0 個），
    直接 hash 位元組的話同一份內容在不同機器上算出不同值，六個 ref
    測試全紅 —— 而那跟「參數有沒有被改」完全無關。

    下面那些期望值是 **LF 版**的 SHA-256，也就是 git 實際存的內容。
    正規化之後仍然抓得到任何真正的內容變更（改參數、改縮排、改鍵順序）。
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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


def test_ref_v6_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v6.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "4be39da54229ccb7c365bf814667ced2208e2413b61666242c192ebb09a7b721"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["n_structures"] == 12
    assert spec["params"]["strawberry_high_demand"] is None


def test_ref_v7_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v7.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "25e83d0142559a87faf368faaa0eb51872418ffd0de4f878cf29afb1d6075aba"
    assert spec["engine_version"] == "1.32.7"
    assert "optimal_assignment" not in spec["params"]


def test_ref_v8_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v8.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "8d6c7855cac0aabc2c8c3ed42b3356a7c30062f83d40b1b0d8459a7fb0941c66"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["optimal_assignment"] is True
    assert "daytime_return_min_value" not in spec["params"]


def test_ref_v9_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v9.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "5cc909c4b31463cacfc65cf9af9c2996262b70c7bacbd78abd12db86c024413d"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["daytime_return_min_value"] == 300
    assert "avoid_last_hour_planting" not in spec["params"]


def test_ref_v10_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v10.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "73b6a0c5eb3fe6b21bc9b38a7a4f8b3cc8caddfcc807a3bafaa72580d9936046"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["avoid_last_hour_planting"] is True
    assert "sell_same_turn_returns" not in spec["params"]


def test_ref_v11_expands_every_gen1_default():
    path = REPO_ROOT / "config/opponents/ref-v11.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "16eb6d2bd5fbdf4691b8c1776af9c1404fbeaece539817705e760afe0220f9ad"
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
