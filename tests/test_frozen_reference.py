from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    LAND_ORDER,
    MARKET_PARAMS,
)

from agents.gen0 import DEFAULT_PARAMS

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


# ⚠️ 2026-08-21 這九個指紋整批換過一次。原因是 `agents/gen1.py` 併進
# `agents/gen0.py`（它本來就只是「gen0 + 一組參數」），所有 config 的
# `entry` 從 `agents.gen1:act` 改成 `agents.gen0:act`。
#
# **params 一個 key 都沒動** —— 改之前逐檔比對過「拿掉 entry 之後的 JSON
# 是否逐項相同」，全部成立。下面每條測試的 params 斷言也都原封不動，
# 它們才是真正在守參數的東西；SHA 守的是「有沒有人手改過這個檔」。


def test_ref_v3_stays_frozen_at_its_original_params():
    """ref-v3 是 2026-08-16 凍結的 Gen1 三地量尺（tiles_per_unit=3）。

    tiles_per_unit 升到 4 之後，「展開現行預設」的角色交給 ref-v4；ref-v3 保留
    成歷史量尺，參數永遠不准動 —— 改了的話 08-16 那批 sweep 結果就沒得比。
    """
    path = REPO_ROOT / "config/opponents/ref-v3.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "f756ed8223a73c971c4d283db5b8988f72b9ec8fb4403806eddff61ca06ca1bc"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["tiles_per_unit"] == 3
    assert spec["params"]["max_quadrants"] == 3
    assert spec["params"]["water_on_demand"] is True


def test_ref_v4_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v4.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "4a5614d7f9f9b5bf596de01a63b0708d3574c9ac6352b2427bbc74948f970eb2"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["n_structures"] == 6
    assert spec["params"]["tiles_per_unit"] == 4
    assert "hire_margin" not in spec["params"]


def test_ref_v5_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v5.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "da922ee4d5eff01daff6fc8884471ff66b83f4bb2692cb3472fefa8be9bff8cc"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["n_structures"] == 9


def test_ref_v6_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v6.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "64bfb81e9be110ea754d6b74e1a94f555c9327b79f493fb4ecf34ba95cc31fcc"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["n_structures"] == 12
    assert spec["params"]["strawberry_high_demand"] is None


def test_ref_v7_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v7.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "a1a6551be112ffaed40055cace8137eeb3ed1c715e7a2dafa3360d23c13d78ff"
    assert spec["engine_version"] == "1.32.7"
    assert "optimal_assignment" not in spec["params"]


def test_ref_v8_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v8.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "bbe21af7e2c2c577a15646b2d9fd91b4a43be6bb40e078503c97f1fa3ef74f5f"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["optimal_assignment"] is True
    assert "daytime_return_min_value" not in spec["params"]


def test_ref_v9_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v9.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "d892426e51996f4ee5c10af1fdc6c458b93c097154a5e392c7104e2391efb9bf"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["daytime_return_min_value"] == 300
    assert "avoid_last_hour_planting" not in spec["params"]


def test_ref_v10_stays_frozen_at_its_original_params():
    path = REPO_ROOT / "config/opponents/ref-v10.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "53a4c9e9de544cd4b8782b4251abc708a665352b76ee143ac30c5b15dcd165ac"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"]["avoid_last_hour_planting"] is True
    assert "sell_same_turn_returns" not in spec["params"]


def test_ref_v11_expands_every_gen1_default():
    path = REPO_ROOT / "config/opponents/ref-v11.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert _sha256(path) == "dce255a627d788de33a036892a78de660e93504ef4781e1ff96362d8db729a39"
    assert spec["engine_version"] == "1.32.7"
    assert spec["params"] == _json_value(DEFAULT_PARAMS)


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
