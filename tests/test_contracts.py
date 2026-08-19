"""`contracts.py` 的 schema 與編碼保證。

這裡釘住的是「陣列的第 k 個位置代表什麼」。訓練好的權重綁死在這上面，
改了不會報錯、只會讓分數莫名其妙掉 —— 所以每一項都要有測試擋著。
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pytest
from kaggle_environments import make

import contracts as C

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 進版控的 replay（只有動作，沒有 observation）。
EPISODE_DIR = REPO_ROOT / "config" / "episodes"
#: 原始 replay（含 observation），`temp/` 被 gitignore，同伴的工作目錄不會有。
RAW_DIR = REPO_ROOT / "temp" / "episodes"


def _fresh_observation():
    """跑一步真實引擎，拿一個貨真價實的 observation。"""
    env = make("kaggriculture", configuration={"seed": 0}, debug=True)
    env.run(["starter", "starter"])
    step = env.steps[200]
    obs = dict(step[0]["observation"])
    obs["player"] = 0
    return obs, env.configuration


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def test_encoder_version_is_bound_to_schema_size():
    """schema 改了就要升版。

    ⚠️ 這個測試紅掉的正確反應是**升 `ENCODER_VERSION` 並改這裡的數字**，
    順便確認舊權重檔已經作廢。不是把數字改掉了事 —— 那等於把保護拆掉。
    """
    assert C.ENCODER_VERSION == 2
    assert C.N_SPATIAL == 38
    assert C.N_SCALAR == 75
    assert C.N_UNIT_FEATURES == 18
    assert C.N_UNIT_OPS == 44
    assert C.N_QTY == 12


def test_schema_names_are_unique():
    """重複的名字會讓索引表悄悄少一個欄位。"""
    for names in (C.SPATIAL_CHANNELS, C.SCALAR_FIELDS, C.UNIT_FEATURES):
        assert len(names) == len(set(names)), f"重複：{names}"
    assert len(C.UNIT_OPS) == len(set(C.UNIT_OPS))


def test_schema_covers_engine():
    """引擎新增品項時要炸掉，逼人升版。import 時就會跑，這裡是明確再跑一次。"""
    C._assert_covers_engine()


# --------------------------------------------------------------------------
# encode
# --------------------------------------------------------------------------

def test_encode_shapes_and_ranges():
    obs, config = _fresh_observation()
    spatial, scalar = C.encode(obs, config)

    assert spatial.shape == (C.N_SPATIAL, 10, 10)
    assert spatial.dtype == np.float32
    assert scalar.shape == (C.N_SCALAR,)
    assert scalar.dtype == np.float32
    assert np.isfinite(spatial).all()
    assert np.isfinite(scalar).all()


def test_encode_units_aligns_with_engine_unit_order():
    """索引 0 是主農夫、1 以後是 hands —— 跟 `action["hands"]` 同順序。"""
    obs, config = _fresh_observation()
    farm = obs["farms"][0]
    pos, feats = C.encode_units(obs, config)

    assert pos.shape == (1 + len(farm["hands"]), 2)
    assert feats.shape == (1 + len(farm["hands"]), C.N_UNIT_FEATURES)
    assert tuple(pos[0]) == tuple(farm["farmer"])
    for i, hand in enumerate(farm["hands"]):
        assert tuple(pos[i + 1]) == tuple(hand)
    assert feats[0, C.UNIT_FEATURES.index("is_farmer")] == 1.0


# --------------------------------------------------------------------------
# 動作編碼 / 解碼
# --------------------------------------------------------------------------

def test_decode_always_supplies_required_arguments():
    """引擎對少帶參數的動作是**靜默忽略**（`["PLACE"]` -> no-op）。"""
    for index, (op, item) in enumerate(C.UNIT_OPS):
        action = C.decode_unit(index, qty_index=0)
        assert action[0] == op
        if op == "PLANT":
            assert len(action) == 2 and action[1] == item
        elif op in ("PICKUP", "PLACE"):
            assert len(action) == 3 and action[1] == item
            assert action[2] in C.QTY_CHOICES
        else:
            assert len(action) == 1


def test_action_round_trip():
    for index in range(C.N_UNIT_OPS):
        for qty_index in (None, 0, 5, C.N_QTY - 1):
            op, item = C.UNIT_OPS[index]
            action = C.decode_unit(index, qty_index)
            back = C.encode_unit_action(action)
            assert back is not None, action
            assert back[0] == index, action
            if op in ("PICKUP", "PLACE"):
                expected = 0 if qty_index is None else qty_index
                assert back[1] == expected, action


def test_unknown_action_returns_none_instead_of_silently_passing():
    """認不得就回 None。當成 PASS 會讓標籤髒掉而且查不出來。"""
    assert C.encode_unit_action(["NOT_AN_OP"]) is None
    assert C.encode_unit_action(["PLANT", "PINEAPPLE"]) is None
    assert C.encode_unit_action([]) is None
    assert C.encode_unit_action(None) is None


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def _corpus_files(directory, limit=None):
    files = sorted(glob.glob(os.path.join(str(directory), "*.json")))
    return files[:limit] if limit else files


def test_vocabulary_covers_every_action_in_the_corpus():
    """60 局 ladder replay 裡的每一個 unit 動作都要編得出來。

    編不出來的動作在訓練時只能整筆丟掉，而丟掉的分布不是隨機的 ——
    會系統性地少學某一類操作。
    """
    files = _corpus_files(EPISODE_DIR)
    if not files:
        pytest.skip(f"{EPISODE_DIR} 是空的")

    unknown = {}
    total = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for row in data["actions"]:
            for action in row:
                if not isinstance(action, dict):
                    continue
                units = [action.get("farmer")] + list(action.get("hands") or [])
                for unit_action in units:
                    if not isinstance(unit_action, list) or not unit_action:
                        continue
                    total += 1
                    if C.encode_unit_action(unit_action) is None:
                        key = tuple(unit_action[:2])
                        unknown[key] = unknown.get(key, 0) + 1

    assert total > 100_000, f"replay 只有 {total} 個 unit-turn，太少"
    assert not unknown, f"詞彙表漏了：{unknown}"


def test_mask_rarely_blocks_what_the_teacher_actually_did():
    """`legal_unit_mask` 的不變式：寧可放寬，絕不擋掉老師做過的動作。

    擋掉 = 網路學不到那個動作。實測歷史：

        14.66%  observation 與 action 錯位一格（steps[t] 的動作是在 t-1 決定的）
         1.62%  HARVEST 只允許作物，漏了動物的 EGG / MILK / WOOL
         0.12%  現況 —— 分散在各 op，是老師自己送出的無效動作

    所以門檻設 0.5%：兩個真正的 bug 都遠超過它，而老師的無效動作遠低於它。
    """
    files = _corpus_files(RAW_DIR, limit=3)
    if not files:
        pytest.skip(f"{RAW_DIR} 沒有原始 replay（temp/ 被 gitignore）")

    blocked = total = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        config = data["configuration"]
        steps = data["steps"]
        for t, entries in enumerate(steps[:-1]):
            shared = entries[0]["observation"]
            for player in (0, 1):
                action = steps[t + 1][player].get("action")
                if not isinstance(action, dict):
                    continue
                obs = dict(shared)
                obs.update(entries[player]["observation"])
                obs["player"] = player
                obs.setdefault("step", t)

                mask = C.legal_unit_mask(obs, config)
                units = [action.get("farmer")] + list(action.get("hands") or [])
                for i, unit_action in enumerate(units):
                    if i >= mask.shape[0]:
                        continue      # 送出的 hands 比實際雇工多，引擎會略過
                    if not isinstance(unit_action, list) or not unit_action:
                        continue
                    decoded = C.encode_unit_action(unit_action)
                    if decoded is None:
                        continue
                    total += 1
                    if not mask[i, decoded[0]]:
                        blocked += 1

    assert total > 10_000
    rate = blocked / total
    assert rate < 0.005, f"擋掉了 {blocked}/{total} = {rate:.3%}"
