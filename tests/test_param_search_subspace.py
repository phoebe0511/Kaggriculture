"""`tools/param_search.py` 的子空間搜尋（`--only`）與暖啟動起點。

盯的是三件「錯了不會報錯，只會安靜地搜錯東西」的事：

1. **沒搜的維度要留在起點**，不能倒回 `gen0` 的預設值。
2. **起點要帶得動 `DEFAULT_PARAMS` 沒有的 key**。`config/params/*.json` 有兩個
   （`whole_turn_assignment` / `priority_step_cost`），掉了的話整輪 CMA-ES 是在
   搜一個沒有那個 +7pp 演算法的 agent。
3. **`--only` 的順序由 `KEYS` 決定**，不是使用者打字的順序 —— 向量位置對不上
   就全錯，而且看起來一切正常。

這裡全部是純函式，不開對局。
"""
from __future__ import annotations

import json

import pytest

from agents.gen0 import DEFAULT_PARAMS
from tools.param_search import REPO_ROOT, load_start, subspace
from tools.param_space import DIM, KEYS, SUBSETS, decode, encode, resolve_subset

SHIPPED = REPO_ROOT / "config/params/cma1-g50-wt.json"


# --------------------------------------------------------------------------
# 1. 沒搜的維度留在起點
# --------------------------------------------------------------------------

def test_expand_only_touches_the_active_dimensions():
    active = ("n_structures", "seed_backlog")
    idx, expand = subspace(active)
    assert idx == [KEYS.index(k) for k in active]

    base = [0.5] * DIM
    x = expand([0.1, 0.9], base)
    assert len(x) == DIM
    for i, v in enumerate(x):
        want = {idx[0]: 0.1, idx[1]: 0.9}.get(i, 0.5)
        assert v == want, f"第 {i} 維（{KEYS[i]}）被動到了"


def test_no_subset_means_the_whole_space():
    idx, expand = subspace(None)
    assert idx is None
    z = [0.25] * DIM
    assert expand(z, [0.75] * DIM) == z, "沒給子集時 base 不該有任何影響"


def test_expand_result_decodes_to_the_start_values_on_frozen_dims():
    """展開後 decode 出來的參數，沒搜的那些 key 要等於起點的值。

    🩸 這是 `--only` 的整個重點：`cma1-g50-wt` 那 26 個維度是上一輪搜出來的，
    倒回預設值等於把那 108 代丟掉。
    """
    x_base, base_params, _how = load_start(SHIPPED)
    active = resolve_subset("s65")
    _idx, expand = subspace(active)

    moved = expand([0.0] * len(active), x_base)
    start_params = decode(x_base, base_params)
    end_params = decode(moved, base_params)

    frozen = [k for k in KEYS if k not in active and "." not in k]
    for k in frozen:
        assert start_params[k] == end_params[k], f"{k} 沒被搜卻變了"


# --------------------------------------------------------------------------
# 2. 起點要帶得動預設表沒有的 key
# --------------------------------------------------------------------------

def test_load_start_from_a_params_config_carries_the_extra_keys():
    x, base, how = load_start(SHIPPED)
    assert len(x) == DIM
    assert "params" in how
    extra = set(base) - set(DEFAULT_PARAMS)
    assert extra == {"whole_turn_assignment", "priority_step_cost"}, extra
    assert decode(x, base)["whole_turn_assignment"] is True


def test_load_start_from_a_best_json_has_no_base(tmp_path):
    """`best.json` 存的是向量，沒有 params 那層 —— base 只能是 None。"""
    bp = tmp_path / "best.json"
    bp.write_text(json.dumps({"x": [0.5] * DIM}), encoding="utf-8")
    x, base, how = load_start(bp)
    assert x == [0.5] * DIM
    assert base is None
    assert "x" in how


def test_load_start_rejects_a_json_that_is_neither(tmp_path):
    bp = tmp_path / "junk.json"
    bp.write_text(json.dumps({"note": "hi"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_start(bp)


def test_the_start_point_round_trips_to_itself():
    """`encode` 之後 `decode` 回來，起點要落在原本那組參數上。

    漂掉的話 CMA-ES 的第 0 代就不是從出貨的那組出發，而 log 上看不出來。
    """
    x, base, _how = load_start(SHIPPED)
    restored = decode(x, base)
    with open(SHIPPED, encoding="utf-8") as f:
        shipped = json.load(f)["params"]
    for k, v in shipped.items():
        if k in ("basket", "late_crew_caps", "crop_share"):
            continue                        # tuple/list 與展開，另有測試
        assert restored[k] == v, k


# --------------------------------------------------------------------------
# 3. 順序與錯誤處理
# --------------------------------------------------------------------------

def test_subset_is_ordered_by_keys_not_by_the_user():
    got = resolve_subset("tiles_per_unit,crop_share.WHEAT,n_structures")
    assert got == tuple(k for k in KEYS if k in set(got))
    assert got.index("crop_share.WHEAT") < got.index("n_structures")


def test_named_subset_and_comma_list_agree():
    assert resolve_subset("s65") == resolve_subset(",".join(SUBSETS["s65"]))


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="不在 SEARCH_SPACE"):
        resolve_subset("max_crop_share")     # 2026-09-01 移出搜尋空間了


def test_empty_subset_is_rejected():
    with pytest.raises(ValueError, match="空的"):
        resolve_subset(",,")


def test_s65_covers_every_dimension_added_this_round():
    """這一輪新加的七個維度一定要在 `s65` 裡 —— 少一個就白跑。"""
    new_dims = {"structure_spread", "seed_backlog"} | {
        f"crop_share.{c}"
        for c in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}
    assert new_dims <= set(SUBSETS["s65"])


def test_encoded_start_is_inside_the_bounds():
    """起點落在界外的話 CMA-ES 的 bounds 會把它夾回來 —— 那就不是那組參數了。"""
    x, _base, _how = load_start(SHIPPED)
    outside = [(KEYS[i], v) for i, v in enumerate(x) if not 0.0 <= v <= 1.0]
    assert not outside, outside
    # 夾在 0 或 1 的維度代表界線訂得比出貨值還窄，encode 會靜靜地截斷
    pinned = [(KEYS[i], v) for i, v in enumerate(x) if v in (0.0, 1.0)]
    allowed = {"structure_spread", "seed_backlog", "cash_reserve_days"}
    assert {k for k, _v in pinned} <= allowed, pinned
