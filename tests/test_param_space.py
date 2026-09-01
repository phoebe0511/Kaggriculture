"""`tools/param_space.py` 的 L0 保護。

盯四件事，全部是「寫錯了會安靜地搜一個錯空間」的類型：

1. **round-trip 不能漂。** `decode(encode(DEFAULT_PARAMS))` 要逐項等於原值 ——
   漂了的話 CMA-ES 的起點就不是人手調過的那組參數，而且不會報錯。
2. **界線兩端要合法。** `decode` 只保證合法不保證合理，但「合法」是硬要求：
   x 全 0 / 全 1 都要跑得完一局。
3. **決定性。** 同一組 params 跑兩次要逐位元組相同 —— 整個 CMA-ES 設計
   （common random numbers、固定 seed 集）建立在這個前提上。
4. **搜尋空間的形狀。** bool / tuple / str / dict 不准混進連續空間。

⚠️ 這裡的對局刻意用短 episode（`episodeSteps=168`，7 天）。L0 的預算是 60 秒，
完整 720 步一局要 3.6 秒（實測），跑四局就吃掉四分之一的預算。168 步已經涵蓋
播種 / 收成 / 市場買賣，足以驗上面四件事。
"""
from __future__ import annotations

import contextlib
import io

import pytest
from kaggle_environments import make

from agents.gen0 import DEFAULT_PARAMS, act
from tools import param_space
from tools.param_space import (DIM, KEYS, SEARCH_SPACE, decode, encode,
                               read_key, split_key, x0)

#: 短局：7 天。理由見模組 docstring。
SHORT_STEPS = 168


def _play(params, steps=SHORT_STEPS, seed=0):
    """自我對戰一局，回傳兩邊的期末現金。引擎的 stderr 很吵，關掉。"""
    def agent(obs, config):
        return act(obs, config, params)

    with contextlib.redirect_stderr(io.StringIO()):
        env = make("kaggriculture",
                   configuration={"seed": seed, "episodeSteps": steps},
                   debug=False)
        env.run([agent, agent])
    return [state.reward for state in env.state]


# --------------------------------------------------------------------------
# 1. round-trip
# --------------------------------------------------------------------------

def test_round_trip_preserves_every_searched_key():
    """`decode(encode(預設))` 每個搜尋維度都要等於 `DEFAULT_PARAMS` 的現值。

    int 走的是 `int(v + 0.5)` 不是 `round()`（banker's rounding 會在界線邊緣
    偏一格），這條測試就是在守那個選擇。

    展開維度（`crop_share.X`）的「現值」走 `DICT_FALLBACK` —— 預設的
    `crop_share` 只列了 STRAWBERRY，另外四種讀的是 `max_crop_share`。
    """
    restored = decode(x0())
    drift = {k: (read_key(DEFAULT_PARAMS, k), read_key(restored, k))
             for k in KEYS
             if read_key(DEFAULT_PARAMS, k) != read_key(restored, k)}
    assert not drift, f"round-trip 漂掉的 key：{drift}"


def test_expanded_dict_round_trip_fills_every_item():
    """展開之後 `crop_share` 要五種全列出來，值等於展開前**實際生效**的值。

    🩸 這是「換空間不換行為」那條線：展開前 WHEAT 走 `max_crop_share` (0.4)，
    展開後 `crop_share["WHEAT"]` 必須也是 0.4。少填一種、或 fallback 抄錯，
    `_greedy_basket` 的配額就變了 —— 而且不會報錯。
    """
    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

    restored = decode(x0())
    assert set(restored["crop_share"]) == set(CROPS)
    for crop in CROPS:
        expected = DEFAULT_PARAMS["crop_share"].get(
            crop, DEFAULT_PARAMS["max_crop_share"])
        assert restored["crop_share"][crop] == pytest.approx(expected), crop


def test_expanded_crop_share_plays_identically_to_the_unexpanded_form():
    """展開只換表示法，不換行為 —— 同一個 seed 的期末現金要完全相同。

    🩸 這是暖啟動的整個前提：`encode(cma1-g50-wt.json)` 那個落點的行為要等於
    那份 config 本身，不然 CMA-ES 的第 0 代就已經不是從出貨的那組參數出發。
    """
    import json as _json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    with open(root / "config/params/cma1-g50-wt.json", encoding="utf-8") as f:
        shipped = _json.load(f)["params"]
    expanded = decode(encode(shipped), shipped)
    shipped["_replace_defaults"] = True
    assert set(expanded["crop_share"]) == {
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
    assert _play(shipped) == _play(expanded)


def test_decode_without_base_drops_keys_that_predefaults_does_not_have():
    """🩸 `decode` 不給 base 就是從 `DEFAULT_PARAMS` 重建 —— config 裡
    **不在預設表**的 key 會安靜地消失。

    2026-09-01 踩到：`cma1-g50-wt.json` 有 `whole_turn_assignment: True`
    和 `priority_step_cost: 1`（`agents/gen0.py:1724` 用 `params.get()` 讀，
    不放進 `DEFAULT_PARAMS` 是因為那會改掉 `gen1` / `ref-v11` 的行為）。
    暖啟動忘了帶 base 的話，整輪 CMA-ES 會在搜一個**沒有**那個 +7pp 演算法的
    agent，分數只是低一截、不會報錯。

    這條測試釘住的是「差別確實存在」，不是「decode 應該怎麼做」——
    所以 base 要給，`tools/param_search.py` 的 `load_start` 負責帶。
    """
    import json as _json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    with open(root / "config/params/cma1-g50-wt.json", encoding="utf-8") as f:
        shipped = _json.load(f)["params"]
    extra = set(shipped) - set(DEFAULT_PARAMS)
    assert extra == {"whole_turn_assignment", "priority_step_cost"}, extra

    x = encode(shipped)
    assert set(decode(x)) & extra == set(), "沒給 base 卻留住了，斷言的前提變了"
    assert extra <= set(decode(x, shipped)), "給了 base 還是掉了"
    assert _play(decode(x)) != _play(decode(x, shipped))


def test_round_trip_preserves_frozen_keys_untouched():
    """凍結的 key 要原封不動地出現在 decode 的產出裡。"""
    restored = decode(x0())
    for key in param_space.FROZEN_KEYS:
        assert restored[key] == DEFAULT_PARAMS[key], f"{key} 被動到了"


def test_decode_returns_full_snapshot_with_replace_flag():
    """decode 要給完整 51 項 + `_replace_defaults`，才不會跟未來的預設值合併。"""
    params = decode(x0())
    assert set(params) == set(DEFAULT_PARAMS) | {"_replace_defaults"}
    assert params["_replace_defaults"] is True


# --------------------------------------------------------------------------
# 2. 界線兩端
# --------------------------------------------------------------------------

@pytest.mark.parametrize("corner", [0.0, 1.0])
def test_bounds_corner_is_within_declared_range(corner):
    params = decode([corner] * DIM)
    for key, (lo, hi, kind) in SEARCH_SPACE.items():
        value = read_key(params, key)
        assert lo <= value <= hi, f"{key}={value} 掉出 [{lo}, {hi}]"
        if kind == "int":
            assert isinstance(value, int) and not isinstance(value, bool), \
                f"{key} 宣告成 int，decode 給了 {type(value).__name__}"
        else:
            assert isinstance(value, float), \
                f"{key} 宣告成 float，decode 給了 {type(value).__name__}"


@pytest.mark.parametrize("corner", [0.0, 1.0])
def test_bounds_corner_survives_a_game(corner):
    """界線兩端的極端參數要跑得完一局。跑不完代表界線訂錯了。"""
    rewards = _play(decode([corner] * DIM))
    assert all(r is not None for r in rewards), f"x 全 {corner} 跑不完一局"


def test_out_of_range_input_is_clipped_not_raised():
    """CMA-ES 的 bounds 已經在管界線，這是第二道 —— 夾回去，不要拋。"""
    params = decode([-5.0] * DIM)
    for key, (lo, _hi, _kind) in SEARCH_SPACE.items():
        assert read_key(params, key) == lo


# --------------------------------------------------------------------------
# 3. 決定性 —— 整個 CMA-ES 設計的前提
# --------------------------------------------------------------------------

def test_same_params_same_seed_give_identical_cash():
    """同一組 params、同一個 seed 跑兩次，期末現金要完全相同。

    🩸 這條垮了整個 Stage 1/2 的設計就垮了：目標函數會變成有噪音的，
    common random numbers 失效，每個候選都要重複評估很多次才排得動。
    """
    params = decode(x0())
    assert _play(params) == _play(params)


def test_different_params_actually_change_the_outcome():
    """反面：參數真的改了，結果要不一樣。

    不然上面那條決定性測試可能只是因為參數根本沒接線 —— 兩條要一起看。
    """
    baseline = _play(decode(x0()))
    tweaked = _play(decode([1.0] * DIM))
    assert baseline != tweaked, "換了整組極端參數但結果一樣，參數大概沒接上"


# --------------------------------------------------------------------------
# 4. 搜尋空間的形狀
# --------------------------------------------------------------------------

def test_search_space_has_no_bool_or_container():
    """bool / tuple / str / dict 不准混進連續空間 —— 那需要別的處理方式。"""
    for key in KEYS:
        value = read_key(DEFAULT_PARAMS, key)
        assert not isinstance(value, (bool, tuple, str, dict)), \
            f"{key} 是 {type(value).__name__}，不該在連續搜尋空間裡"


def test_search_space_covers_exactly_the_numeric_keys():
    """搜尋 key（展開的算它的 base）+ 凍結 key = `DEFAULT_PARAMS` 的全部。"""
    assert DIM == 40
    bases = {split_key(k)[0] for k in KEYS}
    assert bases | set(param_space.FROZEN_KEYS) == set(DEFAULT_PARAMS)
    assert not bases & set(param_space.FROZEN_KEYS)


def test_max_crop_share_is_frozen_now_that_crop_share_is_expanded():
    """🩸 五種作物全展開之後 `share.get(c, max_share)` 永遠命中，
    `max_crop_share` 在 `_greedy_basket` 就再也讀不到。留在搜尋空間裡的話
    CMA-ES 要花代數才學得會忽略一個對目標函數零影響的方向。

    它仍然要留在 `DEFAULT_PARAMS` —— `_adaptive_crop_shares` 和 2026-09-01
    之前凍的每一份 config 都還讀它。
    """
    assert "max_crop_share" not in SEARCH_SPACE
    assert "max_crop_share" in param_space.FROZEN_KEYS
    assert "max_crop_share" in DEFAULT_PARAMS


def test_engine_derived_bounds_follow_the_engine():
    """有硬上界的 key 要跟著引擎走，不能寫死。

    `docs/CLAUDE.md`：規則常數一律 import 引擎、不做鏡像；本機引擎原始碼被
    人手改過（`LAND_ORDER` 曾被改成 `["NE"]`），寫死的話會安靜地搜錯範圍。
    """
    from kaggle_environments.envs.kaggriculture.kaggriculture import LAND_ORDER

    assert SEARCH_SPACE["max_quadrants"][1] == 1 + len(LAND_ORDER)
    assert param_space.DAYS == 30
    assert SEARCH_SPACE["land_min_days_left"][1] == param_space.DAYS


def test_json_params_are_serialisable_and_flagless():
    """`to_json_params` 要吐得出 JSON，而且不含 `_replace_defaults`。"""
    import json

    payload = param_space.to_json_params(decode(x0()))
    assert "_replace_defaults" not in payload
    assert len(payload) == len(DEFAULT_PARAMS)
    round_tripped = json.loads(json.dumps(payload, ensure_ascii=False))
    assert round_tripped["basket"] == list(DEFAULT_PARAMS["basket"])
    assert round_tripped["late_crew_caps"] == [list(t)
                                               for t in DEFAULT_PARAMS["late_crew_caps"]]


def test_encode_accepts_partial_overrides():
    """只給幾個 key 的話，其餘 fall back 到預設 —— Stage 3 翻 bool 時會用到。"""
    x = encode({"max_hands": 20})
    assert decode(x)["max_hands"] == 20
    assert decode(x)["tiles_per_unit"] == DEFAULT_PARAMS["tiles_per_unit"]


# --------------------------------------------------------------------------
# 5. 落地：in-memory 的 tuple 版 vs config JSON 的 list 版
# --------------------------------------------------------------------------

def test_json_round_trip_params_play_identically(tmp_path):
    """Stage 2 用 tuple、Stage 4 落地成 JSON 會變 list —— 行為必須一樣。

    🩸 不一樣的話，CMA-ES 找到的最佳解寫進 `config/opponents/*.json` 之後就
    不是同一個 agent 了，而且不會報錯：`basket` 從 tuple 變 list、
    `late_crew_caps` 從巢狀 tuple 變巢狀 list。
    """
    import json as _json

    from tools.freeze_params import build_spec

    in_memory = decode(x0())
    spec = build_spec(param_space.to_json_params(in_memory), "probe-v0")
    from_json = _json.loads(_json.dumps(spec, ensure_ascii=False))["params"]
    # 凍結對手是靠 build_agent 補這個旗標，這裡手動補上模擬同一條路徑。
    from_json["_replace_defaults"] = True

    assert _play(in_memory) == _play(from_json)


def test_freeze_rejects_incomplete_params():
    """凍結對手少一個 key 就會 fall through 到預設值 —— 要在落地時就擋下來。"""
    from tools.freeze_params import build_spec

    partial = param_space.to_json_params(decode(x0()))
    partial.pop("tiles_per_unit")
    with pytest.raises(SystemExit, match="完整展開"):
        build_spec(partial, "broken-v0")
