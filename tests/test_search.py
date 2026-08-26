"""`search/` 的 L0 保護。

盯三件事，每一件垮掉都會**安靜地**讓 search 產出垃圾：

1. **回溯要乾淨。** `restore()` 之後跑同一串動作，期末現金要逐位元組相同。
   垮了的話每個候選其實是在不同的起點上比，分數完全沒有意義。
2. **候選要合法。** 引擎遇到非法動作是靜默忽略 —— 不驗的話會產出一堆「其實
   沒送出去」的候選，它們的分數會全部擠在同一個值附近。
3. **搜不會比不搜差。** 候選集合含 `gen0` 自己的動作，所以搜出來的 Q 有下界。

⚠️ 一律用短 episode（`episodeSteps=168`，7 天）。L0 預算 60 秒，完整 720 步
一局要 3.6 秒（08-25 實測）。
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pytest

os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")
os.environ.setdefault("KAGGRI_WEIGHTS", "model/weights-e2e-round6.npz")
os.environ.pop("KAGGRI_LOG_FILE", None)

from kaggle_environments import make  # noqa: E402

import contracts as C  # noqa: E402
from agents.gen0 import act as gen0_act  # noqa: E402
from search import candidates as CAND  # noqa: E402
from search import rollout_search as RS  # noqa: E402
from serving.action_validation import (  # noqa: E402
    IllegalAction,
    assert_legal_action,
)

SHORT_STEPS = 168


@pytest.fixture(scope="module")
def board():
    """推到 day 3 hour 12 的短局。回傳 `(env, cfg, snap, obs, gen0 的動作)`。"""
    with contextlib.redirect_stderr(io.StringIO()):
        env = make("kaggriculture",
                   configuration={"seed": 7, "episodeSteps": SHORT_STEPS},
                   debug=False)
    cfg = env.configuration
    for _ in range(3 * 24 + 12):
        env.step([gen0_act(env.state[0].observation, cfg),
                  gen0_act(env.state[1].observation, cfg)])
    snap = RS.snapshot(env)
    obs = env.state[0].observation
    return env, cfg, snap, obs, gen0_act(obs, cfg)


def _gen0(cfg):
    return lambda obs: gen0_act(obs, cfg)


# --------------------------------------------------------------------------
# 1. 回溯
# --------------------------------------------------------------------------

def test_restore_gives_identical_rollouts(board):
    """同一個快照跑三次同樣的 policy，期末現金要完全相同。

    🩸 這是整個離線 search 的地基。垮了的話每個候選是在不同起點上比。
    """
    env, cfg, snap, _obs, _base = board
    act = _gen0(cfg)
    runs = []
    for _ in range(3):
        RS.restore(env, snap)
        runs.append(RS.play_to_end(env, act, act))
    assert runs[0] == runs[1] == runs[2], f"回溯不乾淨：{runs}"


def test_restore_rewinds_steps_and_status(board):
    """`restore()` 要把 `env.steps` 砍回去、`status` 設回 ACTIVE。"""
    env, cfg, snap, _obs, _base = board
    _state, nsteps = snap
    act = _gen0(cfg)
    RS.restore(env, snap)
    RS.play_to_end(env, act, act)
    assert len(env.steps) > nsteps and env.done

    RS.restore(env, snap)
    assert len(env.steps) == nsteps
    assert not env.done
    assert all(s.status == "ACTIVE" for s in env.state)


def test_evaluate_does_not_leak_between_candidates(board):
    """先評估一個「亂搞」的候選，再評估基準線，基準線的分數要跟單獨跑一樣。"""
    env, cfg, snap, obs, base = board
    act = _gen0(cfg)

    RS.restore(env, snap)
    alone = RS.play_to_end(env, act, act)

    n_units = 1 + len(base["hands"])
    junk = {"farmer": ["PASS"], "hands": [["PASS"]] * (n_units - 1), "market": []}
    RS.evaluate(env, snap, junk, act, act)
    after = RS.evaluate(env, snap, base, act, act)
    assert after == alone, f"候選之間有殘留：單獨 {alone}、之後 {after}"


# --------------------------------------------------------------------------
# 2. 候選
# --------------------------------------------------------------------------

def test_all_candidates_are_legal(board):
    """引擎對非法動作是靜默忽略，所以候選一定要先驗過。"""
    _env, cfg, _snap, obs, base = board
    cands, _blocked, _info = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    for label, action in cands:
        assert_legal_action(obs, cfg, action)          # 不合法就拋
        assert set(action) == {"farmer", "hands", "market"}, label


def test_candidates_include_the_baseline(board):
    """基準線一定要在裡面 —— `search_action` 的下界保證靠它。"""
    _env, cfg, _snap, obs, base = board
    cands, _blocked, _info = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    labels = [label for label, _a in cands]
    assert "gen0" in labels
    assert dict(cands)["gen0"] == base


def test_candidates_are_distinct_and_plural(board):
    """去重之後還要夠多 —— 只剩幾個的話等於沒搜。"""
    _env, cfg, _snap, obs, base = board
    cands, _blocked, _info = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    assert len(cands) >= 15, f"只產出 {len(cands)} 個候選"
    assert len({repr(a) for _l, a in cands}) == len(cands), "有重複的候選"


def test_candidate_unit_count_matches_the_board(board):
    """每個候選的 unit 數要跟盤面一致，少一個就是有 unit 沒下指令。"""
    _env, cfg, _snap, obs, base = board
    n_units = C.legal_unit_mask(obs, cfg).shape[0]
    cands, _blocked, _info = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    for label, action in cands:
        assert 1 + len(action["hands"]) == n_units, label


def test_baseline_survives_even_when_illegal(board):
    """基準線非法也要進候選，而且不能拖垮其他候選。

    🩸 2026-08-26 seed 4 就死在這裡：網路想買一顆**買不起**的 MELON 種子。
    市場那半是所有候選共用的，所以 56 個候選被擋掉 55 個，連基準線自己都沒了，
    `search_action` 拋 `ValueError` 整局死掉 —— 而在那之前沒有任何紀錄看得出來。

    基準線是 policy 自己的輸出，不是我們產的候選。引擎對非法動作是靜默忽略，
    所以「不搜會怎樣」仍然有定義，Q 是對的，下界保證仍然成立。
    """
    _env, cfg, _snap, obs, base = board
    healthy, _b, _i = CAND.generate(obs, cfg, base, np.random.default_rng(0))

    poisoned = dict(base)
    poisoned["market"] = list(base.get("market") or []) + [["BUY_SEED", "MELON", 999]]
    with pytest.raises(IllegalAction):               # 前提：這個真的非法
        assert_legal_action(obs, cfg, poisoned)

    cands, _blocked, info = CAND.generate(obs, cfg, poisoned,
                                          np.random.default_rng(0))
    labels = [label for label, _a in cands]
    assert "gen0" in labels, "基準線非法就被丟掉了"
    assert info["base_illegal"], "沒把「基準線非法」記下來"
    assert info["dropped_orders"] >= 1, "沒有清掉買不起的訂單"
    assert len(cands) >= len(healthy) - 2,         f"其餘候選跟著陪葬：{len(cands)} 個，健康時有 {len(healthy)} 個"


# --------------------------------------------------------------------------
# 3. 搜尋
# --------------------------------------------------------------------------

def test_search_never_worse_than_baseline(board):
    """候選含基準線 -> 搜出來的 Q 不可能低於不搜。"""
    env, cfg, snap, obs, base = board
    act = _gen0(cfg)
    cands, _blocked, _info = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    # 只取前 6 個，L0 跑不起 57 個 rollout
    subset = [c for c in cands if c[0] == "gen0"] + \
             [c for c in cands if c[0] != "gen0"][:5]

    _label, _action, cash, scored = RS.search_action(
        env, snap, subset, act, act, base_label="gen0")
    assert cash >= dict(scored)["gen0"]


def test_search_requires_the_baseline_in_candidates(board):
    """沒有基準線就沒有下界保證 —— 要拋錯，不要安靜地搜。"""
    env, cfg, snap, obs, base = board
    act = _gen0(cfg)
    cands, _blocked, _info = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    without = [c for c in cands if c[0] != "gen0"][:3]
    with pytest.raises(ValueError, match="基準線"):
        RS.search_action(env, snap, without, act, act, base_label="gen0")


def test_search_is_reproducible(board):
    """同一組候選搜兩次，選出來的要一樣（決定性）。"""
    env, cfg, snap, obs, base = board
    act = _gen0(cfg)
    cands, _blocked, _info = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    subset = [c for c in cands if c[0] == "gen0"] + \
             [c for c in cands if c[0] != "gen0"][:4]
    first = RS.search_action(env, snap, subset, act, act, base_label="gen0")
    second = RS.search_action(env, snap, subset, act, act, base_label="gen0")
    assert first[0] == second[0] and first[2] == second[2]


# --------------------------------------------------------------------------
# 4. 訓練標籤（Stage 3）
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recorded(board):
    """錄一小段：每 4 個回合當成「search 決定的」，其餘走 gen0 的完整標籤。"""
    from search.expert import SearchRecorder

    env, cfg, snap, _obs, _base = board
    RS.restore(env, snap)
    rec = SearchRecorder(0)
    marks = []
    for t in range(12):
        obs = env.state[0].observation
        action, plan = gen0_act(obs, cfg, None, return_plan=True)
        if t % 4 == 0:
            rec.record_search(obs, cfg, action)        # 只有 action
            marks.append(True)
        else:
            rec.record(obs, cfg, action, plan)         # 完整標籤
            marks.append(False)
        env.step([action, gen0_act(env.state[1].observation, cfg)])
    return rec.finish([50000.0, 40000.0], 999), marks


def test_search_npz_has_the_same_fields_as_rollout(board):
    """欄位要跟 harness/rollout.py 逐項相同，不然 model/train.py 載不進去。

    🩸 對照組要**錄過至少一個回合**才 finish —— 空的 Recorder 在 finish() 裡
    reshape 會炸（`harness/rollout.py:209`），那是測試寫錯不是產品壞掉。
    """
    from harness.rollout import Recorder

    from search.expert import SearchRecorder

    env, cfg, snap, _obs, _base = board
    RS.restore(env, snap)
    obs = env.state[0].observation
    action, plan = gen0_act(obs, cfg, None, return_plan=True)

    reference = Recorder(0)
    reference.record(obs, cfg, action, plan)
    mine = SearchRecorder(0)
    mine.record_search(obs, cfg, action)

    ref = reference.finish([1.0, 2.0], 0)
    got = mine.finish([1.0, 2.0], 0)
    missing = set(ref) - set(got)
    assert not missing, f"少了欄位：{sorted(missing)}"
    for key in ref:
        assert ref[key].dtype == got[key].dtype, f"{key} 的 dtype 不一樣"
        assert ref[key].shape[1:] == got[key].shape[1:], f"{key} 的形狀不一樣"


def test_searched_boards_carry_no_intent_labels(recorded):
    """search 產不出 target / demand，那些要寫成「請忽略」而不是假標籤。

    🩸 `harness/rollout.py:147` 在 plan 缺項時會寫出「原地 PASS」這個看起來
    正常的錯標籤。這條測試就是在守「我們沒有走上那條路」。
    """
    payload, marks = recorded
    blank = ~payload["board_demand_legal_bits"].any(axis=(1, 2))
    assert list(blank) == marks, "demand_legal 全 0 的盤面對不上 search 的回合"

    board_of = payload["unit_board"]
    for i, is_search in enumerate(marks):
        rows = board_of == i
        if is_search:
            assert (payload["unit_target"][rows] == -1).all()
            assert (payload["unit_term_op"][rows] == -1).all()
        else:
            assert (payload["unit_target"][rows] >= 0).any()


def test_searched_boards_keep_the_immediate_action(recorded):
    """op / qty / market 是 gen2_model 真正出手用的三組 head，一定要有真標籤。"""
    payload, marks = recorded
    board_of = payload["unit_board"]
    for i, is_search in enumerate(marks):
        if not is_search:
            continue
        rows = board_of == i
        assert rows.sum() > 0, f"第 {i} 個盤面一個 unit 都沒錄到"
        assert (payload["unit_op"][rows] >= 0).all(), "unit_op 不該有 −1"


def test_search_recorder_counts_searched_turns(recorded):
    payload, marks = recorded
    assert int(payload["searched_turns"][0]) == sum(marks)
