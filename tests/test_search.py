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
from serving.action_validation import assert_legal_action  # noqa: E402

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
    cands, _blocked = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    for label, action in cands:
        assert_legal_action(obs, cfg, action)          # 不合法就拋
        assert set(action) == {"farmer", "hands", "market"}, label


def test_candidates_include_the_baseline(board):
    """基準線一定要在裡面 —— `search_action` 的下界保證靠它。"""
    _env, cfg, _snap, obs, base = board
    cands, _blocked = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    labels = [label for label, _a in cands]
    assert "gen0" in labels
    assert dict(cands)["gen0"] == base


def test_candidates_are_distinct_and_plural(board):
    """去重之後還要夠多 —— 只剩幾個的話等於沒搜。"""
    _env, cfg, _snap, obs, base = board
    cands, _blocked = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    assert len(cands) >= 15, f"只產出 {len(cands)} 個候選"
    assert len({repr(a) for _l, a in cands}) == len(cands), "有重複的候選"


def test_candidate_unit_count_matches_the_board(board):
    """每個候選的 unit 數要跟盤面一致，少一個就是有 unit 沒下指令。"""
    _env, cfg, _snap, obs, base = board
    n_units = C.legal_unit_mask(obs, cfg).shape[0]
    cands, _blocked = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    for label, action in cands:
        assert 1 + len(action["hands"]) == n_units, label


# --------------------------------------------------------------------------
# 3. 搜尋
# --------------------------------------------------------------------------

def test_search_never_worse_than_baseline(board):
    """候選含基準線 -> 搜出來的 Q 不可能低於不搜。"""
    env, cfg, snap, obs, base = board
    act = _gen0(cfg)
    cands, _blocked = CAND.generate(obs, cfg, base, np.random.default_rng(0))
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
    cands, _blocked = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    without = [c for c in cands if c[0] != "gen0"][:3]
    with pytest.raises(ValueError, match="基準線"):
        RS.search_action(env, snap, without, act, act, base_label="gen0")


def test_search_is_reproducible(board):
    """同一組候選搜兩次，選出來的要一樣（決定性）。"""
    env, cfg, snap, obs, base = board
    act = _gen0(cfg)
    cands, _blocked = CAND.generate(obs, cfg, base, np.random.default_rng(0))
    subset = [c for c in cands if c[0] == "gen0"] + \
             [c for c in cands if c[0] != "gen0"][:4]
    first = RS.search_action(env, snap, subset, act, act, base_label="gen0")
    second = RS.search_action(env, snap, subset, act, act, base_label="gen0")
    assert first[0] == second[0] and first[2] == second[2]
