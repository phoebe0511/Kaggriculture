from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path

from kaggle_environments import make

# 🩸 **這裡刻意不是 `main.agent`。**
#
# `baselines.json` 的三個期末現金是「規則式那條路沒有被改壞」的決定性檢查
# （`docs/rules.md` §2.2 的 `assert final_cash == baseline[seed]`）。它要盯的是
# `agents/gen0.py`（2026-08-21 起 gen1 已併入），而那條路是**完全確定性**的。
#
# 2026-08-20 起 `main.py` 換成 `agents/gen4_demand.py`（帶網路），期末現金會
# 隨權重改變 —— 綁在那上面的話，每換一次 checkpoint 這三個數字就要重寫一次，
# 等於把這個保護拆掉。所以基準線改成直接指 `agents.gen0:act`，
# **三個數字一個都沒動**。
#
# submission 那條路由 `test_submission_entry_*` 顧，它不比對現金。
from agents.gen0 import act as rule_agent
from serving.action_validation import (
    assert_legal_action,
    assert_observation_invariants,
)


def submission_agent(obs, config):
    return rule_agent(obs, config)


BASELINES = json.loads(
    (Path(__file__).with_name("baselines.json")).read_text(encoding="utf-8")
)


class TimedAgent:
    """計時 + 每回合驗證動作合法性。

    ⚠️ 驗證**必須放在這裡，不能放在 `main.py`**。submission 帶著它有三個
    問題（成本、比賽當下抓到也沒用、綁死引擎私有 API），理由寫在 `main.py`
    底部。所以 L0 自己補上這一層 —— 開發時每回合都驗，送出去的不驗。

    `durations` 只記 agent 本身的耗時，不含驗證 —— 那個數字要拿來跟
    `actTimeout` 比，混進驗證時間就沒有意義了。
    """

    def __init__(self):
        self.durations = []

    def __call__(self, obs, config):
        started = time.perf_counter()
        action = submission_agent(obs, config)
        self.durations.append(time.perf_counter() - started)
        assert_legal_action(obs, config, action)
        return action


def test_l0_fixed_seed_baselines_finish_under_one_minute():
    started = time.perf_counter()
    for seed_text, expected in BASELINES["seeds"].items():
        audit_agent = TimedAgent()
        env = make(
            "kaggriculture",
            configuration={"seed": int(seed_text)},
            debug=True,
        )
        env.run([audit_agent, "starter"])

        assert len(env.steps) == 720
        assert [state.status for state in env.state] == ["DONE", "DONE"]
        assert env.state[0].reward == expected["cash_agent"]
        assert env.state[1].reward == expected["cash_opponent"]
        assert audit_agent.durations
        assert max(audit_agent.durations) < 0.5

        final_obs = env.steps[-1][0]["observation"]
        assert_observation_invariants(final_obs)

    assert time.perf_counter() - started < 60


def test_engine_version_matches_baseline():
    assert importlib.metadata.version("kaggle-environments") == "1.32.7"
    assert BASELINES["engine"] == "kaggle-environments==1.32.7"


def test_submission_entry_survives_a_whole_game():
    """跑**打包好的那一份**，用 Kaggle 自己的 file-path 載入方式。

    這條測的是上面那條測不到的東西：攤平之後的 import、`weights.npz` 找不找得到、
    `ENCODER_VERSION` 對不對得上、單回合有沒有超過 `actTimeout`。

    ⚠️ **不比對期末現金。** 那個數字會隨權重變 —— 綁上去等於每換一次
    checkpoint 就要改基準線。現金的決定性檢查在上面那條（規則式那條路）。

    ⚠️ 這條吃的是 `submission/` 目前的內容，不是 `agents/` 的原始碼。
    改了 agent 沒重跑 `python -m serving.build_submission` 的話，這裡量到的
    還是舊版 —— 那正是我們想知道的事（上場的是打包好的那一份）。
    """
    entry = Path(__file__).resolve().parents[1] / "submission" / "main.py"
    if not entry.is_file():
        import pytest
        pytest.skip("submission/ 還沒打包（python -m serving.build_submission）")

    started = time.perf_counter()
    env = make("kaggriculture", configuration={"seed": 41001}, debug=True)
    env.run([str(entry), "starter"])

    assert len(env.steps) == 720
    assert [state.status for state in env.state] == ["DONE", "DONE"], \
        "submission 沒跑完 —— 多半是攤平之後某個 import 找不到，" \
        "看 tests/test_submission_package.py::test_no_lazy_imports_of_packaged_modules"
    # PASS 到底也會是 DONE，所以要看它有沒有真的賺到錢
    assert env.state[0].reward > 50_000, \
        f"期末現金只有 {env.state[0].reward} —— agent 大概每回合都在拋錯"
    assert_observation_invariants(env.steps[-1][0]["observation"])
    assert time.perf_counter() - started < 60

