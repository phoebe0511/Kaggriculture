from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path

from kaggle_environments import make

from main import agent as submission_agent
from serving.action_validation import (
    assert_legal_action,
    assert_observation_invariants,
)


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

