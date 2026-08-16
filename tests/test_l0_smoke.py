from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path

from kaggle_environments import make

from main import agent as submission_agent
from serving.action_validation import assert_observation_invariants


BASELINES = json.loads(
    (Path(__file__).with_name("baselines.json")).read_text(encoding="utf-8")
)


class TimedAgent:
    def __init__(self):
        self.durations = []

    def __call__(self, obs, config):
        started = time.perf_counter()
        try:
            return submission_agent(obs, config)
        finally:
            self.durations.append(time.perf_counter() - started)


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

