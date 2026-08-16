"""Kaggle submission 入口。"""

import os

# agents.gen0（由 gen1 共用核心）在 import 時讀取此值，所以必須先設定。
os.environ["KAGGRI_LOG_LEVEL"] = "0"

from agents.gen1 import act  # noqa: E402
from serving.action_validation import assert_legal_action  # noqa: E402


def agent(obs, config):
    action = act(obs, config)
    assert_legal_action(obs, config, action)
    return action
