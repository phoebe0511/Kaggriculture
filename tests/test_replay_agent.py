"""`agents/replay.py` 的正確性依據。

replay agent 存在的唯一理由是「提供一個不是 gen0/gen1 血統的對手」。它一旦
重播得不忠實，就只是一個弱對手，而且**看起來完全正常** —— 不會拋錯、不會
超時、每局照樣跑完 720 步。

所以這裡把「重現原局」釘成斷言。第一版把 `steps[t]["action"]` 當成「在 t
決定的動作」（實際上是在 t-1 決定的），重播出來是 `[30367, 37762]`，
原局是 `[117554, 117668]` —— 慢一拍就掉 3/4 的分數，沒有這個測試抓不到。
"""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_environments import make

from agents.replay import act, load_episode

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 使用者從 Kaggle 下載的真實 ladder 對局，用 `tools/extract_episode.py` 抽過。
EPISODE = "93916293"


def _replay_agent(player):
    return lambda obs, config: act(obs, config, {"episode": EPISODE, "player": player})


def test_episode_file_is_the_ladder_environment():
    """抽出來的 episode 要跟本機環境同一套設定，否則不能拿來當量尺。"""
    data = load_episode(EPISODE)

    assert data["module_version"] == "1.32.7"
    assert data["statuses"] == ["DONE", "DONE"]
    assert data["n_steps"] == 720
    assert data["seed"] is not None, "沒有 seed 就重現不了這一局"

    cfg = data["configuration"]
    # journal 2026-08-17 §A 核對過的欄位。ladder 跟本機預設相同，
    # 所以本機量出來的數字直接有效 —— 這條不成立的話整個量尺失效。
    assert cfg["farmHandCostMult"] == 1
    assert cfg["startingMoney"] == 3000
    assert cfg["turnsPerDay"] == 24
    assert cfg["episodeSteps"] == 720
    assert cfg["shedCapacity"] == 100
    assert cfg["maxMarketOrdersPerTurn"] == 10
    assert cfg["townCenterSellInterval"] == 24
    assert cfg["townShopSellInterval"] == 4
    assert cfg["marketParams"] == {}


def test_replay_reproduces_episode():
    """replay 對 replay 打原 seed，期末現金要一分不差。

    這是 replay agent 唯一的正確性依據。對不上就代表重播不忠實，
    在它上面量到的任何勝率都沒有意義。
    """
    data = load_episode(EPISODE)

    env = make(
        "kaggriculture",
        configuration={"seed": data["seed"]},
        debug=True,
    )
    env.run([_replay_agent(0), _replay_agent(1)])

    assert [state.status for state in env.state] == ["DONE", "DONE"]
    assert [state.reward for state in env.state] == data["rewards"]


def test_extracted_actions_match_source_episode():
    """抽取工具沒有改動內容。

    原始 30 MB 的 episode 檔被 gitignore（`temp/`），所以這個測試在同伴的
    工作目錄上會 skip —— 它擋的是「抽取腳本改壞了但沒人發現」。
    """
    import pytest

    source = REPO_ROOT / "temp" / f"{EPISODE}.json"
    if not source.is_file():
        pytest.skip(f"沒有原始 episode {source}（temp/ 被 gitignore）")

    with open(source, encoding="utf-8") as f:
        raw = json.load(f)
    extracted = load_episode(EPISODE)

    assert len(extracted["actions"]) == len(raw["steps"])
    for t, (row, entries) in enumerate(zip(extracted["actions"], raw["steps"])):
        for p, (action, entry) in enumerate(zip(row, entries)):
            original = entry.get("action")
            if isinstance(original, dict):
                assert action == original, f"step {t} player {p} 的動作被改過"
