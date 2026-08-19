"""重播真實 ladder 對局的動作，當作本機對手。

本機 `config/opponents/` 裡的對手全部是 gen0/gen1 家族，彼此高度相關 ——
Gen1 對它們 100 勝 0 負，但線上排 3000/5000。**那把尺量不出東西了。**
這支 agent 從真實 ladder episode 重播頂端玩家的動作，提供一個不是自家血統
的對手。

用法（`config/opponents/ladder-top-a.json`）：

    {"entry": "agents.replay:act",
     "params": {"episode": "93916293", "player": 0}}

`episode` 可以是 `config/episodes/<id>.json` 的 id，也可以是完整路徑。
小檔由 `tools/extract_episode.py` 從 Kaggle 下載的 30 MB episode 抽出來。

## ⚠️ 這是開迴路重播，不是真正的對手

重播的動作是對方在**原本那一局**做的決定。我們一下場，市場價格就跟原局不同：

    我們的買賣 → 市場價格變 → 它的現金變 → BUY_* / HIRE 訂單可能失敗
                                          → 農場逐漸偏離原軌跡

unit 動作（WATER / PLANT / CARE 都在自己格子上）大多不受影響，市場那條會漂。
所以它是「比 starter 強得多的固定對手」，不是完美對手。

漂移程度用 `tools/replay_drift.py` 量，不要憑感覺假設它小。

## 定位用 day/hour，不是 obs["step"]

`obs["step"]` 只存在於 player 0 的 observation —— kaggle-environments 把共用
欄位只放在第 0 個 entry（實測 `steps[t][1]["observation"]` 沒有 `step`）。
`day` / `hour` 兩邊都有，所以用 `t = day * turnsPerDay + hour`，跟
`gen0._log` 的算法一致。

## ⚠️ steps[t]["action"] 是在 t-1 決定的，不是在 t

kaggle-environments 每一格存的是「這個狀態」加上「**把狀態推到這裡的那個
動作**」。所以時刻 `t` 要送的是 `actions[t + 1]`，不是 `actions[t]`。

實測證據（`temp/93916293.json`）：`steps[1]` 的 money 已經是 22（五個 HIRE
加買動物的錢都付掉了），而 `steps[1]["action"]` 正是那批 HIRE —— 那批只可能
是在 `steps[0]`（money 3000）決定的。`steps[0]["action"]` 是佔位的 PASS。

**第一版寫成 `actions[t]` 時，重播出來是 `[30367, 37762]`，原局是
`[117554, 117668]`。** 整整慢一拍就掉了 3/4 的分數，而且不會報錯 ——
`test_replay_reproduces_episode` 存在的理由就是擋這個。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EPISODE_DIR = REPO_ROOT / "config" / "episodes"

#: 超出重播範圍時送這個。引擎對 hands 長度不符是靜默略過
#: （`kaggriculture.py:318` 的 `pos is None -> return`），所以空 list 安全。
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}

#: 一個 worker 行程只 parse 一次。221 KB 的檔，parse 約幾十毫秒，
#: 但 720 回合 × 每回合呼叫一次的話就不能每次重讀。
_CACHE = {}


def episode_path(episode):
    """把 `episode` 參數解析成實際路徑。

    接受三種寫法：完整路徑、repo 相對路徑、或 `config/episodes/` 下的 id。
    """
    for candidate in (Path(episode),
                      REPO_ROOT / str(episode),
                      EPISODE_DIR / f"{episode}.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"找不到 episode {episode!r}。看過：{EPISODE_DIR}/{episode}.json。\n"
        f"先跑 `python -m tools.extract_episode <下載的 episode JSON>`"
    )


def load_episode(episode):
    """讀（並快取）抽取過的 episode 檔。"""
    path = episode_path(episode)
    key = str(path.resolve())
    if key not in _CACHE:
        with open(path, encoding="utf-8") as f:
            _CACHE[key] = json.load(f)
    return _CACHE[key]


def act(obs, config=None, params=None):
    p = dict(params or {})
    episode = p.get("episode")
    if episode is None:
        raise ValueError(
            "agents.replay 需要 params['episode']，"
            "例如 {\"episode\": \"93916293\", \"player\": 0}"
        )
    player = int(p.get("player", 0))

    data = load_episode(episode)
    actions = data["actions"]

    turns_per_day = 24
    if config is not None:
        turns_per_day = int(config.get("turnsPerDay", 24))
    t = int(obs["day"]) * turns_per_day + int(obs["hour"])

    # +1：steps[i]["action"] 是在 i-1 決定的（見 module docstring）。
    idx = t + 1
    if not (0 <= idx < len(actions)):
        return dict(PASS_ACTION)
    row = actions[idx]
    if player >= len(row):
        return dict(PASS_ACTION)
    return row[player]


def agent(obs, config):
    """kaggle_environments 的進入點。沒有預設 episode，一定要走 params。"""
    return act(obs, config)
