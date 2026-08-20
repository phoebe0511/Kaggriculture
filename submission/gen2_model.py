"""第 2 代：unit 排程交給網路，市場沿用規則式。

網路模仿的是 rating 2979~3229 的真實 ladder 玩家（60 局 replay、842,151 個
unit-turn）。**不是模仿我們自己的 Gen1** —— 2026-08-19 量到 Gen1 對他們
6 勝 154 負，模仿一個在輸的老師沒有意義。

## 只換 unit，市場不換

`act()` 先呼叫 `gen0.act()` 拿一份完整的規則式動作，然後把 `farmer` / `hands`
換成網路的輸出，`market` 原封不動。

⚠️ **已知的不一致**：`gen0._market` 算訂單時會參考它自己排出來的 `unit_actions`
（例如 PICKUP 了多少 WHEAT 就少賣多少），而我們事後把 unit 動作換掉了。
所以市場決策是對「規則式會做什麼」最佳化的，不是對「網路實際做什麼」。

代價量得出來：2026-08-19 的逐件收入歸因顯示 WHEAT + FERTILIZER 佔總差距的
27%，而那兩項主要是市場行為。所以這一版的上限大約是補掉 73% 的差距。

## PLANT 的原子驗證

引擎規則（`engine-notes.md` §10.9）：同一回合所有 unit 對某作物的 PLANT 請求
總數超過手上種子數 → 該作物的**所有** PLANT 請求全部變成 `PASS`。

所以不能直接把網路的 argmax 送出去 —— 超額的話會連本來種得成的那幾格一起
作廢。超出的部分改用該 unit 的第二高分動作。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

import contracts as C
from gen0 import act as gen0_act
from gen1 import DEFAULT_PARAMS as GEN1_DEFAULTS
# 🩸 模組層，不可以搬進 `_policy()` —— 理由見 `agents/gen0.py` 頂端那一段。
# 這裡只是把類別 import 進來，**權重仍然是 lazy 的**（`NumpyPolicy(path)` 才讀檔）。
from npz_forward import NumpyPolicy

#: 權重檔。`KAGGRI_WEIGHTS` 覆寫，方便同時比較好幾個 checkpoint。
#:
#: 沒設的話先找**跟這支檔案同一個目錄**的 `weights.npz` —— submission 是攤平的
#: （`serving/build_submission.py`），權重就躺在旁邊，而且比賽端的工作目錄不是
#: repo root，寫死 `model/weights.npz` 會找不到。
#:
#: ⚠️ 這裡可以用 `__file__`：`main.py` 是被 `exec()` 進去的、沒有 `__file__`，
#: 但這支是正常 `import` 進來的模組，有。
_LOCAL_WEIGHTS = Path(__file__).resolve().parent / "weights.npz"
WEIGHTS_PATH = os.environ.get(
    "KAGGRI_WEIGHTS",
    str(_LOCAL_WEIGHTS) if _LOCAL_WEIGHTS.is_file() else "model/weights.npz")

_POLICY = None
_POLICY_LOCK = threading.Lock()


def _policy():
    """載入一次就好。多行程跑的時候每個 worker 各載自己的。"""
    global _POLICY
    if _POLICY is None:
        with _POLICY_LOCK:
            if _POLICY is None:
                policy = NumpyPolicy(WEIGHTS_PATH)
                if policy.encoder_version != C.ENCODER_VERSION:
                    raise SystemExit(
                        f"{WEIGHTS_PATH} 是 ENCODER_VERSION "
                        f"{policy.encoder_version}，contracts.py 是 "
                        f"{C.ENCODER_VERSION} —— 重新訓練")
                _POLICY = policy
    return _POLICY


def _choose(op_logits, qty_logits, mask, obs):
    """挑動作：先遮掉不合法的，再處理 PLANT 的跨 unit 種子上限。"""
    scores = np.where(mask, op_logits, -np.inf)
    order = np.argsort(-scores, axis=1)          # 每個 unit 的偏好排序

    seeds = dict(obs["private"]["seeds"])
    chosen = []
    for i in range(scores.shape[0]):
        picked = None
        for op_index in order[i]:
            if not np.isfinite(scores[i, op_index]):
                break                             # 後面都是不合法的
            op, item = C.UNIT_OPS[op_index]
            if op == "PLANT":
                # 種子是全體共用的，先搶先贏。搶不到就往下一個選項走 ——
                # 不能硬送，送了會讓該作物**所有** PLANT 一起變 PASS。
                if seeds.get(item, 0) <= 0:
                    continue
                seeds[item] -= 1
            picked = op_index
            break
        if picked is None:
            picked = C.UNIT_OP_INDEX[("PASS", None)]
        qty_index = int(np.argmax(qty_logits[i]))
        chosen.append(C.decode_unit(picked, qty_index))
    return chosen


def act(obs, config=None, params=None):
    resolved = dict(GEN1_DEFAULTS)
    resolved.update(params or {})
    resolved.pop("_replace_defaults", None)

    # 市場、雇工、買地全部沿用規則式的決定
    base = gen0_act(obs, config, resolved)

    policy = _policy()
    spatial, scalar = C.encode(obs, config)
    positions, unit_features = C.encode_units(obs, config)
    # ⚠️ v3 起 policy 多回傳一個 target head。這一版用不到它 —— 它的權重是
    # ENCODER_VERSION 2，`_policy()` 的版本檢查會先擋下來（見上面）。
    op_logits, qty_logits, _target, _value = policy(
        spatial, scalar, positions, unit_features)
    mask = C.legal_unit_mask(obs, config)

    units = _choose(op_logits, qty_logits, mask, obs)
    return {"farmer": units[0], "hands": units[1:], "market": base["market"]}


def agent(obs, config):
    return act(obs, config)
