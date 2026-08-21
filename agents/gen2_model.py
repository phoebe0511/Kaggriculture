"""端到端：網路直接決定每個 unit 這一步做什麼（44 個 `UNIT_OPS` 之一）。

## 🔄 2026-08-21：模仿對象從 ladder 老師換成我們自己的規則式

原本這支是模仿 rating 2979~3229 的真實 ladder 玩家（60 局 replay、842,151 個
unit-turn），而 docstring 舊版寫著「**不是**模仿我們自己的 Gen1 —— 2026-08-19
量到 Gen1 對他們 6 勝 154 負，模仿一個在輸的老師沒有意義」。

**那個判斷被後來的量測推翻了**，理由是「模仿得到誰」比「誰比較強」更關鍵：

- 老師是 **60 份固定 replay，問不到新局面**。網路一走偏就沒有標籤可學，
  實測開局 73 步的腳本只跟得完 22%（journal 2026-08-20 §7）
- 規則式是 **queryable expert** —— 每回合從當下盤面重算，網路把盤面走爛之後
  它仍然答得出「這裡該做什麼」。那正是 DAgger 需要而 replay 給不了的

代價是天花板變成規則式（榜上 7 萬+，本機配對 92,860）。**這一步的目的不是
贏過它**，是產出離線 search 用得動的 policy prior —— 完整動作空間、
top-k 涵蓋率夠高、value head 可靠。超過規則式要靠 search，不是靠模仿。

## 為什麼是端到端，而不是 v5 的 demand map

`agents/gen4_demand.py` 只讓網路輸出「哪一格要做什麼」，配對 / 走路 / 市場
全部還給 `gen0`。**search 只搜得到網路輸出的東西**，所以那條路上市場和配對
永遠碰不到。這支的輸出涵蓋完整動作空間。

## 市場

`model_market=False`（預設）走 `gen0._market`，只為了先隔離 unit 那一半。

⚠️ **已知的不一致**：`gen0._market` 算訂單時會參考它自己排出來的 `unit_actions`
（例如 PICKUP 了多少 WHEAT 就少賣多少），而我們事後把 unit 動作換掉了。
所以市場決策是對「規則式會做什麼」最佳化的，不是對「網路實際做什麼」。
2026-08-19 的逐件收入歸因顯示 WHEAT + FERTILIZER 佔總差距的 27%。

`model_market=True` 才是最終形態（動作空間完整）。

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
from agents.gen0 import act as gen0_act
from agents.gen0 import DEFAULT_PARAMS as RULE_DEFAULTS
# 🩸 模組層，不可以搬進 `_policy()` —— 理由見 `agents/gen0.py` 頂端那一段。
# 這裡只是把類別 import 進來，**權重仍然是 lazy 的**（`NumpyPolicy(path)` 才讀檔）。
from serving.npz_forward import NumpyPolicy

#: 權重檔。`KAGGRI_WEIGHTS` 覆寫，方便同時比較好幾個 checkpoint。
#:
#: 沒設的話先找**跟這支檔案同一個目錄**的 `weights.npz` —— submission 是攤平的
#: （`serving/build_submission.py`），權重就躺在旁邊，而且比賽端的工作目錄不是
#: repo root，寫死 `model/weights.npz` 會找不到。
#:
#: ⚠️ 這裡可以用 `__file__`：`main.py` 是被 `exec()` 進去的、沒有 `__file__`，
#: 但這支是正常 `import` 進來的模組，有。
#:
#: 找不到的話退回 repo 的 `submission/weights.npz` —— 那是 `build_submission`
#: 剛打包進去的同一個檔案，所以開發側跑 `main.py` 跟上場跑的是同一份權重。
#: **不要退回 `model/weights.npz`**：那支是 ENCODER_VERSION 2 的舊檔，
#: 載進來會在第一回合 SystemExit，而錯誤訊息看起來像是 contracts.py 的問題。
_HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = os.environ.get(
    "KAGGRI_WEIGHTS",
    str(next((p for p in (_HERE / "weights.npz",
                          _HERE.parent / "submission" / "weights.npz")
              if p.is_file()), _HERE / "weights.npz")))

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
    """網路直接決定每個 unit 這一步做什麼。

    `model_market`（預設 `False`）決定市場走誰：

    - `False`：`gen0._market`。**先隔離 unit 那一半**，一次只動一個變數。
      ⚠️ 代價寫在模組 docstring 的「已知的不一致」：`gen0._market` 是對
      「規則式會做什麼」最佳化的，而我們把 unit 動作換掉了。
    - `True`：`contracts.decode_market_orders`。動作空間才完整 ——
      離線 search 搜得到市場，這是 v5 的 demand map 給不了的。

    `market_threshold` 是 present head 的 logit 門檻（預設 0，即 sigmoid 0.5）。
    ⚠️ **這個門檻從來沒有被掃過。** `model/train.py` 記著 51.7% 的回合一筆訂單
    都沒有 —— 稀疏正例配 BCE 配 0.5 門檻本來就會系統性少下單，而實測召回率
    只有 0.72~0.79。「學不起來」和「門檻太高」還沒有被分開量過。
    """
    resolved = dict(RULE_DEFAULTS)
    resolved.update(params or {})
    resolved.pop("_replace_defaults", None)
    model_market = bool(resolved.get("model_market", False))
    market_threshold = float(resolved.get("market_threshold", 0.0))

    policy = _policy()
    spatial, scalar = C.encode(obs, config)
    positions, unit_features = C.encode_units(obs, config)
    (op_logits, qty_logits, _target,
     mk_present, mk_qty, _value, _demand) = policy(
        spatial, scalar, positions, unit_features)
    mask = C.legal_unit_mask(obs, config)

    units = _choose(op_logits, qty_logits, mask, obs)

    if model_market:
        market = C.decode_market_orders(
            mk_present, mk_qty, obs, config, threshold=market_threshold)
    else:
        # 只為了拿市場而跑一次完整的規則式排程。浪費，但這一臂本來就是對照組。
        market = gen0_act(obs, config, resolved)["market"]

    return {"farmer": units[0], "hands": units[1:], "market": market}


def agent(obs, config):
    return act(obs, config)
