"""出貨端的 target-head agent：網路只決定「去哪一格、到了做什麼」。

純 numpy（`serving/npz_forward.py`），**不 import torch** —— 這支是要進
submission 的。開發端的對應物是 `agents/ppo_agent.py`（吃 `.pt`、用 torch），
兩支必須輸出**完全相同**的動作，`tests/test_gen3_target.py` 守著這件事。

## 跟 `agents/gen2_model.py` 的差別（很重要，別混用權重）

| | gen2_model | 這一支 |
|---|---|---|
| `labels` | `immediate`：op head 是「這一步做什麼」 | `target`：op head 是「走到終點做什麼」 |
| 走路 | op head 自己輸出 NORTH/SOUTH/… | target head 選格子，`gen0.step_toward` 走 |
| qty head | `argmax(qty_logits)` | 不用（`decode_unit(op, None)`） |

🩸 **載錯 `labels` 不會報錯，只會整局 PASS 拿 0 分**（`gen2_model.require_labels`
的說明，2026-08-21 踩過）。所以這裡也檢查。

## 走路

`gen0.step_toward` 直接用，**不要另外抄一份**（`CLAUDE.md` 硬規則）。
submission 本來就打包了 `gen0.py`。
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import contracts as C                                   # noqa: E402
# 🩸 模組層 import，理由跟 `agents/gen2_model.py` 一樣（權重仍然是 lazy 的）。
from serving.npz_forward import NumpyPolicy             # noqa: E402

try:                                    # submission 是攤平的，沒有 agents/ 這層
    from agents.gen0 import step_toward
except ImportError:                     # noqa: BLE001
    from gen0 import step_toward

#: 權重檔。慣例跟 `agents/gen2_model.py` 一樣：先找同目錄的 `weights.npz`
#: （submission 是攤平的），再退回 repo 的 `submission/weights.npz`。
WEIGHTS_PATH = os.environ.get(
    "KAGGRI_WEIGHTS",
    str(next((p for p in (_HERE / "weights.npz",
                          _HERE.parent / "submission" / "weights.npz")
              if p.is_file()), _HERE / "weights.npz")))

_POLICY = None
_POLICY_LOCK = threading.Lock()


def _policy(path=None):
    global _POLICY
    want = path or WEIGHTS_PATH
    if _POLICY is None or _POLICY[0] != want:
        with _POLICY_LOCK:
            policy = NumpyPolicy(want)
            if policy.encoder_version != C.ENCODER_VERSION:
                raise SystemExit(
                    f"{want} 是 ENCODER_VERSION {policy.encoder_version}，"
                    f"contracts.py 是 {C.ENCODER_VERSION} —— 重新訓練")
            if policy.labels not in (None, "target"):
                raise SystemExit(
                    f"{want} 的 op head 是 `{policy.labels}`，這支要 `target`"
                    f" —— 載錯的話整局 PASS 拿 0 分而且不會報錯")
            _POLICY = (want, policy)
    return _POLICY[1]


def masked_argmax(logits, mask):
    """跟 `model.ppo.masked_log_softmax` 一樣的遮罩語意，再取 argmax。

    🩸 mask 整列全 False 的 unit 是存在的（站在 shed 上、手上沒東西）。
    那一列退回「全部合法」—— 跟訓練端一致，不然同一份權重兩邊會選不同動作。
    """
    m = np.asarray(mask, dtype=bool)
    m = m | ~m.any(axis=-1, keepdims=True)
    return np.argmax(np.where(m, logits, -np.inf), axis=-1)


def act(obs, config=None, params=None):
    p = dict(params or {})
    policy = _policy(p.get("weights"))
    spatial, scalar = C.encode(obs, config)
    positions, unit_features = C.encode_units(obs, config)
    (op_logits, _qty, target_logits,
     mk_present, mk_qty, _value, _demand) = policy(
        spatial, scalar, positions, unit_features)

    t_idx = masked_argmax(target_logits, C.legal_target_mask(obs, config))
    o_idx = masked_argmax(op_logits, C.legal_unit_mask(obs, config))

    board = len(obs["farms"][obs["player"]]["tiles"])
    units = []
    for i in range(len(positions)):
        tx, ty = C.target_xy(int(t_idx[i]), board)
        cur = positions[i]
        if (int(cur[0]), int(cur[1])) != (tx, ty):
            units.append(step_toward((int(cur[0]), int(cur[1])), (tx, ty)))
        else:
            units.append(C.decode_unit(int(o_idx[i]), None))

    # 門檻 0 對應 sigmoid 0.5。PPO 訓練時 present 是 Bernoulli，出貨取
    # 機率大於一半的那些 —— 跟 `agents/ppo_agent.py` 的 greedy 分支同義。
    legal = np.asarray(C.legal_market_mask(obs, config), dtype=bool)
    present = (np.asarray(mk_present) > 0.0) & legal
    qty_onehot = np.zeros((C.N_MARKET_OPS, C.N_MARKET_QTY), np.float32)
    qty_onehot[np.arange(C.N_MARKET_OPS),
               np.argmax(np.asarray(mk_qty).reshape(
                   C.N_MARKET_OPS, C.N_MARKET_QTY), axis=-1)] = 1.0
    market = C.decode_market_orders(
        np.where(present, 1.0, -1.0), qty_onehot, obs, config)
    return {"farmer": units[0], "hands": units[1:], "market": market}


def agent(obs, config):
    return act(obs, config)
