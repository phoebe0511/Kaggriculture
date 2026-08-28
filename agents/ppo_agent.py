"""把 PPO 的 checkpoint 包成 `eval/runner.py` 吃得下的 agent。

    python -m eval.runner --a config/opponents/ppo-warm0.json --ladder \
        config/ladder-top.json --games 40 --log-level 0

spec 長這樣：

    {"name": "ppo-warm0", "entry": "agents.ppo_agent:act",
     "params": {"ckpt": "model/artifacts/ppo-warm0/last.pt", "greedy": true}}

## 🩸 這支是開發側的，不能出貨

它 import torch。submission 那條路是 `agents/gen2_model.py` +
`serving/npz_forward.py`（純 numpy）。要上場的話得先 `serving.export_npz`。

## greedy 還是取樣

PPO 優化的是**隨機** policy，但上場要交一個確定的動作。兩個都留：
`greedy=True` 走 argmax（評估和出貨用），`False` 走跟 rollout 同一條取樣路徑
（要對照訓練時看到的分數時用）。**兩者的分數會不一樣，不要混著比。**
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contracts as C                                   # noqa: E402

#: 每個行程一份，key 是 checkpoint 路徑。eval/runner 的 worker 會重複呼叫。
_CACHE = {}


def _load(path):
    """讀 checkpoint。width / blocks 從檔案裡拿，不要靠呼叫端傳對。"""
    import torch                                        # noqa: PLC0415

    from model.net import KaggricultureNet               # noqa: PLC0415

    if path in _CACHE:
        return _CACHE[path]
    blob = torch.load(path, map_location="cpu", weights_only=False)
    ver = blob.get("encoder_version")
    if ver is not None and ver != C.ENCODER_VERSION:
        raise SystemExit(
            f"{path} 是 ENCODER_VERSION {ver}，現在是 {C.ENCODER_VERSION} "
            f"—— 通道對不上不會報錯，只會表現成打得很爛")
    net = KaggricultureNet(
        n_spatial=C.N_SPATIAL, n_scalar=C.N_SCALAR,
        n_unit_features=C.N_UNIT_FEATURES, n_ops=C.N_UNIT_OPS, n_qty=C.N_QTY,
        n_targets=C.N_TARGET_CELLS, n_market_ops=C.N_MARKET_OPS,
        n_market_qty=C.N_MARKET_QTY, n_task_ops=C.N_TASK_OPS,
        width=blob["width"], unit_hidden=256, n_blocks=blob["blocks"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    torch.set_num_threads(1)          # eval/runner 是多行程的，別互搶
    _CACHE[path] = net
    return net


def act(obs, config=None, params=None):
    import torch                                        # noqa: PLC0415

    from agents.gen0 import step_toward                  # noqa: PLC0415
    from model.ppo import masked_log_softmax, sample_market   # noqa: PLC0415

    p = dict(params or {})
    path = p.get("ckpt") or os.environ.get("KAGGRI_PPO_CKPT")
    if not path:
        raise SystemExit("要給 ckpt（params 或 KAGGRI_PPO_CKPT）")
    net = _load(path)
    greedy = bool(p.get("greedy", True))

    spatial, scalar = C.encode(obs, config)
    pos, feats = C.encode_units(obs, config)
    n = len(pos)
    with torch.no_grad():
        op_logits, _qty, tgt_logits, mk_present, mk_qty, _v, _d = net(
            torch.as_tensor(spatial).unsqueeze(0),
            torch.as_tensor(scalar).unsqueeze(0),
            torch.zeros(n, dtype=torch.long),
            torch.as_tensor(np.asarray(pos)),
            torch.as_tensor(feats))
        op_mask = torch.as_tensor(C.legal_unit_mask(obs, config))
        tgt_mask = torch.as_tensor(C.legal_target_mask(obs, config))
        o_lp = masked_log_softmax(op_logits, op_mask)
        t_lp = masked_log_softmax(tgt_logits, tgt_mask)
        if greedy:
            o_idx = o_lp.argmax(-1)
            t_idx = t_lp.argmax(-1)
            # 門檻 0 對應 sigmoid 0.5，跟 `decode_market_orders` 的預設一致。
            mk_pres = (mk_present[0] > 0.0) & torch.as_tensor(
                C.legal_market_mask(obs, config))
            mk_q = mk_qty[0].argmax(-1)
        else:
            mk_legal = torch.as_tensor(
                C.legal_market_mask(obs, config)).unsqueeze(0)
            t_idx = torch.multinomial(t_lp.exp(), 1).squeeze(-1)
            o_idx = torch.multinomial(o_lp.exp(), 1).squeeze(-1)
            pres, q = sample_market(mk_present, mk_qty, mk_legal)
            mk_pres, mk_q = pres[0], q[0]

    board = len(obs["farms"][obs["player"]]["tiles"])
    t_np, o_np = t_idx.numpy(), o_idx.numpy()
    units = []
    for i in range(n):
        tx, ty = C.target_xy(int(t_np[i]), board)
        cur = tuple(pos[i])
        if (int(cur[0]), int(cur[1])) != (tx, ty):
            units.append(step_toward(cur, (tx, ty)))
        else:
            units.append(C.decode_unit(int(o_np[i]), None))

    qty_onehot = np.zeros((C.N_MARKET_OPS, C.N_MARKET_QTY), np.float32)
    qty_onehot[np.arange(C.N_MARKET_OPS), mk_q.numpy()] = 1.0
    market = C.decode_market_orders(
        np.where(mk_pres.numpy(), 1.0, -1.0), qty_onehot, obs, config)
    return {"farmer": units[0], "hands": units[1:], "market": market}


def agent(obs, config):
    """kaggle_environments 的進入點。checkpoint 從 `KAGGRI_PPO_CKPT` 拿。"""
    return act(obs, config)
