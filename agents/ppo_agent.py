"""把 PPO 的 checkpoint 包成 `eval/runner.py` 吃得下的 agent。

    python -m eval.runner --a config/opponents/ppo-warm0.json --ladder \
        config/ladder-top.json --games 40 --log-level 0

spec 長這樣：

    {"name": "ppo-warm0", "entry": "agents.ppo_agent:act",
     "params": {"ckpt": "model/artifacts/ppo-warm0/last.pt", "greedy": true}}

## 🩸 這支是開發側的，不能出貨

它 import torch。submission 那條路是 `agents/gen2_model.py` +
`serving/npz_forward.py`（純 numpy）。要上場的話得先 `serving.export_npz`。

## 混合模式（`base`）

`params` 給了 `base` 就換一條路：**工人動作由 base policy 出，網路只出
market**，跟 `harness/ppo_rollout.py` 的 `--base-policy` 完全同一個組合。

    {"name": "ppo-hybrid", "entry": "agents.ppo_agent:act",
     "params": {"ckpt": "model/artifacts/ppo-hybrid/best.pt",
                "base": "config/params/cma1-g50-wt.json", "greedy": true}}

`base_side` 決定骨幹負責哪一半：

    "units"（預設）  骨幹出工人動作，網路只出 market
    "market"         骨幹出 market，網路只出工人動作

🩸 `base_side="units"` **用不到 op / target head**，所以 `labels` 是
`immediate` 還是 `target` 在那條路上不影響結果（純網路那條差 31%，§44）。
`base_side="market"` 剛好相反 —— 它**只**用 op / target head，所以 `labels`
一定要跟 checkpoint 對得上。

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

#: 混合模式的骨幹 policy，key 是 spec 路徑。同樣是每個行程一份。
_BASE = {}


def _load_base(spec):
    """混合模式的骨幹。跟 `harness/ppo_rollout.load_base_policy` 同一條路徑。"""
    if spec not in _BASE:
        from harness.ppo_rollout import load_base_policy   # noqa: PLC0415

        _BASE[spec] = load_base_policy(spec)
    return _BASE[spec]


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
    base_spec = p.get("base") or ""
    base_side = p.get("base_side", "units")
    if base_side not in ("units", "market"):
        raise SystemExit(f"base_side 只能是 units / market，收到 {base_side!r}")

    spatial, scalar = C.encode(obs, config)
    pos, feats = C.encode_units(obs, config)
    n = len(pos)
    with torch.no_grad():
        op_logits, qty_logits, tgt_logits, mk_present, mk_qty, _v, _d = net(
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
            # 🩸 op 不在這裡選 —— `_GUARD_OPS` 的合法性會被同一回合稍早的
            # unit 改掉，要逐 unit 縮 mask 再選。見下面的迴圈。
            t_idx = t_lp.argmax(-1)
            q_idx = qty_logits.argmax(-1)
            # 門檻 0 對應 sigmoid 0.5，跟 `decode_market_orders` 的預設一致。
            mk_pres = (mk_present[0] > 0.0) & torch.as_tensor(
                C.legal_market_mask(obs, config))
            mk_q = mk_qty[0].argmax(-1)
        else:
            mk_legal = torch.as_tensor(
                C.legal_market_mask(obs, config)).unsqueeze(0)
            t_idx = torch.multinomial(t_lp.exp(), 1).squeeze(-1)
            q_idx = torch.multinomial(
                torch.softmax(qty_logits, dim=-1), 1).squeeze(-1)
            pres, q = sample_market(mk_present, mk_qty, mk_legal)
            mk_pres, mk_q = pres[0], q[0]

    board = len(obs["farms"][obs["player"]]["tiles"])
    t_np, q_np = t_idx.numpy(), q_idx.numpy()
    # `qty`：PICKUP / PLACE 的數量由網路選（跟 `--qty-factor` 訓出來的
    # checkpoint 配對）。預設關閉 —— 舊的 spec 行為一個位元都不變。
    use_qty = bool(p.get("qty", False))

    qty_onehot = np.zeros((C.N_MARKET_OPS, C.N_MARKET_QTY), np.float32)
    qty_onehot[np.arange(C.N_MARKET_OPS), mk_q.numpy()] = 1.0
    market = C.decode_market_orders(
        np.where(mk_pres.numpy(), 1.0, -1.0), qty_onehot, obs, config)

    # 🩸 同回合衝突靠逐 unit 縮 mask 解決，見 `contracts.guard_mask_row`。
    # 送出引擎會拒絕的動作不是策略選擇，是沒照規則走 —— 無條件修，不做旗標。
    guard = C.turn_guard_state(obs)
    op_mask_np = op_mask.numpy()

    units = []
    for i in range(n) if (not base_spec or base_side == "market") else ():
        tx, ty = C.target_xy(int(t_np[i]), board)
        cur = tuple(pos[i])
        if (int(cur[0]), int(cur[1])) != (tx, ty):
            # 沒站到目標 tile 的 unit 送出的是移動，不會動到任何共用狀態。
            units.append(step_toward(cur, (tx, ty)))
            continue
        # 🩸 逐 unit 縮 mask 再選：`_GUARD_OPS` 的合法性會被同一回合稍早的
        # unit 改掉。訓練側（`harness/ppo_rollout.py`）走同一套，兩邊不能分歧
        # —— 這就是 contracts.py 存在的理由。
        row = C.guard_mask_row(op_mask_np[i], guard, (tx, ty))
        with torch.no_grad():
            row_lp = masked_log_softmax(
                op_logits[i].unsqueeze(0),
                torch.as_tensor(row).unsqueeze(0))[0]
        op_sel = (int(row_lp.argmax()) if greedy
                  else int(torch.multinomial(row_lp.exp(), 1)))
        # 把效果套回 `guard` —— 後面的 unit 看得到改變的唯一途徑。
        C.turn_guard_commit(op_sel, guard, (tx, ty))
        op_i = op_sel
        units.append(C.decode_unit(
            op_i, int(q_np[i]) if use_qty else None))
    if base_spec:
        base = _load_base(base_spec)(obs, config)
        if base_side == "units":
            # 🩸 只換 market。工人動作整個沿用 base，包含它自己的
            # farmer/hands 拆法。
            return {**base, "market": market}
        # 反向：只換 market 以外的部分 —— 工人是網路的，訂單沿用 base。
        return {"farmer": units[0], "hands": units[1:],
                "market": base["market"]}
    return {"farmer": units[0], "hands": units[1:], "market": market}


def agent(obs, config):
    """kaggle_environments 的進入點。checkpoint 從 `KAGGRI_PPO_CKPT` 拿。"""
    return act(obs, config)
