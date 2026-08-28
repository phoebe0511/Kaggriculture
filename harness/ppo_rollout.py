"""PPO 的自對局 rollout —— 收 (盤面, 動作, logprob, value, reward) 軌跡。

    python -m harness.ppo_rollout --games 4 --workers 4      # 只量吞吐量

跟 `harness/rollout.py`（DAgger）不同：那支收「盤面 -> 老師動作」，這支收
policy **自己**的動作和機率，PPO 要算 importance ratio 用。

## 動作設計：target × op，不是原始 44 個 op

每個 unit 選「去哪一格（100）」和「到了做什麼（44）」，走路交給
`gen0.step_toward`。理由是網路不用重新學路徑 —— `contracts.py` 的 v3 標籤
就是這個形狀，`legal_target_mask` / `legal_unit_mask` 都現成。

聯合動作是 13 個 unit 各自獨立取樣的乘積，**不碰 44^13**：

    logprob(joint) = Σ_unit [ logprob(target_u) + logprob(op_u) ]

## 吞吐量（2026-08-28 實測）

    引擎          1.08 秒/局（720 步）= 1.5 ms/步
    contracts.encode  0.15 ms/步
    網路 batch=128 on RTX 4060   0.022 ms/樣本

**瓶頸是引擎，不是網路。** 所以 worker 直接在自己的行程裡跑 torch（CPU 單樣本
2.9 ms/步會變成瓶頸，要批次）—— 先量單行程的實際數字再決定要不要 inference
server。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._quiet import silenced                      # noqa: E402

with silenced():
    import torch
    import contracts as C
    from kaggle_environments import make
    from agents.gen0 import step_toward
    from model.net import KaggricultureNet


def build_net(width=64, blocks=4, unit_hidden=256):
    """跟 `model/train.py` 同一組維度常數，換 width 不用改別的地方。"""
    return KaggricultureNet(
        n_spatial=C.N_SPATIAL, n_scalar=C.N_SCALAR,
        n_unit_features=C.N_UNIT_FEATURES, n_ops=C.N_UNIT_OPS, n_qty=C.N_QTY,
        n_targets=C.N_TARGET_CELLS, n_market_ops=C.N_MARKET_OPS,
        n_market_qty=C.N_MARKET_QTY, n_task_ops=C.N_TASK_OPS,
        width=width, unit_hidden=unit_hidden, n_blocks=blocks)


def masked_sample(logits, mask):
    """一次對所有 unit 取樣。`logits` / `mask` 都是 `[n, K]`。

    回傳 `(idx [n], logprob [n])`，都還在 tensor 上 —— **不要在這裡 `.cpu()`**，
    逐 unit 搬回主機是 rollout 的主要開銷（2026-08-28 實測：逐 unit 迴圈讓每步
    從 2.9 ms 變成 4.1 ms）。

    🩸 mask 整列全 False 的 unit 是存在的（站在 shed 上、手上沒東西）。
    那一列退回「全部合法」而不是拋錯 —— 引擎對非法動作是靜默忽略，
    取樣到非法的等於 PASS。
    """
    m = mask.bool()
    dead = ~m.any(dim=-1, keepdim=True)
    m = m | dead                       # 全 False 的那幾列變成全 True
    logp = torch.log_softmax(logits.masked_fill(~m, -1e9), dim=-1)
    idx = torch.multinomial(logp.exp(), 1).squeeze(-1)
    return idx, logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


class TorchPolicy:
    """把網路包成 `act(obs, config)`，順便吐出 PPO 要的 logprob / value。"""

    def __init__(self, net, device="cpu", seed=0):
        self.net = net.to(device).eval()
        self.device = device
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs, config):
        spatial, scalar = C.encode(obs, config)
        pos, feats = C.encode_units(obs, config)
        n = len(pos)
        dev = self.device
        sp = torch.as_tensor(spatial, device=dev).unsqueeze(0)
        sc = torch.as_tensor(scalar, device=dev).unsqueeze(0)
        ub = torch.zeros(n, dtype=torch.long, device=dev)
        up = torch.as_tensor(np.asarray(pos), device=dev)
        uf = torch.as_tensor(feats, device=dev)
        op_logits, qty_logits, tgt_logits, mk_present, mk_qty, value, _demand = \
            self.net(sp, sc, ub, up, uf)

        op_mask = torch.as_tensor(C.legal_unit_mask(obs, config), device=dev)
        tgt_mask = torch.as_tensor(C.legal_target_mask(obs, config), device=dev)
        board = len(obs["farms"][obs["player"]]["tiles"])

        t_idx, t_lp = masked_sample(tgt_logits, tgt_mask)
        o_idx, o_lp = masked_sample(op_logits, op_mask)
        logps = float((t_lp + o_lp).sum().item())
        t_np = t_idx.cpu().numpy()          # 一次搬回主機，不是逐 unit
        o_np = o_idx.cpu().numpy()

        units = []
        for i in range(n):
            tx, ty = C.target_xy(int(t_np[i]), board)
            cur = tuple(pos[i])
            if (int(cur[0]), int(cur[1])) != (tx, ty):
                # 🩸 step_toward 回傳的已經是動作 list（["EAST"]），不要再包一層
                units.append(step_toward(cur, (tx, ty)))
            else:
                units.append(C.decode_unit(int(o_np[i]), None))
        market = C.decode_market_orders(
            mk_present[0].cpu().numpy(),
            mk_qty[0].cpu().numpy().reshape(C.N_MARKET_OPS, C.N_MARKET_QTY),
            obs, config)
        action = {"farmer": units[0], "hands": units[1:], "market": market}
        return action, logps, float(value[0].item())


def bench(games=4, width=64, blocks=4, device="cpu"):
    """量一局自對局要多久，拆成引擎 / 編碼 / 網路。"""
    net = build_net(width, blocks)
    pol = TorchPolicy(net, device)
    steps = 0
    t0 = time.perf_counter()
    for g in range(games):
        with silenced():
            env = make("kaggriculture", configuration={"seed": 5000 + g}, debug=False)
            env.reset(2)
        while not env.done:
            acts = []
            for p in range(2):
                st = env.steps[-1][p]
                if st["status"] != "ACTIVE":
                    acts.append({})
                    continue
                o = dict(st["observation"])
                o["player"] = p
                a, _lp, _v = pol.act(o, env.configuration)
                acts.append(a)
                steps += 1
            with silenced():
                env.step(acts)
    dt = time.perf_counter() - t0
    print(f"  {games} 局   {dt/games:>6.2f} 秒/局   "
          f"{steps/dt:>7.0f} 個 agent 決策/秒   device={device}")
    return dt / games


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--envs", type=int, default=0,
                    help="用跨 env 批次跑幾個 env（0 = 走單 env 的舊路徑）")
    args = ap.parse_args(argv)
    if args.envs:
        net = build_net(args.width, args.blocks)
        vec = VecRollout(net, n_envs=args.envs, seed0=5000, device=args.device)
        t0 = time.perf_counter()
        steps, cash = vec.run()
        dt = time.perf_counter() - t0
        print(f"  {args.envs} 個 env 平行   {dt:>6.1f} 秒   "
              f"{dt/args.envs:>6.2f} 秒/局   {steps/dt:>7.0f} 個決策/秒   "
              f"device={args.device}")
        return 0
    bench(args.games, args.width, args.blocks, args.device)
    return 0



# --------------------------------------------------------------------------
# 跨 env 批次
# --------------------------------------------------------------------------
#
# 🩸 **單一 env 是死路。** 2026-08-28 實測：batch=1 的前向是 2.9 ms/樣本，
# 一局 8.9 秒；batch=32 在 RTX 4060 上是 0.07 ms/樣本，一局 1.74 秒。
# 向量化逐 unit 的取樣迴圈沒有幫助（4.39 vs 4.19 ms/步）—— 開銷在單樣本前向
# 本身，不在 Python 迴圈。
#
# 所以 rollout 的形狀是「一個行程裡跑 K 個 env，每一步把 2K 個盤面疊成一個
# batch 做一次前向」。K=32 時網路成本降到引擎的 1/20，瓶頸回到引擎。


class VecRollout:
    """K 個 env 一起走。每一步一次批次前向。

    每個 env 每一步有兩個 player 要決策，所以 batch 是 2K。
    `unit_board` 指回每個 unit 屬於 batch 裡第幾個盤面 —— unit 數量各盤面不同
    （雇了幾個人會變），所以不能用固定形狀。
    """

    def __init__(self, net, n_envs=32, seed0=0, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device
        self.n_envs = n_envs
        self.seed0 = seed0
        self.envs = []

    def reset(self):
        self.envs = []
        for i in range(self.n_envs):
            with silenced():
                env = make("kaggriculture",
                           configuration={"seed": self.seed0 + i}, debug=False)
                env.reset(2)
            self.envs.append(env)

    def _gather(self):
        """收集這一步所有還活著的 (env_index, player) 的觀測。"""
        items = []
        for ei, env in enumerate(self.envs):
            if env.done:
                continue
            for p in range(2):
                st = env.steps[-1][p]
                if st["status"] != "ACTIVE":
                    continue
                obs = dict(st["observation"])
                obs["player"] = p
                items.append((ei, p, obs, env.configuration))
        return items

    @torch.no_grad()
    def _policy_batch(self, items):
        """一次前向。回傳每個 item 的 (action, logprob, value)。"""
        dev = self.device
        # 🩸 `C.encode` 是 0.31 ms/次 —— 兩個回傳值要一次拿，不要呼叫兩次。
        enc = [C.encode(o, c) for _, _, o, c in items]
        sp = np.stack([e[0] for e in enc])
        sc = np.stack([e[1] for e in enc])
        pos_list, feat_list, board_list = [], [], []
        for bi, (_, _, o, c) in enumerate(items):
            pos, feats = C.encode_units(o, c)
            pos_list.append(np.asarray(pos))
            feat_list.append(feats)
            board_list.append(np.full(len(pos), bi, dtype=np.int64))
        up = np.concatenate(pos_list)
        uf = np.concatenate(feat_list)
        ub = np.concatenate(board_list)

        op_logits, _qty, tgt_logits, mk_present, mk_qty, value, _d = self.net(
            torch.as_tensor(sp, device=dev), torch.as_tensor(sc, device=dev),
            torch.as_tensor(ub, device=dev), torch.as_tensor(up, device=dev),
            torch.as_tensor(uf, device=dev))

        op_mask = torch.as_tensor(
            np.concatenate([C.legal_unit_mask(o, c) for _, _, o, c in items]),
            device=dev)
        tgt_mask = torch.as_tensor(
            np.concatenate([C.legal_target_mask(o, c) for _, _, o, c in items]),
            device=dev)
        t_idx, t_lp = masked_sample(tgt_logits, tgt_mask)
        o_idx, o_lp = masked_sample(op_logits, op_mask)

        t_np, o_np = t_idx.cpu().numpy(), o_idx.cpu().numpy()
        lp_np = (t_lp + o_lp).cpu().numpy()
        v_np = value.cpu().numpy()
        mp_np, mq_np = mk_present.cpu().numpy(), mk_qty.cpu().numpy()
        ub_np = ub

        out = []
        for bi, (ei, p, o, c) in enumerate(items):
            sel = ub_np == bi
            board = len(o["farms"][p]["tiles"])
            pos = pos_list[bi]
            units = []
            for k, (ti, oi) in enumerate(zip(t_np[sel], o_np[sel])):
                tx, ty = C.target_xy(int(ti), board)
                cur = tuple(pos[k])
                if (int(cur[0]), int(cur[1])) != (tx, ty):
                    units.append(step_toward(cur, (tx, ty)))
                else:
                    units.append(C.decode_unit(int(oi), None))
            market = C.decode_market_orders(
                mp_np[bi], mq_np[bi].reshape(C.N_MARKET_OPS, C.N_MARKET_QTY), o, c)
            action = {"farmer": units[0], "hands": units[1:], "market": market}
            out.append((ei, p, action, float(lp_np[sel].sum()), float(v_np[bi])))
        return out

    def run(self, max_steps=720):
        """跑到全部結束。回傳 (走了幾步, 每個 env 的期末現金)。"""
        self.reset()
        steps = 0
        for _ in range(max_steps):
            items = self._gather()
            if not items:
                break
            decided = self._policy_batch(items)
            acts = {ei: [{}, {}] for ei, _, _, _, _ in decided}
            for ei, p, action, _lp, _v in decided:
                acts[ei][p] = action
            # 引擎自己不印東西，不用每一步都 dup2 —— `silenced()` 一次要 4 個
            # syscall，720 步 × K 個 env 加起來很可觀。
            for ei, pair in acts.items():
                self.envs[ei].step(pair)
            steps += len(decided)
        cash = [[s["reward"] for s in e.steps[-1]] for e in self.envs]
        return steps, cash


if __name__ == "__main__":
    raise SystemExit(main())
