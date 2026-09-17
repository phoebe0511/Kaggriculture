"""PPO 的自對局 rollout —— 收 (盤面, 動作, logprob, value, reward) 軌跡。

    python -m harness.ppo_rollout --games 4 --workers 4      # 只量吞吐量

跟 `harness/rollout.py`（DAgger）不同：那支收「盤面 -> 老師動作」，這支收
policy **自己**的動作和機率，PPO 要算 importance ratio 用。

## 動作設計：target × op，不是原始 44 個 op

每個 unit 選「去哪一格（100）」和「到了做什麼（44）」，走路交給
`gen0.step_toward`。理由是網路不用重新學路徑 —— `contracts.py` 的 v3 標籤
就是這個形狀，`legal_target_mask` / `legal_unit_mask` 都現成。

聯合動作是 13 個 unit 各自獨立取樣的乘積，再乘上 market 那一份，**不碰
44^13**：

    logprob(joint) = Σ_unit [ logprob(target_u) + logprob(op_u) ]
                   + logprob(market)

🩸 market 一定要一起取樣。`contracts.decode_market_orders` 本身是門檻 +
argmax，走那條路 HIRE / BUY_LAND / BUY_SEED / SELL 全部沒有 logprob，
policy gradient 到不了 —— PPO 就只能學「工人走去哪、到了做什麼」，
學不到雇工、買地、種什麼。

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
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 🩸 `agents.gen0` 在 import 時就把 LOG_LEVEL 讀成常數（gen0.py:66），預設 3
# 會把每個決策細節寫到 stderr。當對手跑的時候那是每局 720 次寫入。
# 一定要在 import 之前設，之後改沒有用。
os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")

from tools._quiet import silenced                      # noqa: E402
from model.networth import asset_value                 # noqa: E402

with silenced():
    import torch
    import contracts as C
    from kaggle_environments import make
    from agents.gen0 import step_toward
    from model.net import KaggricultureNet
    from agents import gen0 as _gen0
    from model.ppo import (TrajectoryWriter, market_logp_entropy,
                           masked_log_softmax, sample_market)


def load_base_policy(name):
    """混合模式用的骨幹：`gen0.act` 帶一組凍結參數。

    回傳 `(obs, config) -> action`，跟 `load_opponent` 同一條載入路徑，
    所以「骨幹」和「對手」可以指到同一份 spec。
    """
    return load_opponent(name)


def build_net(width=64, blocks=4, unit_hidden=256):
    """跟 `model/train.py` 同一組維度常數，換 width 不用改別的地方。"""
    return KaggricultureNet(
        n_spatial=C.N_SPATIAL, n_scalar=C.N_SCALAR,
        n_unit_features=C.N_UNIT_FEATURES, n_ops=C.N_UNIT_OPS, n_qty=C.N_QTY,
        n_targets=C.N_TARGET_CELLS, n_market_ops=C.N_MARKET_OPS,
        n_market_qty=C.N_MARKET_QTY, n_task_ops=C.N_TASK_OPS,
        width=width, unit_hidden=unit_hidden, n_blocks=blocks)


def load_opponent(name):
    """用 `eval/runner.py` 的載入器做出一個 `(obs, config) -> action`。

    吃得下 `config/params/cma1-g50-wt.json` 這種凍結的 spec（會自動帶
    `_replace_defaults`），也吃得下 `gen0`、`config/opponents/*.json`。
    """
    from eval.runner import build_agent, load_spec       # noqa: PLC0415
    fn = build_agent(load_spec(name))
    if isinstance(fn, str):
        raise SystemExit(f"{name} 是引擎內建的（{fn}），這裡要的是 callable")
    return fn


def load_league(names):
    """逗號分隔的 spec 名單 -> `(callable list, 名字 list)`。

    🩸 `config/ladder-top.json` 那 8 支是**開迴路 replay**，不會對我們的行為
    反應（`agents/replay.py` 的說明）。放進 league 是為了盤面和市場競爭的
    多樣性，**不能只靠它們** —— 固定的動作序列是可以被背下來的。
    """
    picks = [n.strip() for n in str(names).split(",") if n.strip()]
    return [load_opponent(n) for n in picks], picks


#: 哪些 op 帶數量。`contracts.decode_unit` 只對 PICKUP / PLACE 用 qty_index
#: （PLANT 帶 item 但不帶數量，15 個無參數 op 都不帶）。這張表是從
#: `C.UNIT_OPS` 導出來的，不是另外定義一套。
QTY_OPS = np.array([op in ("PICKUP", "PLACE") for op, _item in C.UNIT_OPS])


def count_plants(farm):
    """一個 farm 上種著作物的格子數。

    判定條件跟 `agents/gen0.py:2250` 的 `crop_count` 一樣（`kind == "PLANT"`）
    —— 兩邊要一致，不然 reward shaping 量的東西跟 agent 看的不是同一個。

    §55.6：6 局 ladder 頂端重播，我們第 0 天是 5 格、頂端 9~19（差 -11.3），
    第 5 天 13 對 18~19（差 -5.7，sd 0.5）。
    """
    return sum(1 for row in farm["tiles"] for t in row
               if isinstance(t, dict) and t.get("kind") == "PLANT")


def masked_sample(logits, mask, greedy=False, gen=None):
    """一次對所有 unit 取樣。`logits` / `mask` 都是 `[n, K]`。

    回傳 `(idx [n], logprob [n])`，都還在 tensor 上 —— **不要在這裡 `.cpu()`**，
    逐 unit 搬回主機是 rollout 的主要開銷（2026-08-28 實測：逐 unit 迴圈讓每步
    從 2.9 ms 變成 4.1 ms）。

    mask 的處理（含整列全 False 的 unit）交給 `model.ppo.masked_log_softmax`
    —— update 重算 logprob 時走的是同一個函式，兩邊分歧會讓 ratio 一開始就
    不是 1。
    """
    logp = masked_log_softmax(logits, mask)
    # greedy 是上場用的：直接取機率最高的那個，沒有隨機性。
    # 🩸 **PPO 優化的是取樣版，上場跑的是 greedy 版，兩者不保證同方向。**
    # 2026-08-28 實測：取樣現金 12,999 -> 31,772（在升），同一段訓練的 greedy
    # 是 44,060 -> 30,740（在降）。所以挑 checkpoint 一定要看 greedy。
    # 🩸 `gen` 不給就走全域 RNG，整段 rollout 不可重現（見 VecRollout.__init__）。
    idx = (logp.argmax(-1) if greedy
           else torch.multinomial(logp.exp(), 1, generator=gen).squeeze(-1))
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

        mk_legal = torch.as_tensor(
            C.legal_market_mask(obs, config)[None], device=dev)

        t_idx, t_lp = masked_sample(tgt_logits, tgt_mask)
        o_idx, o_lp = masked_sample(op_logits, op_mask)
        mk_pres, mk_q = sample_market(mk_present, mk_qty, mk_legal)
        mk_lp, _ = market_logp_entropy(
            mk_present, mk_qty, mk_legal, mk_pres, mk_q)
        logps = float((t_lp + o_lp).sum().item()) + float(mk_lp[0].item())
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
        qty_onehot = np.zeros((C.N_MARKET_OPS, C.N_MARKET_QTY), np.float32)
        qty_onehot[np.arange(C.N_MARKET_OPS), mk_q[0].cpu().numpy()] = 1.0
        market = C.decode_market_orders(
            np.where(mk_pres[0].cpu().numpy(), 1.0, -1.0), qty_onehot,
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
    ap.add_argument("--collect", action="store_true",
                    help="順便收軌跡，印出緩衝區的大小與 reward 分佈")
    ap.add_argument("--opponent", default="",
                    help="固定對手的 spec（空字串 = 自對局）")
    ap.add_argument("--episode-steps", type=int, default=0)
    ap.add_argument("--phi", default="none",
                    choices=("none", "plants", "assets"),
                    help="potential-based shaping 的 Φ 裝什麼")
    ap.add_argument("--recognise", default="strict",
                    choices=("strict", "produce"),
                    help="--phi assets 的認列時點（§83.4）")
    args = ap.parse_args(argv)
    if args.envs:
        net = build_net(args.width, args.blocks)
        vec = VecRollout(net, n_envs=args.envs, seed0=5000, device=args.device,
                         episode_steps=args.episode_steps or None,
                         opponent=load_opponent(args.opponent)
                         if args.opponent else None,
                         phi=args.phi, recognise=args.recognise)
        t0 = time.perf_counter()
        steps, cash, trajs = vec.run(collect=args.collect)
        dt = time.perf_counter() - t0
        print(f"  {args.envs} 個 env 平行   {dt:>6.1f} 秒   "
              f"{dt/args.envs:>6.2f} 秒/局   {steps/dt:>7.0f} 個決策/秒   "
              f"device={args.device}")
        if args.collect:
            from model.ppo import RolloutBatch, step_rewards
            mb = sum(t.nbytes() for t in trajs) / 1e6
            batch = RolloutBatch(trajs)
            r = np.concatenate([step_rewards(t.cash[:, 0], t.cash[:, 1])
                                for t in trajs])
            print(f"  軌跡 {len(trajs)} 條   {len(batch)} 步   "
                  f"{len(batch.unit_step)} 個 unit   {mb:.0f} MB")
            print(f"  reward  mean {r.mean():+.4f}  std {r.std():.4f}  "
                  f"非零 {100*np.mean(r != 0):.1f}%    "
                  f"adv std {batch.adv.std():.3f}  ret 範圍 "
                  f"[{batch.ret.min():+.2f}, {batch.ret.max():+.2f}]")
            pairs = vec.our_cash(cash)
            wins = sum(1 for a, b in pairs if a > b)
            print(f"  期末    我方 {np.mean([a for a, _ in pairs]):>9,.0f}   "
                  f"對手 {np.mean([b for _, b in pairs]):>9,.0f}   "
                  f"勝 {wins}/{len(pairs)}")
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

    def __init__(self, net, n_envs=32, seed0=0, device="cpu",
                 episode_steps=None, opponent=None, half_obs=True,
                 opp_offset=0, base_policy=None, greedy=False,
                 base_side="units", phi="none", recognise="strict",
                 op_exec_only=False, qty_factor=False):
        self.net = net.to(device).eval()
        self.device = device
        self.half_obs = half_obs
        self.n_envs = n_envs
        self.seed0 = seed0
        self.episode_steps = episode_steps
        #: `None` = 自對局（兩邊都是 net，兩條軌跡都收）。
        #: callable `(obs, config) -> action`，或一串 callable（league）——
        #: 第 i 個 env 配 `opponents[i % len]`，只收我方那一條軌跡。
        if opponent is None:
            self.opponents = None
        elif callable(opponent):
            self.opponents = [opponent]
        else:
            self.opponents = list(opponent)
            if not self.opponents:
                raise ValueError("league 是空的")
        self.opp_offset = opp_offset
        #: 混合模式：`gen0.act` 出工人動作，網路只出 market 訂單。
        #:
        #: 🩸 這個切法之所以乾淨，是因為 `gen0.act` 的 `_market(...)` 是在
        #: `unit_actions` **算完之後**才呼叫的（`agents/gen0.py:2543`）——
        #: 工人排程不依賴這一回合的訂單。耦合是延遲一回合的：這回合買了什麼
        #: 種子，下一回合 `_plan_basket` / `tasks` 才看得到。那個回饋迴路正是
        #: 要讓 PPO 學的東西。
        #:
        #: 動作空間從「9 個 unit × (100 格 + 44 op)」縮到「21 個 Bernoulli +
        #: 21 個 categorical(18)」，而且起點是 `cma1-g50-wt` 那條線而不是
        #: 監督式網路。
        self.base_policy = base_policy
        #: 骨幹負責哪一半 —— `"units"` 骨幹出工人（PPO 只練 market head），
        #: `"market"` 骨幹出 market（PPO 只練 unit head）。
        #:
        #: 為什麼要固定一邊：兩邊都讓網路出的話（warm0/1/3）greedy 每次都退，
        #: 而且退了也分不出是哪一個 head 的錯。固定一邊之後分數變化只能來自
        #: 另一邊，梯度也只進那一組 head。
        if base_side not in ("units", "market"):
            raise ValueError(f"base_side 只能是 units / market，收到 {base_side!r}")
        self.base_side = base_side
        #: 上場用的解碼（argmax / logit>0）。訓練時是 False。
        self.greedy = greedy
        #: F2 ablation：unit 沒站到目標格上時，那一步的 op **不會進引擎**
        #: （下面 `_policy_batch` 的 `step_toward` 分支）。開這個旗標就把那些
        #: op 因子從 joint logprob 裡拿掉，policy gradient 不再推它們。
        #:
        #: 🩸 遮罩存進軌跡（`rec["op_exec"]`），update 重算時用**同一份**，
        #: 不在那邊重新判定 —— 重判一次就可能跟 rollout 當下不一致，ratio 會錯。
        self.op_exec_only = bool(op_exec_only)
        #: qty intervention：PICKUP / PLACE 的數量改成**真的由 policy 選**，
        #: 而且進 joint logprob。關掉時 `decode_unit` 收到 None，數量固定 1
        #: （原本的行為，`qty_out` 拿不到梯度）。
        self.qty_factor = bool(qty_factor)
        #: 🩸 動作取樣專用的 RNG，種子從 `seed0` 來。
        #:
        #: 沒有這個的話 `torch.multinomial` / `torch.rand` 走全域 RNG，而
        #: `harness/ppo_pool.py` 的 worker 只呼叫 `torch.set_num_threads(1)`、
        #: **沒有 seed torch** —— 每個 worker 行程拿到的是它啟動時碰巧的種子。
        #:
        #: 2026-08-31 踩到的後果：同一個 checkpoint、同樣的參數跑兩次，
        #: 第 0 輪現金 63,543 vs 60,704、第 10 輪 greedy 77,628 vs 46,490
        #: （差 4.4 個評估標準誤）。**兩次單獨的訓練沒辦法拿來 A/B。**
        #: `--seed0` 本來只餵給環境（決定地圖），沒餵給動作取樣。
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed0))
        #: potential-based shaping 的 Φ 要裝什麼（`model.ppo.step_rewards`）：
        #:
        #:   `none`    不算，reward 就是逐步現金
        #:   `plants`  我方作物格數 − 對手作物格數（§55.6）
        #:   `assets`  我方非現金資產金額（§83.3，`model/networth.py`）
        #:
        #: 🩸 `assets` 只算我方 —— `obs["private"]`（倉庫、手上、種子）是
        #:    自己的才看得到。零和留在現金項。
        if phi not in ("none", "plants", "assets"):
            raise ValueError(f"--phi 只能是 none / plants / assets，收到 {phi!r}")
        self.phi_mode = phi
        self.recognise = recognise
        #: 打固定對手時我方坐哪一邊。單雙數輪流，不然先手／後手的差異會被學進去。
        self.seats = [i % 2 for i in range(n_envs)]
        self.envs = []

    def reset(self):
        self.envs = []
        for i in range(self.n_envs):
            cfg = {"seed": self.seed0 + i}
            if self.episode_steps:
                cfg["episodeSteps"] = self.episode_steps
            with silenced():
                env = make("kaggriculture", configuration=cfg, debug=False)
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

    def _ours(self, ei, p):
        return self.opponents is None or p == self.seats[ei]

    def opp_of(self, ei):
        """第 ei 個 env 配到 league 裡的哪一支。"""
        return self.opponents[self.opp_index(ei)]

    def opp_index(self, ei):
        """🩸 要加 `opp_offset`（主行程給的全域局號）。只用 `ei % len` 的話，
        每個 worker 只有 `envs` 個 env，league 再大也只會用到前面那幾支。"""
        if not self.opponents:
            return -1
        return (self.opp_offset + ei) % len(self.opponents)

    @torch.no_grad()
    def _policy_batch(self, items, collect=False):
        """一次前向。回傳每個 item 的 `(env_index, player, action, rec)`。

        `rec` 是這一步要存進軌跡的東西（`collect=False` 時只有 logp/value，
        沒有觀測）—— 觀測佔 98% 的記憶體，只量吞吐量時不留。
        """
        dev = self.device
        # 🩸 `C.encode` 是 0.31 ms/次 —— 兩個回傳值要一次拿，不要呼叫兩次。
        enc = [C.encode(o, c) for _, _, o, c in items]
        sp = np.stack([e[0] for e in enc])
        sc = np.stack([e[1] for e in enc])
        # spatial 佔軌跡 98% 的位元組（15.2 KB/步）。存 float16 就少一半，
        # 多行程把軌跡送回主行程時省的是同一半。
        # 🩸 **前向也要吃 float16 還原後的值**，不然 update 重算的 logprob 跟
        # rollout 存的會差一點點，ratio 一開始就不是 1。
        sp16 = None
        if collect and self.half_obs:
            sp16 = sp.astype(np.float16)
            sp = sp16.astype(np.float32)
        pos_list, feat_list, board_list = [], [], []
        for bi, (_, _, o, c) in enumerate(items):
            pos, feats = C.encode_units(o, c)
            pos_list.append(np.asarray(pos))
            feat_list.append(feats)
            board_list.append(np.full(len(pos), bi, dtype=np.int64))
        up = np.concatenate(pos_list)
        uf = np.concatenate(feat_list)
        ub = np.concatenate(board_list)

        op_logits, qty_logits, tgt_logits, mk_present, mk_qty, value, _d = self.net(
            torch.as_tensor(sp, device=dev), torch.as_tensor(sc, device=dev),
            torch.as_tensor(ub, device=dev), torch.as_tensor(up, device=dev),
            torch.as_tensor(uf, device=dev))

        op_mask_np = np.concatenate(
            [C.legal_unit_mask(o, c) for _, _, o, c in items])
        tgt_mask_np = np.concatenate(
            [C.legal_target_mask(o, c) for _, _, o, c in items])
        op_mask = torch.as_tensor(op_mask_np, device=dev)
        tgt_mask = torch.as_tensor(tgt_mask_np, device=dev)
        t_idx, t_lp = masked_sample(tgt_logits, tgt_mask, self.greedy, self.gen)
        o_idx, o_lp = masked_sample(op_logits, op_mask, self.greedy, self.gen)
        # 🩸 qty 沒有合法性遮罩（1~12 都送得出去，超過的由引擎自己夾），所以
        # 餵一張全 True 進去 —— 走同一個 `masked_sample` 才跟 update 端一致。
        q_idx = torch.zeros_like(o_idx)
        q_lp = torch.zeros_like(o_lp)
        if self.qty_factor:
            q_mask = torch.ones_like(qty_logits, dtype=torch.bool)
            q_idx, q_lp = masked_sample(qty_logits, q_mask, self.greedy,
                                        self.gen)

        # market 也要取樣，不能用 `decode_market_orders` 的門檻 + argmax ——
        # 那條路沒有 logprob，HIRE / BUY_LAND / BUY_SEED 就拿不到 gradient。
        mk_legal_np = np.stack(
            [C.legal_market_mask(o, c) for _, _, o, c in items])
        mk_legal = torch.as_tensor(mk_legal_np, device=dev)
        mk_pres_act, mk_qty_act = sample_market(mk_present, mk_qty, mk_legal,
                                                self.greedy, self.gen)
        mk_lp, _mk_ent = market_logp_entropy(
            mk_present, mk_qty, mk_legal, mk_pres_act, mk_qty_act)

        t_np, o_np = t_idx.cpu().numpy(), o_idx.cpu().numpy()
        # 🩸 target 和 op 的 logprob 要分開留一份 —— `--op-exec-only` 只把
        # **沒有真的執行到**的 op 那一項排除，target 一律留著。
        t_lp_np, o_lp_np = t_lp.cpu().numpy(), o_lp.cpu().numpy()
        lp_np = t_lp_np + o_lp_np
        q_np, q_lp_np = q_idx.cpu().numpy(), q_lp.cpu().numpy()
        # 只有帶數量的 op 才算 qty 那一項 —— 其他 op 的 qty 不是決策。
        q_use_np = QTY_OPS[o_np] if self.qty_factor else np.zeros(
            len(o_np), dtype=bool)
        v_np = value.cpu().numpy()
        mk_lp_np = mk_lp.cpu().numpy()
        mp_np = mk_pres_act.cpu().numpy()
        mq_np = mk_qty_act.cpu().numpy()
        ub_np = ub
        # 🩸 `turn_guard` 要完整的機率列來找次佳動作，`masked_sample` 只回傳被選
        # 中那一個。多算一次 log_softmax 比重收一批便宜得多。
        o_full_lp_np = masked_log_softmax(op_logits, op_mask).cpu().numpy()

        out = []
        for bi, (ei, p, o, c) in enumerate(items):
            sel = ub_np == bi
            board = len(o["farms"][p]["tiles"])
            pos = pos_list[bi]
            base = self.base_policy(o, c) if self.base_policy else None
            units = []
            # 🩸 `op_exec` 就是下面這個 if 的另一面：站到目標格上才會送 op，
            # 否則送 `step_toward`、那一步的 op 根本不進引擎。判定條件不能另外
            # 寫一份，只能從同一個分支取。
            op_exec = []
            # 取樣結果改寫成 logit 再交給 `decode_market_orders` —— 數量的
            # clamp、HIRE 要送 n 筆這些規則都在那裡面，重寫一份會走鐘。
            # 🩸 要先解碼才知道這回合買幾顆種子（引擎先處理市場訂單）。
            pres_logit = np.where(mp_np[bi], 1.0, -1.0)
            qty_onehot = np.zeros((C.N_MARKET_OPS, C.N_MARKET_QTY), np.float32)
            qty_onehot[np.arange(C.N_MARKET_OPS), mq_np[bi]] = 1.0
            market = C.decode_market_orders(pres_logit, qty_onehot, o, c)
            # 🩸 同回合衝突的重選，見 `contracts.turn_guard`。上場側
            # （`agents/ppo_agent.py`）走同一個函式，兩邊一定要一致。
            #
            # 被改掉的 unit 標成 `op_exec=False`：logprob 仍記**取樣到**的那個
            # op（guard 是動作投影，當成環境的一部分，跟 `step_toward` 同一個
            # 處理方式），但 `--op-exec-only` 開著時不進 joint logprob。
            avail, claimed = C.turn_guard_state(o, market)
            for k, (ti, oi) in enumerate(zip(t_np[sel], o_np[sel])):
                tx, ty = C.target_xy(int(ti), board)
                cur = tuple(pos[k])
                if (int(cur[0]), int(cur[1])) != (tx, ty):
                    units.append(step_toward(cur, (tx, ty)))
                    op_exec.append(False)
                else:
                    gi, changed = C.turn_guard(
                        int(oi), o_full_lp_np[sel][k], op_mask_np[sel][k],
                        avail, (tx, ty), claimed)
                    qi = int(q_np[sel][k]) if self.qty_factor else None
                    units.append(C.decode_unit(gi, qi))
                    op_exec.append(not changed)
            op_exec = np.asarray(op_exec, dtype=bool)
            # 🩸 骨幹選的動作，logprob **不能算進去** —— 那些不是網路選的，
            # 算了就是在對別人的選擇做 policy gradient，而且 update 重算時會
            # 加回來，ratio 就錯了。所以哪一邊是骨幹，那一邊的 logprob 歸 0。
            # `--op-exec-only`：沒執行到的 op 不進 joint logprob。target 照算。
            def _unit_logp():
                q = float((q_lp_np[sel] * q_use_np[sel]).sum())
                if not self.op_exec_only:
                    return float(lp_np[sel].sum()) + q
                return float(t_lp_np[sel].sum()
                             + o_lp_np[sel][op_exec].sum()) + q

            if base is None:
                action = {"farmer": units[0], "hands": units[1:],
                          "market": market}
                unit_logp, mk_logp = _unit_logp(), float(mk_lp_np[bi])
                keep, keep_market = sel, True
            elif self.base_side == "units":
                # 骨幹出工人，網路只出 market。
                action = dict(base)
                action["market"] = market
                unit_logp, mk_logp = 0.0, float(mk_lp_np[bi])
                keep, keep_market = np.zeros_like(sel), True
            else:
                # 反向：骨幹出 market，網路只出工人。
                action = {"farmer": units[0], "hands": units[1:],
                          "market": base["market"]}
                unit_logp, mk_logp = _unit_logp(), 0.0
                keep, keep_market = sel, False
            rec = {"logp": unit_logp + mk_logp,
                   "value": float(v_np[bi])}
            if collect:
                # 🩸 每個切片都要 `.copy()`：`sel` 是布林索引所以本來就是複本，
                # 但 `sp[bi]` 是 view，整批 `sp` 會被下一步蓋掉。
                # 混合模式存 0 個 unit —— update 重算時 `index_add` 就不會把
                # 骨幹選的動作算進 logprob（見上面 unit_logp 那段）。
                mine = keep[sel]              # 這個盤面要留哪幾個 unit
                rec.update(
                    spatial=(sp16 if sp16 is not None else sp)[bi].copy(),
                    scalar=sc[bi].copy(),
                    unit_pos=pos[mine], unit_feats=feat_list[bi][mine],
                    op_mask=op_mask_np[keep], tgt_mask=tgt_mask_np[keep],
                    op_idx=o_np[keep].astype(np.int64),
                    tgt_idx=t_np[keep].astype(np.int64),
                    # update 重算時要用**同一個** mask，不能在那邊重新判定。
                    op_exec=op_exec[mine],
                    qty_idx=q_np[keep].astype(np.int64),
                    qty_used=q_use_np[keep],
                    # 🩸 反向混合把 mk_legal 整列設成 False —— `market_logp_entropy`
                    # 的每一項都乘 `legal`，所以 market 對 logprob 和 entropy 的
                    # 貢獻剛好是 0，update 重算時自然不會把骨幹的訂單算進來。
                    mk_legal=(mk_legal_np[bi] if keep_market
                              else np.zeros_like(mk_legal_np[bi])),
                    mk_present=mp_np[bi],
                    mk_qty=mq_np[bi].astype(np.int64))
            out.append((ei, p, action, rec))
        return out

    def run(self, max_steps=720, collect=False):
        """跑到全部結束。

        回傳 `(走了幾步, 每個 env 的期末現金, 軌跡 list)`。
        `collect=False` 時軌跡是空的 —— 只量吞吐量時不留觀測。
        """
        self.reset()
        steps = 0
        # 每個由 net 控制的 (env, player) 一條軌跡。自對局時同一份權重下兩邊，
        # 所以兩條都能用來訓練，樣本數直接翻倍。
        writers = {(ei, p): TrajectoryWriter()
                   for ei in range(self.n_envs) for p in range(2)
                   if self._ours(ei, p)}
        # Φ(s_T)。`plants` 模式用最後一次觀測到的格子數 —— 期末的 farms
        # 拿不到（引擎結束後只剩 reward），這樣等於期末那一步不發 shaping。
        # `assets` 模式送 0：賣不掉的庫存期末一分不值（期末 reward 只看
        # `farm["money"]`），最後一步一次認列，那也正好滿足 Ng 的
        # Φ(終端)=0（§83.3）。
        last_phi = {}
        for _ in range(max_steps):
            active = self._gather()
            if not active:
                break
            items = [it for it in active if self._ours(it[0], it[1])]
            decided = self._policy_batch(items, collect=collect) if items else []
            acts = {ei: [{}, {}] for ei, _, _, _ in active}
            for ei, p, action, _rec in decided:
                acts[ei][p] = action
            for ei, p, obs, cfg in active:
                if not self._ours(ei, p):
                    acts[ei][p] = self.opp_of(ei)(obs, cfg)
            if collect:
                for (ei, p, obs, cfg), (_ei, _p, _a, rec) in zip(items, decided):
                    # 🩸 期中的 `steps[-1][p]["reward"]` 是 0 —— 引擎只在 DONE
                    # 那一步才填 reward（kaggriculture.py:963）。現金要從
                    # observation 的 farms 拿。
                    farms = obs["farms"]
                    phi = self._phi(obs, cfg, p)
                    last_phi[(ei, p)] = phi
                    writers[(ei, p)].add(
                        rec, float(farms[p]["money"]),
                        float(farms[1 - p]["money"]), phi)
            # 引擎自己不印東西，不用每一步都 dup2 —— `silenced()` 一次要 4 個
            # syscall，720 步 × K 個 env 加起來很可觀。
            for ei, pair in acts.items():
                self.envs[ei].step(pair)
            steps += len(decided)
        cash = [[s["reward"] for s in e.steps[-1]] for e in self.envs]
        trajs = []
        if collect:
            for (ei, p), w in writers.items():
                final = (0.0 if self.phi_mode == "assets"
                         else last_phi.get((ei, p)))
                tr = w.finish(cash[ei][p], cash[ei][1 - p], final)
                if tr is not None:
                    trajs.append(tr)
        return steps, cash, trajs

    def _phi(self, obs, cfg, p):
        """這一步的 Φ。`none` 回 None（shaping 整個關掉）。"""
        if self.phi_mode == "none":
            return None
        if self.phi_mode == "plants":
            farms = obs["farms"]
            return float(count_plants(farms[p]) - count_plants(farms[1 - p]))
        return float(asset_value(
            obs, recognise=self.recognise,
            episode_steps=int(cfg.get("episodeSteps", 720)),
            turns_per_day=int(cfg.get("turnsPerDay", 24))))

    def our_cash(self, cash):
        """把 `run()` 回傳的現金拆成 (我方, 對手) —— 打固定對手時才有意義。"""
        return [(c[self.seats[ei]], c[1 - self.seats[ei]])
                for ei, c in enumerate(cash)]


if __name__ == "__main__":
    raise SystemExit(main())
