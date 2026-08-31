"""PPO 的訓練那一半：GAE、clipped surrogate、value loss、entropy。

    python -m model.ppo --smoke          # 一輪小的，確認跑得動

rollout 在 `harness/ppo_rollout.py`。這裡只吃它吐出來的軌跡。

## 為什麼是 PPO 不是模仿學習

`model/train.py` 的 docstring 記著：監督式拿到 val op 準確率 0.9396
（隨機 0.1609）、逐類別召回率 0.98 以上，**實戰 0 勝 12 負**。

在這個遊戲裡「動作跟老師像」跟「打得贏」幾乎沒有關係 —— 一步走錯不會馬上被
懲罰，但 30 天後現金就是差一截。PPO 直接優化最終報酬，理論上處理得了這個
失敗模式。**但沒有證據，這是這條路最大的風險。**

## 報酬

🩸 **只用期末現金差是 720 步的稀疏訊號，credit assignment 會死。**
用每一步的現金增量當 dense reward，期末再加勝負 bonus：

    r_t = my_cash_t − my_cash_{t−1}
    r_T += terminal_bonus * sign(my_cash_T − opp_cash_T)

現金的量級是 10^5，要除以 `REWARD_SCALE` 才不會讓 value loss 爆掉。

🩸 **原本減掉對手的增量（零和），2026-08-28 量完改掉了** —— 那一項佔 reward
變異數的 86.4%，而且我們幾乎控制不了。細節在 `step_rewards` 的 docstring。

## 動作的 logprob

聯合動作是 13 個 unit 各自獨立取樣的乘積，再乘上 market（見 `ppo_rollout`）：

    logprob(joint) = Σ_unit [ logprob(target_u) + logprob(op_u) ]
                   + logprob(market)

所以 ratio 是 `exp(new_logprob − old_logprob)`，跟單一離散動作一樣用。
⚠️ **unit 數量會變**（雇了幾個人），所以 logprob 是「這一步全部 unit 的和」，
不能事後按 unit 拆開重算。
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._quiet import silenced                      # noqa: E402

with silenced():
    import torch
    import torch.nn.functional as F

#: 現金除以這個數字才進 value head。10^5 量級的報酬會讓 value loss 蓋過 policy。
REWARD_SCALE = 10_000.0

#: 期末勝負的額外報酬（已經是 scale 之後的單位）。
TERMINAL_BONUS = 1.0


def compute_gae(rewards, values, dones, gamma=0.997, lam=0.95):
    """GAE-λ。`rewards` / `values` / `dones` 都是 `[T]` 的 numpy 陣列。

    🩸 `gamma` 要接近 1：一局 720 步，`0.99^720 = 0.07%`，等於看不到期末。
    0.997 的半衰期約 230 步（差不多 10 天），是這個遊戲的合理尺度。

    `values` 要有 T+1 個元素（最後一個是 bootstrap）。
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values[:T]


def ppo_loss(new_logp, old_logp, adv, new_value, ret, entropy,
             clip=0.2, vf_coef=0.5, ent_coef=0.01):
    """一個 minibatch 的損失。回傳 `(total, 各項的 dict)`。

    🩸 advantage 要**在 minibatch 內**標準化，不是全批。全批標準化在
    「大部分步的 advantage 都接近 0、少數很大」的情況下會把訊號壓掉。
    """
    a = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * a
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * a
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(new_value, ret)
    ent = entropy.mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * ent
    return total, {
        "policy": float(policy_loss.item()),
        "value": float(value_loss.item()),
        "entropy": float(ent.item()),
        "clipfrac": float(((ratio - 1).abs() > clip).float().mean().item()),
        "approx_kl": float((old_logp - new_logp).mean().item()),
    }


def step_rewards(cash_a, cash_b, zero_sum=False):
    """逐步現金序列 -> dense reward。`cash_*` 是 `[T+1]`。

    🩸 **預設不是零和的。** 2026-08-28 實測（`ppo-warm1` 的 last.pt、6 局、
    4,314 步、對手 `cma1-g50-wt`）：

        我方現金增量   var    212,497
        對手現金增量   var  1,350,738     <- 6.4 倍
        相關           +0.574
        var(我方 − 對手) = 948,475        <- 比只用我方大 4.5 倍

    對手的收入我們幾乎控制不了（只有市場價格那一點間接影響）。把它減進 reward
    等於在 advantage 裡灌一個**佔 86.4% 變異數的不可控項** —— 相關 0.574 帶來的
    control variate 效果遠遠補不回來。reward std 從 0.0974 降到 0.0461。

    競爭性由期末的勝負 bonus 保留。`zero_sum=True` 是原本的形式，打自對局
    或 league 時可能還是要用（那時對手的行為是我們自己的 policy）。
    """
    da = np.diff(np.asarray(cash_a, dtype=np.float64))
    db = np.diff(np.asarray(cash_b, dtype=np.float64))
    r = (da - db) / REWARD_SCALE if zero_sum else da / REWARD_SCALE
    if len(r):
        r[-1] += TERMINAL_BONUS * np.sign(cash_a[-1] - cash_b[-1])
    return r.astype(np.float32)


# --------------------------------------------------------------------------
# 軌跡緩衝區
# --------------------------------------------------------------------------
#
# 一步的觀測是 spatial [38,10,10] + scalar [75]，float32 共 15.5 KB。
# 一局 720 步 × 2 個 player = 22 MB，16 個 env 一輪 350 MB。
# **spatial 佔 98%** —— 要省記憶體只有它值得動。
#
# unit 層是不定長的（雇了幾個人會變），所以攤平成 [U, ...] 再用 `unit_step`
# 指回第幾步 —— 跟 `net.forward` 的 `unit_board` 同一個形狀。


def masked_log_softmax(logits, mask):
    """非法動作壓到 -inf 之後取 log_softmax。

    🩸 mask 整列全 False 的 unit 是存在的（站在 shed 上、手上沒東西）。
    那一列退回「全部合法」—— 引擎對非法動作是靜默忽略，取樣到非法的等於 PASS。
    rollout 取樣和 update 重算 **一定要走這同一個函式**，否則第一次更新的
    ratio 就不是 1，clipfrac 會無緣無故很高。
    """
    m = mask.bool()
    m = m | ~m.any(dim=-1, keepdim=True)
    return torch.log_softmax(logits.masked_fill(~m, -1e9), dim=-1)


def market_logp_entropy(present_logits, qty_logits, legal_mask,
                        present_act, qty_act):
    """market head 的 logprob 與 entropy，逐盤面加總。回傳 `([B], [B])`。

    🩸 **market head 一定要進取樣分佈。** `contracts.decode_market_orders` 是
    門檻 + argmax，完全決定性 —— 那樣的話 HIRE / BUY_LAND / BUY_SEED / SELL
    全部拿不到 policy gradient，PPO 學不到雇工、買地、種什麼，剩下的只有
    「工人走去哪、到了做什麼」。

    形狀是階層式的：`present` 是每個 op 一個 Bernoulli，`qty` 只在
    `present=1` 時才是決策。

        log p = Σ_{j 合法} log Bern(a_j) + Σ_{j 合法且 a_j=1} log Cat(q_j)

    非法的 op 貢獻 0 —— 它不是決策，`legal_market_mask` 已經先擋掉了。
    entropy 的第二項用機率 `p_j` 加權（條件 entropy 的真值），不是用抽到的
    樣本，變異數小一些。
    """
    legal = legal_mask.bool().float()
    lp1 = F.logsigmoid(present_logits)             # log p
    lp0 = F.logsigmoid(-present_logits)            # log(1 − p)
    a = present_act.float()
    bern = (a * lp1 + (1.0 - a) * lp0) * legal
    p = torch.sigmoid(present_logits)
    bern_ent = -(p * lp1 + (1.0 - p) * lp0) * legal

    qlp = torch.log_softmax(qty_logits, dim=-1)    # [B, ops, buckets]
    q_sel = qlp.gather(-1, qty_act.unsqueeze(-1)).squeeze(-1)
    q_ent = -(qlp.exp() * qlp).sum(-1)
    taken = a * legal
    return ((bern + q_sel * taken).sum(-1),
            (bern_ent + q_ent * p * legal).sum(-1))


def sample_market(present_logits, qty_logits, legal_mask, greedy=False,
                  gen=None):
    """從 market head 取樣。回傳 `(present [B, ops] bool, qty [B, ops] int64)`。

    非法的 op 直接設成不出手 —— `contracts.decode_market_orders` 反正也會擋，
    但這裡先擋掉才不會讓 logprob 算到一個根本送不出去的決策。
    qty 每個 op 都取樣（不管 present），沒被選中的那些 logprob 不計。

    `greedy=True` 走機率大於一半（logit > 0）和 argmax —— 跟
    `agents/ppo_agent.py` 和 `agents/gen3_target.py` 上場時的解碼**必須一致**。

    🩸 `gen` 是取樣用的 `torch.Generator`。**不給的話走全域 RNG，整段 rollout
    就不可重現** —— 2026-08-31 踩到：同一個 checkpoint、同樣的 seed 跑兩次，
    第 0 輪的現金就差 2,839，第 10 輪差 31,138（10 局評估的 SE 才 ~7,000）。
    兩次單獨的訓練沒辦法拿來 A/B，因為 run-to-run 的差異比要量的效果還大。
    """
    if greedy:
        present = (present_logits > 0.0) & legal_mask.bool()
        return present, qty_logits.argmax(dim=-1)
    p = torch.sigmoid(present_logits)
    # `torch.rand_like` 收不到 generator，要寫成 `torch.rand`。
    u = torch.rand(p.shape, dtype=p.dtype, device=p.device, generator=gen)
    present = (u < p) & legal_mask.bool()
    probs = torch.softmax(qty_logits, dim=-1)
    b, o, k = probs.shape
    qty = torch.multinomial(probs.reshape(-1, k), 1,
                            generator=gen).reshape(b, o)
    return present, qty


@dataclasses.dataclass
class Trajectory:
    """一個 (env, player) 的一整局。步層是 [T]，unit 層是攤平的 [U]。

    `cash` 是 [T+1, 2]（我方, 對手）—— 前 T 個是決策當下的現金，最後一個是
    期末。reward 由 `step_rewards` 從它算出來，不另外存。
    """

    spatial: np.ndarray        # [T, N_SPATIAL, 10, 10] float16（見 half_obs）
    scalar: np.ndarray         # [T, N_SCALAR] float32
    value: np.ndarray          # [T] float32
    logp: np.ndarray           # [T] float32（unit 加 market，整步的和）
    cash: np.ndarray           # [T+1, 2] float64
    mk_legal: np.ndarray       # [T, N_MARKET_OPS] bool
    mk_present: np.ndarray     # [T, N_MARKET_OPS] bool
    mk_qty: np.ndarray         # [T, N_MARKET_OPS] int64
    unit_step: np.ndarray      # [U] int64，遞增
    unit_pos: np.ndarray       # [U, 2] int16
    unit_feats: np.ndarray     # [U, N_UNIT_FEATURES] float32
    op_mask: np.ndarray        # [U, N_UNIT_OPS] bool
    tgt_mask: np.ndarray       # [U, N_TARGET_CELLS] bool
    op_idx: np.ndarray         # [U] int64
    tgt_idx: np.ndarray        # [U] int64

    def __len__(self):
        return len(self.value)

    def nbytes(self):
        return sum(getattr(self, f.name).nbytes
                   for f in dataclasses.fields(self))


class TrajectoryWriter:
    """一步一步累積，最後 `finish()` 成 `Trajectory`。

    rollout 每一步只拿得到自己那一格的切片，所以先用 list 接著，收尾才
    concatenate —— 中途 `np.append` 是 O(n^2)。
    """

    def __init__(self):
        self.rows = []
        self.cash = []

    def add(self, rec, my_cash, opp_cash):
        self.rows.append(rec)
        self.cash.append((my_cash, opp_cash))

    def finish(self, final_my_cash, final_opp_cash):
        rows = self.rows
        if not rows:
            return None
        counts = [len(r["op_idx"]) for r in rows]
        return Trajectory(
            spatial=np.stack([r["spatial"] for r in rows]),
            scalar=np.stack([r["scalar"] for r in rows]),
            value=np.array([r["value"] for r in rows], dtype=np.float32),
            logp=np.array([r["logp"] for r in rows], dtype=np.float32),
            cash=np.array(self.cash + [(final_my_cash, final_opp_cash)],
                          dtype=np.float64),
            mk_legal=np.stack([r["mk_legal"] for r in rows]),
            mk_present=np.stack([r["mk_present"] for r in rows]),
            mk_qty=np.stack([r["mk_qty"] for r in rows]),
            unit_step=np.repeat(np.arange(len(rows), dtype=np.int64), counts),
            unit_pos=np.concatenate([r["unit_pos"] for r in rows]),
            unit_feats=np.concatenate([r["unit_feats"] for r in rows]),
            op_mask=np.concatenate([r["op_mask"] for r in rows]),
            tgt_mask=np.concatenate([r["tgt_mask"] for r in rows]),
            op_idx=np.concatenate([r["op_idx"] for r in rows]),
            tgt_idx=np.concatenate([r["tgt_idx"] for r in rows]),
        )


class RolloutBatch:
    """把多條軌跡攤平成一個可以抽 minibatch 的緩衝區。

    minibatch 抽的是**步**，不是 unit —— PPO 的 ratio 是整步聯合動作的，
    把同一步的 unit 拆到不同 minibatch 會算出錯的 logprob。
    """

    def __init__(self, trajs, gamma=0.997, lam=0.95, zero_sum=False):
        trajs = [t for t in trajs if t is not None and len(t)]
        if not trajs:
            raise ValueError("沒有軌跡")
        advs, rets, offs, base = [], [], [], 0
        for tr in trajs:
            T = len(tr)
            rew = step_rewards(tr.cash[:, 0], tr.cash[:, 1], zero_sum)
            dones = np.zeros(T, dtype=np.float32)
            dones[-1] = 1.0
            # 期末 bootstrap 是 0：這一局真的結束了，沒有後續價值。
            v = np.concatenate([tr.value, np.zeros(1, np.float32)])
            a, r = compute_gae(rew, v, dones, gamma, lam)
            advs.append(a)
            rets.append(r)
            offs.append(base)
            base += T

        self.n_steps = base
        self.spatial = np.concatenate([t.spatial for t in trajs])
        self.scalar = np.concatenate([t.scalar for t in trajs])
        self.old_logp = np.concatenate([t.logp for t in trajs])
        self.old_value = np.concatenate([t.value for t in trajs])
        self.adv = np.concatenate(advs)
        self.ret = np.concatenate(rets)
        self.mk_legal = np.concatenate([t.mk_legal for t in trajs])
        self.mk_present = np.concatenate([t.mk_present for t in trajs])
        self.mk_qty = np.concatenate([t.mk_qty for t in trajs])
        self.unit_step = np.concatenate(
            [t.unit_step + o for t, o in zip(trajs, offs)])
        self.unit_pos = np.concatenate([t.unit_pos for t in trajs])
        self.unit_feats = np.concatenate([t.unit_feats for t in trajs])
        self.op_mask = np.concatenate([t.op_mask for t in trajs])
        self.tgt_mask = np.concatenate([t.tgt_mask for t in trajs])
        self.op_idx = np.concatenate([t.op_idx for t in trajs])
        self.tgt_idx = np.concatenate([t.tgt_idx for t in trajs])

        # CSR：每一步的 unit 在攤平陣列裡是連續的一段。
        # 🩸 這只有在 unit_step 遞增時才成立。寫入端是照步序 append 的，但這裡
        # 驗一次 —— 錯了不會報錯，只會安靜地把 unit 配到別步去。
        if len(self.unit_step) and np.any(np.diff(self.unit_step) < 0):
            raise ValueError("unit_step 不是遞增的")
        self.unit_count = np.bincount(
            self.unit_step, minlength=self.n_steps).astype(np.int64)
        self.unit_start = np.concatenate(
            [np.zeros(1, np.int64), np.cumsum(self.unit_count)[:-1]])

    def __len__(self):
        return self.n_steps

    def explained_variance(self):
        """`1 − Var(ret − value) / Var(ret)`，用 rollout 當下的 value 算。

        這是判斷「advantage 到底有沒有內容」最直接的一個數字：

            接近 0   value head 不比「猜平均」好，GAE 出來的 advantage 幾乎是
                     報酬雜訊，policy gradient 在追隨機的東西
            接近 1   value 抓得住報酬，advantage 是真的訊號

        `ppo_loss` 那邊的 `value` 是損失，會隨報酬的量級變 —— 不能拿來判斷
        「學到了沒」。這個可以。
        """
        var = float(self.ret.var())
        if var <= 0:
            return float("nan")
        return 1.0 - float((self.ret - self.old_value).var()) / var

    def _pack(self, idx, device):
        counts = self.unit_count[idx]
        total = int(counts.sum())
        excl = np.concatenate([np.zeros(1, np.int64), np.cumsum(counts)[:-1]])
        # 第 j 步的 unit 在 [start_j, start_j + c_j)，攤平後落在 [excl_j, ...)，
        # 所以 u = (start_j − excl_j) + 攤平位置。
        u = np.repeat(self.unit_start[idx] - excl, counts) + np.arange(total)
        ub = np.repeat(np.arange(len(idx), dtype=np.int64), counts)

        def t(a):
            return torch.as_tensor(a, device=device)

        return {
            # rollout 存的是 float16（見 `VecRollout.half_obs`）；還原成
            # float32 之後跟 rollout 前向吃的位元完全一樣。
            "spatial": t(self.spatial[idx].astype(np.float32, copy=False)),
            "scalar": t(self.scalar[idx]),
            "unit_board": t(ub),
            "unit_pos": t(self.unit_pos[u]),
            "unit_feats": t(self.unit_feats[u]),
            "op_mask": t(self.op_mask[u]),
            "tgt_mask": t(self.tgt_mask[u]),
            "op_idx": t(self.op_idx[u]),
            "tgt_idx": t(self.tgt_idx[u]),
            "mk_legal": t(self.mk_legal[idx]),
            "mk_present": t(self.mk_present[idx]),
            "mk_qty": t(self.mk_qty[idx]),
            "old_logp": t(self.old_logp[idx]),
            "adv": t(self.adv[idx]),
            "ret": t(self.ret[idx]),
        }

    def minibatches(self, size, rng, device="cpu"):
        order = rng.permutation(self.n_steps)
        for i in range(0, self.n_steps, size):
            # 排序只是讓 unit 的 gather 走連續記憶體，對結果沒有影響。
            yield self._pack(np.sort(order[i:i + size]), device)


def evaluate_actions(net, mb):
    """重算 minibatch 裡每一步的 (logprob, entropy, value)。

    unit 的 logprob 用 `index_add_` 加回它所屬的那一步 —— 聯合動作是各 unit
    獨立取樣的乘積，取 log 就是相加；market 那一份再加上去。
    """
    op_logits, _qty, tgt_logits, mk_present, mk_qty, value, _d = net(
        mb["spatial"], mb["scalar"],
        mb["unit_board"], mb["unit_pos"], mb["unit_feats"])
    o_lp = masked_log_softmax(op_logits, mb["op_mask"])
    t_lp = masked_log_softmax(tgt_logits, mb["tgt_mask"])
    lp = (o_lp.gather(-1, mb["op_idx"].unsqueeze(-1)).squeeze(-1)
          + t_lp.gather(-1, mb["tgt_idx"].unsqueeze(-1)).squeeze(-1))
    ent = -((o_lp.exp() * o_lp).sum(-1) + (t_lp.exp() * t_lp).sum(-1))
    n = mb["spatial"].shape[0]
    ub = mb["unit_board"]
    step_lp = torch.zeros(n, device=lp.device, dtype=lp.dtype)
    step_ent = torch.zeros(n, device=ent.device, dtype=ent.dtype)
    mk_lp, mk_ent = market_logp_entropy(
        mk_present, mk_qty, mb["mk_legal"], mb["mk_present"], mb["mk_qty"])
    return (step_lp.index_add(0, ub, lp) + mk_lp,
            step_ent.index_add(0, ub, ent) + mk_ent,
            value)


def update(net, opt, batch, epochs=4, minibatch=512, seed=0, device="cpu",
           clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
           target_kl=0.0):
    """對一批軌跡跑幾輪 PPO 更新。回傳各項損失的平均。

    🩸 **這裡的 KL 是聯合動作的，不是單一 head 的。** 一步有 ~5 個 unit ×
    2 個 head、加 21 個 market Bernoulli 和它們的數量，所以每個成分只動一點點
    也會累成很大的聯合 KL。實測（2026-08-28，隨機初始、epochs=4、lr=3e-4）
    第一輪 approx_kl 0.35、clipfrac 0.79 —— 大部分樣本被 clip 掉，等於沒學。
    `target_kl > 0` 就在超過 1.5 倍時提早收工（`epochs_done` 記實際跑了幾輪）。
    """
    rng = np.random.default_rng(seed)
    net.train()
    acc, n, done_epochs = {}, 0, 0
    for _ in range(epochs):
        done_epochs += 1
        ep_kl, ep_n = 0.0, 0
        for mb in batch.minibatches(minibatch, rng, device):
            new_logp, ent, value = evaluate_actions(net, mb)
            loss, parts = ppo_loss(new_logp, mb["old_logp"], mb["adv"],
                                   value, mb["ret"], ent,
                                   clip=clip, vf_coef=vf_coef,
                                   ent_coef=ent_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            parts["loss"] = float(loss.item())
            parts["grad_norm"] = float(gn)
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1
            ep_kl += parts["approx_kl"]
            ep_n += 1
        # 🩸 2026-08-31：這裡本來是 `acc["approx_kl"] / n`，也就是**所有 epoch
        # 累積起來**的平均。第 0 個 epoch 的 KL 幾乎是 0（實測 0.00136），把
        # 平均一路拉低 —— 8 個 epoch 跑完累積平均只到 0.0272，永遠碰不到
        # `1.5 * 0.02 = 0.03`，而第 7 個 epoch 的實際 KL 已經是 0.0487。
        #
        # 後果：`--target-kl` 在 `ppo-hybrid` / `ppo-cma1-market` 兩次跑裡
        # **一次都沒觸發**（300/300 輪都跑滿 8 個 epoch），信賴區域形同不存在。
        # 反而是從零那次（KL 大到累積平均也超標）一直在觸發。
        if target_kl and ep_kl / max(ep_n, 1) > 1.5 * target_kl:
            break
    net.eval()
    out = {k: v / max(n, 1) for k, v in acc.items()}
    out["epochs_done"] = float(done_epochs)
    # 最後一個 epoch 的 KL —— `approx_kl` 是所有 epoch 的平均，會被前面那些
    # 接近 0 的值拉低，看不出信賴區域有沒有被逼近。提早停止看的是這個。
    out["kl_last_epoch"] = ep_kl / max(ep_n, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="只檢查數學，不跑對局")
    args = ap.parse_args(argv)
    if args.smoke:
        rng = np.random.default_rng(0)
        T = 720
        rewards = rng.normal(0, 0.1, T).astype(np.float32)
        values = rng.normal(0, 0.1, T + 1).astype(np.float32)
        dones = np.zeros(T, dtype=np.float32); dones[-1] = 1.0
        adv, ret = compute_gae(rewards, values, dones)
        print(f"  GAE     adv mean {adv.mean():+.4f}  std {adv.std():.4f}  "
              f"ret mean {ret.mean():+.4f}")
        old = torch.randn(64)
        new = old + torch.randn(64) * 0.05
        loss, parts = ppo_loss(new, old, torch.randn(64),
                               torch.randn(64), torch.randn(64), torch.rand(64))
        print(f"  loss    {loss.item():+.4f}   " +
              "  ".join(f"{k} {v:+.4f}" for k, v in parts.items()))
        r = step_rewards(np.linspace(3000, 90000, 721),
                         np.linspace(3000, 110000, 721))
        print(f"  reward  {len(r)} 步   mean {r.mean():+.5f}   "
              f"最後一步 {r[-1]:+.4f}（含勝負 bonus）")
        return 0
    print("還沒接 rollout —— 先跑 --smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
