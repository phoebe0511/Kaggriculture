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
用每一步的「我方現金增量 − 對手現金增量」當 dense reward（零和），
期末再加勝負 bonus：

    r_t = (my_cash_t − my_cash_{t−1}) − (opp_cash_t − opp_cash_{t−1})
    r_T += terminal_bonus * sign(my_cash_T − opp_cash_T)

現金的量級是 10^5，要除以 `REWARD_SCALE` 才不會讓 value loss 爆掉。

## 動作的 logprob

聯合動作是 13 個 unit 各自獨立取樣的乘積（見 `ppo_rollout` 的說明）：

    logprob(joint) = Σ_unit [ logprob(target_u) + logprob(op_u) ]

所以 ratio 是 `exp(new_logprob − old_logprob)`，跟單一離散動作一樣用。
⚠️ **unit 數量會變**（雇了幾個人），所以 logprob 是「這一步全部 unit 的和」，
不能事後按 unit 拆開重算。
"""
from __future__ import annotations

import argparse
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


def step_rewards(cash_a, cash_b):
    """逐步現金序列 -> 零和的 dense reward。`cash_*` 是 `[T+1]`。"""
    da = np.diff(np.asarray(cash_a, dtype=np.float64))
    db = np.diff(np.asarray(cash_b, dtype=np.float64))
    r = (da - db) / REWARD_SCALE
    if len(r):
        r[-1] += TERMINAL_BONUS * np.sign(cash_a[-1] - cash_b[-1])
    return r.astype(np.float32)


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
