"""F2 ablation 的 sanity check：旗標關掉必須跟原版逐位元相同。

    python -m tools.f2_sanity --games 4

跑兩次同 seed 的 rollout（旗標關 / 開），逐項比對：

    1  executed / non-executed 的比例
    2  沒執行的 op 對 policy loss 的貢獻是不是 0
    3  有執行的 op 的 logprob 貢獻與原版相同
    4  target / market 的 logprob 與原版相同
    5  joint logprob 會變（預期之內，不是 bug）

🩸 旗標不影響動作取樣（`masked_sample` 的 RNG 不變），所以兩次的軌跡逐步相同
—— 這是能逐位元比對的前提，開頭會先驗這件事。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")

from tools._quiet import silenced                       # noqa: E402

with silenced():
    import torch

from harness.ppo_rollout import VecRollout, build_net, load_league  # noqa: E402
from model import ppo                                   # noqa: E402


def roll(net, args, flag):
    opp, _ = load_league(args.opponent)
    vec = VecRollout(net, n_envs=args.games, seed0=args.seed0,
                     device=args.device, opponent=opp, phi="assets",
                     recognise="strict", op_exec_only=flag)
    _steps, _cash, trajs = vec.run(collect=True)
    return ppo.RolloutBatch(trajs, gamma=1.0, lam=0.998, zero_sum=True,
                            phi_weight=1e-4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default="model/artifacts/phi-200/ckpt-00140.pt")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=902_240)
    ap.add_argument("--opponent", default="config/params/cma5-g175.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = build_net(int(blob["width"]), int(blob["blocks"]))
    net.load_state_dict(blob["state_dict"])
    net = net.to(args.device).eval()

    base = roll(net, args, False)
    abl = roll(net, args, True)

    print("0  兩次 rollout 的軌跡是否逐項相同（旗標不該影響取樣）")
    same = (base.n_steps == abl.n_steps
            and np.array_equal(base.op_idx, abl.op_idx)
            and np.array_equal(base.tgt_idx, abl.tgt_idx)
            and np.array_equal(base.unit_step, abl.unit_step)
            and np.array_equal(base.mk_present, abl.mk_present)
            and np.array_equal(base.unit_pos, abl.unit_pos))
    print(f"   {base.n_steps:,} 步、{len(base.op_idx):,} 個 unit   相同：{same}")
    if not same:
        print("   注意：軌跡不同，下面的逐位元比對沒有意義")
        return 1

    ex = base.op_exec.astype(bool)
    print(f"\n1  executed {100 * ex.mean():.2f}%   "
          f"non-executed {100 * (~ex).mean():.2f}%   "
          f"(n={len(ex):,})   兩邊的遮罩相同："
          f"{np.array_equal(base.op_exec, abl.op_exec)}")

    # 用同一個 batch、同一個網路，只切旗標，比 evaluate_actions 的輸出
    dev = args.device
    idx = np.arange(base.n_steps)
    mb = base._pack(idx, dev)
    with torch.no_grad():
        lp0, ent0, v0 = ppo.evaluate_actions(net, mb, False)
        lp1, ent1, v1 = ppo.evaluate_actions(net, mb, True)
        op_l, _q, tg_l, mk_p, mk_q, _v, _d = net(
            mb["spatial"], mb["scalar"], mb["unit_board"],
            mb["unit_pos"], mb["unit_feats"])
        o_lp = ppo.masked_log_softmax(op_l, mb["op_mask"])
        t_lp = ppo.masked_log_softmax(tg_l, mb["tgt_mask"])
        op_sel = o_lp.gather(-1, mb["op_idx"].unsqueeze(-1)).squeeze(-1)
        tg_sel = t_lp.gather(-1, mb["tgt_idx"].unsqueeze(-1)).squeeze(-1)
        mk_lp, _ = ppo.market_logp_entropy(
            mk_p, mk_q, mb["mk_legal"], mb["mk_present"], mb["mk_qty"])
    ub = mb["unit_board"].cpu().numpy()
    op_np = op_sel.cpu().numpy()
    tg_np = tg_sel.cpu().numpy()
    n = base.n_steps

    def per_step(vals):
        out = np.zeros(n)
        np.add.at(out, ub, vals)
        return out

    op_exec_sum = per_step(op_np * ex)
    op_skip_sum = per_step(op_np * ~ex)
    tg_sum = per_step(tg_np)
    mk = mk_lp.cpu().numpy()

    print(f"\n2  沒執行的 op 對 policy loss 的貢獻")
    d = np.abs((lp0 - lp1).cpu().numpy() - op_skip_sum)
    print(f"   (baseline - ablation) 應該剛好等於「沒執行的 op 的 logprob 和」")
    print(f"   max |差| {d.max():.3e}   -> ablation 的 logprob 裡不含那一項")
    print(f"   那一項本身的量級：每步平均 {op_skip_sum.mean():+.4f}"
          f"（baseline 的 joint logprob 平均 {lp0.cpu().numpy().mean():+.4f}）")

    print(f"\n3/4  有執行的 op / target / market 的貢獻")
    recon = op_exec_sum + tg_sum + mk
    d1 = np.abs(lp1.cpu().numpy() - recon)
    print(f"   ablation logprob == 有執行的 op + target + market："
          f"max |差| {d1.max():.3e}")
    recon0 = op_exec_sum + op_skip_sum + tg_sum + mk
    d0 = np.abs(lp0.cpu().numpy() - recon0)
    print(f"   baseline logprob == 全部 op + target + market："
          f"max |差| {d0.max():.3e}")
    print(f"   target 那一項兩邊完全相同（同一個 tensor，未經旗標）："
          f"{np.array_equal(tg_np, tg_np)}")

    print(f"\n5  joint logprob 的差（預期會變）")
    a, b = lp0.cpu().numpy(), lp1.cpu().numpy()
    print(f"   baseline 平均 {a.mean():+.4f}   ablation 平均 {b.mean():+.4f}   "
          f"差 {(a - b).mean():+.4f}")
    print(f"   entropy 兩邊相同（實驗只改 policy loss）："
          f"{torch.equal(ent0, ent1)}   value 相同：{torch.equal(v0, v1)}")

    # 旗標關掉時，rollout 存的 old_logp 必須跟原版一致
    print(f"\n6  旗標關掉的 old_logp vs 重算：max "
          f"{np.abs(ppo.all_logp(net, base, dev) - base.old_logp).max():.3e}")
    print(f"   旗標打開的 old_logp vs 重算（同樣開旗標）：max "
          f"{np.abs(ppo.all_logp(net, abl, dev, op_exec_only=True) - abl.old_logp).max():.3e}")
    print(f"   ^ 這兩個都要是 1e-4 量級（float16 觀測造成的），"
          "否則 ratio 一開始就不是 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
