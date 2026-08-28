"""PPO 的訓練迴圈：rollout -> GAE -> 幾輪 minibatch 更新 -> 存檔。

    python -m model.ppo_train --smoke                 # 兩輪小的，確認接得起來
    python -m model.ppo_train --iters 200 --envs 16 --out model/artifacts/ppo0

rollout 在 `harness/ppo_rollout.py`，損失在 `model/ppo.py`。這支只負責把兩邊
串起來、記數字、存 checkpoint。

## 自對局的兩邊都收

同一份權重下兩邊，所以一個 env 產兩條軌跡，樣本數直接翻倍。reward 是零和的
（`model.ppo.step_rewards`），兩條加起來期望值是 0 —— **所以「平均 reward」
不是進步的指標**，看的是期末現金的平均值（兩邊一起變有錢＝真的在學會種田）
和對 gen0 的勝率。

## 一輪要多久（2026-08-28 實測，RTX 4060 / 20 實體核心）

    16 個 env    3.20 秒/局    -> 一輪 rollout 約 51 秒、23,000 步
    update       epochs=4, minibatch=512

## 🩸 起始權重

`model/weights-e2e-round*.npz` 是監督式學出來的，**value head 預測的是正規化
過的期末現金，不是 PPO 的 return**（零和、除以 `REWARD_SCALE`、有勝負
bonus）。從那裡熱啟動的話 value loss 一開始會很大，前幾輪的 advantage 幾乎
是雜訊。要嘛只載 trunk / policy head，要嘛就接受前幾輪是在重訓 value。
"""
from __future__ import annotations

import argparse
import json
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
    from harness.ppo_rollout import VecRollout, build_net, load_opponent
    from model import ppo


def save_checkpoint(path, net, args, meta):
    """跟 `model/train.py` 同一個形狀 —— 那邊的載入程式碼才吃得下。"""
    torch.save({
        "encoder_version": C.ENCODER_VERSION,
        "state_dict": net.state_dict(),
        "width": args.width,
        "blocks": args.blocks,
        "labels": "target",
        "trainer": "ppo",
        "argv": list(sys.argv[1:]),
        **meta,
    }, path)


def load_init(net, path):
    """熱啟動。只吃形狀對得上的張量，對不上的留隨機初始化並回報。"""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    src = blob["state_dict"] if "state_dict" in blob else blob
    own = net.state_dict()
    taken, skipped = [], []
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            taken.append(k)
        else:
            skipped.append(k)
    net.load_state_dict(own)
    return taken, skipped


def train(args):
    torch.manual_seed(args.seed)
    net = build_net(args.width, args.blocks)
    if args.init:
        taken, skipped = load_init(net, args.init)
        print(f"  熱啟動 {args.init}：吃了 {len(taken)} 個張量，"
              f"跳過 {len(skipped)} 個")
    net = net.to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.jsonl"

    opp = load_opponent(args.opponent) if args.opponent else None
    if opp is None:
        print("  ⚠️ 自對局：隨機初始的兩邊會一起把 $3000 花光，全程 0 對 0，"
              "reward 幾乎沒有訊號（2026-08-28 實測非零步只有 1.9%）。"
              "先用 --opponent 打 gen0。")

    for it in range(args.iters):
        t0 = time.perf_counter()
        # 每一輪換一批 seed —— 固定 16 張地圖會學成背地圖。
        vec = VecRollout(net, n_envs=args.envs,
                         seed0=args.seed0 + it * args.envs,
                         device=args.device,
                         episode_steps=args.episode_steps,
                         opponent=opp)
        steps, cash, trajs = vec.run(collect=True)
        t_roll = time.perf_counter() - t0

        batch = ppo.RolloutBatch(trajs, gamma=args.gamma, lam=args.lam)
        t1 = time.perf_counter()
        parts = ppo.update(net, opt, batch, epochs=args.epochs,
                           minibatch=args.minibatch, seed=args.seed + it,
                           device=args.device, clip=args.clip,
                           vf_coef=args.vf_coef, ent_coef=args.ent_coef,
                           max_grad_norm=args.max_grad_norm,
                           target_kl=args.target_kl)
        t_upd = time.perf_counter() - t1

        pairs = vec.our_cash(cash)
        ours = np.array([a for a, _ in pairs], dtype=np.float64)
        theirs = np.array([b for _, b in pairs], dtype=np.float64)
        units = len(batch.unit_step) / max(len(batch), 1)
        row = {
            "iter": it,
            "steps": steps,
            "cash_mean": float(ours.mean()),
            "cash_max": float(ours.max()),
            "opp_cash_mean": float(theirs.mean()),
            "win_rate": float(np.mean(ours > theirs)),
            "units_per_step": float(units),
            "adv_std": float(batch.adv.std()),
            "ret_mean": float(batch.ret.mean()),
            "roll_s": round(t_roll, 1),
            "upd_s": round(t_upd, 1),
            **{k: round(v, 5) for k, v in parts.items()},
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"  it {it:>4}  現金 {row['cash_mean']:>9,.0f} vs "
              f"{row['opp_cash_mean']:>9,.0f}  勝率 {row['win_rate']:.2f}  "
              f"unit/步 {units:>5.2f}  "
              f"kl {parts['approx_kl']:+.4f}  clip {parts['clipfrac']:.3f}  "
              f"ep {int(parts['epochs_done'])}  "
              f"ent {parts['entropy']:>7.2f}  v {parts['value']:.3f}  "
              f"rollout {t_roll:>5.1f}s  update {t_upd:>5.1f}s", flush=True)

        if args.save_every and (it + 1) % args.save_every == 0:
            save_checkpoint(out / f"ckpt-{it + 1:05d}.pt", net, args, row)
        save_checkpoint(out / "last.pt", net, args, row)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--episode-steps", type=int, default=0,
                    help="0 = 用引擎預設的 720（一整季 30 天）")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.0,
                    help="聯合動作的 approx_kl 超過 1.5 倍就停掉這一輪的 "
                         "epoch。0 = 不管")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=100_000,
                    help="env 的 seed 起點。跟評估用的 seed 錯開")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--init", default="", help="熱啟動用的 .pt")
    ap.add_argument("--opponent", default="config/params/cma1-g50-wt.json",
                    help="固定對手的 spec。空字串 = 自對局（隨機初始時沒訊號）")
    ap.add_argument("--out", default="model/artifacts/ppo0")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="兩輪、4 個 env、短局 —— 只確認接得起來")
    args = ap.parse_args(argv)
    if args.smoke:
        args.iters, args.envs, args.episode_steps = 2, 4, 120
        args.width, args.blocks, args.minibatch = 32, 2, 128
        args.out = args.out + "-smoke"
        args.save_every = 0
    args.episode_steps = args.episode_steps or None
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
