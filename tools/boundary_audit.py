"""argmax 邊界距離：目前的移動量相對於「翻過去所需的質量」還差多少。

    python -m tools.boundary_audit --games 16

沿用 §102~§104 的同一批（driver = ckpt-140、seed0 960,000、16 局，決定性重放）。
**不改 PPO、不跑新的實驗。**

定義（只在 old top-1 沒被翻掉的決策上算）：

    required_transfer = (P1_old - P2_old) / 2
    actual_transfer   = (P1_old - P1_new + P2_new - P2_old) / 2
    movement_ratio    = actual / required          （required > 0 才算）

🩸 `actual_transfer` 是**帶號**的：負值代表往反方向走（把 top-1 推得更遠），
不要取絕對值 —— 那是完全不同的診斷。

⚠️ 不做「還要幾輪」的線性外推（§100 做過，那是未驗證的假設）。
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

from model import ppo                                   # noqa: E402
from harness.ppo_rollout import VecRollout, load_league  # noqa: E402
from tools.factor_audit import greedy_conv, load_ckpt    # noqa: E402

EDGES = [0.05, 0.2, 0.5, 0.9]
#: 「幾乎沒動」的門檻。跟 §100 用的一樣（float32 的機率差在這個量級以下沒有意義）。
ZERO = 1e-4


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def buckets(gap):
    idx = np.digitize(gap, EDGES)
    out = []
    for b in range(len(EDGES) + 1):
        lo = "0" if b == 0 else f"{EDGES[b - 1]:g}"
        hi = "inf" if b == len(EDGES) else f"{EDGES[b]:g}"
        out.append((f"{lo}-{hi}", idx == b))
    return out


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", default="model/artifacts/phi-200/ckpt-00140.pt")
    ap.add_argument("--b", default="model/artifacts/phi-200/ckpt-00200.pt")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=960_000)
    ap.add_argument("--opponent", default="config/params/cma5-g175.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = args.device if torch.cuda.is_available() else "cpu"

    net_a = load_ckpt(args.a, dev)
    net_b = load_ckpt(args.b, dev)
    opp, _ = load_league(args.opponent)
    vec = VecRollout(net_a, n_envs=args.games, seed0=args.seed0, device=dev,
                     opponent=opp, phi="assets", recognise="strict")
    _s, _c, trajs = vec.run(collect=True)
    batch = ppo.RolloutBatch(trajs, gamma=1.0, lam=0.998, zero_sum=True,
                             phi_weight=1e-4)
    del trajs
    print(f"  {batch.n_steps:,} 步、{len(batch.unit_step):,} 個 unit decision、"
          f"{len(batch.traj_start)} 條軌跡（要跟 11,504 / 108,040 / 16 對得上）")
    g = greedy_conv(net_a, net_b, batch, dev)
    adv = batch.adv[batch.unit_step]

    store = {}
    for head, name, act in (("op", "op head", batch.op_idx),
                            ("tgt", "target head", batch.tgt_idx)):
        p1a, p2a = g[f"{head}_p1a"], g[f"{head}_p2a"]
        p1b, p2b = g[f"{head}_p1b"], g[f"{head}_p2b"]
        gap = p1a - p2a
        flip = g[f"{head}_n1b"] != g[f"{head}_t1a"]
        req = gap / 2.0
        act_t = (p1a - p1b + p2b - p2a) / 2.0
        store[head] = dict(p1a=p1a, p2a=p2a, gap=gap, flip=flip, req=req,
                           act=act_t, was_top1=act == g[f"{head}_t1a"],
                           dp1=p1b - p1a, dp2=p2b - p2a)

        hr(f"1-2  {name}：翻轉所需距離 vs 實際移動（只算沒被翻掉的決策）")
        keep = ~flip
        print(f"  {'gap 桶':<10}{'n(未翻)':>9}{'gap 平均':>9}{'gap 中位':>9}"
              f"{'required 平均':>13}{'中位':>8}{'actual 平均':>12}{'中位':>10}"
              f"{'ratio 平均':>11}{'中位':>9}{'P(act>0)':>10}{'flip率':>9}")
        for lab, m in buckets(gap):
            k = m & keep
            if not k.any():
                continue
            r = act_t[k] / np.where(req[k] > 0, req[k], np.nan)
            r = r[np.isfinite(r)]
            print(f"  {lab:<10}{int(k.sum()):>9,}{gap[k].mean():>9.4f}"
                  f"{med(gap[k]):>9.4f}{req[k].mean():>13.4f}{med(req[k]):>8.4f}"
                  f"{act_t[k].mean():>+12.5f}{med(act_t[k]):>+10.5f}"
                  f"{(r.mean() if len(r) else float('nan')):>11.4f}"
                  f"{med(r):>9.4f}{100 * (act_t[k] > 0).mean():>9.2f}%"
                  f"{100 * flip[m].mean():>8.2f}%")

        hr(f"5  {name}：往邊界移 vs 反方向移（沒被翻掉的決策）")
        print(f"  {'gap 桶':<10}{'n':>9}{'actual>0':>10}{'|actual|<1e-4':>15}"
              f"{'actual<0':>10}{'正的平均':>11}{'負的平均':>11}")
        for lab, m in buckets(gap):
            k = m & keep
            if not k.any():
                continue
            a = act_t[k]
            pos, zero, neg = a > ZERO, np.abs(a) <= ZERO, a < -ZERO
            print(f"  {lab:<10}{int(k.sum()):>9,}{100 * pos.mean():>9.2f}%"
                  f"{100 * zero.mean():>14.2f}%{100 * neg.mean():>9.2f}%"
                  f"{(a[pos].mean() if pos.any() else 0):>+11.5f}"
                  f"{(a[neg].mean() if neg.any() else 0):>+11.5f}")

    # ---- 3  真正可能擋住 greedy 的那一群 ----
    for head, name in (("op", "op head"), ("tgt", "target head")):
        d = store[head]
        sel = (adv < 0) & d["was_top1"] & ~d["flip"]
        hr(f"3  {name}：adv<0 且取樣到的就是舊 argmax 且沒翻掉（n={int(sel.sum()):,}）")
        print(f"  佔全部決策 {100 * sel.mean():.2f}%")
        print(f"  {'gap 桶':<10}{'n':>9}{'old P1':>9}{'old gap':>9}"
              f"{'required':>10}{'actual':>11}{'ratio 中位':>11}"
              f"{'ΔP1':>10}{'ΔP2':>10}{'P(act>0)':>10}")
        for lab, m in buckets(d["gap"]):
            k = m & sel
            if not k.any():
                continue
            r = d["act"][k] / np.where(d["req"][k] > 0, d["req"][k], np.nan)
            r = r[np.isfinite(r)]
            print(f"  {lab:<10}{int(k.sum()):>9,}{d['p1a'][k].mean():>9.4f}"
                  f"{d['gap'][k].mean():>9.4f}{d['req'][k].mean():>10.4f}"
                  f"{d['act'][k].mean():>+11.5f}{med(r):>11.4f}"
                  f"{d['dp1'][k].mean():>+10.5f}{d['dp2'][k].mean():>+10.5f}"
                  f"{100 * (d['act'][k] > 0).mean():>9.2f}%")
        # 粗桶：<0.5 / 0.5-0.9 / >0.9
        print(f"\n  粗桶")
        for lab, m in (("gap < 0.5", d["gap"] < 0.5),
                       ("0.5-0.9", (d["gap"] >= 0.5) & (d["gap"] < 0.9)),
                       ("gap > 0.9", d["gap"] >= 0.9)):
            k = m & sel
            if not k.any():
                continue
            r = d["act"][k] / np.where(d["req"][k] > 0, d["req"][k], np.nan)
            r = r[np.isfinite(r)]
            print(f"    {lab:<12}n={int(k.sum()):>7,}   required "
                  f"{d['req'][k].mean():.4f}   actual "
                  f"{d['act'][k].mean():+.5f}   ratio 中位 {med(r):+.4f}"
                  f"   往邊界的比例 {100 * (d['act'][k] > 0).mean():.2f}%")
        # 這一群的 flip rate（含被翻掉的才算得出來）
        sel2 = (adv < 0) & d["was_top1"]
        print(f"    對照：同條件但含被翻掉的 n={int(sel2.sum()):,}，"
              f"flip rate {100 * d['flip'][sel2].mean():.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
