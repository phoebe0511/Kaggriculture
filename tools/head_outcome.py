"""三個 head 的 policy movement 有沒有跟最終 zero-sum 差額對上。

    python -m tools.head_outcome --games 16

用 §102~§104 的同一批 rollout（driver = ckpt-140、seed0 960,000、16 局）。
🩸 那批當時沒存檔，但 rollout 是決定性的（env seed + `torch.Generator` 都固定），
同參數重放得到逐位元相同的一批 —— 開頭會印步數／unit 數讓人對照。

**不改 PPO、不跑新的實驗。** market 的資料結構跟 unit 不同（一步一組 21 個
Bernoulli + qty），所以分開報，不硬湊成 unit-level。

⚠️ advantage 是這次 rollout 現場算的，ckpt-200 是用另外 60 批訓的 —— 那部分
只能回答「學到的東西 generalize 到新狀態時看不看得出方向」。
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
from tools.factor_audit import load_ckpt                 # noqa: E402


def hr(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return corr(np.argsort(np.argsort(a)).astype(float),
                np.argsort(np.argsort(b)).astype(float))


def partial(a, b, z):
    """corr(a, b | z)：把 z 從兩邊迴歸掉再算相關。"""
    if len(a) < 4:
        return float("nan")
    za = a - np.polyval(np.polyfit(z, a, 1), z)
    zb = b - np.polyval(np.polyfit(z, b, 1), z)
    return corr(za, zb)


def heads(net_a, net_b, batch, device, chunk=256):
    """兩個網路在同一批上的逐因子量。unit 層兩份、market 層一份（逐步）。"""
    keys = ("op_dp_s", "op_dlp_s", "op_dp_t1", "op_dp_t2", "op_kl", "op_gap",
            "tgt_dp_s", "tgt_dlp_s", "tgt_dp_t1", "tgt_dp_t2", "tgt_kl",
            "tgt_gap")
    out = {k: [] for k in keys}
    out["mk_dlp"] = []
    out["mk_kl"] = []
    with torch.no_grad():
        for i in range(0, batch.n_steps, chunk):
            idx = np.arange(i, min(i + chunk, batch.n_steps))
            mb = batch._pack(idx, device)
            packs = []
            for net in (net_a, net_b):
                op_l, _q, tg_l, mp, mq, _v, _d = net(
                    mb["spatial"], mb["scalar"], mb["unit_board"],
                    mb["unit_pos"], mb["unit_feats"])
                packs.append((ppo.masked_log_softmax(op_l, mb["op_mask"]),
                              ppo.masked_log_softmax(tg_l, mb["tgt_mask"]),
                              mp, mq))
            (oa, ta, mpa, mqa), (ob, tb, mpb, mqb) = packs
            for pre, la, lb, act in (("op", oa, ob, mb["op_idx"]),
                                     ("tgt", ta, tb, mb["tgt_idx"])):
                top = la.topk(2, dim=-1)
                i1, i2 = top.indices[:, :1], top.indices[:, 1:2]
                sa = la.gather(-1, act.unsqueeze(-1)).squeeze(-1)
                sb = lb.gather(-1, act.unsqueeze(-1)).squeeze(-1)
                out[f"{pre}_dp_s"].append(
                    (sb.exp() - sa.exp()).cpu().numpy())
                out[f"{pre}_dlp_s"].append((sb - sa).cpu().numpy())
                out[f"{pre}_dp_t1"].append(
                    (lb.gather(-1, i1).squeeze(-1).exp()
                     - top.values[:, 0].exp()).cpu().numpy())
                out[f"{pre}_dp_t2"].append(
                    (lb.gather(-1, i2).squeeze(-1).exp()
                     - top.values[:, 1].exp()).cpu().numpy())
                out[f"{pre}_kl"].append(
                    (la.exp() * (la - lb)).sum(-1).cpu().numpy())
                out[f"{pre}_gap"].append(
                    (top.values[:, 0].exp()
                     - top.values[:, 1].exp()).cpu().numpy())
            # market：一步一組，取樣動作的 logprob 差 + 逐步 KL
            lp_a, _ = ppo.market_logp_entropy(
                mpa, mqa, mb["mk_legal"], mb["mk_present"], mb["mk_qty"])
            lp_b, _ = ppo.market_logp_entropy(
                mpb, mqb, mb["mk_legal"], mb["mk_present"], mb["mk_qty"])
            out["mk_dlp"].append((lp_b - lp_a).cpu().numpy())
            legal = mb["mk_legal"].bool().float()
            pa = torch.sigmoid(mpa)
            l1a = torch.nn.functional.logsigmoid(mpa)
            l0a = torch.nn.functional.logsigmoid(-mpa)
            l1b = torch.nn.functional.logsigmoid(mpb)
            l0b = torch.nn.functional.logsigmoid(-mpb)
            kb = pa * (l1a - l1b) + (1 - pa) * (l0a - l0b)
            qa = torch.log_softmax(mqa, dim=-1)
            qb = torch.log_softmax(mqb, dim=-1)
            kc = (qa.exp() * (qa - qb)).sum(-1)
            out["mk_kl"].append(((kb + pa * kc) * legal).sum(-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


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
    # 🩸 cash 的逐步軌跡只在 Trajectory 上，RolloutBatch 只留期末 —— 第 6 節的
    # control 要用中局差額，所以先留一份。
    cash = [t.cash.copy() for t in trajs]
    batch = ppo.RolloutBatch(trajs, gamma=1.0, lam=0.998, zero_sum=True,
                             phi_weight=1e-4)
    del trajs
    print(f"  {batch.n_steps:,} 步、{len(batch.unit_step):,} 個 unit decision、"
          f"{len(batch.traj_start)} 條軌跡"
          "（要跟 §102~§104 的 11,504 / 108,040 / 16 對得上）")

    s = heads(net_a, net_b, batch, dev)
    us = batch.unit_step
    adv_u = batch.adv[us]
    starts = batch.traj_start
    ends = np.append(starts[1:], batch.n_steps)
    traj_of = np.searchsorted(starts, np.arange(batch.n_steps), "right") - 1
    traj_u = traj_of[us]
    margin = batch.traj_cash[:, 0] - batch.traj_cash[:, 1]
    m_u = margin[traj_u]
    print(f"  最終差額：平均 {margin.mean():,.0f}、範圍 "
          f"{margin.min():,.0f} ~ {margin.max():,.0f}")

    # ---------------- 1  decision-level ----------------
    hr("1  decision-level 相關（注意：同一局的決策共用同一個 final margin）")
    print(f"  {'':<26}{'op':>12}{'target':>12}")
    rows = [("corr(ΔP(取樣動作), margin)", "dp_s"),
            ("corr(ΔP(舊 top1), margin)", "dp_t1"),
            ("corr(Δlogp(取樣), margin)", "dlp_s")]
    for lab, key in rows:
        print(f"  {lab:<26}{corr(s[f'op_{key}'], m_u):>12.4f}"
              f"{corr(s[f'tgt_{key}'], m_u):>12.4f}")
    for lab, key in rows:
        print(f"  {'spearman ' + lab[5:]:<26}"
              f"{spearman(s[f'op_{key}'], m_u):>12.4f}"
              f"{spearman(s[f'tgt_{key}'], m_u):>12.4f}")
    print(f"  {'corr(|ΔP|, |margin|)':<26}"
          f"{corr(np.abs(s['op_dp_s']), np.abs(m_u)):>12.4f}"
          f"{corr(np.abs(s['tgt_dp_s']), np.abs(m_u)):>12.4f}")
    print(f"  {'corr(KL, margin)':<26}{corr(s['op_kl'], m_u):>12.4f}"
          f"{corr(s['tgt_kl'], m_u):>12.4f}")
    m_step = margin[traj_of]
    print(f"\n  market（逐步，n={batch.n_steps:,}）"
          f"  corr(Δlogp, margin) {corr(s['mk_dlp'], m_step):.4f}"
          f"   corr(KL, margin) {corr(s['mk_kl'], m_step):.4f}")

    # ---------------- 2  trajectory-level ----------------
    hr("2  trajectory-level（n = %d，這一節才是重點）" % len(margin))
    agg = {}
    for pre in ("op", "tgt"):
        agg[pre] = {
            "mean ΔP(取樣)": np.array(
                [s[f"{pre}_dp_s"][traj_u == t].mean() for t in range(len(margin))]),
            "mean ΔP(舊top1)": np.array(
                [s[f"{pre}_dp_t1"][traj_u == t].mean() for t in range(len(margin))]),
            "mean |ΔP|": np.array(
                [np.abs(s[f"{pre}_dp_s"][traj_u == t]).mean() for t in range(len(margin))]),
            "mean |Δlogp|": np.array(
                [np.abs(s[f"{pre}_dlp_s"][traj_u == t]).mean() for t in range(len(margin))]),
            "sum Δlogp": np.array(
                [s[f"{pre}_dlp_s"][traj_u == t].sum() for t in range(len(margin))]),
            "mean KL": np.array(
                [s[f"{pre}_kl"][traj_u == t].mean() for t in range(len(margin))]),
        }
    agg["mkt"] = {
        "mean Δlogp": np.array(
            [s["mk_dlp"][traj_of == t].mean() for t in range(len(margin))]),
        "mean |Δlogp|": np.array(
            [np.abs(s["mk_dlp"][traj_of == t]).mean() for t in range(len(margin))]),
        "sum Δlogp": np.array(
            [s["mk_dlp"][traj_of == t].sum() for t in range(len(margin))]),
        "mean KL": np.array(
            [s["mk_kl"][traj_of == t].mean() for t in range(len(margin))]),
    }
    n = len(margin)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    print(f"  相關係數的標準誤約 {se:.3f}（n={n}）—— |r| 小於 "
          f"{2 * se:.2f} 都不該當成訊號")
    for pre, lab in (("op", "op head"), ("tgt", "target head"),
                     ("mkt", "market head")):
        print(f"\n  {lab}")
        for k, v in agg[pre].items():
            print(f"    corr({k:<16}, margin)  {corr(v, margin):>+8.4f}"
                  f"   spearman {spearman(v, margin):>+8.4f}")

    # ---------------- 3  outcome bucket ----------------
    hr("3  按最終差額分桶（軌跡層）")
    order = np.argsort(margin)
    qs = np.array_split(order, 5)
    print(f"  {'桶':<10}{'n':>4}{'margin 平均':>13}"
          f"{'op ΔP(取樣)':>14}{'op KL':>10}"
          f"{'tgt ΔP(取樣)':>15}{'tgt KL':>10}{'mkt Δlogp':>12}{'mkt KL':>10}")
    labs = ["最差 20%", "20-40%", "40-60%", "60-80%", "最好 20%"]
    for lab, q in zip(labs, qs):
        print(f"  {lab:<10}{len(q):>4}{margin[q].mean():>13,.0f}"
              f"{agg['op']['mean ΔP(取樣)'][q].mean():>+14.5f}"
              f"{agg['op']['mean KL'][q].mean():>10.5f}"
              f"{agg['tgt']['mean ΔP(取樣)'][q].mean():>+15.5f}"
              f"{agg['tgt']['mean KL'][q].mean():>10.5f}"
              f"{agg['mkt']['mean Δlogp'][q].mean():>+12.5f}"
              f"{agg['mkt']['mean KL'][q].mean():>10.5f}")
    hi = margin > np.median(margin)
    print(f"\n  上下半（各 {int(hi.sum())} 條）")
    for pre, lab, key in (("op", "op ΔP(取樣)", "mean ΔP(取樣)"),
                          ("op", "op KL", "mean KL"),
                          ("tgt", "tgt ΔP(取樣)", "mean ΔP(取樣)"),
                          ("tgt", "tgt KL", "mean KL"),
                          ("mkt", "mkt Δlogp", "mean Δlogp"),
                          ("mkt", "mkt KL", "mean KL")):
        v = agg[pre][key]
        print(f"    {lab:<14}好的一半 {v[hi].mean():>+10.5f}   "
              f"壞的一半 {v[~hi].mean():>+10.5f}   差 {v[hi].mean() - v[~hi].mean():>+10.5f}")

    # ---------------- 4  advantage ----------------
    hr("4  按 advantage 分（只能回答 generalization 的方向，見檔頭）")
    adv_s = batch.adv
    print(f"  {'桶':<12}{'n(unit)':>9}{'op ΔP(取樣)':>14}{'op Δlogp':>12}"
          f"{'op KL':>9}{'tgt ΔP(取樣)':>15}{'tgt Δlogp':>12}{'tgt KL':>9}"
          f"{'mkt Δlogp':>12}{'mkt KL':>9}")
    for lab, mu, ms in (("adv > 0", adv_u > 0, adv_s > 0),
                        ("adv < 0", adv_u < 0, adv_s < 0)):
        print(f"  {lab:<12}{int(mu.sum()):>9,}"
              f"{s['op_dp_s'][mu].mean():>+14.5f}{s['op_dlp_s'][mu].mean():>+12.5f}"
              f"{s['op_kl'][mu].mean():>9.5f}"
              f"{s['tgt_dp_s'][mu].mean():>+15.5f}"
              f"{s['tgt_dlp_s'][mu].mean():>+12.5f}{s['tgt_kl'][mu].mean():>9.5f}"
              f"{s['mk_dlp'][ms].mean():>+12.5f}{s['mk_kl'][ms].mean():>9.5f}")
    for lo, hi_ in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99)):
        for sign, sgn in ((">0", 1), ("<0", -1)):
            mu = (np.abs(adv_u) >= lo) & (np.abs(adv_u) < hi_) & (
                np.sign(adv_u) == sgn)
            ms = (np.abs(adv_s) >= lo) & (np.abs(adv_s) < hi_) & (
                np.sign(adv_s) == sgn)
            if not mu.any():
                continue
            print(f"  {f'{lo}-{hi_} {sign}':<12}{int(mu.sum()):>9,}"
                  f"{s['op_dp_s'][mu].mean():>+14.5f}"
                  f"{s['op_dlp_s'][mu].mean():>+12.5f}"
                  f"{s['op_kl'][mu].mean():>9.5f}"
                  f"{s['tgt_dp_s'][mu].mean():>+15.5f}"
                  f"{s['tgt_dlp_s'][mu].mean():>+12.5f}"
                  f"{s['tgt_kl'][mu].mean():>9.5f}"
                  f"{s['mk_dlp'][ms].mean():>+12.5f}"
                  f"{s['mk_kl'][ms].mean():>9.5f}")

    # ---------------- 6  control ----------------
    hr("6  control：把「局勢本來就好」迴歸掉")
    half = np.array([c[len(c) // 2, 0] - c[len(c) // 2, 1] for c in cash])
    quart = np.array([c[len(c) // 4, 0] - c[len(c) // 4, 1] for c in cash])
    print(f"  中局差額（第 50% 步）平均 {half.mean():,.0f}   "
          f"corr(中局差額, 最終差額) {corr(half, margin):+.4f}")
    print(f"  前段差額（第 25% 步）平均 {quart.mean():,.0f}   "
          f"corr(前段差額, 最終差額) {corr(quart, margin):+.4f}")
    print(f"\n  {'':<22}{'原始 r':>10}{'控制中局後':>12}{'控制前段後':>12}")
    for pre, lab, key in (("op", "op mean KL", "mean KL"),
                          ("tgt", "tgt mean KL", "mean KL"),
                          ("mkt", "mkt mean KL", "mean KL"),
                          ("op", "op ΔP(取樣)", "mean ΔP(取樣)"),
                          ("tgt", "tgt ΔP(取樣)", "mean ΔP(取樣)"),
                          ("mkt", "mkt Δlogp", "mean Δlogp")):
        v = agg[pre][key]
        print(f"  {lab:<22}{corr(v, margin):>+10.4f}"
              f"{partial(v, margin, half):>+12.4f}"
              f"{partial(v, margin, quart):>+12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
