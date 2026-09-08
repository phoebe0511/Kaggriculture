"""三項量測：market clamp、joint KL vs unit 數、沒執行的 op factor 佔多少更新。

    python -m tools.factor_audit --games 16 --dump model/artifacts/phi-gt

**不改 PPO、不重跑訓練。** 三節分別是：

    A  market clamp   取樣出來的數量 vs `clamp_market_qty` 之後真正送出的數量
    B  joint KL       每一步的 KL 對 unit 數分桶（讀 `--dump-batch` 存的 npz）
    C  op factor      unit 沒走到目標格時，那個 op 因子佔了多少 KL / 更新

## 為什麼 C 節的 driver 是**舊**的 checkpoint

要量的是「一次更新往哪裡走」，所以狀態和動作都要來自舊 policy，Δlogp 才是
`new − old`。前幾支診斷（`tools/argmax_changed.py`）driver 用 ckpt-200 是為了
量「同一批盤面上兩個 policy 的差」，目的不同。

🩸 不要在這支裡開 `RolloutPool`（journal §97）。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")

from tools._quiet import silenced                       # noqa: E402

with silenced():
    import torch

import contracts as C                                   # noqa: E402
from model import ppo                                   # noqa: E402
from harness.ppo_rollout import (VecRollout, build_net,  # noqa: E402
                                 load_league)


def hr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def load_ckpt(path, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(int(blob["width"]), int(blob["blocks"]))
    net.load_state_dict(blob["state_dict"])
    return net.to(device).eval()


# --------------------------------------------------------------------------
# A：market clamp
# --------------------------------------------------------------------------

class MarketProbe(VecRollout):
    """每一步把「取樣出來的訂單」跟「`decode_market_orders` 真正送出的」對起來。

    🩸 logprob 用的是**取樣值**：`_policy_batch` 先 `sample_market` 算
    `market_logp_entropy`，再把同一組 present/qty 交給 `decode_market_orders`
    （harness/ppo_rollout.py:478-489）。clamp 發生在 decode 裡面，logprob 不知道。
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.rows = []          # (op_index, raw_qty, exec_qty)
        self.steps = 0
        self.emitted_ok = 0

    def _policy_batch(self, items, collect=False):
        out = super()._policy_batch(items, collect=collect)
        for (ei, p, obs, cfg), (_e, _p, action, rec) in zip(items, out):
            if "mk_present" not in rec:
                continue
            self.steps += 1
            pres = np.asarray(rec["mk_present"], dtype=bool)
            qty = np.asarray(rec["mk_qty"], dtype=np.int64)
            legal = np.asarray(rec["mk_legal"], dtype=bool)
            n_orders = 0
            for j in range(C.N_MARKET_OPS):
                if not (pres[j] and legal[j]):
                    continue
                raw = int(C.market_qty_value(int(qty[j])))
                ex = int(C.clamp_market_qty(j, raw, obs))
                self.rows.append((j, raw, max(ex, 0)))
                if ex > 0:
                    # HIRE / BUY_LAND 是送 n 筆，其餘一筆
                    n_orders += ex if C.MARKET_OPS[j][0] in (
                        "HIRE", "BUY_LAND") else 1
            # 重建的訂單數要跟真正送出去的一致，不然這一節的統計就是錯的
            if n_orders == len(action.get("market") or []):
                self.emitted_ok += 1
        return out


def section_market(net, args):
    hr("A  MARKET CLAMP")
    opp, _ = load_league(args.opponent)
    t0 = time.perf_counter()
    vec = MarketProbe(net, n_envs=args.games, seed0=args.seed0,
                      device=args.device, opponent=opp, phi=args.phi,
                      recognise=args.recognise)
    _steps, _cash, trajs = vec.run(collect=True)
    print(f"  rollout {time.perf_counter() - t0:.1f}s  "
          f"{vec.steps:,} 個決策步（取樣，跟訓練同一條路）")
    print(f"  重建的訂單數與實際送出一致的步：{vec.emitted_ok:,} / "
          f"{vec.steps:,} = {100 * vec.emitted_ok / max(vec.steps, 1):.2f}%"
          "   （不是 100% 的話下面的數字要打折看）")

    rows = np.array(vec.rows, dtype=np.int64)
    if not len(rows):
        print("  這批沒有任何 market 決策")
        return trajs
    op, raw, ex = rows[:, 0], rows[:, 1], rows[:, 2]
    fam = np.array([C.MARKET_OPS[int(j)][0] for j in op])
    print(f"\n  取樣後送進 decode 的 market 決策 {len(rows):,} 個"
          f"（平均每步 {len(rows) / max(vec.steps, 1):.2f} 個）")
    print(f"  {'類別':<14}{'n':>9}{'raw 平均':>10}{'實際平均':>10}"
          f"{'被 clamp':>10}{'clamp 成 0':>11}{'數量改變':>10}")
    for name in ["全部"] + sorted(set(fam.tolist())):
        m = np.ones(len(rows), bool) if name == "全部" else (fam == name)
        if not m.any():
            continue
        cl = ex[m] != raw[m]
        z = ex[m] == 0
        print(f"  {name:<14}{int(m.sum()):>9,}{raw[m].mean():>10.2f}"
              f"{ex[m].mean():>10.2f}{100 * cl.mean():>9.2f}%"
              f"{100 * z.mean():>10.2f}%{100 * cl.mean():>9.2f}%")

    cl = ex != raw
    if cl.any():
        d = (raw - ex)[cl]
        print(f"\n  被 clamp 的幅度（raw - 實際），n={int(cl.sum()):,}")
        print("    百分位 p10 %.0f  p25 %.0f  p50 %.0f  p75 %.0f  p90 %.0f  "
              "最大 %.0f" % tuple(np.percentile(d, [10, 25, 50, 75, 90, 100])))
        print(f"    相對幅度（(raw-實際)/raw）平均 "
              f"{(d / raw[cl]).mean():.3f}")
    print(f"\n  raw 動作 != 實際執行的動作：{100 * cl.mean():.2f}%"
          f"（其中被砍成 0、完全沒送出的：{100 * (ex == 0).mean():.2f}%）")
    print("  PPO 的 logprob 用的是 **raw**（取樣值）："
          "harness/ppo_rollout.py:478-489 先算 market_logp_entropy 再 decode，")
    print("  存進軌跡的也是取樣值（rec['mk_present'] / rec['mk_qty']），"
          "clamp 之後的數量從來沒有回饋給 policy。")
    n_ghost = int((ex == 0).sum())
    print(f"\n  => 每 100 個 market 決策，有 {100 * (ex == 0).mean():.1f} 個的"
          f" policy gradient 對應到一個**完全沒有執行**的動作"
          f"（這批 {n_ghost:,} 個），")
    print(f"     另有 {100 * (cl & (ex > 0)).mean():.1f} 個執行了但數量不是"
          f"取樣的那個。")
    return trajs


# --------------------------------------------------------------------------
# B：joint KL vs unit 數
# --------------------------------------------------------------------------

def section_kl_units(dump_dir):
    hr("B  JOINT KL vs UNITS PER STEP（讀 --dump-batch 的 npz）")
    files = sorted(glob.glob(str(Path(dump_dir) / "batch-*.npz")))
    if not files:
        print(f"  {dump_dir} 沒有 batch-*.npz")
        return
    units, kl, adv = [], [], []
    for fp in files:
        z = np.load(fp)
        n = len(z["old_logp"])
        u = np.bincount(z["unit_step"], minlength=n).astype(np.int64)
        units.append(u)
        # 🩸 這是**整輪更新之後**的 old-new，不是單一 epoch。訓練 log 裡的
        # approx_kl 是每個 minibatch、每個 epoch 的平均，兩者不一樣。
        kl.append(z["old_logp"] - z["new_logp"])
        adv.append(z["adv"])
    units = np.concatenate(units)
    kl = np.concatenate(kl)
    adv = np.concatenate(adv)
    ratio = np.exp(-kl)                       # exp(new - old)
    print(f"  {len(files)} 輪、{len(kl):,} 步（{Path(dump_dir).name}）")
    print(f"  每步 unit 數：平均 {units.mean():.2f}  中位 "
          f"{np.median(units):.0f}  範圍 {units.min()}~{units.max()}")

    edges = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 10 ** 6)]
    print(f"\n  {'units 桶':<10}{'n':>9}{'平均 units':>11}{'平均 KL':>11}"
          f"{'中位 KL':>11}{'KL/unit':>10}{'|ratio-1|>0.2':>15}")
    for lo, hi in edges:
        m = (units >= lo) & (units <= hi)
        if not m.any():
            continue
        lab = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
        print(f"  {lab:<10}{int(m.sum()):>9,}{units[m].mean():>11.2f}"
              f"{kl[m].mean():>11.5f}{np.median(kl[m]):>11.5f}"
              f"{kl[m].mean() / max(units[m].mean(), 1):>10.5f}"
              f"{100 * (np.abs(ratio[m] - 1) > 0.2).mean():>14.2f}%")
    ok = units > 0

    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    # 🩸 平均 KL 是長尾的（0-4 桶平均 0.0212、中位 0.0001），原值的相關係數
    # 被少數極端步主導。秩相關才看得出單調關係。
    print(f"\n  spearman(units_per_step, KL)    "
          f"{spearman(units[ok], kl[ok]):+.4f}   <- 長尾，看這個")
    print(f"  corr(units_per_step, KL)        "
          f"{np.corrcoef(units[ok], kl[ok])[0, 1]:+.4f}")
    print(f"  corr(units_per_step, |KL|)      "
          f"{np.corrcoef(units[ok], np.abs(kl[ok]))[0, 1]:+.4f}")
    print(f"  corr(units_per_step, |adv|)     "
          f"{np.corrcoef(units[ok], np.abs(adv[ok]))[0, 1]:+.4f}"
          "   （對照：unit 多的步 advantage 有沒有比較大）")

    # 迭代層：unit 多的輪是不是比較容易提早停
    tj = Path(dump_dir) / "train.jsonl"
    if tj.exists():
        import json                                      # noqa: PLC0415
        rows = [json.loads(l) for l in open(tj, encoding="utf-8")]
        u = np.array([r["units_per_step"] for r in rows])
        e = np.array([r["epochs_done"] for r in rows])
        k = np.array([r["kl_last_epoch"] for r in rows])
        c = np.array([r["clipfrac"] for r in rows])
        print(f"\n  迭代層（n={len(rows)}）  units_per_step "
              f"{u.mean():.2f}±{u.std():.2f}   提早停 {(e < 8).sum()} 輪")
        print(f"    corr(units_per_step, kl_last_epoch) "
              f"{np.corrcoef(u, k)[0, 1]:+.4f}")
        print(f"    corr(units_per_step, clipfrac)      "
              f"{np.corrcoef(u, c)[0, 1]:+.4f}")
        print(f"    corr(units_per_step, epochs_done)   "
              f"{np.corrcoef(u, e)[0, 1]:+.4f}")


# --------------------------------------------------------------------------
# C：執行 vs 沒執行的 op factor
# --------------------------------------------------------------------------

def both_nets(net_a, net_b, batch, device, chunk=256):
    """同一批 unit 上，兩個網路的逐因子 KL 與被取樣動作的 logprob。"""
    out = {k: [] for k in ("kl_op", "kl_tgt", "kl_mkt", "lp_op_a", "lp_op_b",
                           "lp_tgt_a", "lp_tgt_b",
                           "arg_op_a", "arg_op_b", "arg_tgt_a", "arg_tgt_b")}
    with torch.no_grad():
        for i in range(0, batch.n_steps, chunk):
            idx = np.arange(i, min(i + chunk, batch.n_steps))
            mb = batch._pack(idx, device)
            pack = []
            for net in (net_a, net_b):
                op_l, _q, tg_l, mp, mq, _v, _d = net(
                    mb["spatial"], mb["scalar"], mb["unit_board"],
                    mb["unit_pos"], mb["unit_feats"])
                pack.append((ppo.masked_log_softmax(op_l, mb["op_mask"]),
                             ppo.masked_log_softmax(tg_l, mb["tgt_mask"]),
                             mp, mq))
            (oa, ta, mpa, mqa), (ob, tb, mpb, mqb) = pack
            # market 那一份的 KL（逐步）。階層式：Bernoulli + 以 p 加權的 qty
            # categorical，跟 `ppo.market_logp_entropy` 同一個分解。
            legal = mb["mk_legal"].bool().float()
            pa = torch.sigmoid(mpa)
            lp1a, lp0a = torch.nn.functional.logsigmoid(mpa), \
                torch.nn.functional.logsigmoid(-mpa)
            lp1b, lp0b = torch.nn.functional.logsigmoid(mpb), \
                torch.nn.functional.logsigmoid(-mpb)
            kl_bern = pa * (lp1a - lp1b) + (1 - pa) * (lp0a - lp0b)
            qa = torch.log_softmax(mqa, dim=-1)
            qb = torch.log_softmax(mqb, dim=-1)
            kl_cat = (qa.exp() * (qa - qb)).sum(-1)
            out["kl_mkt"].append(
                ((kl_bern + pa * kl_cat) * legal).sum(-1).cpu().numpy())
            # KL(old || new)，逐因子。被遮掉的動作機率是 0，貢獻 0。
            out["kl_op"].append(
                (oa.exp() * (oa - ob)).sum(-1).cpu().numpy())
            out["kl_tgt"].append(
                (ta.exp() * (ta - tb)).sum(-1).cpu().numpy())
            oi = mb["op_idx"].unsqueeze(-1)
            ti = mb["tgt_idx"].unsqueeze(-1)
            out["lp_op_a"].append(oa.gather(-1, oi).squeeze(-1).cpu().numpy())
            out["lp_op_b"].append(ob.gather(-1, oi).squeeze(-1).cpu().numpy())
            out["lp_tgt_a"].append(ta.gather(-1, ti).squeeze(-1).cpu().numpy())
            out["lp_tgt_b"].append(tb.gather(-1, ti).squeeze(-1).cpu().numpy())
            # funnel 要走 production path：argmax -> step_toward -> 引擎動作
            for key, t in (("arg_op_a", oa), ("arg_op_b", ob),
                           ("arg_tgt_a", ta), ("arg_tgt_b", tb)):
                out[key].append(t.argmax(-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def section_op_factor(net_a, net_b, batch, device):
    hr("C  執行 vs 沒執行的 OP FACTOR")
    s = both_nets(net_a, net_b, batch, device)
    _ = s
    pos = batch.unit_pos
    tx = batch.tgt_idx % C.BOARD_SIZE
    ty = batch.tgt_idx // C.BOARD_SIZE
    arrived = (pos[:, 0].astype(np.int64) == tx) & (
        pos[:, 1].astype(np.int64) == ty)
    adv = batch.adv[batch.unit_step]
    d_op = s["lp_op_b"] - s["lp_op_a"]
    d_tg = s["lp_tgt_b"] - s["lp_tgt_a"]
    n = len(arrived)
    print(f"  {n:,} 個 unit 決策（driver = 舊 checkpoint，取樣）")
    print(f"  站在目標格上（op 這一步真的執行）{100 * arrived.mean():.2f}%")

    print(f"\n  {'':<22}{'executed':>14}{'not executed':>16}{'差':>12}")
    rows = [
        ("n", arrived.sum(), (~arrived).sum(), None),
        ("|adv|", np.abs(adv[arrived]).mean(),
         np.abs(adv[~arrived]).mean(), None),
        ("Δlogp(op) 平均", d_op[arrived].mean(), d_op[~arrived].mean(), None),
        ("|Δlogp(op)| 平均", np.abs(d_op[arrived]).mean(),
         np.abs(d_op[~arrived]).mean(), None),
        ("op 因子 KL 平均", s["kl_op"][arrived].mean(),
         s["kl_op"][~arrived].mean(), None),
        ("target 因子 KL 平均", s["kl_tgt"][arrived].mean(),
         s["kl_tgt"][~arrived].mean(), None),
    ]
    for name, a, b, _ in rows:
        if name == "n":
            print(f"  {name:<22}{int(a):>14,}{int(b):>16,}"
                  f"{int(a) - int(b):>12,}")
        else:
            print(f"  {name:<22}{a:>14.5f}{b:>16.5f}{a - b:>+12.5f}")

    # 更新方向一致率（advantage 的正負 vs Δlogp 的正負）
    # 🩸 E_old[Δlogp] = -KL(old||new) <= 0，跨 60 輪時 Δlogp 幾乎都是負的，
    # 不置中的一致率只是在量那個漂移（§93.1 報的也是置中後的版本）。
    print("")
    for tag, d in (("op", d_op), ("target", d_tg)):
        dc = d - d.mean()
        for lab, m in (("executed", arrived), ("not executed", ~arrived)):
            raw = ((np.sign(adv[m]) == np.sign(d[m])) & (adv[m] != 0)).mean()
            cen = ((np.sign(adv[m]) == np.sign(dc[m])) & (adv[m] != 0)).mean()
            print(f"  方向一致率 {tag:<7}{lab:<14}原始 {100 * raw:6.2f}%   "
                  f"置中後 {100 * cen:6.2f}%   (n={int(m.sum()):,})")

    print(f"\n  每個因子平均 KL   executed {s['kl_op'][arrived].mean():.5f}"
          f"   not-executed {s['kl_op'][~arrived].mean():.5f}"
          f"   比 "
          f"{s['kl_op'][~arrived].mean() / max(s['kl_op'][arrived].mean(), 1e-12):.2f}")
    tot = s["kl_op"].sum()
    print(f"\n  op head 的 KL 總量裡，沒執行的因子佔 "
          f"{100 * s['kl_op'][~arrived].sum() / max(tot, 1e-12):.2f}%")
    both = s["kl_op"] + s["kl_tgt"]
    print(f"  op+target 合計的 KL 裡，沒執行的 op 因子佔 "
          f"{100 * s['kl_op'][~arrived].sum() / max(both.sum(), 1e-12):.2f}%")
    u_per_step = both.sum() / batch.n_steps
    m_per_step = s["kl_mkt"].mean()
    print("\n  逐步的 KL 分解（同樣是 140->200 這 60 輪的移動量）")
    print(f"    unit 因子合計 / 步   {u_per_step:.5f}"
          f"（unit 平均 {len(arrived) / batch.n_steps:.2f} 個/步）")
    print(f"    market 因子 / 步     {m_per_step:.5f}")
    print(f"    market 佔逐步 KL 的  "
          f"{100 * m_per_step / max(u_per_step + m_per_step, 1e-12):.2f}%")
    print(f"    沒執行的 op 佔逐步 KL 的 "
          f"{100 * (s['kl_op'][~arrived].sum() / batch.n_steps) / max(u_per_step + m_per_step, 1e-12):.2f}%")
    return s


def section_funnel(net_a, net_b, batch, device, s=None):
    """target 的移動，有多少真的改到送進引擎的動作。

    沿 production path 比：`argmax -> step_toward -> 引擎吃的 unit 動作`。
    🩸 只比 `old_target != new_target` 會高估 —— 換到同方向的另一格，
    `step_toward` 走的還是同一步。
    """
    from tools.argmax_changed import immediate_action     # noqa: PLC0415

    hr("D  TARGET 的 FUNNEL：argmax -> step_toward -> 引擎動作")
    if s is None:
        s = both_nets(net_a, net_b, batch, device)
    pos = batch.unit_pos
    ta, tb = s["arg_tgt_a"], s["arg_tgt_b"]
    oa, ob = s["arg_op_a"], s["arg_op_b"]
    act_a, arr_a = immediate_action(pos, ta, oa)
    act_b, arr_b = immediate_action(pos, tb, ob)
    n = len(ta)
    adv = np.abs(batch.adv[batch.unit_step])
    d_tg = np.abs(s["lp_tgt_b"] - s["lp_tgt_a"])
    kl_tg = s["kl_tgt"]
    tot_kl = kl_tg.sum()

    tgt_ch = ta != tb
    act_ch = act_a != act_b
    # 「方向」只有在兩邊都還在路上時才定義得了
    moving = (~arr_a) & (~arr_b)
    dir_ch = moving & (act_a != act_b)

    cats = [
        ("1 target unchanged", ~tgt_ch),
        ("2 target 改、動作沒改", tgt_ch & ~act_ch),
        ("3 target 改、動作改了", tgt_ch & act_ch),
        ("4 送進引擎的動作改了", act_ch),
    ]
    print(f"  {n:,} 個 unit 決策（同一批盤面，兩個 checkpoint 各走一次 "
          "production path）")
    print(f"  {'類別':<24}{'n':>9}{'佔全部':>9}{'佔 target 移動':>16}"
          f"{'|Δlogp(tgt)|':>14}{'target KL':>12}{'|adv|':>9}")
    for name, m in cats:
        if not m.any():
            print(f"  {name:<24}{0:>9}")
            continue
        print(f"  {name:<24}{int(m.sum()):>9,}{100 * m.mean():>8.2f}%"
              f"{100 * kl_tg[m].sum() / max(tot_kl, 1e-12):>15.2f}%"
              f"{d_tg[m].mean():>14.5f}{kl_tg[m].mean():>12.5f}"
              f"{adv[m].mean():>9.4f}")

    print(f"\n  轉換率")
    print(f"    target argmax 改變                     "
          f"{100 * tgt_ch.mean():.2f}%  (n={int(tgt_ch.sum()):,})")
    print(f"    -> 實際移動方向改變                    "
          f"{100 * dir_ch.sum() / max(tgt_ch.sum(), 1):.2f}%"
          f"  (n={int(dir_ch.sum()):,}，只算兩邊都在路上的)")
    print(f"    -> 送進引擎的動作改變                  "
          f"{100 * (tgt_ch & act_ch).sum() / max(tgt_ch.sum(), 1):.2f}%"
          f"  (n={int((tgt_ch & act_ch).sum()):,})")

    print(f"\n  為什麼 target 改了動作卻沒改（分開統計）")
    both_move = tgt_ch & moving & ~act_ch
    a_arr = tgt_ch & arr_a & ~act_ch
    b_arr = tgt_ch & arr_b & ~act_ch
    print(f"    兩邊都在路上、step_toward 走同一步   "
          f"{int(both_move.sum()):>8,}"
          f"（佔 target 改變的 {100 * both_move.sum() / max(tgt_ch.sum(), 1):.2f}%）")
    print(f"    其他（一邊已站在目標格上）           "
          f"{int((a_arr | b_arr).sum()):>8,}")
    print(f"\n  兩邊站位狀態的交叉（target 改變的 {int(tgt_ch.sum()):,} 個）")
    for la, ma in (("舊在路上", ~arr_a), ("舊已到", arr_a)):
        for lb, mb in (("新在路上", ~arr_b), ("新已到", arr_b)):
            m = tgt_ch & ma & mb
            if not m.any():
                continue
            print(f"    {la} + {lb:<10}{int(m.sum()):>8,}"
                  f"   其中動作改變 {100 * act_ch[m].mean():6.2f}%")

    # target 的移動量裡，有多少落在「會改動作」的決策上
    share = kl_tg[act_ch].sum() / max(tot_kl, 1e-12)
    print(f"\n  target 的移動量（KL）落在「送進引擎的動作有改變」的決策上："
          f"{100 * share:.2f}%")
    # 接回全部 policy movement（unit 因子 + market 因子）
    total = ((s["kl_op"] + kl_tg).sum() / batch.n_steps
             + s["kl_mkt"].mean())
    tgt_step = kl_tg.sum() / batch.n_steps
    print(f"  target 佔全部 policy movement {100 * tgt_step / total:.2f}%，"
          f"其中會改動作的那部分佔全部 "
          f"{100 * (kl_tg[act_ch].sum() / batch.n_steps) / total:.2f}%")
    return share


def greedy_conv(net_a, net_b, batch, device, chunk=256):
    """每個 unit factor 的機率位移細節，兩個 head 各一份。

    回傳的 key 前綴是 `op_` / `tgt_`：
      t1a/t2a  舊的 top-1 / top-2 索引        p1a/p2a  它們在舊網路的機率
      p1b/p2b  **同兩個動作**在新網路的機率   n1b      新網路的 top-1 索引
      q1b/q2b  新網路自己的 top-1/top-2 機率
      sa/sb    rollout 取樣到的那個動作在新舊網路的機率
    """
    keys = ("t1a", "t2a", "p1a", "p2a", "p1b", "p2b", "n1b", "q1b", "q2b",
            "sa", "sb", "kl")
    out = {f"{h}_{k}": [] for h in ("op", "tgt") for k in keys}
    with torch.no_grad():
        for i in range(0, batch.n_steps, chunk):
            idx = np.arange(i, min(i + chunk, batch.n_steps))
            mb = batch._pack(idx, device)
            lps = []
            for net in (net_a, net_b):
                op_l, _q, tg_l, _mp, _mq, _v, _d = net(
                    mb["spatial"], mb["scalar"], mb["unit_board"],
                    mb["unit_pos"], mb["unit_feats"])
                lps.append((ppo.masked_log_softmax(op_l, mb["op_mask"]),
                            ppo.masked_log_softmax(tg_l, mb["tgt_mask"])))
            for h, k, act in (("op", 0, mb["op_idx"]),
                              ("tgt", 1, mb["tgt_idx"])):
                la, lb = lps[0][k], lps[1][k]
                ta = la.topk(2, dim=-1)
                tb = lb.topk(2, dim=-1)
                i1 = ta.indices[:, :1]
                i2 = ta.indices[:, 1:2]
                grab = {
                    "t1a": ta.indices[:, 0], "t2a": ta.indices[:, 1],
                    "p1a": ta.values[:, 0].exp(), "p2a": ta.values[:, 1].exp(),
                    "p1b": lb.gather(-1, i1).squeeze(-1).exp(),
                    "p2b": lb.gather(-1, i2).squeeze(-1).exp(),
                    "n1b": tb.indices[:, 0],
                    "q1b": tb.values[:, 0].exp(), "q2b": tb.values[:, 1].exp(),
                    "sa": la.gather(-1, act.unsqueeze(-1)).squeeze(-1),
                    "sb": lb.gather(-1, act.unsqueeze(-1)).squeeze(-1),
                    "kl": (la.exp() * (la - lb)).sum(-1),
                }
                for kk, v in grab.items():
                    out[f"{h}_{kk}"].append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def section_greedy_conv(net_a, net_b, batch, device):
    hr("E  GREEDY CONVERSION AUDIT")
    g = greedy_conv(net_a, net_b, batch, device)
    adv = batch.adv[batch.unit_step]
    print(f"  {batch.n_steps:,} 步、{len(batch.unit_step):,} 個 unit 決策"
          "（狀態與取樣動作都來自舊 checkpoint）")

    for head, name, act in (("op", "op head", batch.op_idx),
                            ("tgt", "target head", batch.tgt_idx)):
        p1a, p2a = g[f"{head}_p1a"], g[f"{head}_p2a"]
        p1b, p2b = g[f"{head}_p1b"], g[f"{head}_p2b"]
        gap_a, gap_b = p1a - p2a, p1b - p2b
        flip = g[f"{head}_n1b"] != g[f"{head}_t1a"]
        to_t2 = flip & (g[f"{head}_n1b"] == g[f"{head}_t2a"])
        rest_a = 1.0 - p1a - p2a          # 舊的 top3+ 質量
        rest_b = 1.0 - p1b - p2b
        kl = g[f"{head}_kl"]

        hr(f"E1/E2  {name}：機率位移，按舊的 confidence gap 分桶")
        print(f"  {'gap 桶':<12}{'n':>9}{'P1 舊':>8}{'gap 舊':>8}"
              f"{'ΔP(top1)':>10}{'ΔP(top2)':>10}{'Δgap':>9}"
              f"{'Δtop3+':>9}{'flip':>8}{'flip→top2':>11}{'flip→top3+':>12}")
        for lab, m in gap_buckets_local(gap_a):
            if not m.any():
                continue
            f = flip[m]
            print(f"  {lab:<12}{int(m.sum()):>9,}{p1a[m].mean():>8.4f}"
                  f"{gap_a[m].mean():>8.4f}{(p1b - p1a)[m].mean():>+10.4f}"
                  f"{(p2b - p2a)[m].mean():>+10.4f}"
                  f"{(gap_b - gap_a)[m].mean():>+9.4f}"
                  f"{(rest_b - rest_a)[m].mean():>+9.4f}"
                  f"{100 * f.mean():>7.2f}%"
                  f"{100 * to_t2[m].sum() / max(f.sum(), 1):>10.2f}%"
                  f"{100 * (f.sum() - to_t2[m].sum()) / max(f.sum(), 1):>11.2f}%")

        hr(f"E3  {name}：按 advantage 的方向拆")
        print(f"  {'':<28}{'n':>9}{'ΔP(top1)':>11}{'ΔP(top2)':>11}"
              f"{'Δgap':>10}{'flip':>9}")
        for lab, m in (("adv > 0", adv > 0), ("adv < 0", adv < 0),
                       ("adv>0 且 top1 留著", (adv > 0) & ~flip),
                       ("adv>0 且 top1 被換掉", (adv > 0) & flip),
                       ("adv<0 且 top1 留著", (adv < 0) & ~flip),
                       ("adv<0 且 top1 被換掉", (adv < 0) & flip)):
            if not m.any():
                continue
            print(f"  {lab:<28}{int(m.sum()):>9,}"
                  f"{(p1b - p1a)[m].mean():>+11.4f}"
                  f"{(p2b - p2a)[m].mean():>+11.4f}"
                  f"{(gap_b - gap_a)[m].mean():>+10.4f}"
                  f"{100 * flip[m].mean():>8.2f}%")

        hr(f"E4  {name}：對的更新，greedy 看得到嗎")
        sa, sb = g[f"{head}_sa"], g[f"{head}_sb"]
        dlp = sb - sa                        # 取樣到的那個動作的 Δlogp
        was_top1 = act == g[f"{head}_t1a"]   # 取樣到的就是舊 argmax
        aligned = ((adv > 0) & (dlp > 0)) | ((adv < 0) & (dlp < 0))
        nz = adv != 0
        n = int(nz.sum())
        print(f"  取樣到的動作就是舊 argmax 的比例 {100 * was_top1.mean():.2f}%")
        print(f"  更新方向與 advantage 一致（aligned）"
              f"{100 * aligned[nz].mean():.2f}%   (n={n:,})")
        becomes = g[f"{head}_n1b"] == act    # 新的 argmax 就是取樣到的動作
        groups = [
            ("adv>0 取樣=舊argmax 推高", (adv > 0) & was_top1 & (dlp > 0),
             "greedy 本來就在做"),
            ("adv>0 取樣≠舊argmax 推高 -> 翻過去",
             (adv > 0) & ~was_top1 & (dlp > 0) & becomes, "greedy-visible"),
            ("adv>0 取樣≠舊argmax 推高 -> 沒翻過去",
             (adv > 0) & ~was_top1 & (dlp > 0) & ~becomes, "greedy-hidden"),
            ("adv<0 取樣=舊argmax 壓低 -> 翻掉",
             (adv < 0) & was_top1 & (dlp < 0) & flip, "greedy-visible"),
            ("adv<0 取樣=舊argmax 壓低 -> 沒翻掉",
             (adv < 0) & was_top1 & (dlp < 0) & ~flip, "greedy-hidden"),
            ("adv<0 取樣≠舊argmax 壓低", (adv < 0) & ~was_top1 & (dlp < 0),
             "greedy 本來就不做"),
        ]
        tot_kl = kl.sum()
        al = aligned & nz
        print(f"\n  {'':<34}{'n':>9}{'佔 aligned':>12}{'佔移動量':>10}"
              f"{'|Δlogp|':>10}{'ΔP(取樣動作)':>14}{'舊 gap':>9}  說明")
        for lab, m, note in groups:
            if not m.any():
                continue
            print(f"  {lab:<34}{int(m.sum()):>9,}"
                  f"{100 * m.sum() / max(al.sum(), 1):>11.2f}%"
                  f"{100 * kl[m].sum() / max(tot_kl, 1e-12):>9.2f}%"
                  f"{np.abs(dlp[m]).mean():>10.4f}"
                  f"{(np.exp(sb[m]) - np.exp(sa[m])).mean():>+14.4f}"
                  f"{gap_a[m].mean():>9.4f}  {note}")
        vis = ((adv > 0) & ~was_top1 & (dlp > 0) & becomes) | \
              ((adv < 0) & was_top1 & (dlp < 0) & flip)
        hid = ((adv > 0) & ~was_top1 & (dlp > 0) & ~becomes) | \
              ((adv < 0) & was_top1 & (dlp < 0) & ~flip)
        irr = ((adv > 0) & was_top1 & (dlp > 0)) | \
              ((adv < 0) & ~was_top1 & (dlp < 0))
        print(f"\n  aligned 的更新 {int(al.sum()):,} 個，拆成三塊：")
        for lab, m in (("greedy-visible", vis), ("greedy-hidden", hid),
                       ("greedy 無關（本來就在做／本來就不做）", irr)):
            print(f"    {lab:<40}{int(m.sum()):>9,}"
                  f"{100 * m.sum() / max(al.sum(), 1):>8.2f}%"
                  f"   佔移動量 {100 * kl[m].sum() / max(tot_kl, 1e-12):>6.2f}%")
        # 🩸 這裡的 advantage 是**這次 rollout 現場算的**，而 ckpt-200 是用另外
        # 60 批資料訓出來的 —— 這一筆 advantage 從來沒進過那些更新。所以「一致
        # 率」不能直接讀成「PPO 有沒有往 advantage 的方向走」（那要用
        # `--dump-batch` 的同一批，§93.1 量到步層 81.7%）。這裡能問的是比較弱的
        # 版本：學到的東西**generalize 到新狀態**時，還看不看得出 advantage 的
        # 方向。訊號若存在，應該隨 |adv| 變大而變強。
        print(f"\n  ΔP(取樣動作) 的 advantage 對比（argmax 沒翻的決策）")
        print(f"    {'|adv| 桶':<14}{'n(adv>0)':>10}{'ΔP|adv>0':>11}"
              f"{'n(adv<0)':>10}{'ΔP|adv<0':>11}{'對比':>10}")
        keep = ~flip
        for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99)):
            mm = keep & (np.abs(adv) >= lo) & (np.abs(adv) < hi)
            pos, neg = mm & (adv > 0), mm & (adv < 0)
            if not (pos.any() and neg.any()):
                continue
            dp = np.exp(sb) - np.exp(sa)
            print(f"    {f'{lo}-{hi}':<14}{int(pos.sum()):>10,}"
                  f"{dp[pos].mean():>+11.5f}{int(neg.sum()):>10,}"
                  f"{dp[neg].mean():>+11.5f}"
                  f"{dp[pos].mean() - dp[neg].mean():>+10.5f}")
        if hid.any():
            print(f"\n    greedy-hidden 的擋路距離：舊 gap 平均 "
                  f"{gap_a[hid].mean():.4f}、中位 {np.median(gap_a[hid]):.4f}")
            print(f"    這 60 輪把 P(取樣動作) 推高 "
                  f"{(np.exp(sb[hid]) - np.exp(sa[hid])).mean():+.5f}，"
                  f"距離 top-1 還差 "
                  f"{(p1b[hid] - np.exp(sb[hid])).mean():.4f}")


def gap_buckets_local(gap):
    edges = [0.05, 0.2, 0.5, 0.9]
    idx = np.digitize(gap, edges)
    out = []
    for b in range(len(edges) + 1):
        lo = "-inf" if b == 0 else f"{edges[b - 1]:g}"
        hi = "inf" if b == len(edges) else f"{edges[b]:g}"
        out.append((f"[{lo},{hi})", idx == b))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", default="model/artifacts/phi-200/ckpt-00140.pt")
    ap.add_argument("--b", default="model/artifacts/phi-200/ckpt-00200.pt")
    ap.add_argument("--dump", default="model/artifacts/phi-gt")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=960_000)
    ap.add_argument("--opponent", default="config/params/cma5-g175.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phi", default="assets")
    ap.add_argument("--phi-weight", type=float, default=1e-4)
    ap.add_argument("--recognise", default="strict")
    ap.add_argument("--report", default="temp/factor.txt")
    args = ap.parse_args(argv)
    args.device = args.device if torch.cuda.is_available() else "cpu"

    net_a = load_ckpt(args.a, args.device)
    net_b = load_ckpt(args.b, args.device)
    print(f"  舊 {args.a}\n  新 {args.b}")

    section_kl_units(args.dump)
    trajs = section_market(net_a, args)
    batch = ppo.RolloutBatch(trajs, gamma=1.0, lam=0.998, zero_sum=True,
                             phi_weight=args.phi_weight)
    st = section_op_factor(net_a, net_b, batch, args.device)
    section_funnel(net_a, net_b, batch, args.device, st)
    section_greedy_conv(net_a, net_b, batch, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
