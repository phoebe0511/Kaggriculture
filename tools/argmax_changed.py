"""140 -> 200 真正改掉 argmax 的那些 decision，是不是重要的 decision。

    python -m tools.argmax_changed --games 16 --out temp/changed.npz

§92 量到「每 5 輪換掉約 4.2% 的 unit 決策、60 輪累積 14.11% 真的改變主意」，
但 greedy 差額只動 554。這支要回答的是**那 14.11% 落在哪裡**：小 advantage /
接近 tie / episode 後段 / 低影響，還是重要的決策。

## 為什麼要自己收一批軌跡

`--dump-batch` 存的 npz 沒有觀測（`model/ppo_train.py:dump_batch`），
重算 argmax 一定要 spatial / scalar / unit_pos / unit_feats / mask。

🩸 **不要在腳本裡自己開 `RolloutPool`** —— 2026-09-07 踩了兩次，worker 各跑
1~3 秒就不動，等 40 分鐘也不出來（journal §97）。這支走單行程 `VecRollout`，
慢但不會卡。

## 兩個 checkpoint 看的是同一批盤面

軌跡由 `--driver`（預設 ckpt-00200）取樣驅動，另一個 checkpoint **不重跑**，
只在同一批 (盤面, unit) 上重算一次前向。所以「argmax 不同」量的純粹是
policy 的差異，沒有摻進不同軌跡帶來的盤面差異。

advantage 是這批軌跡自己的 GAE（value 來自 driver），旗標對齊 phi-200 那次
訓練：`--zero-sum --gamma 1.0 --lam 0.998 --phi assets --phi-weight 1e-4`。

⚠️ advantage 是**每步一個**，同一步的 unit 共用 —— 所以 unit 層的 n 不是
有效樣本數。步層和軌跡層的統計另外印。
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

os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")

from tools._quiet import silenced                       # noqa: E402

with silenced():
    import torch

from harness.ppo_rollout import (VecRollout, build_net,  # noqa: E402
                                 load_league)
import contracts as C                                   # noqa: E402
from model import ppo                                   # noqa: E402


def load_ckpt(path, device):
    """載一個 PPO checkpoint。

    🩸 不走 `ppo_train.load_init` —— 那支會對 `market_present_out` 除以
    `--market-temp`。phi-200 的權重**已經**是除過的（熱啟動時做的那一次），
    這裡再除一次就不是同一個 policy 了。
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(int(blob["width"]), int(blob["blocks"]))
    net.load_state_dict(blob["state_dict"])
    return net.to(device).eval(), int(blob.get("iter", -1)) + 1


def head_stats(net, batch, device, chunk=256, gather=None):
    """整批重算每個 unit 的 top-1 / top-2。順序跟 `batch` 的 unit 一致。

    `gather` 是 `{名字: (op 索引陣列, target 索引陣列)}` —— 對**指定的動作**
    再回報一次機率和排名（排名 0 就是 argmax）。要回答「argmax 沒翻，但機率
    有沒有往另一個動作移」就得看這個：只看 top-1 / top-2 的話，reference 的
    動作排第 7 名時它的機率變化根本不在視野裡。

    🩸 `gather` 的索引是**逐 unit**的，所以切 chunk 時要跟著 unit 的偏移量走，
    不能用步的索引。`u0` 就是那個累計偏移。
    """
    keys = ("op_top1", "op_top2", "op_top3", "op_p1", "op_p2", "op_p3",
            "tgt_top1", "tgt_top2", "tgt_top3", "tgt_p1", "tgt_p2", "tgt_p3")
    out = {k: [] for k in keys}
    gather = dict(gather or {})
    for g in gather:
        for head in ("op", "tgt"):
            out[f"{head}_p_{g}"] = []
            out[f"{head}_rank_{g}"] = []
    u0 = 0
    with torch.no_grad():
        for i in range(0, batch.n_steps, chunk):
            idx = np.arange(i, min(i + chunk, batch.n_steps))
            mb = batch._pack(idx, device)
            op_logits, _q, tgt_logits, _mp, _mq, _v, _d = net(
                mb["spatial"], mb["scalar"], mb["unit_board"],
                mb["unit_pos"], mb["unit_feats"])
            # 一定要走 `ppo.masked_log_softmax` —— 全 False 的 mask 退回
            # 「全部合法」那條規則在裡面，自己寫一份會跟 rollout 不一致。
            o_lp = ppo.masked_log_softmax(op_logits, mb["op_mask"])
            t_lp = ppo.masked_log_softmax(tgt_logits, mb["tgt_mask"])
            n_u = int(o_lp.shape[0])
            for name, lp in (("op", o_lp), ("tgt", t_lp)):
                top = lp.topk(3, dim=-1)
                p = top.values.exp().cpu().numpy()
                for r in range(3):
                    out[f"{name}_top{r + 1}"].append(
                        top.indices[:, r].cpu().numpy())
                    out[f"{name}_p{r + 1}"].append(p[:, r])
            for g, (op_i, tgt_i) in gather.items():
                for head, lp, src in (("op", o_lp, op_i), ("tgt", t_lp, tgt_i)):
                    # -1（認不得的動作）先當 0 取，呼叫端用同一個遮罩排除。
                    want = np.maximum(src[u0:u0 + n_u], 0)
                    ix = torch.as_tensor(want, device=lp.device).unsqueeze(-1)
                    at = lp.gather(-1, ix).squeeze(-1)
                    out[f"{head}_p_{g}"].append(at.exp().cpu().numpy())
                    rank = (lp > at.unsqueeze(-1)).sum(-1)
                    out[f"{head}_rank_{g}"].append(rank.cpu().numpy())
            u0 += n_u
    return {k: np.concatenate(v) for k, v in out.items()}


def traj_tables(batch):
    """每一步屬於哪條軌跡、在軌跡裡第幾步、往後實際拿到多少 reward。

    🩸 future return 要從 `rew` 沿軌跡往後加，不能用 `batch.ret` ——
    `ret = adv + old_value`，拿它當 ground truth 是循環論證（§93.2）。
    `gamma=1.0` 所以就是後綴和。
    """
    starts = batch.traj_start
    n = batch.n_steps
    ends = np.append(starts[1:], n)
    traj_of = np.searchsorted(starts, np.arange(n), side="right") - 1
    t_in = np.arange(n) - starts[traj_of]
    length = (ends - starts)[traj_of]
    fut = np.zeros(n, dtype=np.float64)
    for s, e in zip(starts, ends):
        fut[s:e] = np.cumsum(batch.rew[s:e][::-1])[::-1]
    margin = batch.traj_cash[:, 0] - batch.traj_cash[:, 1]
    return traj_of, t_in, length, fut, margin


class ExpertProbe(VecRollout):
    """跟 `VecRollout` 一模一樣，只是每一步多問一次 reference 該怎麼做。

    🩸 用 subclass 而不是改 `harness/ppo_rollout.py` —— 那支 PPO 訓練在用，
    加旗標就多一條會影響訓練的路徑。這裡只覆寫 `_policy_batch`，先呼叫
    `super()`（軌跡照原樣收），再對同一個 `obs` 問一次 expert。

    🩸 **unit 的順序**：`_policy_batch` 和這裡都走 `C.encode_units(obs, config)`，
    `plan["plan"][i]` 也是照那個順序（見 `harness/rollout.Recorder.record`）。
    對齊不能靠假設 —— `main` 收尾時會逐格比對 `unit_pos`，對不上就拋錯。
    """

    def __init__(self, *args, experts=(), **kw):
        super().__init__(*args, **kw)
        #: [(名字, act 函式, params)]。可以一次問多個 —— rollout 只跑一次，
        #: 每個 reference 看到的就一定是同一批盤面。
        self.experts = list(experts)
        self.rows = {}

    def _policy_batch(self, items, collect=False):
        out = super()._policy_batch(items, collect=collect)
        if collect and self.experts:
            for ei, p, obs, cfg in items:
                self.rows.setdefault((ei, p), []).append(self._ask(obs, cfg))
        return out

    def _ask(self, obs, config):
        """每個 reference 對這個盤面每個 unit 的答案：(終點格, 到了做什麼)。

        ⚠️ 認不得的動作記 -1 而不是跳過 —— 跳過會讓 unit 的位置對不上。
        """
        positions, _feats = C.encode_units(obs, config)
        ans = {}
        for name, fn, params in self.experts:
            with silenced():
                _act, plan = fn(obs, config, params, return_plan=True)
            board = plan["board"]
            tgt = np.full(len(positions), -1, dtype=np.int64)
            op = np.full(len(positions), -1, dtype=np.int64)
            for i in range(len(positions)):
                entry = plan["plan"][i] if i < len(plan["plan"]) else None
                if entry is None:
                    # 閒置：reference 說「留在原地什麼都不做」。這是答案，
                    # 不是缺答案。
                    dest = (int(positions[i][0]), int(positions[i][1]))
                    final = ["PASS"]
                else:
                    dest, final = entry
                tgt[i] = C.target_index(dest[0], dest[1], board)
                enc = C.encode_unit_action(final)
                if enc is not None:
                    op[i] = enc[0]
            ans[name] = (tgt, op)
        return np.asarray(positions, dtype=np.int64), ans

    def expert_arrays(self):
        """照 `run()` 交出 `trajs` 的順序攤平。

        回傳 `(unit_pos, {名字: (tgt, op)})`。
        """
        keys = [(ei, p) for ei in range(self.n_envs) for p in range(2)
                if self._ours(ei, p) and self.rows.get((ei, p))]
        pos = []
        acc = {name: ([], []) for name, _fn, _p in self.experts}
        for k in keys:
            for p_arr, ans in self.rows[k]:
                pos.append(p_arr)
                for name, (tgt, op) in ans.items():
                    acc[name][0].append(tgt)
                    acc[name][1].append(op)
        return (np.concatenate(pos),
                {n: (np.concatenate(t), np.concatenate(o))
                 for n, (t, o) in acc.items()})


def immediate_action(unit_pos, tgt_top1, op_top1, board=None):
    """argmax 解碼之後，這一步**實際送進引擎**的 unit 動作。

    規則抄自 `harness/ppo_rollout._policy_batch`：還沒站到目標格就往那裡走一步
    （`gen0.step_toward`：先對齊 x 再對齊 y），站到了才做 op。所以

      - unit 還沒走到的時候，op head 的 argmax 這一步對引擎沒有作用；
      - target 換到同一個方向的另一格時，這一步走的方向不變。

    回傳 `(動作碼, 是否已經站在目標格)`。動作碼 0~3 是 EAST/WEST/SOUTH/NORTH，
    100+op 是到了要做的事。

    ⚠️ 這只量**這一步**。target 改了但方向沒改的 unit，後面幾步還是會分岔 ——
    所以這個數字是「行為差異」的下界，不是「沒有影響」的證明。
    """
    board = board or C.BOARD_SIZE
    x = unit_pos[:, 0].astype(np.int64)
    y = unit_pos[:, 1].astype(np.int64)
    tx, ty = tgt_top1 % board, tgt_top1 // board
    code = np.where(tx > x, 0, np.where(tx < x, 1, np.where(ty > y, 2, 3)))
    arrived = (tx == x) & (ty == y)
    return np.where(arrived, 100 + op_top1.astype(np.int64), code), arrived


def cell_dist(a, b, board=None):
    """兩個 target 格子的曼哈頓距離。"""
    board = board or C.BOARD_SIZE
    return (np.abs(a % board - b % board)
            + np.abs(a // board - b // board))


def expert_cross(f, name, ch, gap_old, a_top1, b_top1, e_idx):
    """policy 有沒有改 × reference 同不同意，以及分 gap 桶的翻轉率。

    ⚠️ `cma1-g50-wt` 是 reference，不是 ground truth。「disagree」只代表
    兩邊選的不一樣，不代表網路錯。
    """
    ok = e_idx >= 0                     # 認得出動作的才算得了同不同意
    ch, gap_old, a1, b1, e = (ch[ok], gap_old[ok], a_top1[ok],
                              b_top1[ok], e_idx[ok])
    agree = a1 == e                     # ckpt-00140 跟 reference 一樣嗎
    n = len(ch)
    print(f"\n== {name}：policy movement × reference ==", file=f)
    print(f"  可比對的 decision {n:,}（認不得動作而排除 "
          f"{int((~ok).sum()):,}）", file=f)
    print(f"  {'':<18}{'expert agree':>16}{'expert disagree':>18}"
          f"{'合計':>12}", file=f)
    for lab, sel in (("policy unchanged", ~ch), ("policy changed", ch)):
        print(f"  {lab:<18}{int((sel & agree).sum()):>16,}"
              f"{int((sel & ~agree).sum()):>18,}{int(sel.sum()):>12,}", file=f)
    print(f"  {'合計':<18}{int(agree.sum()):>16,}"
          f"{int((~agree).sum()):>18,}{n:>12,}", file=f)
    print(f"  reference disagreement rate  {100 * (~agree).mean():.2f}%",
          file=f)
    print(f"  flip rate | agree            {100 * ch[agree].mean():.2f}%",
          file=f)
    print(f"  flip rate | disagree         {100 * ch[~agree].mean():.2f}%",
          file=f)

    edges = [0.05, 0.2, 0.5, 0.9]
    for tag, sel in (("disagree", ~agree), ("agree（對照）", agree)):
        print(f"\n  {name} / reference {tag}，按舊 policy 的 gap 分桶", file=f)
        print(f"    {'gap 桶':<16}{'n':>10}{'flip':>10}{'unchanged':>12}",
              file=f)
        idx = np.digitize(gap_old, edges)
        for b in range(len(edges) + 1):
            s = (idx == b) & sel
            if not s.any():
                continue
            lo = "-inf" if b == 0 else f"{edges[b - 1]:g}"
            hi = "inf" if b == len(edges) else f"{edges[b]:g}"
            print(f"    [{lo}, {hi})".ljust(20)
                  + f"{int(s.sum()):>10,}{100 * ch[s].mean():>9.2f}%"
                  + f"{100 * (~ch[s]).mean():>11.2f}%", file=f)

    hi_gap = gap_old > 0.9
    print(f"\n  {name} 三個關鍵數字", file=f)
    print(f"    A 全部 decision 的 flip rate           "
          f"{100 * ch.mean():.2f}%   (n={n:,})", file=f)
    print(f"    B reference disagree 的 flip rate      "
          f"{100 * ch[~agree].mean():.2f}%   (n={int((~agree).sum()):,})",
          file=f)
    s = ~agree & hi_gap
    print(f"    C disagree 且舊 gap>0.9 的 flip rate   "
          f"{100 * ch[s].mean():.2f}%   (n={int(s.sum()):,})", file=f)
    s = agree & hi_gap
    print(f"    對照 agree 且舊 gap>0.9                "
          f"{100 * ch[s].mean():.2f}%   (n={int(s.sum()):,})", file=f)
    s = ~agree & (gap_old < 0.05)
    print(f"    對照 disagree 且舊 gap<0.05            "
          f"{100 * ch[s].mean():.2f}%   (n={int(s.sum()):,})", file=f)

    # 翻掉的那些，是往 reference 靠過去還是走開？
    print(f"\n  {name}：更新的方向（相對 reference）", file=f)
    print(f"    ckpt 舊 跟 reference 一致  {100 * agree.mean():.2f}%", file=f)
    print(f"    ckpt 新 跟 reference 一致  {100 * (b1 == e).mean():.2f}%",
          file=f)
    fixed = int((ch & ~agree & (b1 == e)).sum())
    broke = int((ch & agree & (b1 != e)).sum())
    print(f"    flip 之後變成一致（原本不一致）{fixed:,}", file=f)
    print(f"    flip 之後變成不一致（原本一致）{broke:,}", file=f)
    print(f"    其餘 flip（本來就不一致、改完還是不一致）"
          f"{int(ch.sum()) - fixed - broke:,}", file=f)


GAP_EDGES = [0.05, 0.2, 0.5, 0.9]


def gap_buckets(gap):
    """回傳 [(標籤, 布林遮罩)]，桶界固定成 GAP_EDGES。"""
    idx = np.digitize(gap, GAP_EDGES)
    out = []
    for b in range(len(GAP_EDGES) + 1):
        lo = "-inf" if b == 0 else f"{GAP_EDGES[b - 1]:g}"
        hi = "inf" if b == len(GAP_EDGES) else f"{GAP_EDGES[b]:g}"
        out.append((f"[{lo}, {hi})", idx == b))
    return out


def prob_move(f, name, gap_a, pa1, pa2, pb_a1, pb_a2, b_p1, b_p2, ch):
    """argmax 沒翻的地方，機率往哪裡動了。

    `pb_a1` / `pb_a2` 是**舊網路的 top1 / top2 這兩個動作**在新網路下的機率
    —— 追的是同一組動作，不是新網路自己的 top1 / top2。兩個都報。
    """
    print(f"\n== {name}：機率有沒有移動（按 ckpt 舊的 top1-top2 gap 分桶）==",
          file=f)
    print(f"  {'gap 桶':<14}{'n':>9}{'flip':>8}"
          f"{'P(舊top1) 舊→新':>22}{'Δ':>9}"
          f"{'P(舊top2) 舊→新':>22}{'Δ':>9}"
          f"{'同組 gap 舊→新':>20}{'新自己的 gap':>14}", file=f)
    for lab, s in gap_buckets(gap_a):
        if not s.any():
            continue
        d1 = (pb_a1 - pa1)[s].mean()
        d2 = (pb_a2 - pa2)[s].mean()
        print(f"  {lab:<14}{int(s.sum()):>9,}{100 * ch[s].mean():>7.2f}%"
              f"{pa1[s].mean():>11.4f}{pb_a1[s].mean():>11.4f}{d1:>+9.4f}"
              f"{pa2[s].mean():>11.4f}{pb_a2[s].mean():>11.4f}{d2:>+9.4f}"
              f"{gap_a[s].mean():>10.4f}{(pb_a1 - pb_a2)[s].mean():>10.4f}"
              f"{(b_p1 - b_p2)[s].mean():>14.4f}", file=f)
    print(f"\n  {'gap 桶':<14}{'ΔP(舊top1)<0 的比例':>22}"
          f"{'ΔP(舊top2)>0 的比例':>22}{'|ΔP(舊top1)| 中位':>20}", file=f)
    for lab, s in gap_buckets(gap_a):
        if not s.any():
            continue
        print(f"  {lab:<14}{100 * ((pb_a1 - pa1)[s] < 0).mean():>21.2f}%"
              f"{100 * ((pb_a2 - pa2)[s] > 0).mean():>21.2f}%"
              f"{np.median(np.abs((pb_a1 - pa1)[s])):>20.4f}", file=f)


#: 新排名的分桶（1-based）。11+ 對 op（44 個動作）和 target（100 格）都夠用。
RANK_BINS = ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"),
             (6, 10, "6-10"), (11, 10 ** 9, "11+"))


def rank_transition(f, title, rows, mask=None):
    """舊排名的那幾個動作，在新網路排到哪裡去。

    `rows` 是 `[(標籤, 新排名陣列)]`，排名是 1-based。

    🩸 排名用「有幾個動作的機率嚴格大於它」算，所以完全同分的動作會拿到同一
    個排名。合法動作之間剛好同分很少見，但不是不可能。
    """
    print(f"\n  {title}", file=f)
    head = "".join(lab.rjust(9) for _lo, _hi, lab in RANK_BINS)
    print(f"    {'新排名':<14}{head}{'n':>12}", file=f)
    for lab, r in rows:
        rr = r if mask is None else r[mask]
        if not len(rr):
            continue
        cells = "".join(
            f"{100 * ((rr >= lo) & (rr <= hi)).mean():8.2f}%"
            for lo, hi, _l in RANK_BINS)
        print(f"    {lab:<14}{cells}{len(rr):>12,}", file=f)


def swap_stats(f, r_a1, r_a2, r_b1, mask=None):
    """top1 / top2 是不是就地交換，還是被第三個動作插進來。

    `r_a1` / `r_a2`：舊 top1 / top2 在**新**網路的排名。
    `r_b1`：新 top1 在**舊**網路的排名 —— 反方向才看得出新的 argmax 是從
    哪裡來的（舊的第 2 名，還是原本排不上的動作）。
    """
    m = slice(None) if mask is None else mask
    a1, a2, b1 = r_a1[m], r_a2[m], r_b1[m]
    n = len(a1)
    if not n:
        return
    flip = a1 != 1
    print(f"    top1 -> top2                {100 * (a1 == 2).mean():6.2f}%",
          file=f)
    print(f"    top2 -> top1                {100 * (a2 == 1).mean():6.2f}%",
          file=f)
    print(f"    top1 留在 top3              {100 * (a1 <= 3).mean():6.2f}%",
          file=f)
    print(f"    top1 掉出 top3              {100 * (a1 > 3).mean():6.2f}%",
          file=f)
    print(f"    純交換（top1<->top2 同時）  "
          f"{100 * ((a1 == 2) & (a2 == 1)).mean():6.2f}%", file=f)
    if flip.any():
        print(f"    argmax 翻掉的 {int(flip.sum()):,} 個裡：", file=f)
        print(f"      是純交換                {100 * ((a1 == 2) & (a2 == 1))[flip].mean():6.2f}%",
              file=f)
        print(f"      新 top1 原本排第 2      {100 * (b1[flip] == 2).mean():6.2f}%",
              file=f)
        print(f"      新 top1 原本排 3~5      "
              f"{100 * ((b1[flip] >= 3) & (b1[flip] <= 5)).mean():6.2f}%",
              file=f)
        print(f"      新 top1 原本排 6 以後   {100 * (b1[flip] > 5).mean():6.2f}%",
              file=f)
        print(f"      舊 top1 掉到 top3 之外  {100 * (a1[flip] > 3).mean():6.2f}%",
              file=f)


def ref_move(f, name, ref, gap_a, e_idx, a_top1, ch, pa1, pb_a1,
             p_ref_a, p_ref_b, rank_a, rank_b, traj):
    """機率有沒有往 reference 的動作移動。分 agree / disagree、分 gap 桶。"""
    ok = e_idx >= 0
    agree = (a_top1 == e_idx) & ok
    dis = (~agree) & ok
    print(f"\n== {name} × {ref}：機率往 reference 移動了嗎 ==", file=f)
    for tag, sel in (("disagree", dis), ("agree（對照）", agree)):
        print(f"\n  {tag}", file=f)
        print(f"    {'gap 桶':<14}{'n':>9}{'flip':>8}"
              f"{'ΔP(舊top1)':>12}{'ΔP(ref)':>10}"
              f"{'P(ref) 舊→新':>18}{'ref 排名 舊→新(中位)':>22}"
              f"{'ΔP(ref)>0':>11}", file=f)
        for lab, s0 in gap_buckets(gap_a):
            s = s0 & sel
            if not s.any():
                continue
            print(f"    {lab:<14}{int(s.sum()):>9,}"
                  f"{100 * ch[s].mean():>7.2f}%"
                  f"{(pb_a1 - pa1)[s].mean():>+12.4f}"
                  f"{(p_ref_b - p_ref_a)[s].mean():>+10.4f}"
                  f"{p_ref_a[s].mean():>9.4f}{p_ref_b[s].mean():>9.4f}"
                  f"{np.median(rank_a[s]):>11.1f}{np.median(rank_b[s]):>11.1f}"
                  f"{100 * ((p_ref_b - p_ref_a)[s] > 0).mean():>10.2f}%",
                  file=f)

    # 重點那一群：有把握、而且 reference 不同意。
    s = dis & (gap_a > 0.9)
    if not s.any():
        return
    d1 = (pb_a1 - pa1)[s]
    dr = (p_ref_b - p_ref_a)[s]
    close = d1 - dr                       # 兩者的差距每 60 輪縮小多少
    print(f"\n  ★ {name} / {ref}：舊 gap>0.9 且 disagree（n={int(s.sum()):,}）",
          file=f)
    print(f"    ΔP(舊 top1)      平均 {d1.mean():+.5f}   中位 "
          f"{np.median(d1):+.5f}   下降的比例 {100 * (d1 < 0).mean():.2f}%",
          file=f)
    print(f"    ΔP(reference)    平均 {dr.mean():+.5f}   中位 "
          f"{np.median(dr):+.5f}   上升的比例 {100 * (dr > 0).mean():.2f}%",
          file=f)
    print(f"    P(舊 top1)  {pa1[s].mean():.4f} -> {pb_a1[s].mean():.4f}"
          f"     P(reference) {p_ref_a[s].mean():.5f} -> "
          f"{p_ref_b[s].mean():.5f}", file=f)
    print(f"    reference 的排名（0=argmax）中位 {np.median(rank_a[s]):.0f}"
          f" -> {np.median(rank_b[s]):.0f}，平均 {rank_a[s].mean():.2f}"
          f" -> {rank_b[s].mean():.2f}", file=f)
    # 🩸 434,486 個 decision 不是 434,486 個獨立樣本 —— 同一局同一步的 unit
    # 高度相關。頭條數字用軌跡當單位再算一次標準誤。
    per = np.array([d1[traj[s] == t].mean() for t in np.unique(traj[s])
                    if (traj[s] == t).any()])
    se = per.std(ddof=1) / np.sqrt(len(per))
    print(f"    以軌跡為單位（n={len(per)}）ΔP(舊 top1) {per.mean():+.5f}"
          f" ± {se:.5f}（1 SE）", file=f)
    gap_now = (pa1 - p_ref_a)[s].mean()
    rate = -close.mean()
    print(f"    現在 P(舊top1) - P(ref) = {gap_now:.4f}，這 60 輪縮小 "
          f"{rate:+.5f}", file=f)
    if rate > 0:
        print(f"    注意 線性外推（假設速率不變，未驗證）：還要約 "
              f"{60 * gap_now / rate:,.0f} 輪才會跨過 argmax 邊界", file=f)
    else:
        print("    速率是往反方向（差距擴大），沒有跨過去的軌道", file=f)


def compare(tag, changed, rows, f):
    """changed / unchanged 兩組的並排統計。`rows` 是 (名稱, 陣列, 小數位)。"""
    m, u = changed, ~changed
    print(f"\n== {tag} ==  changed {m.sum():,} / {len(m):,} "
          f"= {100 * m.mean():.2f}%", file=f)
    print(f"  {'量':<24}{'changed':>12}{'unchanged':>12}{'差':>12}", file=f)
    for name, arr, d in rows:
        a, b = float(arr[m].mean()), float(arr[u].mean())
        print(f"  {name:<24}{a:>12.{d}f}{b:>12.{d}f}{a - b:>+12.{d}f}", file=f)
    for name, arr, d in rows:
        if name.startswith("adv") or name.startswith("|adv"):
            a, b = float(np.median(arr[m])), float(np.median(arr[u]))
            print(f"  {name + ' 中位':<24}{a:>12.{d}f}{b:>12.{d}f}"
                  f"{a - b:>+12.{d}f}", file=f)


def bucket_rate(tag, key, changed, edges, f):
    """按 `key` 分桶，印每桶的翻轉率。"""
    print(f"\n  翻轉率 by {tag}", file=f)
    idx = np.digitize(key, edges)
    for b in range(len(edges) + 1):
        s = idx == b
        if not s.any():
            continue
        lo = "-inf" if b == 0 else f"{edges[b - 1]:g}"
        hi = "inf" if b == len(edges) else f"{edges[b]:g}"
        print(f"    [{lo}, {hi})".ljust(24)
              + f"n={int(s.sum()):>9,}  {100 * changed[s].mean():6.2f}%",
              file=f)


def main(argv=None):
    # 🩸 只取第一段：docstring 裡的 🩸 / ⚠️ 在 cp950 主控台印不出來，
    # `--help` 會拋 UnicodeEncodeError（跟 §96.2 的 `%` 同一類問題）。
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", default="model/artifacts/phi-200/ckpt-00140.pt",
                    help="舊的 checkpoint")
    ap.add_argument("--b", default="model/artifacts/phi-200/ckpt-00200.pt",
                    help="新的 checkpoint")
    ap.add_argument("--driver", default="b", choices=("a", "b"),
                    help="哪一個 checkpoint 取樣驅動軌跡（§92 用的是 200）")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=950_000,
                    help="避開訓練用過的 900,000~903,184")
    ap.add_argument("--opponent", default="config/params/cma5-g175.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.998)
    ap.add_argument("--phi", default="assets")
    ap.add_argument("--phi-weight", type=float, default=1e-4)
    ap.add_argument("--recognise", default="strict")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--expert", default="",
                    help="reference agent（例如 config/params/cma1-g50-wt.json）"
                         "。給了就在同一批盤面上多做一次交叉統計")
    ap.add_argument("--out", default="temp/changed.npz")
    ap.add_argument("--report", default="temp/changed.txt")
    args = ap.parse_args(argv)

    dev = args.device if torch.cuda.is_available() else "cpu"
    net_a, it_a = load_ckpt(args.a, dev)
    net_b, it_b = load_ckpt(args.b, dev)
    driver = net_b if args.driver == "b" else net_a
    print(f"  a = {args.a}（it {it_a}）")
    print(f"  b = {args.b}（it {it_b}）")
    print(f"  驅動軌跡的是 {args.driver}，{args.games} 局 vs {args.opponent}")

    opp, _names = load_league(args.opponent)
    experts = []
    if args.expert:
        from harness.rollout import _load                 # noqa: PLC0415
        for spec in args.expert.split(","):
            spec = spec.strip()
            if not spec:
                continue
            fn, params = _load(spec)
            experts.append((Path(spec).stem, fn, dict(params)))
        print(f"  reference = {', '.join(n for n, _f, _p in experts)}"
              "（每一步多問一次，rollout 會變慢）")
    t0 = time.perf_counter()
    vec = ExpertProbe(driver, n_envs=args.games, seed0=args.seed0, device=dev,
                      opponent=opp, phi=args.phi, recognise=args.recognise,
                      experts=experts)
    steps, _cash, trajs = vec.run(collect=True)
    print(f"  rollout {time.perf_counter() - t0:.1f}s  "
          f"{steps:,} 步  {len(trajs)} 條軌跡")

    batch = ppo.RolloutBatch(trajs, gamma=args.gamma, lam=args.lam,
                             zero_sum=True, phi_weight=args.phi_weight)
    del trajs
    refs = {}
    if experts:
        e_pos, refs = vec.expert_arrays()
        # 🩸 對齊要驗，不能假設。錯位不會報錯，只會安靜地把 A 的 reference
        # 答案配到 B 的決策上，整張交叉表就是雜訊。
        if e_pos.shape != batch.unit_pos.shape or not np.array_equal(
                e_pos, batch.unit_pos.astype(np.int64)):
            raise AssertionError(
                f"reference 的 unit 順序跟 batch 對不上："
                f"{e_pos.shape} vs {batch.unit_pos.shape}")
        print(f"  reference 對齊驗過：{len(e_pos):,} 個 unit 的位置逐格相同")
    t0 = time.perf_counter()
    # 三次前向，順序有相依：
    #   1. 新網路（先跑）—— 拿到它的 top1，才能問「新的 argmax 在舊網路排第幾」
    #   2. 舊網路 —— 量 reference 和新 top1 在舊網路下的機率／排名
    #   3. 新網路（再一次）—— 量舊網路 top1/top2/top3 在新網路的排名
    # 一次約 2.5 秒，比 rollout 便宜兩個數量級，不值得為了省一次而繞。
    gather_ref = {name: (op, tgt) for name, (tgt, op) in refs.items()}
    pre = head_stats(net_b, batch, dev, args.chunk, gather=gather_ref)
    gather_a = dict(gather_ref)
    gather_a["b1"] = (pre["op_top1"], pre["tgt_top1"])
    sa = head_stats(net_a, batch, dev, args.chunk, gather=gather_a)
    # 新網路：除了 reference，還要量**舊網路的 top1/top2/top3 這三個動作**在
    # 新網路下的機率和排名 —— argmax 沒翻的時候，機率往哪裡動只有這樣看得到。
    gather_b = dict(gather_ref)
    for r in (1, 2, 3):
        gather_b[f"a{r}"] = (sa[f"op_top{r}"], sa[f"tgt_top{r}"])
    sb = head_stats(net_b, batch, dev, args.chunk, gather=gather_b)
    if not (np.array_equal(pre["op_top1"], sb["op_top1"])
            and np.array_equal(pre["tgt_top1"], sb["tgt_top1"])):
        raise AssertionError("同一個網路兩次前向的 argmax 不一致")
    print(f"  三次前向 {time.perf_counter() - t0:.1f}s  "
          f"{len(sa['op_top1']):,} 個 unit decision")

    traj_of, t_in, length, fut, margin = traj_tables(batch)
    us = batch.unit_step                      # 每個 unit 屬於第幾步
    u_adv = batch.adv[us]
    u_fut = fut[us]
    step_pos = t_in / np.maximum(length - 1, 1)
    u_pos = step_pos[us]
    u_traj = traj_of[us]
    u_margin = margin[u_traj]

    ch_op = sa["op_top1"] != sb["op_top1"]
    ch_tg = sa["tgt_top1"] != sb["tgt_top1"]
    ch_any = ch_op | ch_tg
    gap_a_op = sa["op_p1"] - sa["op_p2"]
    gap_b_op = sb["op_p1"] - sb["op_p2"]
    gap_a_tg = sa["tgt_p1"] - sa["tgt_p2"]
    gap_b_tg = sb["tgt_p1"] - sb["tgt_p2"]
    act_a, arr_a = immediate_action(batch.unit_pos, sa["tgt_top1"],
                                    sa["op_top1"])
    act_b, arr_b = immediate_action(batch.unit_pos, sb["tgt_top1"],
                                    sb["op_top1"])
    ch_act = act_a != act_b

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, unit_step=us, adv=u_adv, fut=u_fut, pos=u_pos,
        traj=u_traj, margin=u_margin, step_adv=batch.adv, step_rew=batch.rew,
        traj_start=batch.traj_start, traj_cash=batch.traj_cash,
        t_in=t_in, length=length,
        **{f"a_{k}": v for k, v in sa.items()},
        **{f"b_{k}": v for k, v in sb.items()},
        **{f"e_{n}_{k}": v for n, (tgt, op) in refs.items()
           for k, v in (("tgt", tgt), ("op", op))})

    with open(args.report, "w", encoding="utf-8") as f:
        print(f"# argmax changed：{args.a} -> {args.b}", file=f)
        print(f"# 驅動 {args.driver}、{args.games} 局、seed0 {args.seed0}、"
              f"對手 {args.opponent}", file=f)
        print(f"# {batch.n_steps:,} 步、{len(ch_op):,} 個 unit decision、"
              f"{len(margin)} 條軌跡", file=f)
        print(f"# explained_var {batch.explained_variance():+.4f}  "
              f"最終差額平均 {margin.mean():,.0f}", file=f)
        print(f"\n翻轉率  op {100 * ch_op.mean():.2f}%  "
              f"target {100 * ch_tg.mean():.2f}%  "
              f"任一 {100 * ch_any.mean():.2f}%", file=f)

        for tag, ch, ga, gb, pa, pb in (
                ("op head", ch_op, gap_a_op, gap_b_op,
                 sa["op_p1"], sb["op_p1"]),
                ("target head", ch_tg, gap_a_tg, gap_b_tg,
                 sa["tgt_p1"], sb["tgt_p1"])):
            compare(tag, ch, [
                ("advantage", u_adv, 4),
                ("|advantage|", np.abs(u_adv), 4),
                ("episode 位置 t/T", u_pos, 4),
                ("absolute step", t_in[us].astype(float), 1),
                ("top-1 P 舊", pa, 4),
                ("top-1 P 新", pb, 4),
                ("top1-top2 gap 舊", ga, 4),
                ("top1-top2 gap 新", gb, 4),
                ("實際 future return", u_fut, 4),
                ("該局最終差額", u_margin, 0),
            ], f)
            bucket_rate("|advantage|", np.abs(u_adv), ch,
                        [0.1, 0.25, 0.5, 1.0], f)
            bucket_rate("top1-top2 gap（舊）", ga, ch,
                        [0.05, 0.2, 0.5, 0.9], f)
            bucket_rate("episode 位置", u_pos, ch, [0.2, 0.4, 0.6, 0.8], f)
            print("\n  接近 tie 的比例（舊 checkpoint 的 gap）", file=f)
            for thr in (0.01, 0.05, 0.1, 0.2, 0.5):
                print(f"    gap < {thr:<6}"
                      f"changed {100 * (ga[ch] < thr).mean():6.2f}%   "
                      f"unchanged {100 * (ga[~ch] < thr).mean():6.2f}%",
                      file=f)

        # 接近 tie 的決策是不是也剛好是 advantage 小的決策？如果是，「改的都是
        # 平手的」就等於「改的都不重要」；如果不是，兩件事要分開講。
        print("\n== 接近 tie 的決策，advantage 有比較小嗎 ==", file=f)
        print(f"  {'gap 桶（舊, target head）':<26}{'n':>10}"
              f"{'|adv| 平均':>12}{'翻轉率':>10}", file=f)
        edges = [0.05, 0.2, 0.5, 0.9]
        idx = np.digitize(gap_a_tg, edges)
        for b in range(len(edges) + 1):
            s = idx == b
            if not s.any():
                continue
            lo = "-inf" if b == 0 else f"{edges[b - 1]:g}"
            hi = "inf" if b == len(edges) else f"{edges[b]:g}"
            print(f"  [{lo}, {hi})".ljust(28)
                  + f"{int(s.sum()):>10,}{np.abs(u_adv[s]).mean():>12.4f}"
                  + f"{100 * ch_tg[s].mean():>9.2f}%", file=f)
        print(f"  corr(|adv|, gap 舊 target) "
              f"{np.corrcoef(np.abs(u_adv), gap_a_tg)[0, 1]:+.4f}", file=f)

        # 行為層：argmax 變了不等於送進引擎的動作變了。
        print("\n== 行為層：這一步實際送進引擎的動作 ==", file=f)
        print(f"  站在目標格上（op 這一步才作數）  "
              f"{100 * arr_b.mean():.2f}%", file=f)
        print(f"  argmax 有變（任一 head）        "
              f"{100 * ch_any.mean():.2f}%", file=f)
        print(f"  這一步的動作有變                "
              f"{100 * ch_act.mean():.2f}%", file=f)
        print(f"  -> 每 100 個 argmax flip 只有 "
              f"{100 * ch_act.sum() / max(ch_any.sum(), 1):.1f} 個改到這一步",
              file=f)
        print(f"  op flip 裡站在目標格上的        "
              f"{100 * arr_b[ch_op].mean():.2f}%  "
              f"（沒站到的那一步 op 不進引擎）", file=f)
        moved = ~arr_a & ~arr_b
        print(f"  兩邊都還在路上的 unit，方向不同  "
              f"{100 * (act_a != act_b)[moved].mean():.2f}%  "
              f"(n={int(moved.sum()):,})", file=f)
        print(f"  |adv|  動作有變 {np.abs(u_adv[ch_act]).mean():.4f}   "
              f"動作沒變 {np.abs(u_adv[~ch_act]).mean():.4f}", file=f)

        # target flip 換去的格子有多遠 —— 換到隔壁格和換去半張地圖外，
        # 都算「argmax 改了」，但不是同一回事。
        d = cell_dist(sa["tgt_top1"], sb["tgt_top1"])
        d2 = cell_dist(sa["tgt_top1"], sa["tgt_top2"])
        print("\n== target flip 換到多遠 ==", file=f)
        print(f"  {'曼哈頓距離':<14}{'flip 換到的格子':>16}"
              f"{'對照：舊的 top1 vs top2':>26}", file=f)
        for lo, hi, lab in ((1, 1, "1 格"), (2, 2, "2 格"), (3, 4, "3~4 格"),
                            (5, 99, "5 格以上")):
            a = ((d[ch_tg] >= lo) & (d[ch_tg] <= hi)).mean()
            b = ((d2 >= lo) & (d2 <= hi)).mean()
            print(f"  {lab:<14}{100 * a:>15.2f}%{100 * b:>25.2f}%", file=f)
        print(f"  中位距離        {np.median(d[ch_tg]):>14.1f}"
              f"{np.median(d2):>25.1f}", file=f)

        # top-1 / top-2 對調，還是換上了原本排不上名的動作？
        print("\n== flip 是不是 top-1 / top-2 對調 ==", file=f)
        for name, ch, a1, a2, b1 in (
                ("op", ch_op, sa["op_top1"], sa["op_top2"], sb["op_top1"]),
                ("target", ch_tg, sa["tgt_top1"], sa["tgt_top2"],
                 sb["tgt_top1"])):
            print(f"  {name:<8}新的 top1 原本就是舊的 top2："
                  f"{100 * (b1[ch] == a2[ch]).mean():.2f}%", file=f)
        verbs = np.array([C.UNIT_OPS[i][0] for i in range(C.N_UNIT_OPS)])
        same_verb = verbs[sa["op_top1"][ch_op]] == verbs[sb["op_top1"][ch_op]]
        print(f"  op flip 留在同一個動詞（PLANT A -> PLANT B 這種）："
              f"{100 * same_verb.mean():.2f}%", file=f)

        # 步層：同一步的 unit 共用 advantage，所以 unit 層的 n 是灌水的。
        n_unit = np.bincount(us, minlength=batch.n_steps)
        step_ch = (np.bincount(us, weights=ch_any.astype(float),
                               minlength=batch.n_steps)
                   / np.maximum(n_unit, 1))
        ok = n_unit > 0
        print(f"\n== 步層（n = {int(ok.sum()):,} 步）==", file=f)
        for name, arr in (("|adv|", np.abs(batch.adv[ok])),
                          ("adv", batch.adv[ok]),
                          ("t/T", step_pos[ok])):
            c = np.corrcoef(step_ch[ok], arr)[0, 1]
            print(f"  corr(該步翻轉比例, {name})".ljust(34) + f"{c:+.4f}",
                  file=f)

        # 軌跡層：final margin 一條軌跡只有一個，不能當成每個 decision 一個。
        print(f"\n== 軌跡層（n = {len(margin)} 條）==", file=f)
        tr_rate = np.array([ch_any[u_traj == i].mean()
                            for i in range(len(margin))])
        print(f"  corr(該局翻轉比例, 最終差額) "
              f"{np.corrcoef(tr_rate, margin)[0, 1]:+.4f}", file=f)
        hi = tr_rate > np.median(tr_rate)
        print(f"  翻轉多的一半 最終差額 {margin[hi].mean():>12,.0f}", file=f)
        print(f"  翻轉少的一半 最終差額 {margin[~hi].mean():>12,.0f}", file=f)

        # argmax 沒翻的地方，機率往哪裡動了。
        print("\n\n######## 機率位移（argmax 之外）########", file=f)
        for name, gap, pa1, pa2, k1, k2, bp1, bp2, ch in (
                ("op head", gap_a_op, sa["op_p1"], sa["op_p2"],
                 "op_p_a1", "op_p_a2", sb["op_p1"], sb["op_p2"], ch_op),
                ("target head", gap_a_tg, sa["tgt_p1"], sa["tgt_p2"],
                 "tgt_p_a1", "tgt_p_a2", sb["tgt_p1"], sb["tgt_p2"], ch_tg)):
            prob_move(f, name, gap, pa1, pa2, sb[k1], sb[k2], bp1, bp2, ch)

        # rank transition：機率動了之後，動作的**名次**有沒有換位置。
        print("\n\n######## rank transition（140 -> 200）########", file=f)
        print("# 「舊 top1 -> 新排名 2」跟「舊 top2 -> 新排名 1」如果數量相當，"
              "就是就地交換；\n# 要是新 top1 來自舊的第 6 名以後，才是 mass 真的"
              "移到別的動作上。", file=f)
        for name, head in (("op head", "op"), ("target head", "tgt")):
            r1 = sb[f"{head}_rank_a1"] + 1
            r2 = sb[f"{head}_rank_a2"] + 1
            r3 = sb[f"{head}_rank_a3"] + 1
            rb1 = sa[f"{head}_rank_b1"] + 1
            print(f"\n== {name}：全部 decision ==", file=f)
            rank_transition(f, "舊 rank 1/2/3 在新網路排到哪", [
                ("old rank 1", r1), ("old rank 2", r2), ("old rank 3", r3)])
            swap_stats(f, r1, r2, rb1)

        for ref in refs:
            e_tgt, e_op = refs[ref]
            print(f"\n\n######## reference = {ref} ########", file=f)
            print("# reference 只是參照，不是 ground truth：disagree 只代表兩邊"
                  "選的不一樣，不代表網路錯。", file=f)
            expert_cross(f, "op head", ch_op, gap_a_op,
                         sa["op_top1"], sb["op_top1"], e_op)
            expert_cross(f, "target head", ch_tg, gap_a_tg,
                         sa["tgt_top1"], sb["tgt_top1"], e_tgt)
            ref_move(f, "op head", ref, gap_a_op, e_op, sa["op_top1"], ch_op,
                     sa["op_p1"], sb["op_p_a1"],
                     sa[f"op_p_{ref}"], sb[f"op_p_{ref}"],
                     sa[f"op_rank_{ref}"], sb[f"op_rank_{ref}"], u_traj)
            ref_move(f, "target head", ref, gap_a_tg, e_tgt, sa["tgt_top1"],
                     ch_tg, sa["tgt_p1"], sb["tgt_p_a1"],
                     sa[f"tgt_p_{ref}"], sb[f"tgt_p_{ref}"],
                     sa[f"tgt_rank_{ref}"], sb[f"tgt_rank_{ref}"], u_traj)

            # 重點那一群的 rank transition。
            for name, head, gap, e_idx in (("op head", "op", gap_a_op, e_op),
                                           ("target head", "tgt", gap_a_tg,
                                            e_tgt)):
                a1 = sa[f"{head}_top1"]
                m = (gap > 0.9) & (e_idx >= 0) & (a1 != e_idx)
                if not m.any():
                    continue
                r1 = sb[f"{head}_rank_a1"] + 1
                r2 = sb[f"{head}_rank_a2"] + 1
                r3 = sb[f"{head}_rank_a3"] + 1
                rb1 = sa[f"{head}_rank_b1"] + 1
                print(f"\n== {name} / {ref}：舊 gap>0.9 且 disagree"
                      f"（n={int(m.sum()):,}）==", file=f)
                rank_transition(f, "舊 rank 1/2/3 在新網路排到哪", [
                    ("old rank 1", r1), ("old rank 2", r2),
                    ("old rank 3", r3)], mask=m)
                rank_transition(f, "reference 那個動作的排名（舊 vs 新）", [
                    ("舊網路", sa[f"{head}_rank_{ref}"] + 1),
                    ("新網路", sb[f"{head}_rank_{ref}"] + 1)], mask=m)
                swap_stats(f, r1, r2, rb1, mask=m)

    # 🩸 主控台是 cp950，報告裡有它編不出來的字元就整支炸在最後一行
    # （§96.2 的 U+2212 是同一類）。檔案已經寫好了，印不出來不該失敗。
    text = open(args.report, encoding="utf-8").read()
    enc = getattr(sys.stdout, "encoding", "") or "utf-8"
    print(text.encode(enc, "replace").decode(enc, "replace"))
    print(f"  原始資料 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
