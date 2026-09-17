"""PPO 到底「不知道」還是「知道但沒學會」PLANT 之後要 WATER。

    python -m tools.water_probe --ckpt model/artifacts/phi-200/ckpt-00140.pt --games 8

作法：收一批帶 observation 的 rollout，對每一個 unit decision 讀出**它站的那一格**
的作物狀態（直接從 `batch.spatial` 用 `unit_pos` 取），再用 `argmax_changed.head_stats`
的 gather 機制拿到 WATER 這個動作的機率與排名。

🩸 不要在腳本裡自己開 RolloutPool（journal §97 卡死兩次），走單行程 VecRollout。
🩸 引擎：WHEAT 的加產窗口是 age 2~4 天，`crop_age = (day-planted_day)/max_yield_day`
   所以窗口在 crop_age ∈ [0.5, 1.0]。`consecutive_unwatered >= 2` 變雜草。
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tools._quiet import silenced                        # noqa: E402
with silenced():
    import torch
from harness.ppo_rollout import VecRollout, load_league   # noqa: E402
import contracts as C                                    # noqa: E402
from model import ppo                                    # noqa: E402
from tools.argmax_changed import load_ckpt, head_stats, traj_tables  # noqa: E402

SP = {n: i for i, n in enumerate(C.SPATIAL_CHANNELS)}
OPS = [f"{o}:{a}" if a else o for o, a in C.UNIT_OPS]
WATER = OPS.index("WATER")
CROP_CH = [SP[f"crop_{c}"] for c in C.CROP_ORDER]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default="model/artifacts/phi-200/ckpt-00140.pt")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=950_000)
    ap.add_argument("--opponent", default="config/params/cma5-g175.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", default="temp/water_probe.txt")
    args = ap.parse_args(argv)

    OUT = open(args.out, "w", encoding="utf-8")
    def P(*a):
        print(*a, file=OUT); OUT.flush(); print(*a)

    dev = args.device if torch.cuda.is_available() else "cpu"
    net, it = load_ckpt(args.ckpt, dev)
    opp, _ = load_league(args.opponent)
    P(f"ckpt {args.ckpt}（it {it}）  {args.games} 局 vs {args.opponent}")

    vec = VecRollout(net, n_envs=args.games, seed0=args.seed0, device=dev,
                     opponent=opp, phi="assets", recognise="strict")
    steps, _cash, trajs = vec.run(collect=True)
    batch = ppo.RolloutBatch(trajs, gamma=1.0, lam=0.998, zero_sum=True,
                             phi_weight=1e-4)
    del trajs
    U = len(batch.unit_pos)
    P(f"  {steps:,} 步   {U:,} 個 unit decision")

    st = head_stats(net, batch, dev, args.chunk,
                    gather={"water": (np.full(U, WATER, np.int64),
                                      np.zeros(U, np.int64))})
    us = batch.unit_step
    # 🩸 unit_pos 是 (x, y)：net.py:144 用的是 features[..., unit_pos[:,1], unit_pos[:,0]]。
    ux = batch.unit_pos[:, 0].astype(int); uy = batch.unit_pos[:, 1].astype(int)
    sp = batch.spatial
    g = lambda ch: sp[us, ch, uy, ux].astype(np.float64)

    has_crop = np.zeros(U, bool)
    for ch in CROP_CH:
        has_crop |= g(ch) > 0.5
    age = g(SP["crop_age"]); wtd = g(SP["crop_watered_today"]) > 0.5
    unw = g(SP["crop_unwatered"]); yf = g(SP["crop_yield_frac"])
    legal = batch.op_mask[np.arange(U), WATER]
    pw = st["op_p_water"]; rk = st["op_rank_water"]; top1 = st["op_top1"]
    chose = top1 == WATER

    traj_of, t_in, length, fut, margin = traj_tables(batch)
    adv = batch.adv[us]; ufut = fut[us]

    def block(tag, m):
        if m.sum() == 0:
            P(f"  {tag:34}  n=0"); return
        P(f"  {tag:34}{m.sum():>9,}{100*legal[m].mean():>9.1f}%"
          f"{pw[m].mean():>10.4f}{np.median(pw[m]):>10.4f}"
          f"{100*chose[m].mean():>9.1f}%{np.median(rk[m]):>7.0f}")

    ok = has_crop & ~wtd
    agree = (legal[ok].mean() if ok.sum() else float("nan"))
    P(f"  自我驗證：站在作物格且今天沒澆 -> WATER 合法率 {100*agree:.1f}%"
      f"（contracts.py:936 的規則是 kind==PLANT and not watered_today，應為 100%%）")
    if agree < 0.99:
        P("  ⚠️ 不到 100%%，座標或遮罩對應仍有問題，下面的數字不要採用")
    P("")
    P("=" * 96)
    P("  1. 按「這個 unit 站的那一格」分類")
    P("=" * 96)
    P(f"  {'狀態':34}{'n':>9}{'WATER合法':>10}{'P平均':>10}{'P中位':>10}{'選WATER':>10}{'排名':>7}")
    block("全部 unit decision", np.ones(U, bool))
    block("站在非作物格", ~has_crop)
    block("站在作物格", has_crop)
    block("  作物 今天已澆", has_crop & wtd)
    block("  作物 今天沒澆", has_crop & ~wtd)
    block("    且 WATER 合法", has_crop & ~wtd & legal)
    block("    且 age 在加產窗口 [0.5,1]", has_crop & ~wtd & legal & (age >= 0.5) & (age <= 1.0))
    block("    且 age 窗口前 (<0.5)", has_crop & ~wtd & legal & (age < 0.5))
    block("    且 快枯死 unwatered>=0.5", has_crop & ~wtd & legal & (unw >= 0.5))

    P("")
    P("=" * 96)
    P("  2. 「站在沒澆水的作物上、WATER 合法」時，policy 選了什麼")
    P("=" * 96)
    m = has_crop & ~wtd & legal
    cnt = np.bincount(top1[m], minlength=len(OPS))
    for i in np.argsort(-cnt)[:10]:
        if cnt[i] == 0:
            continue
        P(f"    {OPS[i]:<24}{cnt[i]:>9,}{100*cnt[i]/m.sum():>8.1f}%")

    P("")
    P("=" * 96)
    P("  2b. 曝光度：unit 有多常站在可以澆的作物上")
    P("=" * 96)
    ng = args.games
    P(f"    unit decision/局        {U/ng:>10,.0f}")
    P(f"    站在作物格/局            {has_crop.sum()/ng:>10,.0f}   {100*has_crop.mean():>5.1f}%")
    P(f"    站在沒澆的作物/局         {(has_crop&~wtd).sum()/ng:>10,.0f}   {100*(has_crop&~wtd).mean():>5.1f}%")
    P(f"    實際 WATER/局            {(chose).sum()/ng:>10,.0f}")

    HV = OPS.index("HARVEST")
    hvm = batch.op_mask[np.arange(U), HV]
    ch_hv = top1 == HV
    P("")
    P("=" * 96)
    P("  2c. 收成時機：HARVEST 合法時，policy 在什麼 crop_age 收")
    P("=" * 96)
    P(f"  {chr(39)}狀態{chr(39):34}{chr(39)}n{chr(39):>9}{chr(39)}選HARVEST{chr(39):>11}{chr(39)}age平均{chr(39):>10}{chr(39)}yield_frac{chr(39):>12}")
    for tag, mm in [("HARVEST 合法且站在作物上", hvm & has_crop),
                    ("  age < 0.5（加產窗口前）", hvm & has_crop & (age < 0.5)),
                    ("  age 0.5~1.0（窗口內）", hvm & has_crop & (age >= 0.5) & (age <= 1.0)),
                    ("  age > 1.0（過熟）", hvm & has_crop & (age > 1.0))]:
        if mm.sum() == 0:
            P(f"  {tag:34}  n=0"); continue
        P(f"  {tag:34}{mm.sum():>9,}{100*ch_hv[mm].mean():>11.1f}%"
          f"{age[mm].mean():>10.3f}{yf[mm].mean():>12.3f}")
    P("")
    P("=" * 96)
    P("  3. learning signal：同一批 state 裡，選 WATER 與沒選的比較")
    P("=" * 96)
    P(f"  {'':22}{'n':>9}{'advantage':>12}{'未來報酬':>12}")
    for tag, mm in [("選了 WATER", m & chose), ("沒選 WATER", m & ~chose)]:
        P(f"  {tag:22}{mm.sum():>9,}{adv[mm].mean():>+12.4f}{ufut[mm].mean():>+12.4f}")
    P("  🩸 advantage 是**每步一個**、同一步的 unit 共用，所以 unit 層的 n 不是"
      "有效樣本數；而且這是相關不是因果。")
    OUT.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
