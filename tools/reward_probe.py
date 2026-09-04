"""現金版 vs 資產版 reward：哪一個在短視窗裡看得到最終結果（§83.8）。

    python -m tools.reward_probe --games 10 --workers 10
    python -m tools.reward_probe --a cma5-g175 --b dmitry-larko --games 20

## 在問什麼

PPO 的 advantage 是 GAE 的截斷和，等效視窗 `1/(1-gamma*lam)`：lam=0.95 是
19 步、0.998 是 200 步。前五輪非得把 lam 開到 0.998 才有效（§56）——
現金版 reward 在收穫之前一路是負的，真正的訊號在 240 步外。

資產版（§83.3）把投資記成資產，理論上短視窗就看得到。這支工具**不訓練**，
只拿現成的對局把兩種 reward 都算出來，量兩件事：

    1. Φ_t 對「剩下的 margin」（margin_T − margin_t）的相關。value head 要學
       的就是這個。§81.4 我當時回歸錯目標（用 margin_T），這裡是改對的。
    2. 從 t 起算 W 步的 reward 累積，跟「剩下的 margin」的相關係數。
       現金版在 W=20 接近 0 而資產版不是 -> lam 不用 0.998。

## 沒有算進去的

`TERMINAL_BONUS`（期末勝負的階梯函數）不算 —— 它跟視窗長度的問題無關，
放進來只會在最後 W 步製造一個假的相關。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.ppo import REWARD_SCALE                     # noqa: E402
from tools._quiet import silenced                      # noqa: E402

#: 作物格數換算成錢的匯率。`--plant-weight 0.01` 就是一格 $100。
PLANT_DOLLARS = 100.0

#: 表格的欄位順序，四種 reward 共用。
COLUMNS = ("現金版", "資產 strict", "資產 produce", "作物格")


def _play(job):
    """跑一局，回傳逐步的現金和 Φ。top-level 才 pickle 得動（Windows spawn）。"""
    spec_a, spec_b, seed, episode_steps = job
    with silenced():
        from kaggle_environments import make
    from eval.runner import build_agent
    from harness.ppo_rollout import count_plants
    from model.networth import asset_value

    a_slot = seed % 2                # 座位輪流，先手／後手的差異不要混進結果
    specs = [spec_a, spec_b] if a_slot == 0 else [spec_b, spec_a]
    cfg = {"seed": seed}
    if episode_steps:
        cfg["episodeSteps"] = episode_steps
    with silenced():
        env = make("kaggriculture", configuration=cfg, debug=False)
        env.run([build_agent(specs[0]), build_agent(specs[1])])

    ep = int(env.configuration.get("episodeSteps", 720))
    tpd = int(env.configuration.get("turnsPerDay", 24))
    me, opp = a_slot, 1 - a_slot
    keys = ("cash_a", "cash_b", "strict", "produce", "opp_strict", "plants")
    out = {k: [] for k in keys}
    for st in env.steps[:-1]:
        obs_me, obs_opp = st[me]["observation"], st[opp]["observation"]
        farms = obs_me["farms"]
        out["cash_a"].append(float(farms[me]["money"]))
        out["cash_b"].append(float(farms[opp]["money"]))
        out["strict"].append(asset_value(obs_me, recognise="strict",
                                         episode_steps=ep, turns_per_day=tpd))
        out["produce"].append(asset_value(obs_me, recognise="produce",
                                          episode_steps=ep, turns_per_day=tpd))
        out["opp_strict"].append(asset_value(obs_opp, recognise="strict",
                                             episode_steps=ep,
                                             turns_per_day=tpd))
        out["plants"].append(PLANT_DOLLARS * (count_plants(farms[me])
                                              - count_plants(farms[opp])))
    # 期末：現金用引擎給的 reward，Φ 是 0（賣不掉的庫存一分不值）。
    out["cash_a"].append(float(env.steps[-1][me]["reward"]))
    out["cash_b"].append(float(env.steps[-1][opp]["reward"]))
    for k in ("strict", "produce", "opp_strict", "plants"):
        out[k].append(0.0)
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def windowed(r, W):
    """`S[t] = sum(r[t:t+W])`，長度跟 `r` 一樣。用累積和算，O(T)。"""
    c = np.concatenate([[0.0], np.cumsum(r)])
    hi = np.minimum(np.arange(len(r)) + W, len(r))
    return c[hi] - c[:-1]


def corr(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def reward_series(game):
    """一局 -> {欄位名: 逐步 reward}。都是零和基礎 + 各自的 Φ shaping（γ=1）。"""
    margin = game["cash_a"] - game["cash_b"]
    r_cash = np.diff(margin) / REWARD_SCALE
    out = {"現金版": r_cash}
    for label, key in (("資產 strict", "strict"), ("資產 produce", "produce"),
                       ("作物格", "plants")):
        out[label] = r_cash + np.diff(game[key]) / REWARD_SCALE
    return out


def analyse(games, windows=(20, 50, 240), tpd=24):
    """`games` 是 `_play` 的回傳 list。印表，順便回傳 dict 給測試用。"""
    res = {}
    phi_cols = {"資產 strict": "strict", "資產 produce": "produce",
                "作物格 x$100": "plants", "資產 我−對手": None}
    day, phi, remain = [], {k: [] for k in phi_cols}, []
    for g in games:
        margin = g["cash_a"] - g["cash_b"]
        remain.append((margin[-1] - margin)[:-1])
        day.append(np.arange(len(margin) - 1) // tpd)
        for label, key in phi_cols.items():
            v = (g["strict"] - g["opp_strict"]) if key is None else g[key]
            phi[label].append(v[:-1])
    day = np.concatenate(day)
    remain = np.concatenate(remain)
    phi = {k: np.concatenate(v) for k, v in phi.items()}

    print(f"\n{len(games)} 局 × {len(remain) // len(games)} 步   "
          f"剩下的 margin sd = {remain.std():,.0f}")
    print("\nΦ_t 對「剩下的 margin」（margin_T − margin_t）的相關係數")
    print(f"  {'':<12}" + "".join(f"{k:>14}" for k in phi_cols))
    for name in ("全部", "day 0~9", "day 10~19", "day 20~29"):
        if name == "全部":
            m = np.ones(len(day), dtype=bool)
        else:
            d0 = int(name.split()[1].split("~")[0])
            m = (day >= d0) & (day < d0 + 10)
        row = {k: corr(phi[k][m], remain[m]) for k in phi_cols}
        res[f"phi_corr_{name}"] = row
        print(f"  {name:<12}" + "".join(f"{row[k]:>+14.3f}" for k in phi_cols))

    series, rems = {}, []
    for g in games:
        margin = g["cash_a"] - g["cash_b"]
        rems.append((margin[-1] - margin)[:-1] / REWARD_SCALE)
        for k, v in reward_series(g).items():
            series.setdefault(k, []).append(v)
    rem_all = np.concatenate(rems)

    print("\nW 步內的 reward 累積 對「剩下的 margin」的相關係數")
    print(f"  {'W':<8}" + "".join(f"{k:>14}" for k in COLUMNS))
    for W in list(windows) + [0]:
        label = "全程" if W == 0 else str(W)
        row = {k: corr(np.concatenate([windowed(v, W or len(v)) for v in vs]),
                       rem_all) for k, vs in series.items()}
        res[f"window_{label}"] = row
        print(f"  {label:<8}" + "".join(f"{row[k]:>+14.3f}" for k in COLUMNS))

    # 分十天看。§83.3 的機制主張是「前段把正確的投資記成扣分」，那要在
    # day 0~9 的平均 reward 上看得到；短視窗的訊號也是那一段最缺。
    d_step = np.concatenate([np.arange(len(v)) // tpd
                             for v in series[COLUMNS[0]]])
    win20 = {k: np.concatenate([windowed(v, 20) for v in vs])
             for k, vs in series.items()}
    flat = {k: np.concatenate(vs) for k, vs in series.items()}
    print("\n分十天：每步 reward 的平均 / W=20 的累積對「剩下的 margin」的相關")
    print(f"  {'day':<12}" + "".join(f"{k:>14}" for k in COLUMNS))
    for d0 in (0, 10, 20):
        m = (d_step >= d0) & (d_step < d0 + 10)
        means = {k: flat[k][m].mean() for k in COLUMNS}
        cs = {k: corr(win20[k][m], rem_all[m]) for k in COLUMNS}
        res[f"mean_day{d0}"] = means
        res[f"win20_day{d0}"] = cs
        print(f"  {d0}~{d0 + 9:<9}" + "".join(
            f"{means[k]:>+9.4f}/{cs[k]:>+5.2f}" for k in COLUMNS))

    # potential-based shaping 真正的賣點：它不改變最優策略，改的是 value head
    # 要學的目標。γ=1 時那個目標就是「剩下的 margin 減掉 Φ_t」。sd 掉得越多，
    # critic 的工作越輕。
    print("\nvalue head 的目標 sd（剩下的 margin − Φ_t，單位是錢）")
    print(f"  {'day':<12}{'現金版':>14}" + "".join(
        f"{k:>14}" for k in COLUMNS[1:]))
    tgt = {"現金版": remain,
           "資產 strict": remain - phi["資產 strict"],
           "資產 produce": remain - phi["資產 produce"],
           "作物格": remain - phi["作物格 x$100"]}
    for name, m in (("全部", np.ones(len(day), dtype=bool)),
                    ("day 0~9", (day < 10)),
                    ("day 10~19", (day >= 10) & (day < 20)),
                    ("day 20~29", (day >= 20))):
        res[f"target_sd_{name}"] = {k: float(v[m].std()) for k, v in tgt.items()}
        print(f"  {name:<12}" + "".join(
            f"{tgt[k][m].std():>14,.0f}" for k in COLUMNS))

    print("\n每一步 reward 的散佈（scaled，1.0 = $10,000）")
    for k in COLUMNS:
        v = np.concatenate(series[k])
        res[f"sd_{k}"] = float(v.std())
        print(f"  {k:<14} sd {v.std():>7.4f}   負的比例 "
              f"{100 * np.mean(v < 0):>5.1f}%   |r| 中位數 "
              f"{np.median(np.abs(v)):>7.4f}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="現金版 vs 資產版 reward 的訊號量")
    ap.add_argument("--a", default="config/params/cma5-g175.json")
    ap.add_argument("--b", default="dmitry-larko")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=770_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--episode-steps", type=int, default=0)
    ap.add_argument("--out", default="", help="把逐步序列存成 .npz")
    args = ap.parse_args(argv)

    # Windows 的主控台是 cp950，減號 U+2212 編不出來會讓整個 print 拋
    # UnicodeEncodeError（`tools/action_dist.py:133` 同一個坑）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):     # 不是真的 TextIOWrapper 就算了
            pass

    from eval.runner import load_spec
    spec_a, spec_b = load_spec(args.a), load_spec(args.b)
    jobs = [(spec_a, spec_b, args.seed0 + i, args.episode_steps)
            for i in range(args.games)]
    print(f"{spec_a['name']} vs {spec_b['name']}   {args.games} 局   "
          f"seed {args.seed0}~{args.seed0 + args.games - 1}")
    if args.workers > 1 and args.games > 1:
        with mp.Pool(min(args.workers, args.games)) as pool:
            games = pool.map(_play, jobs)
    else:
        games = [_play(j) for j in jobs]

    analyse(games)
    if args.out:
        np.savez_compressed(args.out, **{f"{k}_{i}": g[k]
                                         for i, g in enumerate(games)
                                         for k in g})
        print(f"\n逐步序列 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
