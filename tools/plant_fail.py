"""PLANT 為什麼沒種成：逐回合比對「發出的 PLANT」與「真的長出來的作物」。

    python -m tools.plant_fail --a temp/specs/ppo-phi200-140.json --games 8

mask 規則是 `tile is None and seeds.get(item, 0) > 0`（contracts.py:931），
而 mask 用的是**該步開始前**的狀態 —— 同一回合多個 unit 種同一種作物時，
會一起看到同一份種子存量。這支就是要看失敗是不是集中在那種回合。
"""
from __future__ import annotations
import argparse, collections, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")


def _one(job):
    from eval.runner import build_agent, _quiet_make
    spec_a, spec_b, seed = job
    make = _quiet_make()
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    try:
        env.run([build_agent(spec_a), build_agent(spec_b)])
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    rows = []
    prev = None
    for t, step in enumerate(env.steps[:720]):
        try:
            farm = step[0]["observation"]["farms"][0]
            tiles = farm.get("tiles") or []
        except (KeyError, TypeError, IndexError):
            continue
        cur = {}
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    cur[(y, x)] = tile.get("planted_day")
        if prev is not None:
            # 🩸 同一格收成後立刻重種也算新的一株，所以要比 planted_day，
            #    不能只比 key 在不在。
            born = sum(1 for k, pd in cur.items()
                       if k not in prev or prev[k] != pd)
        else:
            born = 0
        act = step[0].get("action") if isinstance(step[0].get("action"), dict) else {}
        want = collections.Counter()
        for ua in [act.get("farmer"), *(act.get("hands") or [])]:
            if isinstance(ua, list) and len(ua) > 1 and ua[0] == "PLANT":
                want[ua[1]] += 1
        # 🩸 對齊實測（2026-09-17）：不用錯開。steps[t]["action"] 與 steps[t] 出現的
        #    新作物同回合，相關 +0.947；錯開 ±1 只有 +0.34。
        rows.append((t, sum(want.values()), born, max(want.values()) if want else 0,
                     len(want)))
        prev = cur
    return np.array(rows, dtype=np.int64), None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default="config/params/cma5-g175.json")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    from eval.runner import load_spec
    sa, sb = load_spec(args.a), load_spec(args.b)
    jobs = [(sa, sb, args.seed0 + i) for i in range(args.games)]
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        res = pool.map(_one, jobs)
    R = np.concatenate([r for r, e in res if r is not None])
    want = R[:, 1]; born = R[:, 2]; mx = R[:, 3]
    m = want > 0
    print(f"{args.a}")
    print(f"  有發 PLANT 的回合 {m.sum():,}   發出 {want[m].sum():,} 次"
          f"   長出 {born[m].sum():,} 株   成功率 {born[m].sum()/max(want[m].sum(),1):.3f}")
    print()
    print(f"  {'同回合發幾次 PLANT':22}{'回合數':>8}{'發出':>9}{'長出':>9}{'成功率':>9}")
    for lo, hi, tag in [(1, 2, "1 次"), (2, 3, "2 次"), (3, 5, "3~4 次"),
                        (5, 9, "5~8 次"), (9, 999, "9 次以上")]:
        s = m & (want >= lo) & (want < hi)
        if s.sum() == 0:
            continue
        print(f"  {tag:22}{s.sum():>8,}{want[s].sum():>9,}{born[s].sum():>9,}"
              f"{born[s].sum()/max(want[s].sum(),1):>9.3f}")
    print()
    print(f"  {'同回合同一種作物最多幾次':22}{'回合數':>8}{'發出':>9}{'長出':>9}{'成功率':>9}")
    for lo, hi, tag in [(1, 2, "1"), (2, 3, "2"), (3, 5, "3~4"), (5, 999, "5 以上")]:
        s = m & (mx >= lo) & (mx < hi)
        if s.sum() == 0:
            continue
        print(f"  {tag:22}{s.sum():>8,}{want[s].sum():>9,}{born[s].sum():>9,}"
              f"{born[s].sum()/max(want[s].sum(),1):>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
