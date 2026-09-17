"""發出的 unit 動作，引擎到底有沒有執行。ground truth 是引擎，不需要對照組。

    python -m tools.op_waste --a temp/specs/ppo-phi200-140.json --games 8

作法：每一回合記下每個 unit 站的格子與它送出的 op，再看下一個 observation 裡
那一格（或該方的存量）有沒有出現這個 op 該有的變化。沒有 = 這個 unit 整回合白做。

🩸 只看**有站到目標格**的 unit —— 沒站到的送的是 step_toward，那一步的 op
   根本不進引擎（`harness/ppo_rollout.py:524` 的 op_exec 同一條）。

各 op 的判定（引擎行為見 docs/games/engine-notes.md）：
    PLANT       那一格出現新作物（planted_day 變了也算，收成後立刻重種）
    WATER       那一格 watered_today False -> True
    HARVEST     那一格 yield_units 下降，或整格消失
    FERTILIZE   那一格 fertilized_until_day 變大
    FEED        那一格動物 fed_today False -> True
    CARE        那一格動物 cared_today False -> True
    DIG         那一格變成 None
    BUILD_COOP / BUILD_PASTURE   那一格出現建物
⚠️ PICKUP / PLACE / DROP / COLLECT_FERTILIZER 牽涉 unit 自己的 inventory 與 shed，
   observation 看不到逐 unit 的 inventory，這支不判定，另外列出次數。
"""
from __future__ import annotations
import argparse, collections, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")

TILE_OPS = ("PLANT", "WATER", "HARVEST", "FERTILIZE", "FEED", "CARE",
            "DIG", "BUILD_COOP", "BUILD_PASTURE")
SKIP = ("PICKUP", "PLACE", "DROP", "COLLECT_FERTILIZER")
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def _snap(farm):
    """把 tiles 抓成 {(x,y): dict}。

    unit 的位置（`farm["farmer"]` / `farm["hands"][i]`）是 **(x, y)**，而
    `tiles` 是 `[y][x]`。2026-09-17 實測：照 (x,y) 查對上 169 次、照 (y,x)
    只有 21 次（那 21 次是對角線上的巧合）。跟 `net.py:144` 的 unit_pos
    同一個慣例，這個坑踩過兩次了。
    """
    out = {}
    for y, row in enumerate(farm.get("tiles") or []):
        for x, t in enumerate(row):
            out[(x, y)] = t
    return out


def _did(op, before, after):
    """引擎有沒有執行。看不出來就回 None（不計入分母）。"""
    b, a = before, after
    bd = b if isinstance(b, dict) else {}
    ad = a if isinstance(a, dict) else {}
    if op == "PLANT":
        if not (isinstance(a, dict) and a.get("kind") == "PLANT"):
            return False
        return (not isinstance(b, dict) or b.get("kind") != "PLANT"
                or b.get("planted_day") != a.get("planted_day"))
    if op == "WATER":
        return bool(ad.get("watered_today")) and not bool(bd.get("watered_today"))
    if op == "HARVEST":
        if a is None and isinstance(b, dict):
            return True
        return ad.get("yield_units", 0) < bd.get("yield_units", 0)
    if op == "FERTILIZE":
        return ad.get("fertilized_until_day", -1) > bd.get("fertilized_until_day", -1)
    if op == "FEED":
        return bool(ad.get("fed_today")) and not bool(bd.get("fed_today"))
    if op == "CARE":
        return bool(ad.get("cared_today")) and not bool(bd.get("cared_today"))
    if op == "DIG":
        return a is None and b is not None
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        want = "COOP" if op == "BUILD_COOP" else "PASTURE"
        return ad.get("kind") == want and bd.get("kind") != want
    return None


def _one(job):
    from eval.runner import build_agent, _quiet_make
    spec_a, spec_b, seed = job
    make = _quiet_make()
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    try:
        env.run([build_agent(spec_a), build_agent(spec_b)])
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    issued = collections.Counter(); worked = collections.Counter()
    dup = collections.Counter(); other = collections.Counter()
    prev_tiles = None
    for t, step in enumerate(env.steps[:720]):
        farm = step[0]["observation"]["farms"][0]
        tiles = _snap(farm)
        # 對齊：`steps[t]["action"]` 產生的就是 `steps[t]` 的狀態
        # （2026-09-17 實測相關 +0.947，錯開 +-1 掉到 +0.34）。
        # 所以拿第 t 步的動作比 tiles[t-1] -> tiles[t]。
        act = step[0].get("action") or {}
        ulist = [act.get("farmer"), *(act.get("hands") or [])]
        poss = [tuple(farm.get("farmer") or (0, 0))]
        for h in (farm.get("hands") or []):
            poss.append(tuple(h) if isinstance(h, (list, tuple)) else (0, 0))
        if prev_tiles is not None:
            seen = collections.Counter()
            for j, ua in enumerate(ulist):
                if not (isinstance(ua, list) and ua and j < len(poss)):
                    continue
                op, pp = ua[0], poss[j]
                if op in SKIP or op in MOVES or op == "PASS":
                    other[op] += 1
                    continue
                issued[op] += 1
                seen[(op, pp)] += 1
                if seen[(op, pp)] > 1:
                    dup[op] += 1
                if _did(op, prev_tiles.get(pp), tiles.get(pp)):
                    worked[op] += 1
        prev_tiles = tiles
    return (issued, worked, dup, other), None


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
    I = collections.Counter(); W = collections.Counter()
    D = collections.Counter(); O = collections.Counter()
    for r, e in res:
        if r is None:
            print("  失敗:", e); continue
        I += r[0]; W += r[1]; D += r[2]; O += r[3]
    ng = sum(1 for r, e in res if r is not None)
    print(f"{args.a}   {ng} 局")
    print()
    print(f"  {'op':<18}{'發出':>9}{'生效':>9}{'生效率':>9}{'同格重複':>10}{'每局白做':>10}")
    for op in sorted(I, key=lambda k: -(I[k] - W[k])):
        waste = I[op] - W[op]
        print(f"  {op:<18}{I[op]:>9,}{W[op]:>9,}{W[op]/max(I[op],1):>9.3f}"
              f"{D[op]:>10,}{waste/max(ng,1):>10.1f}")
    tot_i, tot_w = sum(I.values()), sum(W.values())
    print(f"  {'合計':<18}{tot_i:>9,}{tot_w:>9,}{tot_w/max(tot_i,1):>9.3f}"
          f"{sum(D.values()):>10,}{(tot_i-tot_w)/max(ng,1):>10.1f}")
    print()
    print("  ⚠️ 不判定（observation 看不到 unit 的 inventory / shed）：")
    print("   ", "  ".join(f"{k} {v:,}" for k, v in O.most_common()
                           if k not in MOVES and k != "PASS"))
    print("  移動與 PASS：", "  ".join(f"{k} {O[k]:,}" for k in MOVES + ("PASS",)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
