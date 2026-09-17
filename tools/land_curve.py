"""逐步錄製雙方的土地組成，供空地率時間曲線分析。

    python -m tools.land_curve --a phi200 --b cma5-g175 --games 20 --workers 10 \
           --out temp/curve_phi200_vs_cma5.npz

result.json 只存整局彙總（`eval/runner._farm_history`），沒有逐步曲線 ——
AUC、「第一次低於 X%」、以及空地率與 margin 的領先落後都需要逐步資料。

分母跟 `_farm_history` 一致：**已解鎖 tile-turns**（LOCKED 不計），所以晚買的地
不會被 LOCKED 期間稀釋。

輸出 npz，形狀都是 [game, player, step]：
    open / empty / crop / structure / weed / money
外加 [game] 的 seed、[game, player] 的 final_cash。
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")

N_STEP = 720
PHASES = tuple((d * 24, (d + 1) * 24) for d in range(30))   # 引擎的每日結算週期


def _one(job):
    """跑一局，回傳逐步土地組成。top-level 才 pickle 得動（Windows spawn）。"""
    from eval.runner import build_agent, load_spec, _quiet_make
    spec_a, spec_b, seed = job
    make = _quiet_make()
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    try:
        env.run([build_agent(spec_a), build_agent(spec_b)])
    except Exception as exc:
        return seed, None, f"{type(exc).__name__}: {exc}"

    from tools.behav_ops import tally, N_FIELD
    out = np.zeros((2, N_STEP, 6), dtype=np.float32)
    ops = np.zeros((2, len(PHASES), N_FIELD), dtype=np.float64)
    for t, step in enumerate(env.steps[:N_STEP]):
        try:
            farms = step[0]["observation"]["farms"]
        except (KeyError, TypeError, IndexError):
            continue
        ph = next((k for k, (a_, b_) in enumerate(PHASES) if a_ <= t < b_), None)
        for pid in range(2):
            if ph is not None:
                try:
                    tally(step[pid].get("action"), ops[pid, ph])
                except (TypeError, IndexError, AttributeError):
                    pass
        for pid, farm in enumerate(farms[:2]):
            tiles = farm.get("tiles") or []
            o = e = c = s_ = w = 0
            for row in tiles:
                for tile in row:
                    if tile == "LOCKED":
                        continue
                    o += 1
                    if tile is None:
                        e += 1
                    elif isinstance(tile, dict):
                        k = tile.get("kind")
                        if k == "PLANT":
                            c += 1
                        elif k in ("COOP", "PASTURE"):
                            s_ += 1
                        elif k == "WEED":
                            w += 1
            out[pid, t] = (o, e, c, s_, w, farm.get("money", 0.0))
    cash = [float(st.reward or 0) for st in env.state]
    return seed, (out, ops), cash


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from eval.runner import load_spec
    sa, sb = load_spec(args.a), load_spec(args.b)
    jobs = [(sa, sb, args.seed0 + i) for i in range(args.games)]

    import multiprocessing as mp
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            res = pool.map(_one, jobs)
    else:
        res = [_one(j) for j in jobs]

    seeds, arrs, opss, cashes, bad = [], [], [], [], 0
    for seed, arr, extra in res:
        if arr is None:
            bad += 1; print(f"  seed {seed} 失敗: {extra}"); continue
        seeds.append(seed); arrs.append(arr[0]); opss.append(arr[1]); cashes.append(extra)
    if not arrs:
        print("全部失敗"); return 1
    A = np.stack(arrs)                       # [game, player, step, 6]
    np.savez_compressed(
        args.out, seed=np.array(seeds), cash=np.array(cashes, dtype=np.float64),
        ops=np.stack(opss),
        open=A[..., 0], empty=A[..., 1], crop=A[..., 2],
        structure=A[..., 3], weed=A[..., 4], money=A[..., 5],
        a_name=str(sa.get("name", args.a)), b_name=str(sb.get("name", args.b)))
    er = A[..., 1] / np.maximum(A[..., 0], 1)
    print(f"{len(arrs)} 局（{bad} 失敗）-> {args.out}")
    print(f"  整局平均空地率  a {er[:,0].mean():.4f}   b {er[:,1].mean():.4f}")
    print(f"  期末空地率      a {er[:,0,-1].mean():.4f}   b {er[:,1,-1].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
