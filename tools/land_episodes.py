"""重播 config/episodes/ 的所有 episode（**雙方都重播**），錄逐步土地組成。

    python -m tools.land_episodes --workers 10 --out temp/curve_episodes.npz

雙方都重播 + 原局 seed = 逐位元重現原局（2026-09-17 驗過：93916293 的期末現金
差 0）。單方重播是開迴路、會漂（見 agents/replay.py 的警告），這裡不是。

所以這批是**真實的高分/低分軌跡**，不是近似。分數跨度 36,335 ~ 165,141。
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")


def _ep_job(job):
    """跑一個 episode 的雙方重播。top-level 才 pickle 得動。"""
    from tools.land_curve import _one
    eid, seed = job
    sa = {"name": f"rp{eid}-p0", "entry": "agents.replay:act",
          "params": {"episode": str(eid), "player": 0}}
    sb = {"name": f"rp{eid}-p1", "entry": "agents.replay:act",
          "params": {"episode": str(eid), "player": 1}}
    return (eid,) + _one((sa, sb, seed))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="temp/curve_episodes.npz")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    eps = []
    for f in sorted(glob.glob("config/episodes/*.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("n_steps") != 720 or not d.get("rewards"):
            continue
        eps.append((d["episode_id"], d["seed"], d["rewards"], d["team_names"]))
    if args.limit:
        eps = eps[:args.limit]
    print(f"{len(eps)} 個 episode，雙方都重播")

    jobs = [(eid, seed) for eid, seed, _, _ in eps]
    import multiprocessing as mp
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            res = pool.map(_ep_job, jobs)
    else:
        res = [_ep_job(j) for j in jobs]

    meta = {eid: (rw, tn) for eid, _, rw, tn in eps}
    ids, arrs, cash, want, bad = [], [], [], [], 0
    for eid, seed, arr, extra in res:
        if arr is None:
            bad += 1; print(f"  {eid} 失敗: {extra}"); continue
        ids.append(eid); arrs.append(arr); cash.append(extra)
        want.append(meta[eid][0])
    A = np.stack(arrs)
    cash = np.array(cash, dtype=np.float64); want = np.array(want, dtype=np.float64)
    ok = np.abs(cash - want).max(axis=1) == 0
    print(f"  成功 {len(arrs)}（失敗 {bad}）；逐位元重現 {ok.sum()}/{len(ok)}")
    if not ok.all():
        w = np.where(~ok)[0]
        print(f"  ⚠️ 沒重現的 episode: {[ids[i] for i in w][:8]}")
        print(f"     最大現金誤差 {np.abs(cash-want).max():,.0f}")
    np.savez_compressed(
        args.out, episode=np.array(ids), cash=cash, want=want, exact=ok,
        open=A[..., 0], empty=A[..., 1], crop=A[..., 2],
        structure=A[..., 3], weed=A[..., 4], money=A[..., 5])
    er = A[..., 1] / np.maximum(A[..., 0], 1)
    print(f"  -> {args.out}")
    print(f"  整局平均空地率 {er.mean():.4f}   期末 {er[..., -1].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
