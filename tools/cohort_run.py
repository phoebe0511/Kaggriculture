"""跑對局並錄 crop cohort + 按品項拆開的 SELL。

    python -m tools.cohort_run --episodes --out temp/coh_kaggle.npz
    python -m tools.cohort_run --a temp/specs/ppo-phi200-140.json \
        --b config/params/cma5-g175.json --games 20 --seed0 900000 \
        --out temp/coh_ppo140.npz

🩸 不是重收資料：這些對局是決定性的（雙方重播已驗過逐位元重現原局），
   同一批軌跡只是多讀出 tile 層的欄位。
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER")
PIDX = {p: i for i, p in enumerate(PRODUCTS)}
N_STEP = 720


def _one(job):
    """跑一局，回傳兩方的 cohort 與 SELL 拆解。top-level 才 pickle 得動。"""
    from eval.runner import build_agent, _quiet_make
    from tools.crop_cohort import Tracker
    spec_a, spec_b, seed, tag = job
    make = _quiet_make()
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    try:
        env.run([build_agent(spec_a), build_agent(spec_b)])
    except Exception as exc:
        return tag, None, None, f"{type(exc).__name__}: {exc}"

    trk = [Tracker(0), Tracker(1)]
    sell = np.zeros((2, len(PRODUCTS)))
    harv = np.zeros(2)
    for t, step in enumerate(env.steps[:N_STEP]):
        try:
            obs = step[0]["observation"]
            farms = obs["farms"]
            day = obs.get("day", t // 24)
        except (KeyError, TypeError, IndexError):
            continue
        for pid, farm in enumerate(farms[:2]):
            trk[pid].feed(farm.get("tiles") or [], t, day)
        for pid in range(2):
            try:
                act = step[pid].get("action")
            except (TypeError, IndexError):
                continue
            if not isinstance(act, dict):
                continue
            for ua in [act.get("farmer"), *(act.get("hands") or [])]:
                if isinstance(ua, list) and ua and ua[0] == "HARVEST":
                    harv[pid] += 1
            for m in (act.get("market") or []):
                if isinstance(m, list) and len(m) > 2 and m[0] == "SELL":
                    i = PIDX.get(m[1])
                    if i is not None:
                        sell[pid, i] += m[2]
    coh = [np.array(trk[p].finish(N_STEP), dtype=np.float64).reshape(-1, 12)
           for p in range(2)]
    cash = [float(s.reward or 0) for s in env.state]
    return tag, coh, (sell, harv, cash), None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", action="store_true")
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    jobs = []
    if args.episodes:
        for f in sorted(glob.glob("config/episodes/*.json")):
            d = json.load(open(f, encoding="utf-8"))
            if d.get("n_steps") != 720:
                continue
            eid = d["episode_id"]
            mk = lambda p: {"name": f"rp{eid}-p{p}", "entry": "agents.replay:act",
                            "params": {"episode": str(eid), "player": p}}
            jobs.append((mk(0), mk(1), d["seed"], eid))
    else:
        from eval.runner import load_spec
        sa, sb = load_spec(args.a), load_spec(args.b)
        jobs = [(sa, sb, args.seed0 + i, args.seed0 + i) for i in range(args.games)]
    print(f"{len(jobs)} 局")

    import multiprocessing as mp
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            res = pool.map(_one, jobs)
    else:
        res = [_one(j) for j in jobs]

    C, S, H, CA, KEY, bad = [], [], [], [], [], 0
    for gi, (tag, coh, extra, err) in enumerate(res):
        if coh is None:
            bad += 1; print(f"  {tag} 失敗: {err}"); continue
        sell, harv, cash = extra
        for pid in range(2):
            c = coh[pid]
            if len(c):
                c[:, 0] = len(KEY)          # traj id 重編
                C.append(c)
            S.append(sell[pid]); H.append(harv[pid]); CA.append(cash[pid])
            KEY.append((tag, pid))
    C = np.concatenate(C) if C else np.zeros((0, 12))
    np.savez_compressed(args.out, cohort=C, sell=np.array(S), harvest=np.array(H),
                        cash=np.array(CA), key=np.array(KEY),
                        products=np.array(PRODUCTS))
    print(f"  {len(KEY)} 條軌跡（{bad} 局失敗），{len(C):,} 筆 cohort -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
