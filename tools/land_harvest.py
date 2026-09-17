"""把 temp/ 所有 eval 對局的 result.json 收成一張表：空地率 vs 表現。

每局每一方一列。空地率的分母是**已解鎖 tile-turns**（`eval/runner._farm_history`），
晚買的地只從解鎖後計入，不會被 LOCKED 期間稀釋。

    empty_rate   empty_tile_turns / open_tile_turns    整局平均
    final_empty  final.empty / final.open              期末
    margin       自己的期末現金 − 對手的

🩸 這裡沒有逐步曲線 —— result.json 只存整局彙總。AUC 和「第一次低於 X%」要逐步
   資料，得另外跑（見 tools/land_curve.py）。
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")

rows = []
for rj in sorted(glob.glob("temp/*/result.json")):
    try:
        d = json.load(open(rj, encoding="utf-8"))
    except Exception:
        continue
    a_name = (d.get("a") or {}).get("name") or "?"
    b_name = (d.get("b") or {}).get("name") or None
    res = d.get("results") or {}
    items = res.items() if isinstance(res, dict) else [("?", res)]
    for opp, games in items:
        if not isinstance(games, list):
            continue
        for g in games:
            if not isinstance(g, dict) or g.get("error"):
                continue
            for side, me, you in (("a", "a", "b"), ("b", "b", "a")):
                fh = g.get(f"farm_{me}")
                if not isinstance(fh, dict) or not fh.get("open_tile_turns"):
                    continue
                fin = fh.get("final") or {}
                cash = g.get(f"cash_{me}"); ocash = g.get(f"cash_{you}")
                if cash is None or ocash is None:
                    continue
                act = g.get(f"actions_{me}") or {}
                rows.append(dict(
                    run=os.path.basename(os.path.dirname(rj)),
                    agent=(a_name if side == "a" else (opp if opp != "?" else (b_name or "?"))),
                    side=side, seed=g.get("seed"),
                    cash=float(cash), margin=float(cash) - float(ocash),
                    empty_rate=fh["empty_tile_turns"] / fh["open_tile_turns"],
                    crop_occ=fh.get("crop_occupancy", np.nan),
                    managed_occ=fh.get("managed_occupancy", np.nan),
                    weed_rate=fh.get("weed_rate", np.nan),
                    final_empty=(fin.get("empty", 0) / fin["open"]) if fin.get("open") else np.nan,
                    open_tt=fh["open_tile_turns"],
                    n_land_buys=len(g.get(f"land_{me}") or []),
                    move_rate=act.get("move_rate", np.nan),
                    prod_rate=act.get("productive_rate", np.nan),
                ))

if not rows:
    print("沒有找到 result.json"); sys.exit(0)

import collections
print(f"{len(rows):,} 列（局 × 方），來自 {len(set(r['run'] for r in rows))} 個對局目錄")
print()
E = np.array([r["empty_rate"] for r in rows])
M = np.array([r["margin"] for r in rows])
C = np.array([r["cash"] for r in rows])
FE = np.array([r["final_empty"] for r in rows])

def pe(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    s = x.std() * y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / s) if s else np.nan

print("  全樣本相關")
print(f"    empty_rate  vs margin  {pe(E, M):+.4f}      vs cash  {pe(E, C):+.4f}")
print(f"    final_empty vs margin  {pe(FE, M):+.4f}      vs cash  {pe(FE, C):+.4f}")
print()
print("  按 agent 彙總（n>=10 的）")
by = collections.defaultdict(list)
for r in rows:
    by[r["agent"]].append(r)
print(f"    {'agent':<26}{'n':>5}{'empty_rate':>12}{'final_empty':>13}{'cash':>11}{'margin':>11}")
for ag, rs in sorted(by.items(), key=lambda kv: -np.mean([r["cash"] for r in kv[1]])):
    if len(rs) < 10:
        continue
    f = lambda k: np.nanmean([r[k] for r in rs])
    print(f"    {ag[:25]:<26}{len(rs):>5}{f('empty_rate'):>12.4f}"
          f"{f('final_empty'):>13.4f}{f('cash'):>11,.0f}{f('margin'):>+11,.0f}")

np.save("temp/land_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
print()
print("原始列存到 temp/land_rows.npy")
