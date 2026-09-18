"""`legal_unit_mask` 說的合法，跟引擎收不收，對不對得起來。

    python -m tools.mask_audit --a temp/specs/ppo-phi200-140.json --games 4

## 為什麼不能「重寫一份規則再跟 mask 比」

那只是比對我自己寫的兩份，兩份一起錯就看不出來。這支工具拿**引擎**當 ground
truth：深複製一份狀態，直接呼叫引擎的 `_apply_unit_action`，看它到底動不動。

對抽樣到的每一個回合、每一個 unit、全部 44 個 op 各試一次：

    偽陽性   mask 說合法、引擎其實靜默 no-op   -> policy 會學到做不成的動作
    偽陰性   mask 說不合法、引擎其實會執行     -> policy 永遠學不到這個動作

兩種都是 bug。判準是引擎收不收，跟老師做過什麼無關。

## 判「引擎有沒有執行」的方式

比對這一次呼叫前後的四樣東西：這個 unit 站的 tile、它自己的 inventory、
shed、seeds，外加它的位置（移動只改位置）。`_apply_unit_action` 對非法動作
一律是靜默 no-op、什麼都不碰（引擎行 313 的 docstring），所以「有沒有任何
改變」就是精確的判準。

🩸 FERTILIZE 例外：引擎行 478 的 `_inv_take` 在行 481 的 `max(...)` 之前，
同一天重複施肥「肥料扣掉了但 `fertilized_until_day` 沒變大」—— inventory 有變，
用通則會誤判成合法。這個 op 只認 `fertilized_until_day` 有沒有變大。
（`tools/rule_probe.py` 的 S2 實測過：那兩列「引擎執行 否」旁邊都寫著
「肥料扣掉 1」。）

## qty

一律用 1（`decode_unit` 的預設）。`legal_unit_mask` 本來就不看數量 ——
PICKUP 的 mask 只問「shed 有沒有貨」，引擎則是 `n = min(n, 存量)` 做部分成交。
數量對不對是另一個題目（journal §132 的清單）。

⚠️ 每一次試都要深複製，所以是**抽樣**跑（預設一天抽一個回合），不是每一步都跑。
"""
from __future__ import annotations
import argparse
import collections
import copy
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")


def _sig(farm, priv, idx):
    """這個 unit 動得到的那幾樣東西，抓一份可以比對的快照。"""
    pos = (tuple(farm["farmer"]) if idx == 0
           else tuple(farm["hands"][idx - 1]))
    x, y = pos
    tiles = farm["tiles"]
    tile = tiles[y][x] if (0 <= y < len(tiles) and 0 <= x < len(tiles[y])) else None
    invs = priv["inventories"]
    return (
        pos,
        copy.deepcopy(tile) if isinstance(tile, dict) else tile,
        dict(invs[idx]) if idx < len(invs) else {},
        dict(priv["shed"]),
        dict(priv["seeds"]),
    )


def _fert_until(farm, idx):
    pos = (tuple(farm["farmer"]) if idx == 0
           else tuple(farm["hands"][idx - 1]))
    x, y = pos
    tile = farm["tiles"][y][x]
    return tile.get("fertilized_until_day", -1) if isinstance(tile, dict) else -1


def _tile_label(farm, idx):
    """偽陽性 / 偽陰性的分類用：這個 unit 站在什麼上面。"""
    pos = (tuple(farm["farmer"]) if idx == 0
           else tuple(farm["hands"][idx - 1]))
    x, y = pos
    tiles = farm["tiles"]
    if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
        return "界外"
    tile = tiles[y][x]
    if tile is None:
        return "空地"
    if tile == "LOCKED":
        return "LOCKED"
    if isinstance(tile, dict):
        if tile.get("animal"):
            return f"{tile.get('kind')}+動物"
        if tile.get("kind") == "PLANT":
            return f"作物 {tile.get('crop')}"
        return str(tile.get("kind"))
    return str(tile)


def _one(job):
    from eval.runner import build_agent, _quiet_make
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    import contracts as C

    spec_a, spec_b, seed, player, every = job
    env = _quiet_make()("kaggriculture", configuration={"seed": seed}, debug=True)

    tried = collections.Counter()           # op -> 試了幾次
    fp = collections.Counter()              # op -> 偽陽性
    fn = collections.Counter()              # op -> 偽陰性
    fp_where = collections.defaultdict(collections.Counter)
    fn_where = collections.defaultdict(collections.Counter)

    cfg = env.configuration
    tpd = int(cfg.get("turnsPerDay", 24))
    cap = int(cfg.get("shedCapacity", 100))
    orig_interp = env.interpreter
    box = {"t": 0}

    def audit(state):
        obs = dict(state[0].observation)
        obs.update(state[player].observation)
        obs["player"] = player
        # ⚠️ `env.reset()` 會先跑一次 interpreter 建初始盤面，那一次 `farms`
        # 還沒有東西（core.py:358）。跳過。
        farms = obs.get("farms") or []
        if len(farms) <= player or not obs.get("private"):
            return
        farm0 = farms[player]
        priv0 = obs["private"]
        if not farm0.get("tiles"):
            return
        board = len(farm0["tiles"])
        day = int(obs.get("day", 0))

        mask = C.legal_unit_mask(obs, cfg)
        n_units = min(mask.shape[0], 1 + len(farm0["hands"]))

        for i in range(n_units):
            for j in range(C.N_UNIT_OPS):
                op = C.UNIT_OPS[j][0]
                # PASS 依定義什麼都不做，用「有沒有改到狀態」判會一律變成
                # 偽陽性。它永遠合法也永遠無效，不是稽核的對象。
                if op == "PASS":
                    continue
                f2 = copy.deepcopy(farm0)
                p2 = copy.deepcopy(priv0)
                before = _sig(f2, p2, i)
                b_until = _fert_until(f2, i) if op == "FERTILIZE" else None
                K._apply_unit_action(f2, p2, i, C.decode_unit(j),
                                     board, day, tpd, cap)
                if op == "FERTILIZE":
                    works = _fert_until(f2, i) > b_until
                else:
                    works = _sig(f2, p2, i) != before

                said = bool(mask[i, j])
                tried[op] += 1
                if said and not works:
                    fp[op] += 1
                    fp_where[op][_tile_label(farm0, i)] += 1
                elif works and not said:
                    fn[op] += 1
                    fn_where[op][_tile_label(farm0, i)] += 1

    def interp(state, env_):
        if box["t"] % every == 0:
            audit(state)
        box["t"] += 1
        return orig_interp(state, env_)

    env.interpreter = interp
    try:
        env.run([build_agent(spec_a), build_agent(spec_b)])
    except Exception:
        import traceback
        return None, traceback.format_exc()
    return (tried, fp, fn, fp_where, fn_where), None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="temp/specs/ppo-phi200-140.json")
    ap.add_argument("--b", default="config/params/cma5-g175.json")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--player", type=int, default=0)
    ap.add_argument("--every", type=int, default=24,
                    help="每幾步抽一個回合來稽核（預設 24 = 一天一次）")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    from eval.runner import load_spec
    sa, sb = load_spec(args.a), load_spec(args.b)
    jobs = [(sa, sb, args.seed0 + i, args.player, args.every)
            for i in range(args.games)]
    import multiprocessing as mp
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_one, jobs)

    T = collections.Counter(); FP = collections.Counter(); FN = collections.Counter()
    FPW = collections.defaultdict(collections.Counter)
    FNW = collections.defaultdict(collections.Counter)
    ng = 0
    for r, e in res:
        if r is None:
            print("  失敗:", e)
            continue
        ng += 1
        T += r[0]; FP += r[1]; FN += r[2]
        for op, c in r[3].items():
            FPW[op] += c
        for op, c in r[4].items():
            FNW[op] += c
    if not ng:
        return 1

    print(f"{args.a}   {ng} 局   player {args.player}   每 {args.every} 步抽一個回合")
    print()
    print(f"  {'op':<20}{'試了':>9}{'偽陽性':>9}{'偽陰性':>9}   "
          f"偽陽性 = mask 放行但引擎 no-op")
    bad = 0
    for op in sorted(T, key=lambda k: -(FP[k] + FN[k])):
        if not (FP[op] or FN[op]):
            continue
        bad += FP[op] + FN[op]
        print(f"  {op:<20}{T[op]:>9,}{FP[op]:>9,}{FN[op]:>9,}")
    if not bad:
        print("  （沒有任何 op 對不上）")
    print(f"  {'合計':<20}{sum(T.values()):>9,}"
          f"{sum(FP.values()):>9,}{sum(FN.values()):>9,}")

    if FP:
        print("\n  偽陽性：mask 說合法、引擎其實靜默 no-op（unit 站在什麼上面）")
        for op in sorted(FP, key=lambda k: -FP[k]):
            print(f"    {op}  {FP[op]:,}")
            for where, n in FPW[op].most_common(5):
                print(f"      {n:>7,}  {where}")
    if FN:
        print("\n  偽陰性：mask 說不合法、引擎其實會執行（policy 學不到）")
        for op in sorted(FN, key=lambda k: -FN[k]):
            print(f"    {op}  {FN[op]:,}")
            for where, n in FNW[op].most_common(5):
                print(f"      {n:>7,}  {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
