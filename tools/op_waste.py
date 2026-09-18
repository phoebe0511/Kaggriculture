"""發出的 unit 動作，引擎到底有沒有執行。ground truth 是引擎，不需要對照組。

    python -m tools.op_waste --a temp/specs/ppo-phi200-140.json --games 8

## 作法：掛在引擎上，不從 before/after 猜

舊版是比對「回合前的 tiles」與「回合後的 tiles」。🩸 那個作法對**同格互斥**的 op
量不出東西：兩個 unit 站同一格都送 WATER，回合開始時兩個都看到
`watered_today == False`、回合結束時兩個都看到 `True`，兩個都被算成功
（2026-09-17 §127.2 標的星號就是這件事）。

這一版包住引擎自己的兩個函式：

    env.interpreter      每一步記下雙方送出的原始動作、算出 PLANT 原子驗證會擋掉
                         哪些作物，並記下 farms 的 id 好分辨玩家
    _apply_unit_action   引擎逐 unit 執行動作的地方。在**單一次呼叫**的前後各抓
                         一份狀態，所以是誰做成的完全確定

⚠️ `kaggle_environments/core.py:625` 用 `self.interpreter.__code__.co_argcount`
決定傳幾個參數，所以 interpreter 的 wrapper 必須是 `def f(state, env)` 兩個位置
參數，不能用 `*args`（argcount 會變成 0）。

## 各欄的定義

    發出      引擎真的收到這個 op 的 unit-回合數。沒走到目標格的 unit 送的是移動，
              本來就不會出現在這裡（跟 harness/ppo_rollout.py 的 op_exec 同義）
    生效      那一次呼叫有沒有改到狀態。看的是這個 unit 站的 tile、它自己的
              inventory、shed、seeds 四樣的 diff —— `_apply_unit_action` 對非法
              動作一律是**靜默 no-op**、什麼都不碰（引擎行 313 的 docstring：
              `Invalid / illegal actions are silent no-ops`），所以
              「有沒有任何改變」就是精確的判準，不需要逐 op 寫規則
    已佔格    那個 tile 這一回合已經被前面的 unit 佔走（`turn_guard` 的 `claimed`
              是同一件事）。這個 unit 不是第一個動它的。
              **跨 op 計** —— A 收成把 tile 清掉、B 的 WATER 落空，這種舊版完全
              沒算到（舊版的 dup 只比對同一個 op）。
              ⚠️ 已佔格**不等於**白做：FEED -> HARVEST 在同一個 tile 上兩個都
              會成立。所以另外分出「其中白做」，那一欄才是 turn_guard 擴充之後
              修得掉的量
    原子擋掉  PLANT 專屬。引擎行 919-931：某作物的 PLANT 請求數 > 回合開始的種子
              數 -> 該作物這回合的 PLANT **全部**被換成 PASS。被換掉的動作到不了
              `_apply_unit_action`，所以不會出現在「發出」裡，要另外算

## 已知還沒處理的

PICKUP / PLACE / DROP 的「已佔格」沒有意義（多個 unit 在 shed 旁各拿各的，
不衝突），所以同格那兩欄只計會改 tile 的 op。PLACE 放動物那條分支會改 tile，
但失敗時分不出走的是哪一條分支，一律不計。

「誰擋掉誰」記的是那一格**第一個**動它的 op，不是緊鄰的前一個。三個以上 unit
擠同一格時，中間那個不會出現在配對表裡。
"""
from __future__ import annotations
import argparse, collections, copy, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")

#: 會改「unit 站的那個 tile」的 op —— 已佔格只計這些。
TILE_OPS = ("PLANT", "WATER", "HARVEST", "FERTILIZE", "FEED", "CARE",
            "DIG", "BUILD_COOP", "BUILD_PASTURE", "COLLECT_FERTILIZER")


def _why(op, item, tile, inv, shed, seeds, day, near_shed, CROPS, ANIMALS):
    """這一次呼叫為什麼沒改到狀態。照引擎的 guard 逐條回推，回傳短字串。"""
    td = tile if isinstance(tile, dict) else {}
    kind = td.get("kind")

    if op in ("PICKUP", "PLACE", "DROP"):
        if op == "PICKUP":
            if not near_shed:
                return "不在 shed 旁"
            if shed.get(item, 0) <= 0:
                return "shed 沒貨"
            return "?"
        if op == "DROP":
            if not near_shed:
                return "不在 shed 旁"
            if not inv:
                return "inventory 空的"
            return "shed 滿了?"
        housing = (item in ANIMALS and kind == ANIMALS[item]["structure"]
                   and "animal" not in td)
        if housing:
            return "手上沒有那隻動物"
        if not near_shed:
            return "不在 shed 旁、tile 也不是對應的空建物"
        if inv.get(item, 0) <= 0:
            return "手上沒有那個物品"
        return "shed 滿了?"

    if tile == "LOCKED":
        return "LOCKED"

    if op == "PLANT":
        if tile is not None:
            return "tile 已被佔（" + (kind or "非空") + "）"
        if seeds.get(item, 0) <= 0:
            return "沒種子"
        return "?"
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        if tile is not None:
            return "tile 已被佔（" + (kind or "非空") + "）"
        return "?"
    if op == "WATER":
        if kind != "PLANT":
            return "tile 上不是作物"
        if td.get("watered_today"):
            return "今天已經澆過"
        return "?"
    if op == "FERTILIZE":
        if kind != "PLANT":
            return "tile 上不是作物"
        if inv.get("FERTILIZER", 0) <= 0:
            return "手上沒肥料"
        return "肥料已扣但 fertilized_until_day 沒變大（同一天重複施肥）"
    if op == "HARVEST":
        if not isinstance(tile, dict):
            return "tile 空了"
        if td.get("yield_units", 0) <= 0:
            return "沒有可收的量"
        if kind == "PLANT":
            cd = CROPS.get(td.get("crop")) or {}
            need = cd.get("first_yield_day", 0)
            if day - td.get("planted_day", 0) < need:
                return "還沒到 first_yield_day（" + str(td.get("crop")) + " 要 " + str(need) + " 天）"
        return "?"
    if op == "DIG":
        if tile is None:
            return "tile 已經是空的"
        if "animal" in td:
            return "建物裡有動物"
        return "?"
    if op == "FEED":
        if "animal" not in td:
            return "tile 上沒有動物"
        if td.get("fed_today"):
            return "今天已經餵過"
        if inv.get("WHEAT", 0) <= 0:
            return "手上沒有 WHEAT"
        return "?"
    if op == "CARE":
        if "animal" not in td:
            return "tile 上沒有動物"
        if td.get("cared_today"):
            return "今天已經照顧過"
        return "?"
    if op == "COLLECT_FERTILIZER":
        if "animal" not in td:
            return "tile 上沒有動物"
        if not td.get("fertilizer_available"):
            return "今天的肥料已被拿走"
        return "?"
    return "?"


def _one(job):
    from eval.runner import build_agent, _quiet_make
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    from contracts import _shed_tiles      # 跟 legal_unit_mask 同一個定義

    spec_a, spec_b, seed, player = job
    make = _quiet_make()
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)

    issued = collections.Counter()
    worked = collections.Counter()
    why = collections.defaultdict(collections.Counter)
    dup = collections.Counter()
    dup_waste = collections.Counter()
    dup_pairs = collections.Counter()
    ok_pairs = collections.Counter()
    atomic = collections.Counter()
    other = collections.Counter()      # 移動 / PASS：沒有 op 送進引擎的 unit-回合
    turn = {"farm_ids": [], "tile_seen": {}}

    orig_unit = K._apply_unit_action
    orig_interp = env.interpreter
    board = int(env.configuration.get("boardSize", 10))
    sheds = set(_shed_tiles(board))

    def unit_wrapper(farm, private, idx, action, board_size, day,
                     turns_per_day, shed_capacity=100):
        try:
            me = turn["farm_ids"].index(id(farm))
        except ValueError:
            me = -1
        op = action[0] if isinstance(action, list) and action else None
        if me != player or op is None:
            return orig_unit(farm, private, idx, action, board_size, day,
                             turns_per_day, shed_capacity)
        if op in MOVES or op == "PASS":
            other[op] += 1
            return orig_unit(farm, private, idx, action, board_size, day,
                             turns_per_day, shed_capacity)

        item = action[1] if len(action) > 1 else None
        pos = (tuple(farm["farmer"]) if idx == 0
               else tuple(farm["hands"][idx - 1]))
        x, y = pos
        invs = private["inventories"]
        b_tile = copy.deepcopy(farm["tiles"][y][x])
        b_inv = dict(invs[idx]) if idx < len(invs) else {}
        b_shed = dict(private["shed"])
        b_seeds = dict(private["seeds"])

        orig_unit(farm, private, idx, action, board_size, day,
                  turns_per_day, shed_capacity)

        a_tile = farm["tiles"][y][x]
        a_inv = dict(invs[idx]) if idx < len(invs) else {}
        if op == "FERTILIZE":
            # 🩸 例外：引擎行 478 的 `_inv_take` 在行 481 的 `max(...)` 之前，
            # 所以同一天重複施肥「肥料扣掉了但沒有延長」—— inventory 有變，
            # 用通則會誤判成生效。這個 op 只認 fertilized_until_day 有沒有變大。
            bd = b_tile if isinstance(b_tile, dict) else {}
            ad = a_tile if isinstance(a_tile, dict) else {}
            changed = (ad.get("fertilized_until_day", -1)
                       > bd.get("fertilized_until_day", -1))
        else:
            changed = (a_tile != b_tile or a_inv != b_inv
                       or dict(private["shed"]) != b_shed
                       or dict(private["seeds"]) != b_seeds)

        issued[op] += 1
        if changed:
            worked[op] += 1
        else:
            why[op][_why(op, item, b_tile, b_inv, b_shed, b_seeds, day,
                         pos in sheds, K.CROPS, K.ANIMALS)] += 1

        if op in TILE_OPS:
            prev = turn["tile_seen"].get(pos)
            if prev is None:
                turn["tile_seen"][pos] = op
            else:
                dup[op] += 1                       # 已佔格（不等於白做）
                if not changed:
                    dup_waste[op] += 1             # 已佔格 而且 白做
                    dup_pairs[(prev, op)] += 1
                else:
                    ok_pairs[(prev, op)] += 1      # 已佔格 但照樣生效

    def interp_wrapper(state, env_):
        farms = state[0].observation["farms"]
        turn["farm_ids"] = [id(f) for f in farms]
        turn["tile_seen"] = {}
        # PLANT 原子驗證（引擎行 919-931）：用回合開始的種子數
        s = state[player]
        act = s.action if isinstance(s.action, dict) else {}
        units = [act.get("farmer", ["PASS"]), *(act.get("hands") or [])]
        demand = collections.Counter(
            a[1] for a in units
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT")
        priv = state[player].observation.get("private") or {}
        seeds = priv.get("seeds") or {}
        for crop, n in demand.items():
            if n > seeds.get(crop, 0):
                atomic[crop] += n
        return orig_interp(state, env_)

    K._apply_unit_action = unit_wrapper
    env.interpreter = interp_wrapper
    try:
        env.run([build_agent(spec_a), build_agent(spec_b)])
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        K._apply_unit_action = orig_unit
    return (issued, worked, why, dup, dup_waste, dup_pairs, ok_pairs,
            atomic, other), None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default="config/params/cma5-g175.json")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--player", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    from eval.runner import load_spec
    sa, sb = load_spec(args.a), load_spec(args.b)
    jobs = [(sa, sb, args.seed0 + i, args.player) for i in range(args.games)]
    import multiprocessing as mp
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_one, jobs)

    I = collections.Counter(); W = collections.Counter()
    D = collections.Counter(); DW = collections.Counter()
    P = collections.Counter(); OK = collections.Counter()
    A = collections.Counter(); O = collections.Counter()
    Y = collections.defaultdict(collections.Counter)
    for r, e in res:
        if r is None:
            print("  失敗:", e)
            continue
        I += r[0]; W += r[1]; D += r[3]; DW += r[4]
        P += r[5]; OK += r[6]; A += r[7]; O += r[8]
        for op, c in r[2].items():
            Y[op] += c
    ng = sum(1 for r, e in res if r is not None)
    if not ng:
        return 1

    print(f"{args.a}   {ng} 局   player {args.player}   "
          f"seed {args.seed0}~{args.seed0 + ng - 1}")
    print()
    print(f"  {'op':<20}{'發出':>8}{'生效':>8}{'生效率':>8}"
          f"{'白做':>8}{'每局白做':>10}{'佔格造成':>10}{'%':>6}")
    for op in sorted(I, key=lambda k: -(I[k] - W[k])):
        waste = I[op] - W[op]
        frac = f"{100 * DW[op] / waste:.0f}%" if waste else "—"
        print(f"  {op:<20}{I[op]:>8,}{W[op]:>8,}{W[op] / max(I[op], 1):>8.3f}"
              f"{waste:>8,}{waste / ng:>10.1f}{DW[op]:>10,}{frac:>6}")
    ti, tw = sum(I.values()), sum(W.values())
    tws = ti - tw
    tfrac = f"{100 * sum(DW.values()) / tws:.0f}%" if tws else "—"
    print(f"  {'合計':<20}{ti:>8,}{tw:>8,}{tw / max(ti, 1):>8.3f}"
          f"{tws:>8,}{tws / ng:>10.1f}{sum(DW.values()):>10,}{tfrac:>6}")
    print("  「佔格造成」= 白做裡，那個 tile 這一回合已經被前面的 unit 佔走的部分。"
          "把 contracts.turn_guard")
    print("  擴充成「任何 tile 專屬的 op 都佔格」之後消得掉的就是這些，"
          "「%」是它佔白做的比例。")
    extra = sum(D.values()) - sum(DW.values())
    print(f"  （另有 {extra:,} 次是被佔格之後仍然生效的，例如 FEED -> HARVEST，"
          f"不是白做，所以不列）")

    print("\n  PLANT 被原子驗證擋掉（動作被換成 PASS，不在上表的「發出」裡）")
    if A:
        for crop, n in A.most_common():
            print(f"    {crop:<14}{n:>8,}{n / ng:>10.1f} 次/局")
        tot = sum(A.values())
        print(f"    {'合計':<14}{tot:>8,}{tot / ng:>10.1f} 次/局")
    else:
        print("    0")

    print("\n  白做的原因")
    for op in sorted(Y, key=lambda k: -sum(Y[k].values())):
        tot = sum(Y[op].values())
        print(f"    {op}  {tot:,} 次（{tot / ng:.1f}/局）")
        for reason, n in Y[op].most_common(6):
            print(f"      {n:>7,}  {100 * n / tot:>5.1f}%  {reason}")

    mv = sum(O[k] for k in MOVES)
    ps = O.get("PASS", 0)
    total = ti + mv + ps
    print(f"\n  一局的 unit-回合全貌（{total:,} 次 = {total / ng:.0f}/局）")
    print(f"    送 op 且生效      {tw:>8,}{tw / ng:>9.1f}/局{100 * tw / total:>7.1f}%")
    print(f"    送 op 但白做      {tws:>8,}{tws / ng:>9.1f}/局{100 * tws / total:>7.1f}%")
    print(f"    走路（沒到目標）  {mv:>8,}{mv / ng:>9.1f}/局{100 * mv / total:>7.1f}%")
    print(f"    PASS              {ps:>8,}{ps / ng:>9.1f}/局{100 * ps / total:>7.1f}%")
    print("    ⚠️ 走路和 PASS 引擎都正常處理，不是白做。這張表只判斷"
          "「引擎有沒有執行」，")
    print("    不判斷「值不值得做」。")

    print("\n  同一個 tile 上，先動的 op -> 後動的 op")
    print(f"    [後面的白做]  共 {sum(P.values()):,} 次")
    for (a, b), n in P.most_common(12):
        print(f"      {n:>7,}  {n / ng:>6.1f}/局   {a} -> {b}")
    print(f"    [後面的照樣生效]  共 {sum(OK.values()):,} 次"
          f" —— claimed 改成一律佔格會誤擋這些")
    for (a, b), n in OK.most_common(12):
        print(f"      {n:>7,}  {n / ng:>6.1f}/局   {a} -> {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
