"""DAgger：讓 policy 去走，在它**走到的局面**上問 expert 該做什麼。

    python -m harness.rollout --policy gen1 --expert gen1 \
           --opponents ladder-top-a,ref-v4,starter --games 60 --out data/dagger/round0

## 為什麼要這支

2026-08-20 量到：模仿 ladder 頂端玩家的 replay，每一步 0.868 的準確率，實戰
0 勝 4 負、期末現金 2,061 對 140,210。原因是 **behavior cloning 的錯誤是累乘的**
—— 老師的開局前 73 步是逐字相同的腳本，我們第 6 步就分歧（`1/(1−0.868) = 7.6`
正好預測到），而 `0.868^73 = 0.003%`。一步走錯就進入老師沒走過的局面，
那裡沒有任何標籤。

DAgger 就是為這個病設計的：**訓練資料來自 policy 自己會走到的局面**，
所以不再有「沒標籤的區域」。

之前用不了是因為老師是 60 份固定 replay、問不到新局面。`gen1` 問得到 ——
它每回合從當下盤面重算，任何局面都給得出答案。

⚠️ **DAgger 的天花板就是 expert。** 第一輪 expert = gen1，得到的是「gen1 裝進
網路裡」（約 $73k），榜上分數不會動。它的價值是驗證閉迴路修得掉，以及產出
self-play 資料 —— 第二輪要拿那份資料訓一個能分辨盤面好壞的 value head，
再把 expert 換成離線 search。

## v5：多錄一份逐格需求

`board_demand_bits` / `board_demand_legal_bits` 是「哪一格要做什麼」，直接讀
`gen0._tasks()` 掃出來的任務清單（`plan["tasks"]`）。v4 的逐 unit target 標籤
仍然照錄 —— 兩組標籤都在同一份 npz 裡，`agents/gen3_target.py` 和
`agents/gen4_demand.py` 可以用同一份權重做 A/B。

每個回合都驗「expert 的需求 ⊆ `legal_demand_mask`」，不合就拋錯。

## 標籤直接讀規劃器，不反推

`agents/gen0.act(return_plan=True)` 會交出每個 unit 的**終點格與到了要做的動作**
（`gen0.task_destination()`），那正好就是 v3 的標籤格式。不必像
`harness/build_dataset.segment_labels()` 那樣從動作序列反推 MOVE 段落 ——
那是為了老師的 replay 才做的事，我們自己的 agent 沒有理由丟掉已經知道的答案。

## 輸出跟 build_dataset 完全一樣

一局一個 npz，欄位與 dtype 跟 `harness/build_dataset.py` 逐項相同，
所以 `model/train.py` 不用改就吃得下，也可以跟 replay 資料混著訓。

## 對手要有多樣性

`--opponents` 預設是強／中／弱各一（`ladder-top-a` / `ref-v4` / `starter`），
輪流配。理由：value head 的訓練目標是期末現金，只跟同一個對手打的話結果都差不多
——現有的 replay 資料就是這樣（120 個 player-episode 的目標範圍只有
0.363~1.402，低於 0.5 的只有 3 個），訓出來的 value head 分辨不出盤面好壞。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contracts as C                                    # noqa: E402
from harness.build_dataset import dense_market           # noqa: E402

#: 強／中／弱各一。理由見 module docstring。
DEFAULT_OPPONENTS = ("ladder-top-a", "ref-v4", "starter")


def demand_from_tasks(tasks, board):
    """expert 的任務清單 -> `[N_TASK_OPS, N_TARGET_CELLS]` 的 bool（v5 標籤）。

    `FETCH_*` 不進去 —— 那三個是從 FEED / FERTILIZE / PLACE 的格數導出的
    （`agents/gen0._tasks()` 的補給那一段），目標格永遠是 shed 那四格，
    不是「哪一格該做什麼」的答案。網路預測導出量只會多一個對不上的機會。

    標籤直接讀規劃器的輸出，不從動作反推 —— 跟 v3 的 target 標籤同一個理由。
    """
    grid = np.zeros((C.N_TASK_OPS, C.N_TARGET_CELLS), dtype=bool)
    for _pri, op, x, y, _arg in tasks:
        index = C.TASK_OP_INDEX.get(op)
        if index is not None:
            grid[index, C.target_index(x, y, board)] = True
    return grid


def unpack_demand(bits):
    """`board_demand_bits` -> `[..., N_TASK_OPS, N_TARGET_CELLS]` 的 uint8（0/1）。

    `count` 一定要給 —— 不給的話最後一個 byte 的 4 個 padding bit 會被當成
    第 101~104 格，形狀變 104 而不是 100，之後所有的 reshape 都會錯位。
    """
    return np.unpackbits(bits, axis=-1, count=C.N_TARGET_CELLS)


class Recorder:
    """一局之內累積樣本；`finish()` 補上期末現金後吐出 arrays。"""

    def __init__(self, player):
        self.player = player
        self.spatial, self.scalar, self.step = [], [], []
        self.unit_board, self.unit_pos, self.unit_feat = [], [], []
        self.unit_op, self.unit_qty = [], []
        self.unit_target, self.unit_term_op, self.unit_term_qty = [], [], []
        self.market_board, self.market_op, self.market_qty = [], [], []
        self.demand, self.demand_legal = [], []
        self.skipped = 0

    def record(self, obs, config, action, plan):
        """一個回合：局面 + expert 的答案。"""
        spatial, scalar = C.encode(obs, config)
        positions, feats = C.encode_units(obs, config)
        board = plan["board"]

        index = len(self.spatial)
        self.spatial.append(spatial.astype(np.float16))
        self.scalar.append(scalar)
        self.step.append(int(obs.get("step", 0)))

        # v5：逐格需求。expert 掃出來的每一筆都必須落在 mask 裡面 ——
        # 擋掉的話網路連學都學不到那筆需求，而且不會有人告訴你
        # （`contracts.legal_demand_mask` 的不變式）。所以這裡直接拋錯。
        demand = demand_from_tasks(plan["tasks"], board)
        legal = C.legal_demand_mask(obs, config)
        outside = demand & ~legal
        if outside.any():
            op_index, cell = (int(a[0]) for a in np.nonzero(outside))
            raise AssertionError(
                f"step {obs.get('step')}：expert 要求 "
                f"{C.TASK_OPS[op_index]} @ {C.target_xy(cell, board)}，"
                f"但 legal_demand_mask 擋掉了 —— 放寬 mask，不要改這裡")
        self.demand.append(np.packbits(demand, axis=-1))
        self.demand_legal.append(np.packbits(legal, axis=-1))

        emitted = [action.get("farmer")] + list(action.get("hands") or [])
        for i in range(len(positions)):
            entry = plan["plan"][i] if i < len(plan["plan"]) else None
            if entry is None:
                # 閒置 —— expert 決定「留在原地什麼都不做」。這是誠實的標籤，
                # 網路會忠實學會 gen1 那 18.1% 的閒置（老師是 9.1%）。
                # ⚠️ DAgger 不會修掉 expert 自己的缺點，那要靠換 expert。
                dest, final = (int(positions[i][0]), int(positions[i][1])), ["PASS"]
            else:
                dest, final = entry

            term = C.encode_unit_action(final)
            now = C.encode_unit_action(emitted[i]) if i < len(emitted) else None
            if term is None or now is None:
                self.skipped += 1
                continue

            self.unit_board.append(index)
            self.unit_pos.append(positions[i])
            self.unit_feat.append(feats[i])
            self.unit_op.append(now[0])
            self.unit_qty.append(-1 if now[1] is None else now[1])
            self.unit_target.append(C.target_index(dest[0], dest[1], board))
            self.unit_term_op.append(term[0])
            self.unit_term_qty.append(-1 if term[1] is None else term[1])

        for order in action.get("market") or []:
            decoded = C.encode_market_action(order)
            if decoded is None:
                continue
            self.market_board.append(index)
            self.market_op.append(decoded[0])
            self.market_qty.append(decoded[1])

    def finish(self, rewards, episode_id):
        n = len(self.spatial)
        mine, theirs = rewards[self.player], rewards[1 - self.player]
        present, bucket = dense_market(
            self.market_board, self.market_op, self.market_qty, n)
        return {
            "board_spatial": np.asarray(self.spatial, dtype=np.float16),
            "board_scalar": np.asarray(self.scalar, dtype=np.float32),
            "board_reward": np.full(n, mine / 100_000.0, dtype=np.float32),
            "board_win": np.full(
                n, 1 if mine > theirs else (-1 if mine < theirs else 0), dtype=np.int8),
            "board_step": np.asarray(self.step, dtype=np.int16),
            "board_player": np.full(n, self.player, dtype=np.int8),
            "unit_board": np.asarray(self.unit_board, dtype=np.int32),
            "unit_pos": np.asarray(self.unit_pos, dtype=np.int16).reshape(-1, 2),
            "unit_feat": np.asarray(self.unit_feat, dtype=np.float32).reshape(
                -1, C.N_UNIT_FEATURES),
            "unit_op": np.asarray(self.unit_op, dtype=np.int16),
            "unit_qty": np.asarray(self.unit_qty, dtype=np.int8),
            "unit_target": np.asarray(self.unit_target, dtype=np.int16),
            "unit_term_op": np.asarray(self.unit_term_op, dtype=np.int16),
            "unit_term_qty": np.asarray(self.unit_term_qty, dtype=np.int8),
            "market_board": np.asarray(self.market_board, dtype=np.int32),
            "market_op": np.asarray(self.market_op, dtype=np.int8),
            "market_qty": np.asarray(self.market_qty, dtype=np.int16),
            "board_market_present": present,
            "board_market_qty": bucket,
            # v5 的 demand 標籤與遮罩。存的是 `np.packbits` 之後的 `[N, 11, 13]`
            # 而不是 `[N, 11, 100]` —— 8 倍的差別，而且 `model/train.py` 本來就
            # 要以 packed 的形式放在記憶體裡（一輪 39 萬個盤面，dense 是 427 MB
            # 一份、要兩份）。名字帶 `_bits` 就是提醒不要當 dense 用，
            # 解開用 `unpack_demand()`。
            "board_demand_bits": np.asarray(self.demand, dtype=np.uint8).reshape(
                n, C.N_TASK_OPS, -1),
            "board_demand_legal_bits": np.asarray(
                self.demand_legal, dtype=np.uint8).reshape(n, C.N_TASK_OPS, -1),
            "encoder_version": np.asarray([C.ENCODER_VERSION], dtype=np.int32),
            "episode_id": np.asarray([episode_id], dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
        }


def _load(name):
    """`--policy` / `--expert` 的值 -> 一個 `act(obs, config, ...)`。"""
    from eval.runner import load_spec
    spec = load_spec(name)
    if "builtin" in spec:
        raise SystemExit(f"{name} 是引擎內建 agent，問不到 plan，不能當 expert")
    module_path, _, attr = spec["entry"].partition(":")
    module = __import__(module_path, fromlist=[attr or "act"])
    fn = getattr(module, attr or "act")
    return fn, dict(spec.get("params") or {})


def _play(job):
    """跑一局並錄下 expert 的答案。top-level 才 pickle 得動（Windows spawn）。"""
    policy_name, expert_name, opponent, seed, out_dir = job
    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    os.environ.pop("KAGGRI_LOG_FILE", None)

    from eval.runner import build_agent, load_spec, _quiet_make
    policy_fn, policy_params = _load(policy_name)
    expert_fn, expert_params = _load(expert_name)
    same = policy_name == expert_name

    rec = Recorder(player=0)
    t0 = time.perf_counter()
    err = None

    def me(obs, config):
        expert_action, plan = expert_fn(obs, config, expert_params, return_plan=True)
        rec.record(obs, config, expert_action, plan)
        # 第 0 輪 policy 就是 expert，不必再算一次
        return expert_action if same else policy_fn(obs, config, policy_params)

    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            env = _quiet_make()("kaggriculture",
                                configuration={"seed": seed}, debug=False)
            env.run([me, build_agent(load_spec(opponent))])
        rewards = [float(s.reward or 0.0) for s in env.state]
    except Exception as exc:                    # 單局炸掉不該拖垮整批
        return {"seed": seed, "opponent": opponent, "error":
                f"{type(exc).__name__}: {exc}", "cash": 0.0, "boards": 0}

    arrays = rec.finish(rewards, episode_id=seed)
    dest = Path(out_dir) / f"{policy_name}-{opponent}-{seed:06d}.npz"
    np.savez_compressed(dest, **arrays)
    return {"seed": seed, "opponent": opponent, "error": err,
            "cash": rewards[0], "opp_cash": rewards[1],
            "boards": len(arrays["board_scalar"]),
            "units": len(arrays["unit_op"]), "skipped": rec.skipped,
            # demand 的密度：`model/train.py` 的 dummy 對照組（「所有合法的格子
            # 都做」）的精確率就是這兩個數的比值 —— 太接近 1 代表 mask 已經把
            # 答案框死了，網路沒有判斷可做。
            "demand_on": int(unpack_demand(arrays["board_demand_bits"]).sum()),
            "demand_legal_on": int(
                unpack_demand(arrays["board_demand_legal_bits"]).sum()),
            "bytes": dest.stat().st_size, "elapsed": time.perf_counter() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="gen1", help="誰開車（走出來的局面）")
    ap.add_argument("--expert", default="gen1", help="誰貼標籤（要能吃 return_plan）")
    ap.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="data/dagger/round0")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)

    jobs = [(args.policy, args.expert, opponents[i % len(opponents)],
             args.seed0 + i, str(out_dir)) for i in range(args.games)]

    print(f"policy {args.policy}   expert {args.expert}   "
          f"對手 {opponents}   {args.games} 局   {workers} workers")
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = []
        for n, row in enumerate(pool.imap_unordered(_play, jobs), 1):
            results.append(row)
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}", flush=True)
    elapsed = time.perf_counter() - t0

    bad = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]
    if not ok:
        raise SystemExit(f"全部失敗，第一個錯誤：{bad[0]['error'] if bad else '?'}")

    cash = np.array([r["cash"] for r in ok])
    size = sum(r["bytes"] for r in ok)
    print(f"\n{len(ok)} 局 -> {out_dir}（失敗 {len(bad)}）")
    print(f"  board 樣本   {sum(r['boards'] for r in ok):,}")
    print(f"  unit 樣本    {sum(r['units'] for r in ok):,}"
          f"（編不出來而跳過 {sum(r['skipped'] for r in ok):,}）")
    on = sum(r["demand_on"] for r in ok)
    legal_on = sum(r["demand_legal_on"] for r in ok)
    boards = sum(r["boards"] for r in ok)
    print(f"  demand        每個盤面 {on / max(1, boards):.1f} 筆需求 / "
          f"{legal_on / max(1, boards):.1f} 個合法格"
          f"（dummy 精確率 {on / max(1, legal_on):.1%}）")
    print(f"  磁碟          {size / 2**20:,.0f} MB")
    print(f"  吞吐          {len(ok) / elapsed * 3600:,.0f} 局/小時")
    # value head 要靠這個範圍才學得到東西 —— replay 資料只有 0.363~1.402
    print(f"  期末現金      min {cash.min():,.0f}  中位 {np.median(cash):,.0f}  "
          f"max {cash.max():,.0f}  標準差 {cash.std():,.0f}")
    for opp in opponents:
        rows = [r["cash"] for r in ok if r["opponent"] == opp]
        if rows:
            print(f"    vs {opp:<16}{len(rows):>4} 局  平均 {np.mean(rows):>9,.0f}")
    for r in bad[:3]:
        print(f"  ⚠️ seed {r['seed']}：{r['error']}", file=sys.stderr)
    print(f"  ENCODER_VERSION {C.ENCODER_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
