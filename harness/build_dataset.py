"""把 ladder replay 轉成訓練用的陣列。

    python -m harness.build_dataset temp/episodes --out data/dataset

## 老師是誰

不是我們自己的 self-play —— 是 rating 2979~3229 的真實 ladder 玩家。
2026-08-19 量到 Gen1 對他們 6 勝 154 負，所以模仿 Gen1 沒有意義
（imitation learning 的天花板就是被模仿的對象）。詳見
`docs/memory/journal/2026-08-19.md`。

## ⚠️ observation 與 action 差一格

kaggle-environments 每一格存的是「這個狀態」加上「**把狀態推到這裡的那個
動作**」。所以在 `steps[t]` 的局面下做的決定，記在 `steps[t + 1]["action"]`。

錯位的資料拿去訓練幾乎驗不出來 —— loss 照樣會降，網路只是學到「上一回合的
局面配這一回合的動作」。2026-08-19 這個坑踩了兩次（`agents/replay.py` 一次、
這裡一次），是 `contracts.legal_unit_mask` 的不變式把它抓出來的：錯位時
14.66% 的老師動作會落在 mask 之外，配對正確時只有 0.12%。

## 輸出結構：board 兩層、unit 一層

每回合的 unit 數不固定（1~15），padding 會浪費空間又要另外記長度。改成
兩層扁平：

    board_spatial  [S, 38, 10, 10]  float16   S = 719 步 × 2 玩家
    board_scalar   [S, 75]          float32
    board_reward   [S]              float32   該玩家的期末現金 / 100k
    board_win      [S]              int8      1 贏 / 0 平 / -1 輸
    board_step     [S]              int16
    board_player   [S]              int8

    unit_board     [U]              int32     指回上面第幾個 board 樣本
    unit_pos       [U, 2]           int16
    unit_feat      [U, 18]          float32
    unit_op        [U]              int16     44 選 1 的標籤（**這一步**做什麼）
    unit_qty       [U]              int8      PICKUP / PLACE 的數量，其餘 -1

## v3 的 segment 標籤

`unit_op` 是「這一步做什麼」，而老師 47.6% 的 unit-turn 是 MOVE —— 那些步的
標籤裡沒有「要去哪」。「這個 unit 要去哪」不在任何 observation 欄位裡，所以只看
盤面的網路每回合重新猜目標，猜的一變就掉頭（實測方向反轉率 32.9%，老師 9.2%）。

所以另外抽一組標籤：一段連續 MOVE 加上結尾那個非 MOVE 動作叫一個 **segment**，
segment 裡每一步的標籤都是「**終點在哪一格、到了做什麼、做多少**」。

    unit_target    [U]              int16     終點格 = y * board + x，沒有終點是 -1
    unit_term_op   [U]              int16     終點動作（44 選 1）
    unit_term_qty  [U]              int8      終點動作的數量，其餘 -1

實測（3 局 × 2 玩家、41,484 個 unit-turn）：segment 平均 1.79 步、56.4% 只有
1 步；4.2% 的 MOVE 走到一半就日終、沒有終點動作，那些 `unit_target = -1`，
訓練時遮掉不算 loss（跟 `unit_qty` 同一套做法）。

⚠️ **unit 身分只在一天之內成立。** 引擎每天結算會 `farm["hands"] = []` 且
`private["inventories"] = [{}]`（`kaggriculture.py:880`），主農夫也回到
`_default_spawn`。天之內 `_do_hire` 是 append（`kaggriculture.py:708`），
所以 hand index 當天穩定。**重建 segment 一定要按天切，不可以跨天串。**

市場標籤存兩份 —— 稀疏的原始訂單，以及訓練直接吃的稠密版：

    market_board          [M]        int32
    market_op             [M]        int8      MARKET_OPS 的索引
    market_qty            [M]        int16     實際數量（不是桶）

    board_market_present  [S, 21]    int8      這回合有沒有下這個 op
    board_market_qty      [S, 21]    int16     數量桶，沒下的是 -1

## v4：為什麼市場也要模仿

v1~v3 只有 unit 走網路，`action["market"]` 沿用 `agents/gen0.py` 的規則式。
2026-08-20 量到那是致命的：老師整季買 553 個 WHEAT 餵動物、14 隻動物，
gen0 只買 13 個 WHEAT、3 隻 COW。**網路的 unit 動作是在「老師那種農場」上學的，
卻被放到一個 gen0 蓋出來的農場上跑。**

按天比對狀態分布：day 6 動物就跌破老師的 p5（我們 1 隻、老師 p5 是 6 隻），
day 14 起現金在第 0.0 百分位。而動物是市場買的，不是 unit 決定的。

`spatial` 用 float16 是為了大小：float32 每個樣本 15.2 KB、60 局約 1.3 GB。
通道多半是 0/1，float16 完全表示得了，連續的那幾個（距離、age）也只需要
兩三位有效數字。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contracts as C  # noqa: E402

#: v4 起這份表住在 `contracts.py`（比賽端也要用它把網路輸出轉回訂單）。
#: 這裡的別名留著，舊的 import 路徑不用改。順序完全沒動。
MARKET_OPS = C.MARKET_OPS
MARKET_OP_INDEX = C.MARKET_OP_INDEX

#: MOVE 的動作編號。segment 就是「連續這幾個，直到出現別的」。
MOVE_OP_INDICES = frozenset(
    C.UNIT_OP_INDEX[(move, None)] for move in C.MOVES)


def dense_market(market_board, market_op, market_qty, n_boards):
    """稀疏的市場訂單 -> 逐盤面的 `(present [S,21] int8, qty 桶 [S,21] int16)`。

    ⚠️ 同一回合同一個 op 出現兩次以上的佔 **9.6%**，數量要**相加**不是取後者。
    引擎逐單位重新報價，`SELL 5` + `SELL 5` 與 `SELL 10` 完全等價；`HIRE` 兩筆
    就是雇兩個人。相加之後才轉桶。

    `harness/rollout.py` 也用這支 —— DAgger 產的資料要跟 replay 抽的長一樣，
    `model/train.py` 才吃得下同一份 schema。
    """
    dense_qty = np.zeros((n_boards, C.N_MARKET_OPS), dtype=np.int32)
    for b, op, q in zip(market_board, market_op, market_qty):
        dense_qty[int(b), int(op)] += int(q)
    present = dense_qty > 0
    bucket = np.full((n_boards, C.N_MARKET_OPS), -1, dtype=np.int16)
    for b, op in zip(*np.nonzero(present)):
        bucket[b, op] = C.market_qty_bucket(dense_qty[b, op])
    return present.astype(np.int8), bucket


def segment_labels(rows, turns_per_day, board):
    """把逐步標籤反推成 segment 標籤。

    `rows` 是 `(step, unit_slot, op_index, qty_index, x, y)` 的序列，**必須照
    step 遞增**。回傳跟 `rows` 等長的 `(target_index, term_op, term_qty)` 清單，
    沒有終點動作的是 `(-1, -1, -1)`。

    做法是按 `(day, unit_slot)` 分組後**往回掃**：看到非 MOVE 就把
    「這一格 + 這個動作」記成 carry，看到 MOVE 就套用 carry。

    兩個地方會把 carry 清掉：

    - 跨天（hands 每天被清空，index 指的不是同一個人）
    - step 不連續（中間那一步的動作 `encode_unit_action` 認不得，被跳過了）。
      寧可讓前面那幾步變成沒有標籤，也不要讓它們指到一個隔著未知動作的終點
    """
    groups = {}
    for i, (step, slot, _op, _qty, _x, _y) in enumerate(rows):
        groups.setdefault((step // turns_per_day, slot), []).append(i)

    labels = [(-1, -1, -1)] * len(rows)
    for indices in groups.values():
        carry = (-1, -1, -1)
        next_step = None
        for i in reversed(indices):
            step, _slot, op, qty, x, y = rows[i]
            if next_step is not None and next_step != step + 1:
                carry = (-1, -1, -1)          # 中間有認不得的動作，鏈斷掉
            next_step = step
            if op in MOVE_OP_INDICES:
                labels[i] = carry
            else:
                carry = (C.target_index(x, y, board), op,
                         -1 if qty is None else qty)
                labels[i] = carry
    return labels


def build_episode(path, engine="1.32.7", team=None):
    """一局 replay -> arrays dict。引擎版本不符或沒有指定隊伍時回 `None`。

    `team` 是隊伍名字。給了就只抽那一方的樣本 —— 8 支頂端隊伍雖然打同一套，
    細節仍有分歧（tetsuya 15 建物還養鵝、ReCurSiON 12 建物種瓜），而網路
    分不出樣本來自誰，學到的是平均。老師之間有分歧時，平均出來的動作可能
    兩邊都不像。只用一支就沒有這個問題。
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("module_version")
    if engine and version != engine:
        return None, f"引擎 {version} != {engine}"

    names = (data.get("info") or {}).get("TeamNames") or []
    if team:
        players = [i for i, n in enumerate(names) if n == team]
        if not players:
            return None, f"沒有 {team}（這局是 {names}）"
    else:
        players = [0, 1]

    config = data["configuration"]
    steps = data["steps"]
    rewards = data.get("rewards") or [0.0, 0.0]

    board_spatial, board_scalar = [], []
    board_reward, board_win, board_step, board_player = [], [], [], []
    unit_board, unit_pos, unit_feat, unit_op, unit_qty = [], [], [], [], []
    market_board, market_op, market_qty = [], [], []
    # segment 標籤要按 (player, day, unit slot) 分組回推，先把這三個記下來
    row_player, row_step, row_slot = [], [], []
    skipped_units = 0
    board = C.BOARD_SIZE

    for t, entries in enumerate(steps[:-1]):
        shared = entries[0]["observation"]
        for player in players:
            # ⚠️ +1：見 module docstring。
            action = steps[t + 1][player].get("action")
            if not isinstance(action, dict):
                continue

            obs = dict(shared)
            obs.update(entries[player]["observation"])
            obs["player"] = player
            obs.setdefault("step", t)

            spatial, scalar = C.encode(obs, config)
            positions, feats = C.encode_units(obs, config)
            board = len(obs["farms"][player]["tiles"])

            index = len(board_spatial)
            board_spatial.append(spatial.astype(np.float16))
            board_scalar.append(scalar)
            board_reward.append(rewards[player] / 100_000.0)
            mine, theirs = rewards[player], rewards[1 - player]
            board_win.append(1 if mine > theirs else (-1 if mine < theirs else 0))
            board_step.append(t)
            board_player.append(player)

            units = [action.get("farmer")] + list(action.get("hands") or [])
            for i, unit_action in enumerate(units):
                if i >= len(positions):
                    # 送出的 hands 比實際雇工多。引擎靜默略過
                    # （`kaggriculture.py:318` 的 `pos is None -> return`），
                    # 沒有對應的 unit，不能當訓練樣本。
                    continue
                decoded = C.encode_unit_action(unit_action)
                if decoded is None:
                    skipped_units += 1
                    continue
                op_index, qty_index = decoded
                unit_board.append(index)
                unit_pos.append(positions[i])
                unit_feat.append(feats[i])
                unit_op.append(op_index)
                unit_qty.append(-1 if qty_index is None else qty_index)
                row_player.append(player)
                row_step.append(t)
                row_slot.append(i)

            for order in action.get("market") or []:
                decoded = C.encode_market_action(order)
                if decoded is None:
                    continue
                market_board.append(index)
                market_op.append(decoded[0])
                market_qty.append(decoded[1])

    # --- v3：往回推 segment 標籤 ------------------------------------------
    # 一定要按 player 分開推，否則兩個玩家同一天同一個 slot 會被串成一條。
    turns_per_day = int(config.get("turnsPerDay", 24))
    unit_target = [-1] * len(unit_op)
    unit_term_op = [-1] * len(unit_op)
    unit_term_qty = [-1] * len(unit_op)
    for player in players:
        picked = [i for i in range(len(unit_op)) if row_player[i] == player]
        rows = [(row_step[i], row_slot[i], unit_op[i], unit_qty[i],
                 int(unit_pos[i][0]), int(unit_pos[i][1])) for i in picked]
        labels = segment_labels(rows, turns_per_day, board)
        for j, i in enumerate(picked):
            unit_target[i], unit_term_op[i], unit_term_qty[i] = labels[j]

    arrays = {
        "board_spatial": np.asarray(board_spatial, dtype=np.float16),
        "board_scalar": np.asarray(board_scalar, dtype=np.float32),
        "board_reward": np.asarray(board_reward, dtype=np.float32),
        "board_win": np.asarray(board_win, dtype=np.int8),
        "board_step": np.asarray(board_step, dtype=np.int16),
        "board_player": np.asarray(board_player, dtype=np.int8),
        "unit_board": np.asarray(unit_board, dtype=np.int32),
        "unit_pos": np.asarray(unit_pos, dtype=np.int16).reshape(-1, 2),
        "unit_feat": np.asarray(unit_feat, dtype=np.float32).reshape(
            -1, C.N_UNIT_FEATURES),
        "unit_op": np.asarray(unit_op, dtype=np.int16),
        "unit_qty": np.asarray(unit_qty, dtype=np.int8),
        "unit_target": np.asarray(unit_target, dtype=np.int16),
        "unit_term_op": np.asarray(unit_term_op, dtype=np.int16),
        "unit_term_qty": np.asarray(unit_term_qty, dtype=np.int8),
        "market_board": np.asarray(market_board, dtype=np.int32),
        "market_op": np.asarray(market_op, dtype=np.int8),
        "market_qty": np.asarray(market_qty, dtype=np.int16),
        "encoder_version": np.asarray([C.ENCODER_VERSION], dtype=np.int32),
        "episode_id": np.asarray([data.get("episode_id") or 0], dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
    }
    present, bucket = dense_market(
        market_board, market_op, market_qty, len(board_scalar))
    arrays["board_market_present"] = present
    arrays["board_market_qty"] = bucket

    dangling = sum(1 for v in unit_target if v < 0)
    return arrays, f"skipped_units={skipped_units} dangling={dangling}"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="裝著原始 replay JSON 的目錄")
    ap.add_argument("--out", default="data/dataset", help="輸出目錄")
    ap.add_argument("--engine", default="1.32.7",
                    help="只收這個引擎版本，空字串關掉過濾")
    ap.add_argument("--team", default=None,
                    help="只抽這支隊伍的樣本，例如 --team カワシギ")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 局（除錯用）")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.source, "*.json")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"{args.source} 裡沒有 JSON")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_boards = total_units = total_orders = total_dangling = 0
    written = skipped = 0
    for n, path in enumerate(files, 1):
        arrays, note = build_episode(path, args.engine, args.team)
        if arrays is None:
            print(f"  skip {os.path.basename(path)}：{note}", file=sys.stderr)
            skipped += 1
            continue
        dest = out_dir / (Path(path).stem + ".npz")
        np.savez_compressed(dest, **arrays)
        total_boards += len(arrays["board_scalar"])
        total_units += len(arrays["unit_op"])
        total_orders += len(arrays["market_op"])
        total_dangling += int((arrays["unit_target"] < 0).sum())
        written += 1
        if n % 5 == 0 or n == len(files):
            print(f"  {n}/{len(files)}  board {total_boards:,}  "
                  f"unit {total_units:,}  market {total_orders:,}", flush=True)

    size = sum(p.stat().st_size for p in out_dir.glob("*.npz"))
    print(f"\n{written} 局 -> {out_dir}（跳過 {skipped}）")
    print(f"  board 樣本   {total_boards:,}")
    print(f"  unit 樣本    {total_units:,}")
    print(f"  market 訂單  {total_orders:,}")
    # 健康檢查：走到一半就日終、沒有終點動作的比例。實測 4.2%（3 局 × 2 玩家）。
    # 明顯偏高就是 segment 按天切錯或 unit slot 對錯了。
    frac = total_dangling / max(1, total_units)
    print(f"  無終點的 unit {total_dangling:,}（{frac:.1%}，實測基準 4.2%）")
    print(f"  磁碟          {size / 2**20:,.0f} MB")
    print(f"  ENCODER_VERSION {C.ENCODER_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
