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
    board_scalar   [S, 66]          float32
    board_reward   [S]              float32   該玩家的期末現金 / 100k
    board_win      [S]              int8      1 贏 / 0 平 / -1 輸
    board_step     [S]              int16
    board_player   [S]              int8

    unit_board     [U]              int32     指回上面第幾個 board 樣本
    unit_pos       [U, 2]           int16
    unit_feat      [U, 18]          float32
    unit_op        [U]              int16     44 選 1 的標籤
    unit_qty       [U]              int8      PICKUP / PLACE 的數量，其餘 -1

市場標籤照樣存（v1 不訓練，但存下來的話要加第二個 head 不用重跑）：

    market_board   [M]              int32
    market_op      [M]              int8      MARKET_OPS 的索引
    market_qty     [M]              int16

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

#: 市場動作詞彙。順序寫死，跟 `contracts.UNIT_OPS` 同樣是 schema 的一部分。
MARKET_OPS = (
    ("HIRE", None),
    ("BUY_LAND", None),
    *(("SELL", item) for item in C.PRODUCT_ORDER),
    # 引擎只讓 WHEAT / FERTILIZER 用 BUY_PRODUCT（`engine-notes.md` §1）
    ("BUY_PRODUCT", "WHEAT"),
    ("BUY_PRODUCT", "FERTILIZER"),
    *(("BUY_SEED", crop) for crop in C.CROP_ORDER),
    *(("BUY_ANIMAL", animal) for animal in C.ANIMAL_ORDER),
)
MARKET_OP_INDEX = {key: i for i, key in enumerate(MARKET_OPS)}


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
    skipped_units = 0

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

            for order in action.get("market") or []:
                if not isinstance(order, list) or not order:
                    continue
                key = (order[0], order[1] if len(order) > 1 else None)
                if key not in MARKET_OP_INDEX:
                    key = (order[0], None)
                if key not in MARKET_OP_INDEX:
                    continue
                market_board.append(index)
                market_op.append(MARKET_OP_INDEX[key])
                market_qty.append(int(order[2]) if len(order) > 2 else 1)

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
        "market_board": np.asarray(market_board, dtype=np.int32),
        "market_op": np.asarray(market_op, dtype=np.int8),
        "market_qty": np.asarray(market_qty, dtype=np.int16),
        "encoder_version": np.asarray([C.ENCODER_VERSION], dtype=np.int32),
        "episode_id": np.asarray([data.get("episode_id") or 0], dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
    }
    return arrays, f"skipped_units={skipped_units}"


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

    total_boards = total_units = total_orders = 0
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
        written += 1
        if n % 5 == 0 or n == len(files):
            print(f"  {n}/{len(files)}  board {total_boards:,}  "
                  f"unit {total_units:,}  market {total_orders:,}", flush=True)

    size = sum(p.stat().st_size for p in out_dir.glob("*.npz"))
    print(f"\n{written} 局 -> {out_dir}（跳過 {skipped}）")
    print(f"  board 樣本   {total_boards:,}")
    print(f"  unit 樣本    {total_units:,}")
    print(f"  market 訂單  {total_orders:,}")
    print(f"  磁碟          {size / 2**20:,.0f} MB")
    print(f"  ENCODER_VERSION {C.ENCODER_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
