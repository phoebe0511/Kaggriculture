"""v4 的市場模仿：`legal_market_mask` 的不變式與稠密標籤的往返。

市場是**集合型**輸出 —— 一回合 0~10 筆訂單、每筆帶數量、而且逐筆相依
（前面的買單把錢花光，後面的就失敗）。`contracts.py` 的 docstring 早就標了
「跟 unit 的 44 選 1 是完全不同的問題」，所以這裡的測試要比 unit 那邊更嚴：
標籤只要在稀疏↔稠密之間掉一筆，訓練訊號就少一塊而且不會報錯。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest

import contracts as C
from harness.build_dataset import build_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "temp" / "episodes"


def _episodes(limit=2):
    return sorted(glob.glob(str(RAW_DIR / "*.json")))[:limit]


@pytest.fixture(scope="module")
def arrays():
    files = _episodes()
    if not files:
        pytest.skip(f"{RAW_DIR} 沒有原始 replay（temp/ 被 gitignore）")
    out = [a for a, _ in (build_episode(p) for p in files) if a is not None]
    if not out:
        pytest.skip("沒有引擎版本相符的 replay")
    return out


# --------------------------------------------------------------------------
# 詞彙表與桶
# --------------------------------------------------------------------------

def test_market_vocabulary_covers_every_order_in_the_corpus(arrays):
    """老師下過的每一筆訂單都要編得出來。

    編不出來就整筆丟掉，而丟掉的分布不是隨機的 —— 會系統性地少學某一類採購。
    """
    files = _episodes()
    unknown, total = {}, 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for entries in data["steps"]:
            for entry in entries:
                action = entry.get("action")
                if not isinstance(action, dict):
                    continue
                for order in action.get("market") or []:
                    if not isinstance(order, list) or not order:
                        continue
                    total += 1
                    if C.encode_market_action(order) is None:
                        key = tuple(order[:2])
                        unknown[key] = unknown.get(key, 0) + 1

    assert total > 1_000, f"只有 {total} 筆訂單，樣本太少"
    assert not unknown, f"市場詞彙表漏了：{unknown}"


def test_qty_buckets_round_down_never_up():
    """數量桶一律往下取。

    賣少一點只是少賺；賣爆會把價格砸到地板（`engine-notes.md` §6：MELON 的
    `above_func = sq / 3.60`，賣 40 個價格砍半）。所以寧可低估。
    """
    for qty in range(1, 200):
        value = C.market_qty_value(C.market_qty_bucket(qty))
        assert value <= qty, f"qty {qty} 被放大成 {value}"
    # 小數量要精確 —— 91.8% 的訂單在這個範圍
    for qty in range(1, 13):
        assert C.market_qty_value(C.market_qty_bucket(qty)) == qty


# --------------------------------------------------------------------------
# mask 不變式
# --------------------------------------------------------------------------

def test_market_mask_rarely_blocks_what_the_teacher_actually_did():
    """跟 `legal_unit_mask` 同一條不變式、同一個門檻（0.5%）。

    ⚠️ 這條紅掉的正確反應是**放寬 mask**，不是調高門檻。擋掉老師下過的訂單
    等於網路連學都學不到那一類採購 —— 而 2026-08-20 量到的致命洞正是
    「動物買太少」。
    """
    files = _episodes(limit=2)
    if not files:
        pytest.skip(f"{RAW_DIR} 沒有原始 replay")

    blocked = total = 0
    by_op = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        config = data["configuration"]
        steps = data["steps"]
        for t, entries in enumerate(steps[:-1]):
            shared = entries[0]["observation"]
            for player in (0, 1):
                action = steps[t + 1][player].get("action")
                if not isinstance(action, dict):
                    continue
                obs = dict(shared)
                obs.update(entries[player]["observation"])
                obs["player"] = player
                obs.setdefault("step", t)

                mask = C.legal_market_mask(obs, config)
                for order in action.get("market") or []:
                    decoded = C.encode_market_action(order)
                    if decoded is None:
                        continue
                    total += 1
                    if not mask[decoded[0]]:
                        blocked += 1
                        key = C.MARKET_OPS[decoded[0]]
                        by_op[key] = by_op.get(key, 0) + 1

    assert total > 1_000
    rate = blocked / total
    assert rate < 0.005, f"擋掉了 {blocked}/{total} = {rate:.3%}，分布 {by_op}"


# --------------------------------------------------------------------------
# 稀疏 <-> 稠密
# --------------------------------------------------------------------------

def test_dense_labels_match_the_summed_sparse_orders(arrays):
    """稠密標籤要等於「同一回合同一個 op 的數量相加後轉桶」。

    ⚠️ 相加而不是取後者：引擎逐單位重新報價，`SELL 5` + `SELL 5` 與 `SELL 10`
    完全等價；`HIRE` 兩筆就是雇兩個人。實測同一回合同一個 op 重複的佔 9.6%，
    取後者的話那 9.6% 的數量標籤全部偏低。
    """
    for data in arrays:
        n_boards = len(data["board_scalar"])
        summed = np.zeros((n_boards, C.N_MARKET_OPS), dtype=np.int64)
        for b, op, q in zip(data["market_board"], data["market_op"],
                            data["market_qty"]):
            summed[int(b), int(op)] += int(q)

        present = data["board_market_present"].astype(bool)
        assert (present == (summed > 0)).all(), "present 跟稀疏訂單對不上"

        rows, cols = np.nonzero(present)
        assert len(rows) > 100
        for b, op in zip(rows, cols):
            assert int(data["board_market_qty"][b, op]) == \
                C.market_qty_bucket(summed[b, op])
        # 沒下單的地方一律 -1，不能是 0（0 是「數量 1」那個桶）
        assert (data["board_market_qty"][~present] == -1).all()


def test_duplicate_orders_are_common_enough_to_matter(arrays):
    """相加這件事不是理論顧慮 —— 實測 9.6% 的 (回合, op) 組合是重複的。

    這條在測「上面那個相加測試有沒有真的被觸發」。重複率掉到 0 就表示
    抽取那邊悄悄改成取後者了，上面那條會變成恆真。
    """
    dup = combos = 0
    for data in arrays:
        seen = {}
        for b, op in zip(data["market_board"], data["market_op"]):
            key = (int(b), int(op))
            seen[key] = seen.get(key, 0) + 1
        combos += len(seen)
        dup += sum(1 for v in seen.values() if v > 1)
    assert combos > 500
    assert dup / combos > 0.03, f"重複率只有 {dup / combos:.1%}，實測基準 9.6%"


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------

def test_decode_always_supplies_required_arguments():
    """帶參數的 op 一定要把參數補上 —— 引擎對格式不對的訂單是**靜默忽略**。

    跟 `decode_unit` 同一個坑：`gen0` 第一版漏了 `["PLACE"]` 的物品名，
    整季 337 個回合白丟、6 個雞舍全空。
    """
    for j, (op, item) in enumerate(C.MARKET_OPS):
        order = C.decode_market(j, 5)
        assert order[0] == op
        if op in ("HIRE", "BUY_LAND"):
            assert len(order) == 1, f"{op} 不該帶參數：{order}"
        else:
            assert len(order) == 3 and order[1] == item, f"{op} 少了參數：{order}"
            assert order[2] >= 1


def test_clamp_never_sells_more_than_the_shed_holds():
    """`SELL` 的數量夾到 shed 實有量；買單**不夾現金**。

    不夾現金是因為引擎逐筆結算，而 `MARKET_OPS` 讓 `SELL` 排在 `BUY_*` 前面
    —— 賣單先進帳，後面的買單才有錢。用送出當下的現金去夾會夾掉本來會成功的。
    """
    obs = {
        "player": 0,
        "farms": [{"money": 100, "unlocked_quadrants": ["NW"], "hires_today": 0}],
        "private": {"shed": {"WHEAT": 3}},
        "market": {"prices": {}},
    }
    sell_wheat = C.MARKET_OP_INDEX[("SELL", "WHEAT")]
    assert C.clamp_market_qty(sell_wheat, 80, obs) == 3
    sell_milk = C.MARKET_OP_INDEX[("SELL", "MILK")]
    assert C.clamp_market_qty(sell_milk, 5, obs) == 0

    # 現金只有 $100、COW 要 $400，仍然照送 —— 同一回合的賣單會補上
    cow = C.MARKET_OP_INDEX[("BUY_ANIMAL", "COW")]
    assert C.clamp_market_qty(cow, 10, obs) == 10

    # HIRE 例外：價格是當天第 n 個的 fib，所以「還雇得起幾個」算得出來。
    # 實測老師 57.6% 的雇工回合一次送 5 筆 —— hands 每天被清空，開工那一刻
    # 要一次補滿，只送 1 筆等於每天上午都在慢慢補人。
    hire = C.MARKET_OP_INDEX[("HIRE", None)]
    assert C.clamp_market_qty(hire, 5, obs) == 5          # $100 雇得起 5 個以上
    obs["farms"][0]["money"] = 3
    # 價格是 1, 1, 2, 3, ...：$3 付得起前兩個（1+1），第三個要 2 元只剩 1 元
    assert C.clamp_market_qty(hire, 5, obs) == 2
