"""產候選聯合動作 —— 網路當提案者，`gen0` 的動作當基準線。

## 為什麼要重做一組

`tools/value_probe.py` 的候選只有 8 個（[value_probe.py:118-120](tools/value_probe.py#L118)）：

    argmax（不改）+ 6 個「把第 k 個 unit 換成它的**第 2 名**」+ 1 個強迫買種子

只換**一個** unit、只換到**第 2 名**、不換目標格、基準線是網路自己。
journal 08-24 §5 記的「改一個回合的上限 +2.42%」是**那 8 個窄候選**的上限，
不是聯合動作空間（13 units × 44 個 `UNIT_OPS` × 目標格 + 市場訂單）的上限。

這裡把候選擴大，並且**把基準線換成 `gen0` 自己的動作** —— 這樣才量得到
「網路跟 gen0 意見不同時誰對」，那是 value_probe 結構上量不到的東西。

## 五類候選

1. `gen0`      —— 基準線，`gen0` 自己的動作
2. `net`       —— 網路的完整聯合動作（`gen2_model` 實際會送出的那個）
3. `u{i}r{r}`  —— 只把第 i 個 unit 換成網路偏好的第 r 名，其餘照 `gen0`
4. `m{m}s{s}`  —— 隨機挑 m 個 unit，各換成一個**真正跟 gen0 不同**的替代動作
5. `mk-*`      —— 市場訂單的變體（網路的、強迫買種子、強迫賣、全開、門檻 ±3、清空）

## 🩸 合法性一定要驗

引擎遇到非法動作是**靜默忽略**。不驗的話會產出一堆「其實沒生效」的候選，
它們的分數會全部等於某個基準，看起來像「這些改動沒差」——
實際上是根本沒送出去。所以每個候選都過 `serving.action_validation.assert_legal_action`，
不過的直接丟掉並計數。

種子是**全體 unit 共用、先搶先贏**的（`gen2_model._choose` 的註解）：把某個 unit
換成沒有種子的 `PLANT`，引擎會靜默忽略。`assert_legal_action` 抓得到這個
（`_validate_units` 會數同回合的 `PLANT` 需求對上 `private["seeds"]`）。
"""
from __future__ import annotations

import numpy as np

import contracts as C
from agents import gen2_model as G
from serving.action_validation import IllegalAction, assert_legal_action

#: `BUY_SEED` / `SELL` 的 op index，市場變體要用。
SEED_OPS = tuple(j for j, (op, _i) in enumerate(C.MARKET_OPS) if op == "BUY_SEED")
SELL_OPS = tuple(j for j, (op, _i) in enumerate(C.MARKET_OPS) if op.startswith("SELL"))


def heads(obs, config):
    """跑一次網路，回傳 7 個 head。"""
    spatial, scalar = C.encode(obs, config)
    positions, feats = C.encode_units(obs, config)
    return G._policy()(spatial, scalar, positions, feats)


def _as_action(units, market):
    return {"farmer": units[0], "hands": list(units[1:]), "market": list(market)}


def _unit_ranked(op_logits, qty_logits, mask, i, n_rank):
    """第 i 個 unit 依網路偏好排序的前 n_rank 個**合法**動作。"""
    scores = np.where(mask[i], op_logits[i], -np.inf)
    order = np.argsort(-scores)
    qty_index = int(np.argmax(qty_logits[i]))
    out = []
    for op_index in order[:n_rank]:
        if not np.isfinite(scores[op_index]):
            break                                    # 後面都不合法
        out.append(C.decode_unit(int(op_index), qty_index))
    return out


def _legal_market_subset(obs, config, units, orders):
    """只留下驗得過的市場訂單 —— 逐筆加進去試，成交不了的丟掉。

    🩸 市場那半是**所有候選共用**的。基準線裡有一筆買不起的訂單，就會讓
    `assert_legal_action` 把幾乎所有候選一起擋掉。2026-08-26 seed 4 實測：
    day 1 hour 12 的 `['BUY_SEED', 'MELON', 1]` 買不起，56 個候選被擋掉 55 個，
    連基準線自己都沒了，`search_action` 直接拋 `ValueError` 整局死掉。
    """
    kept = []
    for order in orders:
        try:
            assert_legal_action(obs, config, _as_action(units, kept + [order]))
        except IllegalAction:
            continue
        kept.append(order)
    return kept


def _market_variants(obs, config, mk_present, mk_qty, base_market,
                     base_label="gen0"):
    """市場訂單的候選。回傳 `[(標籤, orders), ...]`。"""
    def decode(threshold):
        return C.decode_market_orders(mk_present, mk_qty, obs, config,
                                      threshold=threshold)

    default = G.market_thresholds()
    seed_forced = default.copy()
    seed_forced[list(SEED_OPS)] = -99.0
    sell_forced = default.copy()
    sell_forced[list(SELL_OPS)] = -99.0
    all_forced = default.copy()
    all_forced[:] = -99.0

    # 🩸 門檻要拉開才看得到差別。2026-08-25 在 seed 7 / day 12 實測 ±1.0 的
    # `loose` / `tight` / `net` / `none` **全部都是 0 筆訂單**，跟 gen0 一樣，
    # 去重之後只剩 `seed` 一個候選。市場那半等於沒有搜到。
    return [
        # 🩸 這一筆的名字要跟 `generate()` 的 `base_label` 一致，否則
        # `base_label != "gen0"` 時會多產一個跟基準線重複的 `mk-gen0`。
        (base_label, list(base_market)),
        ("net", decode(default)),
        ("seed", decode(seed_forced)),
        ("sell", decode(sell_forced)),
        ("all", decode(all_forced)),
        ("loose3", decode(default - 3.0)),           # 門檻放鬆 = 下更多單
        ("tight3", decode(default + 3.0)),           # 門檻收緊 = 下更少單
        ("none", []),
    ]


def generate(obs, config, base_action, rng, n_rank=4, multi_sizes=(2, 3, 5),
             n_multi=6, base_label="gen0"):
    """產候選。回傳 `([(label, action), ...], 被擋掉的數量)`。

    `base_action` 是 `gen0` 在這個盤面的動作 —— **一定會出現在結果裡**，
    `search.rollout_search.search_action` 靠它保證「搜不會比不搜差」。
    """
    op_l, qty_l, _target, mk_present, mk_qty, _value, _demand = heads(obs, config)
    mask = C.legal_unit_mask(obs, config)

    base_units = [base_action["farmer"]] + list(base_action.get("hands") or [])
    base_market = list(base_action.get("market") or [])
    n_units = min(len(base_units), mask.shape[0])

    # 🩸 基準線是 policy 自己的輸出，**不是我們產的候選**。
    # `assert_legal_action` 存在的理由是擋我們自己產的候選有 bug（引擎對非法
    # 動作靜默忽略，不驗的話會搜到一堆「其實沒生效」的候選）。基準線非法時
    # 引擎同樣會忽略那部分，而那個結果**就是**「不搜會怎樣」——
    # 所以基準線一律進候選、不驗，`search_action` 的下界保證仍然成立。
    #
    # 但衍生候選共用的市場單要先清乾淨，不然它們會集體陪葬（見
    # `_legal_market_subset` 的 docstring）。
    base_illegal = None
    try:
        assert_legal_action(obs, config, _as_action(base_units, base_market))
    except IllegalAction as exc:
        base_illegal = str(exc)
    shared_market = (
        _legal_market_subset(obs, config, base_units, base_market)
        if base_illegal else base_market)

    net_units = G._choose(op_l, qty_l, mask, obs)
    markets = _market_variants(obs, config, mk_present, mk_qty, shared_market,
                               base_label=base_label)
    market_by_name = dict(markets)

    raw = [(base_label, base_units, base_market)]

    # 網路的完整聯合動作 —— 一個「完全不同的意見」。
    raw.append(("net", list(net_units), market_by_name["net"]))
    # 拆開來看是 unit 的意見還是市場的意見在起作用。
    raw.append(("net-units", list(net_units), shared_market))

    # 單一 unit 換成網路偏好的前幾名（跟 gen0 相同的那個跳過）。
    for i in range(n_units):
        for rank, alt in enumerate(_unit_ranked(op_l, qty_l, mask, i, n_rank)):
            if alt == base_units[i]:
                continue                             # 跟 gen0 一樣，不是新候選
            units = list(base_units)
            units[i] = alt
            raw.append((f"u{i}r{rank}", units, shared_market))

    # 多個 unit 同時換。
    # 🩸 換成網路的**第一名**沒有用 —— 網路和 gen0 常常同意，換了等於沒換，
    # 去重之後 12 個 multi 候選只活下來 3 個（2026-08-25 實測）。改成從
    # 「真正跟 gen0 不同的替代動作」裡抽。
    alts = {}
    for i in range(n_units):
        alts[i] = [a for a in _unit_ranked(op_l, qty_l, mask, i, n_rank)
                   if a != base_units[i]]
    swappable = [i for i in range(n_units) if alts[i]]
    for m in multi_sizes:
        if m > len(swappable):
            continue
        for s in range(n_multi):
            picked = rng.choice(swappable, size=m, replace=False)
            units = list(base_units)
            for i in picked:
                choices = alts[int(i)]
                units[int(i)] = choices[int(rng.integers(len(choices)))]
            raw.append((f"m{m}s{s}", units, shared_market))

    # 市場變體（unit 那半維持 gen0）。
    for name, orders in markets:
        if name == base_label:
            continue
        raw.append((f"mk-{name}", list(base_units), orders))

    out, blocked, reasons = [], 0, []
    seen = set()
    for label, units, market in raw:
        action = _as_action(units, market)
        key = repr(action)
        if key in seen and label != base_label:
            continue                                 # 去重：撞到一樣的組合
        if label != base_label:                      # 基準線不驗，見上面
            try:
                assert_legal_action(obs, config, action)
            except IllegalAction as exc:
                blocked += 1
                if len(reasons) < 3:
                    reasons.append(f"{label}: {exc}")
                continue
        seen.add(key)
        out.append((label, action))
    info = {
        "blocked": blocked,
        "n_raw": len(raw),
        "base_illegal": base_illegal,                # None = 合法
        "dropped_orders": len(base_market) - len(shared_market),
        "reasons": reasons,
    }
    return out, blocked, info
