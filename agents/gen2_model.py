"""端到端：網路直接決定每個 unit 這一步做什麼（44 個 `UNIT_OPS` 之一）。

## 🔄 2026-08-21：模仿對象從 ladder 老師換成我們自己的規則式

原本這支是模仿 rating 2979~3229 的真實 ladder 玩家（60 局 replay、842,151 個
unit-turn），而 docstring 舊版寫著「**不是**模仿我們自己的 Gen1 —— 2026-08-19
量到 Gen1 對他們 6 勝 154 負，模仿一個在輸的老師沒有意義」。

**那個判斷被後來的量測推翻了**，理由是「模仿得到誰」比「誰比較強」更關鍵：

- 老師是 **60 份固定 replay，問不到新局面**。網路一走偏就沒有標籤可學，
  實測開局 73 步的腳本只跟得完 22%（journal 2026-08-20 §7）
- 規則式是 **queryable expert** —— 每回合從當下盤面重算，網路把盤面走爛之後
  它仍然答得出「這裡該做什麼」。那正是 DAgger 需要而 replay 給不了的

代價是天花板變成規則式（榜上 7 萬+，本機配對 92,860）。**這一步的目的不是
贏過它**，是產出離線 search 用得動的 policy prior —— 完整動作空間、
top-k 涵蓋率夠高、value head 可靠。超過規則式要靠 search，不是靠模仿。

## 為什麼是端到端，而不是 v5 的 demand map

`agents/gen4_demand.py` 只讓網路輸出「哪一格要做什麼」，配對 / 走路 / 市場
全部還給 `gen0`。**search 只搜得到網路輸出的東西**，所以那條路上市場和配對
永遠碰不到。這支的輸出涵蓋完整動作空間。

## 市場也在網路裡（2026-08-21 起沒有退路）

這支**完全不 import `agents/gen0.py`**。曾經有一個 `model_market` 開關可以把
市場退回 `gen0._market`，拿掉了 —— 它會生出好看的假數字（同一份權重，市場走
gen0 是 gen1 的 81.5%、走網路是 17.0%），而且 DAgger 應該從真正要出貨的
policy 收狀態，不是從混合版。

⚠️ **市場是目前唯一的瓶頸。** 逐 op 的召回率（round2 權重、固定驗證集）：

    HIRE              1.000      BUY_SEED WHEAT       0.380
    SELL FERTILIZER   0.961      BUY_SEED CARROT      0.250
    BUY_ANIMAL SHEEP  0.941      BUY_SEED TOMATO      0.222
    BUY_LAND          0.833      BUY_PRODUCT WHEAT    0.422   <- 飼料

`BUY_SEED` 整組沒學會 -> 田是空的 -> unit 沒事幹就 PASS（26.6%，gen1 是 17.1%）。
**調門檻治不了這個**（實測 -1.0 只讓現金 18,194 -> 19,138）。

## PLANT 的原子驗證

引擎規則（`engine-notes.md` §10.9）：同一回合所有 unit 對某作物的 PLANT 請求
總數超過手上種子數 → 該作物的**所有** PLANT 請求全部變成 `PASS`。

所以不能直接把網路的 argmax 送出去 —— 超額的話會連本來種得成的那幾格一起
作廢。超出的部分改用該 unit 的第二高分動作。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

import contracts as C
# 🩸 模組層，不可以搬進 `_policy()` —— 理由見 `agents/gen0.py` 頂端那一段。
# 這裡只是把類別 import 進來，**權重仍然是 lazy 的**（`NumpyPolicy(path)` 才讀檔）。
from serving.npz_forward import NumpyPolicy

#: 權重檔。`KAGGRI_WEIGHTS` 覆寫，方便同時比較好幾個 checkpoint。
#:
#: 沒設的話先找**跟這支檔案同一個目錄**的 `weights.npz` —— submission 是攤平的
#: （`serving/build_submission.py`），權重就躺在旁邊，而且比賽端的工作目錄不是
#: repo root，寫死任何 `model/...` 的路徑都會找不到。
#:
#: ⚠️ 這裡可以用 `__file__`：`main.py` 是被 `exec()` 進去的、沒有 `__file__`，
#: 但這支是正常 `import` 進來的模組，有。
#:
#: 找不到的話退回 repo 的 `submission/weights.npz` —— 那是 `build_submission`
#: 剛打包進去的同一個檔案，所以開發側跑 `main.py` 跟上場跑的是同一份權重。
#: **不要退回 `model/artifacts/weights.npz`**：那支是 ENCODER_VERSION 2 的舊檔，
#: 載進來會在第一回合 SystemExit，而錯誤訊息看起來像是 contracts.py 的問題。
#: 0 安靜 / 2 每回合一筆 / 3 再加上網路的候選動作與 logits。
#:
#: ⚠️ **import 時讀一次**，跟 `agents/gen0.py:66` 一樣 —— `main.py` 靠的就是
#: 這個（先設 `KAGGRI_LOG_LEVEL=0` 再 import）。`eval/runner.py` 是在開 worker
#: 之前設環境變數，子行程 import 時才讀得到。
LOG_LEVEL = int(os.environ.get("KAGGRI_LOG_LEVEL", "3"))

_HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = os.environ.get(
    "KAGGRI_WEIGHTS",
    str(next((p for p in (_HERE / "weights.npz",
                          _HERE.parent / "submission" / "weights.npz")
              if p.is_file()), _HERE / "weights.npz")))

_POLICY = None
_POLICY_LOCK = threading.Lock()


def _policy():
    """載入一次就好。多行程跑的時候每個 worker 各載自己的。"""
    global _POLICY
    if _POLICY is None:
        with _POLICY_LOCK:
            if _POLICY is None:
                policy = NumpyPolicy(WEIGHTS_PATH)
                if policy.encoder_version != C.ENCODER_VERSION:
                    raise SystemExit(
                        f"{WEIGHTS_PATH} 是 ENCODER_VERSION "
                        f"{policy.encoder_version}，contracts.py 是 "
                        f"{C.ENCODER_VERSION} —— 重新訓練")
                _POLICY = policy
    return _POLICY


def require_labels(policy, want):
    """權重的 op head 語意必須對得上，不然整局 PASS 拿 0 分而且不報錯。

    🩸 2026-08-21 踩到：`--a e2e` 沒設 `KAGGRI_WEIGHTS`，退回去載
    `submission/weights.npz`（v5，`labels: target`）。`ENCODER_VERSION` 都是 5
    所以版本檢查放行了，但 `target` 的 op head 預測的是「段落終點動作」——
    永遠不會輸出 MOVE。實測 MOVE 0.0%、PASS 80.8%、期末現金 0，**零錯誤訊息**。

    - `immediate`：op head 是「這個 unit 這一步做什麼」-> `agents/gen2_model.py`
    - `target`：op head 是「走到終點要做什麼」-> `agents/gen3_target.py`、
      `agents/gen4_demand.py`
    """
    got = policy.labels
    if got == want:
        return
    if got is None:
        raise SystemExit(
            f"{WEIGHTS_PATH} 沒有 `labels` 欄位（2026-08-21 之前匯出的）。"
            f"重跑 `python -m serving.export_npz` 就會補上。")
    raise SystemExit(
        f"{WEIGHTS_PATH} 的 op head 是 `{got}`，這支要 `{want}` —— "
        f"載錯的話整局 PASS 拿 0 分而且不會報錯。換權重或換 agent。")


def _choose(op_logits, qty_logits, mask, obs):
    """挑動作：先遮掉不合法的，再處理 PLANT 的跨 unit 種子上限。"""
    scores = np.where(mask, op_logits, -np.inf)
    order = np.argsort(-scores, axis=1)          # 每個 unit 的偏好排序

    seeds = dict(obs["private"]["seeds"])
    chosen = []
    for i in range(scores.shape[0]):
        picked = None
        for op_index in order[i]:
            if not np.isfinite(scores[i, op_index]):
                break                             # 後面都是不合法的
            op, item = C.UNIT_OPS[op_index]
            if op == "PLANT":
                # 種子是全體共用的，先搶先贏。搶不到就往下一個選項走 ——
                # 不能硬送，送了會讓該作物**所有** PLANT 一起變 PASS。
                if seeds.get(item, 0) <= 0:
                    continue
                seeds[item] -= 1
            picked = op_index
            break
        if picked is None:
            picked = C.UNIT_OP_INDEX[("PASS", None)]
        qty_index = int(np.argmax(qty_logits[i]))
        chosen.append(C.decode_unit(picked, qty_index))
    return chosen


#: 「補貨型」買單的 op index —— `BUY_SEED` ×5 與 `BUY_PRODUCT` ×2。
#:
#: 它們跟其他 op 的**錯誤成本不對稱**：少買一次種子/飼料，整片田空著好幾天
#: （2026-08-21 實測期末作物 3.6 對 gen1 的 14.0，PASS 衝到 26.6%）；多買一份
#: 只花 $10~$100。所以這幾個該用比較鬆的門檻。
#:
#: **`BUY_LAND` 和 `BUY_ANIMAL` 不在裡面** —— 大額支出，而且它們的召回率本來
#: 就有 0.83~0.94，放寬只會亂買。
RESTOCK_OPS = tuple(
    j for j, (op, _item) in enumerate(C.MARKET_OPS)
    if op in ("BUY_SEED", "BUY_PRODUCT"))


def market_thresholds(base=0.0, restock=None):
    """`[N_MARKET_OPS]` 的逐 op 門檻。`restock` 沒給就整排都是 `base`。"""
    th = np.full(C.N_MARKET_OPS, float(base))
    if restock is not None:
        th[list(RESTOCK_OPS)] = float(restock)
    return th


_LOG_FH = None
_LOG_FH_PATH = None


def _log_sink():
    """回傳要寫進去的 file object。

    跟 `agents/gen0.py` 同一套做法（那邊有完整說明），這裡刻意**複製而不是
    import** —— 這支的設計前提是完全不依賴 `agents/gen0.py`，而
    `serving/build_submission.py` 打包時是把檔案攤平的，多一條 import
    就多一個要進 `FILE_MAP` 的檔案。
    """
    global _LOG_FH, _LOG_FH_PATH
    path = os.environ.get("KAGGRI_LOG_FILE")
    if not path:
        import sys
        return sys.stderr
    if path != _LOG_FH_PATH:
        if _LOG_FH is not None:
            _LOG_FH.close()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _LOG_FH = open(path, "a", encoding="utf-8")
        _LOG_FH_PATH = path
    return _LOG_FH


def close_log():
    """關掉 log 的 file handle。

    `eval/runner.py` 跑完一局要把 `.jsonl` 改名（把期末現金加進檔名），
    Windows 不讓你改一個開著的檔 —— 不關的話 `PermissionError`。
    """
    global _LOG_FH, _LOG_FH_PATH
    if _LOG_FH is not None:
        try:
            _LOG_FH.close()
        except OSError:
            pass
    _LOG_FH = None
    _LOG_FH_PATH = None


def _board_counts(tiles):
    """一次數完整片田。回傳 `(animals, empty_structs, crops, weeds, tiles_open)`。

    ⚠️ `crops` 是**真的種下去的格數**（`kind == "PLANT"`），跟
    `agents/gen0.py` 記在 `budget["active_crop_count"]` 的那個數字**不一樣**
    —— 後者是規則式「打算種幾格」的計畫值。這支沒有計畫值可記，而且
    `tools/state_dist.py` 的老師基準用的是 `contracts.SCALAR_FIELDS`
    的 `n_crop_tiles`，那也是實際格數。所以這裡對齊的是老師那一邊。
    """
    animals, empty_structs, crops, weeds, open_tiles = {}, {}, 0, 0, 0
    for row in tiles:
        for tile in row:
            if tile == "LOCKED":
                continue
            open_tiles += 1
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crops += 1
            elif kind == "WEED":
                weeds += 1
            elif kind in ("COOP", "PASTURE"):
                if tile.get("animal"):
                    animals[tile["animal"]] = animals.get(tile["animal"], 0) + 1
                else:
                    empty_structs[kind] = empty_structs.get(kind, 0) + 1
    return animals, empty_structs, crops, weeds, open_tiles


def _top_ops(op_logits, mask, i, k=3):
    """第 i 個 unit 分數最高的 k 個**合法** op，附 logit。"""
    scores = np.where(mask[i], op_logits[i], -np.inf)
    out = []
    for j in np.argsort(-scores)[:k]:
        if not np.isfinite(scores[j]):
            break
        op, item = C.UNIT_OPS[j]
        out.append([op, item, round(float(op_logits[i, j]), 3)])
    return out


def _log(obs, action, op_logits, mask, mk_present, thresholds, value):
    """一回合一筆 JSON。

    欄位跟 `agents/gen0._log` 對齊到 `tools/plot_prices.py` 和
    `tools/state_dist.py` 讀得到的程度；規則式才有的
    `budget` / `task_counts` / `top5` / `assigned` 換成網路這邊對應的東西
    （`units` 的候選動作、`market_logits`）。
    """
    farm = obs["farms"][obs["player"]]
    private = obs["private"]
    animals, empty_structs, crops, weeds, tiles_open = _board_counts(farm["tiles"])
    rec = {
        # 多局的 log 合併之後靠 tag 分辨是哪一局哪一邊。引擎會把 seed 從
        # observation 抹掉，所以只能由 runner 從外面塞進來。
        "tag": os.environ.get("KAGGRI_LOG_TAG", ""),
        "player": obs["player"],
        "t": obs["day"] * 24 + obs["hour"],
        "day": obs["day"],
        "hour": obs["hour"],
        "cash": farm["money"],
        "crew": len(farm["hands"]),
        "quadrants": list(farm["unlocked_quadrants"]),
        "tiles_open": tiles_open,
        "animals": animals,
        "empty_structs": empty_structs,
        "crops": crops,
        "weeds": weeds,
        # value head 預測的期末分數（正規化後）。這是 search 要用的那顆頭，
        # 值得逐回合留著跟真實結果對。
        "value": round(float(np.ravel(value)[0]), 4),
        "prices": dict(obs["market"]["prices"]),
        "seeds": {k: v for k, v in private["seeds"].items() if v},
        "shed": {k: v for k, v in private["shed"].items() if v},
        "carry": [{k: v for k, v in inv.items() if v}
                  for inv in private["inventories"]],
        "action": action,
    }
    if LOG_LEVEL >= 3:
        # 「網路本來想做什麼」——被 `legal_unit_mask` 遮掉、或被 PLANT 種子
        # 上限擋掉的時候，送出去的動作跟第一名不同，只看 action 看不出來。
        rec["units"] = [_top_ops(op_logits, mask, i)
                        for i in range(op_logits.shape[0])]
        # 市場只記**接近門檻**的那些（logit > 門檻 - 2），全部 21 個太吵。
        near = []
        for j, (op, item) in enumerate(C.MARKET_OPS):
            logit = float(mk_present[j])
            if logit > thresholds[j] - 2.0:
                near.append([op, item, round(logit, 3),
                             round(float(thresholds[j]), 2)])
        rec["market_logits"] = near
    fh = _log_sink()
    fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    fh.flush()


def act(obs, config=None, params=None):
    """網路決定每個 unit 這一步做什麼，**以及所有市場訂單**。

    這支不 import `agents/gen0.py`。曾經有一個 `model_market` 開關可以把市場
    退回 `gen0._market`，2026-08-21 拿掉了，理由有兩個：

    1. **那個開關會生出好看的假數字。** 同一份 round2 權重，市場走 gen0 是
       75,469（gen1 的 81.5%）、走網路是 18,194（17.0%）。前者被當成「端到端
       網路的成績」報了一整天，但它有一半是規則式的功勞。
    2. **DAgger 應該從真正要出貨的 policy 收狀態。** round1 / round2 的
       rollout 都是混合版走出來的，那不是這支 agent 會遇到的分布。

    `market_threshold` 是 present head 的 logit 門檻（預設 0 = sigmoid 0.5）；
    `market_threshold_restock` 只套用在 `BUY_SEED` / `BUY_PRODUCT`（見
    `RESTOCK_OPS`）。
    2026-08-21 在固定驗證集上掃過：門檻 0 每盤面下 0.35 筆訂單、真實是 0.43 筆，
    -1.0 剛好對上（召回率 0.747 -> 0.834）。**但實測 -1.0 只讓現金從 18,194 變
    19,138** —— 降門檻是齊頭式的，治不了 `BUY_SEED` 那組 0.22~0.62 的召回率。
    """
    params = params or {}
    thresholds = market_thresholds(
        params.get("market_threshold", 0.0),
        params.get("market_threshold_restock"))

    policy = _policy()
    require_labels(policy, "immediate")
    spatial, scalar = C.encode(obs, config)
    positions, unit_features = C.encode_units(obs, config)
    (op_logits, qty_logits, _target,
     mk_present, mk_qty, value, _demand) = policy(
        spatial, scalar, positions, unit_features)

    mask = C.legal_unit_mask(obs, config)
    units = _choose(op_logits, qty_logits, mask, obs)
    market = C.decode_market_orders(
        mk_present, mk_qty, obs, config, threshold=thresholds)
    action = {"farmer": units[0], "hands": units[1:], "market": market}
    if LOG_LEVEL >= 2:
        _log(obs, action, op_logits, mask, mk_present, thresholds, value)
    return action


def agent(obs, config):
    return act(obs, config)
