# CLAUDE.md

Kaggriculture 比賽用 AI。目標：**上榜（前 10）。**

先讀 `README.md`、`docs/ml-primer.md`、`docs/architecture.md`。

---

## 讀者背景

**團隊兩人都是資深軟體工程師，但沒做過 ML。**

所以：

- ML 名詞第一次出現時要用一句白話解釋
- 能用軟體工程的類比就用（例如：訓練 vs 推論 ≈ 編譯 vs 執行）
- 不要假設對方知道 PyTorch 的慣例
- 反過來，**系統工程的東西不用解釋**——多行程、queue、GIL、序列化、效能剖析都是他們的日常

---

## 語言

- 一律**繁體中文（台灣用語）**
- 技術名詞保留英文原文，不要硬翻
  - ✅ policy head、beam search、checkpoint
  - ❌ 策略頭、束搜尋、檢查點
- 不要翻譯腔

---

## 硬規則

### 證據

**不確定的事不要用推論帶過。** 每個常數只有三種狀態：

```python
X = 48      # VERIFIED: interpreter.py 第 204 行
Y = 120     # UNVERIFIED — 從 replay 反推，見 unknowns.md #7
Z = None    # UNKNOWN — 擋住 T11
```

- 不知道就說不知道，並給出查證方法
- **不准用聽起來合理的推論填空**
- 「找不到證據」≠「證據顯示不存在」
- 寫出「通常是」「往往」「一般來說」= 你沒查，去查

### 測試

| 層 | 時間 | 何時 |
|---|---|---|
| L0 smoke | **< 60 秒** | 每次改動 |
| L1 quick | ~5 分鐘 | 確認沒搞砸 |
| L2 判定 | 30~60 分鐘 | 整合日、提交前 |

- **改完一定跑 L0**。超過 60 秒就是設計錯了
- **不要動不動跑 200 局**。用配對種子 + SPRT，明顯的改動 30~50 局就判完
- L0 **不看勝率**（3 局是純噪音），只看斷言
- 期末現金跟 `tests/baselines.json` 比，**不是跟 0 比**
- 🚨 **測試沒過就去修 code，絕不調 baseline 或放寬斷言**

### Log

- **改東西就加 log**，而且要 log 判斷依據不只是結果
- 每回合一筆 JSON，**必須有 `top5` 候選動作和分數** — 要 debug 的是「為什麼沒選那個更好的」
- `LOG_LEVEL`：0 安靜 / 1 每日 / 2 每回合 / 3 決策細節
- **submission 固定 `LOG_LEVEL = 0`**
- 不變式檢查（現金非負、庫存非負）永遠開著

### 介面

`contracts.py` 凍結，不准單方面改。

```
encoding channel：✅ 只能往後加   ❌ 不准調換 / 刪除
```

每次加 → `ENCODER_VERSION += 1`。存模型寫入版本，載入不符直接拋錯。

### Submission

- **不准 import torch**（載入好幾秒，第一回合就可能超時）
- 模型存 `.npz`，用 numpy 做前向
- 動作送出前必過 `legal_mask` — **引擎遇到非法動作會靜默忽略**
- 搜尋必須 anytime：網路前向先產保底解 → 有時間才展開 → 到點交出

---

## 方法路線（已定案）

Expert Iteration：規則式 AI 暖身 → 訓練網路 → 網路 + beam search → 再訓練，共 4~6 代。

規則式的東西只出現在兩個地方，都是這套方法的零件：
1. 第 0 代的暖身 AI
2. 超時的保底動作

**不要建議改用純規則式方案。不要建議 PPO from scratch。不要建議用 LSTM 預測價格**（引擎公式讀得到）。

---

## 檔案所有權

```
encoding/  harness/  serving/  main.py   → A
agents/  model/  eval/                   → B
contracts.py                             → 凍結
```

改別人的檔案前先問。

---

## 效能

瓶頸是 **CPU 端的遊戲模擬**，不是 GPU 訓練。

一代 = 5,000 局 × 720 回合 = 360 萬次模擬，引擎是純 Python。

- self-play worker **不要 import torch**，走 queue 給 inference server 做 batch
- 目標：規則式 > 5,000 局/小時；帶網路 > 1,500 局/小時
- 搜尋要**增量修補上一回合的解**，不要每回合從零重算

---

## 現在的狀態

🟡 **Phase 1 進行中**（最後更新 2026-08-16）。

### 已經有的

- 引擎規則讀完了，寫在 `docs/games/engine-notes.md`（含行號）與 `docs/games/op-flows.md`。
  `unknowns.md` #1~#9 全部結案。**⚠️ T00 仍待第二人獨立讀完再對答案。**
- 引擎固定 `kaggle-environments==1.32.7`。升級引擎要重核 `engine-notes.md` 並重跑 L0。
- `main.py` = Gen1 三地版（`agents/gen1.py` 疊在 `agents/gen0.py` 核心上）
  + `serving/action_validation.py` 每個動作嚴格驗證。
- 凍結量尺：`ref-v2`（Gen0）、`ref-v3`（Gen1 t3）、`ref-v4`（Gen1 t4 = 現行預設）。
  **params 完整展開，不指向 `DEFAULT_PARAMS`** —— 對手池指向預設值的話，
  每次調預設就等於換了量尺。改了預設就新增一版，不要改舊的。
- L0 = `pytest`（9 項，約 10 秒）；baseline 在 `tests/baselines.json`。
- 成績：30 seeds 對 starter 平均 `$86,306`；80 局 paired 對 ref-v3 勝率 `76.3%`。

### 還沒有的

**Phase 2 那條線一行都還沒寫**：`encoding/`、`model/`、`harness/`、`contracts.py`
都不存在。`architecture.md` 的模組清單是**計畫，不是現況**——別照著它找檔案。

### 現在的工程重點

路徑規劃（MOVE 仍佔 52%、土地利用率只有 61%）與對戰產線吞吐。

### 兩條還在生效的警告

- **規則常數一律 `import` 引擎，不做鏡像。** 抄一份會有「抄錯」和「上游改版沒跟上」
  兩個風險，兩者都不會報錯。
- **本機引擎原始碼被人手改過。**（`LAND_ORDER` 曾被改成 `["NE"]`，害查了四輪。）
  引擎常數不能當永久事實，依賴它的邏輯要動態走，例如照 `len(LAND_ORDER)`。
