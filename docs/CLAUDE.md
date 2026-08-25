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

🟢 **Phase 2 進行中**（最後更新 2026-08-24）。

### 路線

`docs/CLAUDE.md` 的方法路線（Expert Iteration）沒有改：
**規則式暖身 → 訓練網路 → 網路 + search → 再訓練**。

現在在「訓練網路」那一步的尾聲。網路是**端到端**的
（`agents/gen2_model.py`）：每個 unit 這一步做什麼（44 個 `UNIT_OPS`，
含走路方向）以及所有市場訂單都由網路決定，**不 import `agents/gen0.py`**。

> 🚫 **不要提議「退回規則式」或「把某一段還給 gen0」當作結論。**
> 那條路已經走過兩輪並量到天花板：模仿架構的上限就是被模仿的對象，
> 而規則式對 ladder 頂端只有 56~58%（2026-08-24 用 100 局配對重量：
> `gen1` 對 `ladder-top-a` 是 **56.8%**，64,030 vs 112,719）。
> 要超過它只能靠 search。詳見 `docs/memory/journal/2026-08-21.md` §1。
>
> `agents/gen1.py` **已經不存在**（2026-08-21 併進 `agents/gen0.py`）——
> 它本來就只是「gen0 + 一組調過的參數」。任何說「改用 gen1」的結論都是過期的。

### 已經有的

- 引擎規則讀完了，寫在 `docs/games/engine-notes.md`（含行號）與 `docs/games/op-flows.md`。
  `unknowns.md` #1~#9 全部結案。**⚠️ T00 仍待第二人獨立讀完再對答案。**
- 引擎固定 `kaggle-environments==1.32.7`。升級引擎要重核 `engine-notes.md` 並重跑 L0。
- **規則式**：`agents/gen0.py`（gen1 已併入，`DEFAULT_PARAMS` 51 個 key）。
  榜上 7 萬+，本機對 `ladder-top-a` 是 56~58%。
- **Phase 2 全線都在**：`contracts.py`（`ENCODER_VERSION` 5）、`model/`、
  `harness/`（rollout + build_dataset）、`serving/`（npz 前向、export、打包）、
  `eval/runner.py`、`tools/`（`action_dist` / `state_dist` / `eval_table`）。
- L0 = `pytest`（81 項，約 45 秒）；baseline 在 `tests/baselines.json`
  （盯的是規則式那條路，`agents.gen0:act`）。
- 分數記錄：`docs/eval-results.md`（`python -m tools.eval_table` 重新產生）。
- **權重**：`model/weights-e2e-round6.npz` 是目前最好的（`--epochs 24
  --width 128 --blocks 8`，1,599,159 參數，對 `gen1` 現金比 93%）。
  指標性權重的去留規則寫在 `model/README.md`。
- **打包好的網路版**：`submission/e2e-round6/`（`submission/main.py` 仍是
  規則式，沒有動）。要測打包好的東西只能用 `config/opponents/*.json` 的
  `builtin`（`--a e2e-round6-sub` / `--a submission`）——
  直接給 `.py` 路徑會走 `__import__` 然後 `ModuleNotFoundError`。

### 還沒有的

**`search/` 2026-08-25 開工了**（`rollout_search.py` / `candidates.py`，
加上 `tools/search_probe.py` / `tools/search_game.py`）。這是路線圖上
「網路 + search」那一步，也是唯一能超過規則式的東西。forward model 直接用引擎本身
（離線跑，沒有 `actTimeout` 限制，也沒有「呼叫引擎私有函式導致換版本每回合
TypeError」的風險）。

⚠️ **離線 search 的產出上不了場** —— 每個決策要幾十次打到底的 rollout。
它是用來離線產「比 expert 更好的動作」，再靠 DAgger 蒸餾回網路的。

2026-08-24 量到三件事把它的形式收窄了（細節見當天 journal §5/§6/§7），
但第 3 點已被 08-25 推翻：

1. **線上即時搜尋不可行。** `actTimeout = 1` 秒（8 個真實 episode config
   與本機預設全部一樣）。一步 rollout 約 10.4 ms，換最便宜的 `gen0.act`
   當 rollout policy 也要 4.2 ms —— horizon 24 只夠 10 次 simulation。
2. **不能用 value head 剪枝。** `tools/value_probe.py` 用 round6 權重重跑，
   H=1/8/24 三個 horizon 全部還是比不搜差。每個候選都得真的打到底。
3. ~~搜「某一回合的動作」上限只有 2.4%~~ —— **2026-08-25 推翻，不要再引用。**
   那個 oracle（round5 +2.62% / round6 +2.42%）的候選集合**只有 8 個**
   （`tools/value_probe.py:118-120`：不改、6 個「單一 unit 換成第 2 名」、
   1 個強迫買種子），而且基準線是網路自己的 argmax。
   候選擴大到 57 個、基準線換成 `gen0` 的動作之後，**oracle 是 +8.37%**
   （12 個盤面，`temp/search-probe-stage1.json`）。leverage 集中在 day 6~12，
   day 18 之後幾乎沒有搜的價值。詳見 08-25 journal §9~§11。

### 現在的工程重點

🟢 **2026-08-25：離線 search 的兩個 gate 都過了，這是目前最好的一條路。**

    對 ladder-top-a（seed 0~9、a_slot 0、同一組 seed）的現金比
      round6 網路          51.5%   63,546
      gen1 / gen0 規則式    55.1%   67,944
      gen0 + 離線 search   76.9%   94,892     <- 10/10 局，p=0.00195
      ladder-top-a 自己             123,323

缺口從 45% 縮到 23%。**但還沒追過 `ladder-top-a`。** 兩個未解的問題見
08-25 journal §15：(1) 對手的期末現金沒有記錄，分不出增益是「自己多賺」還是
「壓低開迴路的 replay」（工具已修、還沒重跑）；(2) 產出上不了場，一定要
Stage 3 蒸餾回網路。

增益的來源可以看懂：**day 0 佔 46.5%，而且是「不要種 day 10 才首收的
STRAWBERRY / MELON，改種 day 2 就收的 CARROT」**（journal §16）。那正好是
08-24 §4 診斷出來但用參數修不掉的「前六天賺得不夠」。

---

**（以下是 08-24 寫的，模仿路線那一段仍然成立）**

**模仿這條路已經接近它的天花板，瓶頸不在網路。**

    round6 對 gen1 的現金比          93%    模仿得夠像了
    gen1 對 ladder-top-a 的現金比    56.8%  被模仿的對象只有這樣
    round6 對 ladder-top-a           50.7%

就算 round7 模仿到完美，終點是 gen1 的成績。而 2026-08-24 實測 +200 局
DAgger 資料的效果是 **p=0.599（判不出差別）**，訓練曲線也從欠訓練變成
overfit（loss 掉 7.7%、驗證 `op` 只升 0.3 個百分點）。**加資料和加大模型
都已經到頂。**

真正的差距在整條軌跡的策略：day 6 時 `ladder-top-a` 手上有約 $1,762，
`gen0` 只有 $958。但**單點介入修不了**——2026-08-24 把 `land_first_min_day`
從 3 改成 0（讓它 day 0 就買地），結果是 **0/60 全輸、−58%**。參數之間
互相咬合。

> **market head 的校準：訓練指標上已經不是問題，但沒有端到端驗證過。**
> round6 的 market recall（驗證集）是 **0.9345**。先前寫在這裡的
> 0.25~0.38 有兩個混在一起的東西：舊架構（96/6）的實際召回率，以及
> 2026-08-21 §15 澄清過的「0.3767 是量測假象」（量錯對象）。
> ⚠️ **沒有測過**拿掉 `agents/gen2_model.RESTOCK_OPS` 的逐 op 門檻補償
> 會不會變差 —— 訓練指標好不等於實戰可以拿掉那層補償。要下這個結論
> 得跑一次 A/B。

### ⚠️ 凍結量尺的保護沒有真的裝上

`ref-v3`…`ref-v11` 各有 19 個 key **沒有展開**，會 fall through 到
`gen0.DEFAULT_PARAMS`。改那些預設值 = 所有凍結尺同時位移，而且不會報錯。
（`ref-v11` 是唯一完整展開 51 項的。）Gen0 線的 9 個對手已於 2026-08-21 刪除。

### 兩條還在生效的警告

- **規則常數一律 `import` 引擎，不做鏡像。** 抄一份會有「抄錯」和「上游改版沒跟上」
  兩個風險，兩者都不會報錯。
- **本機引擎原始碼被人手改過。**（`LAND_ORDER` 曾被改成 `["NE"]`，害查了四輪。）
  引擎常數不能當永久事實，依賴它的邏輯要動態走，例如照 `len(LAND_ORDER)`。
