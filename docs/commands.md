# 常用指令

> 語法是 **PowerShell**（本機預設 shell）。Git Bash 的話環境變數寫成
> `KAGGRI_WEIGHTS=... python -m ...` 放在同一行。

## ⚠️ 跑之前先確認這三件事

| | |
|---|---|
| **`--workers` 上限 16** | `cpu_count()` 是 28，但 `harness/rollout.py:300` 的預設是 `cpu_count - 2 = 26`，會吃掉整台機器。**每個指令都要明寫。** |
| **`--games N` 是 N 個配對種子** | 實際跑 **2N 局**（`build_jobs(swap=True)` 每個 seed 正反各一次）。 |
| **`KAGGRI_WEIGHTS` 沒設會退回 `submission/weights.npz`** | 那可能是別版的。2026-08-21 之前載到語意不符的權重會**整局 PASS 拿 0 分且零錯誤訊息**；現在 npz 存了 `labels`，對不上會直接 `SystemExit`。 |

---

## `--a` / `--b` 是什麼

`--a` 是對戰的 A 方，吃**三種**寫法：

```
--a e2e                     對手池的名字 -> config/opponents/e2e.json
--a agents.gen0:act         module:function，直接指函式，不經 config
--a submission/main.py      檔案路徑，走 kaggle_environments 的 file-path 載入
                            （跟 Kaggle 上場同一條路，是唯一測得到攤平後
                             import 的方式）
```

名字怎麼變成程式碼：

```
--a e2e
  └─> config/opponents/e2e.json
        entry:  agents.gen2_model:act     <- 真正被呼叫的函式
        params: 無                        <- 不覆寫任何參數
              └─> agents/gen2_model.py 的 act()
                    └─> 權重讀 KAGGRI_WEIGHTS
```

### 對手池裡有什麼

| 名字 | 是什麼 |
|---|---|
| `e2e` | **端到端網路**（`agents/gen2_model.py`）：每個 unit 的當下動作 + 所有市場訂單都走網路 |
| `e2e-t10` / `e2e-t20` | 同一支，掃 market present 的全域門檻 |
| `e2e-restock10`…`30` | 同一支，只掃 `BUY_SEED` / `BUY_PRODUCT` 的門檻 |
| `gen1` | **規則式**（`agents/gen0.py` + 預設參數），榜上這一版 |
| `gen1-three-land` / `gen1-four-land` | 規則式改象限數 |
| `ref-v3` … `ref-v11` | 凍結量尺，都是規則式、參數寫死 |
| `ladder-top-a` / `ladder-top-b` | 真實 ladder 對局重播 |
| `starter` / `random` / `pass` | 引擎內建 |

---

## 對戰

```powershell
$env:KAGGRI_WEIGHTS = "model/weights-e2e-round3.npz"

# 端到端網路 vs 規則式（主要的判定）
python -m eval.runner --a e2e --b gen1 --games 20 --workers 16

# vs ladder 頂端
python -m eval.runner --a e2e --b ladder-top-a --games 10 --workers 16

# 一次打一整排對手（--ladder 吃一個 sweep 檔）
python -m eval.runner --a e2e --ladder config/sweep-hire.json --games 20 --workers 5
```

> ⚠️ `--ladder config/sweep-hire.json` 是拿 A 去打「不同 `max_hands` 的規則式」。
> `hire-0` / `hire-2` 贏了只代表比殘廢版強；**`hire-12` 才是真正的規則式**
> （12 是預設值），那一列才有意義。

> ⚠️ **不同對手的分數不能互相比較** —— 市場是兩家共用的。規則式對
> `ladder-top-a` 是 66,540、對 `starter` 是 119,701，同一支 agent。

### 輸出

- `temp/<時間>_<A>_vs_<B>/summary.txt` —— 勝負、平均與 **min/max** 現金、買地、田況、動作分布
- `logs/*.jsonl`（要 `--log-level 2`）—— 一局一個檔，檔名帶期末現金：
  `seed0000_e2e_vs_gen1_a71035_b93538.jsonl`。
  **一個檔裡兩邊都有**，用 `player` 欄位分（A 是 player 0）。
  `--log-level 3` 會再記 `agents/gen2_model.py` 每個 unit 的前三名候選動作
  與 logits —— 動作被 `legal_unit_mask` 或 PLANT 種子上限改掉時，
  光看 `action` 看不出網路本來想做什麼。
- `python -m tools.eval_table` —— 把所有 run 整理進 `docs/eval-results.md`

---

## Kill switch

```powershell
python -m tools.action_dist temp/<run 目錄>    # 動作分布 vs 對手
python -m tools.state_dist  temp/<run 目錄>    # 狀態分布，按天
```

> ⚠️ **`state_dist` 對現在這條線沒有判定力。** 它的基準是**老師**的逐日 p5，
> 而規則式自己也過不了（動物 day 1、作物 day 6、現金 day 14）——
> ① 跟它的 expert 逐項相同是模仿到位，不是失敗。要有判定力得換成規則式的分布。

> 🩸 **2026-08-21 之前的 e2e run 用不了 `state_dist`。** 那時候
> `agents/gen2_model.py` 不寫 log，檔案裡只有規則式那一邊 ——
> 而 `state_dist` 沒篩 player，於是把**對手的**數字印成「我們」。
> 現在兩邊都寫 log、也篩 player 了；讀到舊 run 會直接報錯而不是給錯的數字。
> 要看網路版的狀態分布得**重跑一次** `eval.runner --log-level 2`。

---

## DAgger 一輪

```powershell
$env:KAGGRI_WEIGHTS = "model/weights-e2e-round3.npz"

# 1. 收資料：網路開車、規則式出答案
python -m harness.rollout --policy e2e --expert gen1 `
       --games 200 --workers 16 --out data/dagger/e2e-round4

# 2. 訓練（aggregate 全部輪次）
python -m model.train `
       --data data/dagger/e2e-round0,data/dagger/e2e-round1,data/dagger/e2e-round2,data/dagger/e2e-round3,data/dagger/e2e-round4 `
       --labels immediate --val-from data/dagger/e2e-round0 `
       --out model/ckpt-e2e-round4 --epochs 8 --width 96 --blocks 6

# 3. 匯出
python -m serving.export_npz --ckpt model/ckpt-e2e-round4/best.pt `
       --out model/weights-e2e-round4.npz
```

> 🩸 **`--val-from` 一定要帶。** train/val 切分吃 `len(paths)`，每加一輪資料夾
> 抽到的驗證集就換一批 —— round0/round1 的指標曾經因此看起來像退步
> （連 dummy 都變了）。釘在 `e2e-round0` 就每輪都考同一份。
>
> 🩸 **`--policy` 要用真正要出貨的那一支。** DAgger 的價值在「在你自己會走到
> 的爛盤面上問 expert」。2026-08-21 的 round1/round2 是用混合版收的，
> 那個 agent 已經不存在了。
>
> 🩸 **`--labels immediate`**（當下這一步）。`target`（段落終點動作）是
> 已刪的 v3/v5 那條線用的，餵給 `gen2_model` 會整局 PASS。

---

## 打包 submission

```powershell
python -m serving.build_submission --tar

# 🩸 打包完一定要用 file-path 載入再驗一次 —— 那是唯一測得到
#    「函式內 import 上場會 ModuleNotFoundError」的路徑（2026-08-21 踩過兩次）
python -m eval.runner --a submission --b gen1 --games 3 --workers 6
```

> ⚠️ `build_submission` 會 **`rmtree` 整個 `submission/`**，跑之前先看裡面
> 有沒有別人放的備份。
>
> ⚠️ 現在 `main.py` 是規則式，所以只打包 `main.py` + `gen0.py`、**不帶權重**。
> 要換成網路版的話，`serving/build_submission.py` 的 `FILE_MAP` 要加回
> `contracts.py` / `npz_forward.py` / `gen2_model.py`，並用 `--weights` 指定 npz。

---

## L0

```powershell
python -m pytest -q          # 81 項，約 45 秒
```
