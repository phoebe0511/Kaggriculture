# 常用指令

> 語法是 **PowerShell**（本機預設 shell）。Git Bash 的話環境變數寫成
> `KAGGRI_WEIGHTS=... python -m ...` 放在同一行。

## ⚠️ 跑之前先確認這三件事

| | |
|---|---|
| **`--workers` 上限 16** | `cpu_count()` 是 28，但 `harness/rollout.py:300` 的預設是 `cpu_count - 2 = 26`，會吃掉整台機器。**每個指令都要明寫。** |
| **`--games N` 是 N 個配對種子** | 實際跑 **2N 局**（`build_jobs(swap=True)` 每個 seed 正反各一次）。 |
| **`KAGGRI_WEIGHTS` 沒設會退回 agent 檔案旁邊的 `weights.npz`** | 順序是 `agents/weights.npz` -> `submission/weights.npz`（兩個現在都不存在，所以 `--a e2e` 沒設就會載入失敗）。2026-08-21 之前載到語意不符的權重會**整局 PASS 拿 0 分且零錯誤訊息**；現在 npz 存了 `labels`，對不上會直接 `SystemExit`。 |
| **`KAGGRI_WEIGHTS` 有設會蓋過打包好的權重** | 測 `--a e2e-round6-sub` 這種自帶 `weights.npz` 的東西時，**一定要先清掉**（`Remove-Item Env:KAGGRI_WEIGHTS`），否則量到的不是要出貨的那份。而且 `result.json` 的 `weights` 欄位這時只記 `"(未設)"`，事後翻 run 目錄考古看不出實際載了什麼。 |
| **檔名打錯不會報錯** | 2026-08-24 把 `round6` 打成 `wound6`，結果是整張表 `nan`、0/0/0，判定欄照樣印「❌ 確實較弱」。跑完先看局數對不對。 |

---

## `--a` / `--b` 是什麼

`--a` 是對戰的 A 方，吃**三種**寫法：

```
--a e2e                     對手池的名字 -> config/opponents/e2e.json
--a agents.gen0:act         module:function，直接指函式，不經 config
--a submission              對手池的名字，但那個 JSON 裡是 "builtin": "<.py 路徑>"
                            -> 走 kaggle_environments 的 file-path 載入
                            （跟 Kaggle 上場同一條路，是唯一測得到攤平後
                             import 的方式）
```

名字怎麼變成程式碼，兩條分支：

```
--a e2e
  └─> config/opponents/e2e.json
        entry:  agents.gen2_model:act     <- 真正被呼叫的函式
        params: 無                        <- 不覆寫任何參數
              └─> agents/gen2_model.py 的 act()
                    └─> 權重讀 KAGGRI_WEIGHTS

--a submission
  └─> config/opponents/submission.json
        builtin: submission/main.py       <- build_agent 直接把字串交給 env.run
              └─> kaggle_environments 的 exec + sys.path.append(exec_dir)
                    └─> 攤平後的 `from gen0 import` 才找得到同目錄的檔案
```

> 🩸 **不要直接把 `.py` 路徑丟給 `--a`。** `--a submission/main.py` 走的是
> `load_spec` 的 `.py` 分支 —— 那條把路徑轉成 module path 再 `__import__`，
> **不是** file-path 載入。攤平後的檔案沒有 package 前綴，於是：
>
>     ⚠️ seed 0: ModuleNotFoundError: No module named 'gen0'
>     局數 0   ⚠️ 作廢 2 局   ...   判定 ❌ 確實較弱
>
> 2026-08-24 實測。**注意最後那行**——0 局跑完照樣印「確實較弱」，
> 跟載錯權重那類靜默失敗是同一個模式（見下面「跑之前先確認」第三格）。
> 要測打包好的東西，一律經 `config/opponents/*.json` 的 `builtin`。

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
| `submission` | **打包好的那一份**：`submission/main.py`（現在是規則式 gen0）。走 `builtin` |
| `e2e-round6-sub` | **打包好的網路版**：`submission/e2e-round6/main.py` + 自帶 round6 權重。走 `builtin` |

> ⚠️ `submission` 和 `e2e-round6-sub` 量到的是**磁碟上打包好的檔案**，不是
> `agents/` 的原始碼。改了 agent 沒重新打包的話，這裡跑的還是舊版 ——
> 那正是我們要它測的事。

---

## 對戰

```powershell
$env:KAGGRI_WEIGHTS = "model/weights-e2e-round5.npz"

# 端到端網路 vs 規則式（主要的判定）
python -m eval.runner --a e2e --b gen1 --games 20 --workers 16

# vs ladder 頂端
python -m eval.runner --a e2e --b ladder-top-a --games 10 --workers 16

# 一次打一整排對手（--ladder 吃一個 sweep 檔）
python -m eval.runner --a e2e --ladder config/sweep-hire.json --games 20 --workers 5
```

測**打包好的**那一份（自帶權重，所以要先把環境變數清掉）：

```powershell
Remove-Item Env:KAGGRI_WEIGHTS -ErrorAction SilentlyContinue

python -m eval.runner --a e2e-round6-sub --b gen1 --games 20 --workers 16
python -m eval.runner --a e2e-round6-sub --ladder .\config\sweep-hire.json --games 20 --workers 8
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
$env:KAGGRI_WEIGHTS = "model/weights-e2e-round5.npz"

# 1. 收資料：網路開車、規則式出答案
python -m harness.rollout --policy e2e --expert gen1 `
       --games 200 --workers 16 --out data/dagger/e2e-round6

# 2. 訓練（aggregate 全部輪次）
python -m model.train `
       --data data/dagger/e2e-round0,data/dagger/e2e-round1,data/dagger/e2e-round2,data/dagger/e2e-round3,data/dagger/e2e-round4,data/dagger/e2e-round5,data/dagger/e2e-round6 `
       --labels immediate --val-from data/dagger/e2e-round0 `
       --out model/artifacts/ckpt-e2e-round6 --epochs 24 --width 128 --blocks 8

# 3. 匯出
python -m serving.export_npz --ckpt model/artifacts/ckpt-e2e-round6/best.pt `
       --out model/artifacts/weights-e2e-round6.npz
```

> 🩸 **產物寫進 `model/artifacts/`，不要寫進 `model/`。** `model/` 底下只放
> 挑過的指標性權重（現在是 round0 / round2 / round5），`artifacts/` 整個
> 可以刪掉重跑。規則寫在 `model/README.md`。

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
>
> 🩸 **`--epochs 24 --width 128 --blocks 8` 是 2026-08-24 起的規格**
> （1,599,159 參數）。round0~round5 是 `--epochs 8 --width 96 --blocks 6`
> （867k），2026-08-21 §17 量到那個組合**欠訓練** —— loss 還在掉 4%、
> 驗證還在升就停了。換規格之後對 `gen1` 從 5.0% 變 20.0%。
> npz 裡存了 `width` / `blocks`，載入時不需要對得上命令列。

---

## 打包 submission

現在有**兩份**打包好的東西，指哪一個給 Kaggle 就是哪一個上場：

| 路徑 | 內容 | 對手池名字 |
|---|---|---|
| `submission/` | 規則式 `gen0`，不帶權重。**榜上現行版本** | `submission` |
| `submission/e2e-round6/` | 端到端網路 + round6 權重（5.97 MB） | `e2e-round6-sub` |

### 規則式那份（`serving/build_submission.py` 產生）

```powershell
python -m serving.build_submission --tar

# 🩸 打包完一定要用 file-path 載入再驗一次 —— 那是唯一測得到
#    「函式內 import 上場會 ModuleNotFoundError」的路徑（2026-08-21 踩過兩次）
python -m eval.runner --a submission --b gen1 --games 3 --workers 6
```

> ⚠️ `build_submission` 會 **`rmtree` 整個 `submission/`**，跑之前先看裡面
> 有沒有別人放的備份 —— **`submission/e2e-round6/` 會一起被刪掉。**
>
> ⚠️ 它的 `FILE_MAP` 只有 `main.py` + `gen0.py`、`DEFAULT_WEIGHTS` 是 `None`。
> 要讓它直接產網路版的話，`FILE_MAP` 要加回 `contracts.py` /
> `npz_forward.py` / `gen2_model.py`，並用 `--weights` 指定 npz。

### 網路版那份（2026-08-24 手動組出來的）

沒有走 `build_submission`（它的 `FILE_MAP` 是規則式那組）。攤平用的是同一支
檔案的 `_flatten()`，所以 import 改寫規則一致：

```
main.py         入口，import gen2_model 的 agent
gen2_model.py   agents/gen2_model.py     （唯一的改寫：serving.npz_forward -> npz_forward）
npz_forward.py  serving/npz_forward.py   （無改寫）
contracts.py    contracts.py             （無改寫，ENCODER_VERSION 5）
weights.npz     model/artifacts/weights-e2e-round6.npz  （原樣複製）
```

驗過的事（2026-08-24）：四支檔案跟來源逐行相同（`npz_forward.py` 只差
CRLF/LF）、`weights.npz` sha256 一致、`ENCODER_VERSION` 與 npz 內的
`encoder_version` 都是 5、隔離 `sys.path` 之後 import 解析到的是子資料夾裡
那幾支、同一個 obs 餵進去 action 與 `--a e2e` + round6 **逐位元組相同**。

```powershell
Remove-Item Env:KAGGRI_WEIGHTS -ErrorAction SilentlyContinue
python -m eval.runner --a e2e-round6-sub --b gen1 --games 20 --workers 16

kaggle competitions submit kaggriculture `
  -f submission/e2e-round6/submission.tar.gz `
  -m "端到端網路 round6"
```

> ⚠️ **這份還沒過 `submission/main.py` 寫的換版門檻**（對 `gen1` 只有 20.0%，
> [10.5%, 34.8%]）。2026-08-24 是在看過這個數字之後決定仍要送的。
>
> ⚠️ `submission/e2e-round6/{weights.npz, submission.tar.gz}` 共約 12 MB，
> **不在 `.gitignore` 的任何規則裡**（頂層的 `submission/weights.npz` 才有
> 專門放行）。`git add -A` 會把它們掃進去。

---

## L0

```powershell
python -m pytest -q          # 81 項，約 45 秒
```
