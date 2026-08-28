# `model/` 放什麼

    model/
      net.py  train.py           程式碼
      weights-*.npz              **指標性權重** —— 要留下來的那幾份
      artifacts/                 產物，隨時可以整個刪掉重跑

## 為什麼要分

`model/` 以前是程式碼、checkpoint、匯出的 `.npz` 混在一起，13 個 `ckpt-*/`
加 13 個 `.npz` 一共 88 MB，要找「現在最好的是哪一份」得看 mtime。

規則：**`artifacts/` 裡的東西，只要 `data/` 還在，就重跑得出來**（`model.train`
\+ `serving.export_npz`，一輪約 15 分鐘，隨機性讓數字不會完全一樣）。

🩸 **「重跑得出來」只在這台機器上成立。** `data/`（396 MB 的 DAgger 資料）
也在 `.gitignore` 裡，所以**從 clone 出來的 repo 產不出任何一份權重**。
硬碟掛掉的話 `artifacts/` 是真的沒了，不是「重跑 15 分鐘」。
指標性權重會進 git 就是為了這個。

## 現在留著哪幾份，為什麼

| 檔案 | 為什麼留 |
|---|---|
| `weights-e2e-round0.npz` | **純 BC 的起點**。要回答「DAgger 到底帶來多少」只能跟它比 |
| `weights-e2e-round2.npz` | 市場診斷是拿這一份做的（`agents/gen2_model.py` 的 docstring、`README.md`、journal §10/§11 的門檻掃描全部引用它） |
| `weights-e2e-round5.npz` | **舊架構（`--width 96 --blocks 6`，867k 參數）最好的一份**，對 gen1 現金比 84.4%（`docs/eval-results.md`）。留著是為了回答「加大模型到底帶來多少」 |
| `weights-e2e-round6.npz` | **目前最好的**。2026-08-24 換規格重練（`--epochs 24 --width 128 --blocks 8`，1,599,159 參數）。對 `gen1` 現金比 **93%**（81,802 / 87,877）、得分率 20.0% [10.5%, 34.8%]，40 局配對（`temp/20260824-124546_e2e_vs_gen1`）。同日 `weights-e2e-big24.npz`（同架構、少 200 局資料）配對比較 Wilcoxon p=0.599 —— **兩者判不出高下**，留 round6 是因為資料較多、`op` 0.8996 與 market recall 0.9345 略高 |

其餘的（`v4` / `v5-round*` / `dagger*` / `kawashigi`）在 `artifacts/` 裡。
🩸 **它們現在載不起來** —— 對應的 agent（`gen3_target.py` / `gen4_demand.py`）
已於 2026-08-21 刪除，而且 op head 的語意是 `target` 不是 `immediate`
（`agents/gen2_model.require_labels()` 會擋下來）。留著只是歷史。

`artifacts/weights.npz` 是 ENCODER_VERSION 2 的舊檔，**載進去會在第一回合
SystemExit**，錯誤訊息看起來像 `contracts.py` 的問題。不要當成預設值。

## git

`.gitignore` 擋掉所有 `*.npz` / `*.pt` 與整個 `model/artifacts/`，然後
**逐檔**放行上面那四份（不是萬用字元）。加一份新的指標性權重 = 在
`.gitignore` 的放行清單多寫一行 + 在上面那張表多寫一列說明為什麼留。

`submission/weights.npz` 也放行 —— 那是要上場的那一份，由
`serving/build_submission.py` 從選定的 `model/weights-*.npz` 複製過去。
⚠️ 現在那個檔**不存在**：`main.py` 出貨的是規則式，`DEFAULT_WEIGHTS` 是
`None`，換成網路版才會產生。

### 不要讓 .git 再漲回去

2026-08-21 曾經有 23 個權重檔被 `git add -A` 掃進來，`.git` 漲到 72 MB。
防線有三道：

1. `*.npz` 仍然擋住所有東西，放行是逐檔寫的
2. **一份權重進 git 之後就不再改那個檔** —— 新的一輪用新檔名。
   binary 沒有 delta 壓縮，改一次等於在 history 裡多存一份 3.1 MB
3. 產物寫進 `artifacts/`，那整個目錄是擋掉的

## CMA-ES 的狀態檔（`cma1-*.pkl`）

⚠️ **這些不是網路權重**，是規則式 `agents/gen0.py` 那 51 項參數的搜尋狀態
（`tools/param_search.py` 產的 `cma.CMAEvolutionStrategy` pickle）。放在這裡是
因為 `model/` 已經有「大的二進位產物 + `.gitignore` 逐檔放行」的慣例。

| 檔 | 是什麼 | 為什麼留 |
|---|---|---|
| `cma1-g50.pkl` | run `temp/cma/20260827-164222` 第 50 代 | **holdout / 九隊最好的那一組**。落地的參數在 `config/params/cma1-g50.json` |
| `cma1-g68.pkl` | 同一輪第 68 代 | train 分數最好（−8,654）但 holdout 輸給第 50 代。留著當對照 |

**要用參數的話讀 `config/params/*.json` 就好**，那裡 51 項完整展開。pkl 只有
兩個用途：

1. `--warm-start`：借它的 `es.best.x` 當下一輪的起點（covariance 不沿用）
2. 重現：`es.countiter` / `es.sigma` / covariance 都在裡面

`.pkl` 不在 `.gitignore` 的擋掉清單裡，所以是預設進版控的。**一份進 git 之後
就不要再改那個檔**（同上面那條，binary 沒有 delta 壓縮）—— 新的一輪用新檔名
（`cma2-*`）。單檔 ~310 KB。
