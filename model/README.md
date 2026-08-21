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
| `weights-e2e-round5.npz` | **目前最好的 ①**，對 gen1 是 84.4%（`docs/eval-results.md`） |

其餘的（`v4` / `v5-round*` / `dagger*` / `kawashigi`）在 `artifacts/` 裡。
🩸 **它們現在載不起來** —— 對應的 agent（`gen3_target.py` / `gen4_demand.py`）
已於 2026-08-21 刪除，而且 op head 的語意是 `target` 不是 `immediate`
（`agents/gen2_model.require_labels()` 會擋下來）。留著只是歷史。

`artifacts/weights.npz` 是 ENCODER_VERSION 2 的舊檔，**載進去會在第一回合
SystemExit**，錯誤訊息看起來像 `contracts.py` 的問題。不要當成預設值。

## git

`.gitignore` 擋掉所有 `*.npz` / `*.pt` 與整個 `model/artifacts/`，然後
**逐檔**放行上面那三份（不是萬用字元）。加一份新的指標性權重 = 在
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
