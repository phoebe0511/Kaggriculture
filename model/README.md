# `model/` 放什麼

    model/
      net.py  train.py           程式碼
      weights-*.npz              **指標性權重** —— 要留下來的那幾份
      artifacts/                 產物，隨時可以整個刪掉重跑

## 為什麼要分

`model/` 以前是程式碼、checkpoint、匯出的 `.npz` 混在一起，13 個 `ckpt-*/`
加 13 個 `.npz` 一共 88 MB，要找「現在最好的是哪一份」得看 mtime。

現在的規則很簡單：**`artifacts/` 裡的東西全部可以重新產生**（重跑
`model.train` + `serving.export_npz`），刪掉只是浪費時間，不會弄丟資訊。
放在 `model/` 底下的那幾份不行 —— 對應的訓練資料還在，但重跑要 15 分鐘，
而且隨機性讓數字不會完全一樣。

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

`.gitignore` 擋掉所有 `*.npz` / `*.pt` 與整個 `model/artifacts/`。
**`model/` 底下那幾份指標性權重也不在 git 裡** —— 要上場的那一份由
`serving/build_submission.py` 複製成 `submission/weights.npz`，那是唯一
被追蹤的權重（`!submission/weights.npz`）。真的要進 git 就 `git add -f`。
