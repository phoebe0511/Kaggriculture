"""Kaggle submission 入口 —— 端到端網路版（`agents/gen2_model.py`），round6 權重。

## ⚠️ 這份還沒過 `submission/main.py` 寫的換版門檻

`submission/main.py`（規則式 gen0，榜上現行版本）明講：換網路版之前要先在
`eval.runner` 打贏門檻，在那之前榜上維持規則式。

**round6 的實測是輸的，不是贏的：**

    KAGGRI_WEIGHTS=model/artifacts/weights-e2e-round6.npz \
      python -m eval.runner --a e2e --b gen1 --games 20 --workers 16

    對 gen1（榜上現行的那個 agent）：40 局配對，8/0/32，得分率 20.0%
    [10.5%, 34.8%]，現金差 -6,075。跟同日的 big24（20.0%）配對比較，
    Wilcoxon p=0.599 —— 判不出誰比較強。

2026-08-24：使用者已經看過這個數字，明確決定仍要把這份包成 submission 上場。
這個資料夾是那個決定的產物，**不是**「門檻已經打贏」的紀錄。

## 為什麼是獨立資料夾，不是改掉 `submission/main.py`

`submission/main.py`（gen0）沒有動 —— 那是 A 的檔案，而且它自己的
docstring 講的規則沒有變。這裡是另一份完整攤平的 submission，指到哪一個
就是哪一個上場：

    kaggle competitions submit kaggriculture \\
      -f submission/e2e-round6/submission.tar.gz \\
      -m "端到端網路 round6"

## 內容

跟 `serving/build_submission.py` 的攤平規則一樣（見那支檔案的 docstring
解釋為什麼要攤平），只是這次帶的是網路版三支檔案 + 權重：

    main.py         這支
    gen2_model.py   `agents/gen2_model.py` 攤平（`from serving.npz_forward
                    import` 改寫成 `from npz_forward import`）
    npz_forward.py  `serving/npz_forward.py` 攤平（無需改寫）
    contracts.py    `contracts.py` 攤平（無需改寫，`ENCODER_VERSION` 5，
                    跟 `weights.npz` 裡存的 `encoder_version` 對過，一致）
    weights.npz     `model/artifacts/weights-e2e-round6.npz` 的原樣複製
"""

import os

# gen2_model 在 import 時讀取此值，所以必須先設定。
os.environ["KAGGRI_LOG_LEVEL"] = "0"

from gen2_model import agent  # noqa: E402,F401

# 這裡**故意不呼叫** `serving.action_validation.assert_legal_action`。
# 理由跟 `submission/main.py` 底部那段一樣：比賽當下抓到 bug 也不能改，
# 而且私有 API 會隨引擎版本改（1.29.3 vs 1.32.7 參數不同，多傳會每回合
# TypeError）。驗證留在開發側：`tests/test_l0_smoke.py` 每回合都會呼叫。
