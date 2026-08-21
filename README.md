# Kaggriculture

Kaggriculture competition agent。架構、規則與工作流程見 [`docs/`](docs/README.md)。

## 本機環境

需要 Python 3.11～3.13。遊戲引擎固定為 `kaggle-environments==1.32.7`；升級引擎時
必須重新核對 `docs/games/engine-notes.md` 並重跑 L0 baseline。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## 常用指令

```bash
# L0 smoke（3 個固定種子，目標 60 秒內）
KAGGRI_LOG_LEVEL=0 python -m pytest -q

# 規則式基準（agents/gen0.py，gen1 已於 2026-08-21 併入）
KAGGRI_LOG_LEVEL=0 python -m eval.runner --a gen1 --b starter --games 20 --workers 16

# 端到端網路（unit 動作與市場訂單全部走網路）
KAGGRI_WEIGHTS=model/weights-e2e-round5.npz   python -m eval.runner --a e2e --b gen1 --games 20 --workers 16

# 把所有跑過的分數整理成表
python -m tools.eval_table

# 產生 submission/submission.tar.gz
KAGGRI_LOG_LEVEL=0 python -m serving.build_submission --tar

# 完成 `kaggle auth login` 後提交
kaggle competitions submit kaggriculture \
  -f submission/submission.tar.gz \
  -m "規則式 gen0（榜上版本）"
```

⚠️ **`--workers` 上限 16。** `cpu_count()` 是 28，但 `harness/rollout.py` 的預設
`cpu_count - 2 = 26` 會吃掉整台機器。每個指令都要明寫。

## 目前狀態（engine 1.32.7）

**分數的權威記錄在 [`docs/eval-results.md`](docs/eval-results.md)**（由
`python -m tools.eval_table` 從 `temp/*/result.json` 產生，不要手抄）。

方法路線是 Expert Iteration：規則式暖身 → 訓練網路 → **網路 + search** → 再訓練。
現在在「訓練網路」那一步的尾聲，`search/` 還沒開始寫。

| | 對 `ladder-top-a` | 對規則式 |
|---|---|---|
| 規則式 `agents/gen0.py`（榜上這一版） | 56~58% | — |
| 端到端網路 `agents/gen2_model.py` | 未量 | 42% |

> 🚫 **不要把「退回規則式」或「把某一段還給 gen0」當成結論。**
> 模仿架構的天花板就是被模仿的對象，而規則式對 ladder 頂端只有 56~58%。
> 要超過它只能靠 search。`agents/gen1.py` 已經不存在（併進 `agents/gen0.py`）。

### 規則式那條路的已知失敗模式：品項過度集中

格數變多會讓 `_plan_basket` / `_plan_animals` 把配額集中到少數品項，同時放掉
當局稀缺、單價高的品項。seed 41003 對 starter 從 `$86,263` 掉到 `$49,892`，
逐件重建收入後的歸因是：

| item | t3 量 → t4 量 | 收入差 |
|---|---|---|
| STRAWBERRY | 120 → 46 | **−18,285** |
| MILK | 108 → 156 | **−14,921**（賣多 44%、收入少 62%） |
| WOOL | 58 → 0 | **−12,394** |
| CARROT | 186 → 336 | +4,867 |
| MELON | 170 → 181 | **+713** |

**MELON 是清白的** —— 期末價 `$43 → $10` 只是 `above_func=sq` 在那個位置太陡的
快照假象，收入其實還多了 $713。真正的洞是 STRAWBERRY 減種、MILK 自我灌爆、
WOOL 整個消失。
