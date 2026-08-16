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

# 主版本本機評估（Gen1 + 嚴格 action validator）
KAGGRI_LOG_LEVEL=0 python -m eval.runner --a main:agent --b starter --games 20

# 四地候選；不影響 main.py
KAGGRI_LOG_LEVEL=0 python -m eval.runner --a gen1-four-land --b starter --games 20

# 產生 dist/submission.tar.gz
KAGGRI_LOG_LEVEL=0 python -m serving.build_submission

# 完成 `kaggle auth login` 後提交
kaggle competitions submit kaggriculture \
  -f dist/submission.tar.gz \
  -m "Gen1 active-tiles on-demand-water"
```

## 目前基準（engine 1.32.7）

- 30 個固定 seeds 對 starter：平均現金 `$83,667`（Gen0/ref-v2 為 `$77,336`）。
- 60 局 paired 對 ref-v2：`55 / 0 / 5`，勝率 `91.7%`。
- MOVE 比例：`68.1% → 52.2%`。
- `main.py` 使用 Gen1 三地版；條件式四地版保留在
  `config/opponents/gen1-four-land.json`，尚未升主版。
