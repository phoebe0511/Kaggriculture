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

`main.py` = Gen1 三地版，`tiles_per_unit=4`（active tiles 上限 52 格 / 可種 69 格）。

- 30 個固定 seeds 對 starter：平均現金 `$86,306`
  （t3 版 `$83,667`、Gen0/ref-v2 `$77,336`）。
- 80 局 paired 對 ref-v3（t3 凍結量尺），兩組獨立種子：
  `32/8` + `29/11` = 勝率 `76.3%`，CI 都不跨 50%。
- 動作分布：MOVE `61.2%`／生產 `28.4%`／PASS `10.4%`；全季作物覆蓋 `56.7%`。
- 條件式四地版保留在 `config/opponents/gen1-four-land.json`，尚未升主版。

⚠️ **已知的失敗模式**：格數變多會連帶讓 `max_crop_share` 分到更多 MELON 格。
MELON 不在任何 shop 的需求清單、`above_func=sq` 最凶，量大時會自己把價格砸到地板。
對手很弱（市場供給幾乎全是自己的）時最明顯 —— seed 41003 對 starter 從
`$86,263` 掉到 `$49,892`，期末 MELON 價 `$43 → $10`、MILK `$275 → $84`。
對 ref-v3 這類強對手沒有這條左尾。`max_crop_share` 待跟 `tiles_per_unit` 一起重掃。
