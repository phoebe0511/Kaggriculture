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

# 產生 submission/submission.tar.gz
KAGGRI_LOG_LEVEL=0 python -m serving.build_submission --tar

# 完成 `kaggle auth login` 後提交
kaggle competitions submit kaggriculture \
  -f submission/submission.tar.gz \
  -m "Gen1 active-tiles on-demand-water"
```

## 目前基準（engine 1.32.7）

`main.py` = Gen1 第二輪三地版：`tiles_per_unit=4`、12 格動物建物，加入肥料
貨幣 ROI、季末 10→8 人縮編、同優先序任務的全域最短配對、白天精準回倉，
並禁止在每天最後一小時播下必定立即變成雜草的新苗。

- 對 starter：30 個 paired seeds／60 局全勝，平均現金 `$120,841`。
- 對第一輪凍結版 ref-v7：60 局全勝，平均 `$94,551 vs $78,720`，差 `+$15,831`。
- 凍結 ladder：10 個對手 × 5 paired seeds，共 100 局全勝。
- 對 starter 的動作分布：MOVE `56.4%`／生產 `26.5%`／PASS `17.0%`；
  全季管理中土地 `63.0%`，雜草率 `2.2%`。
- 條件式四地版仍保留在 `config/opponents/gen1-four-land.json`；13～15 格動物建物
  重掃也全數輸給 12 格，因此都沒有升主版。

⚠️ **已知的失敗模式：品項過度集中。** 格數變多會讓 `_plan_basket` /
`_plan_animals` 把配額集中到少數品項，同時放掉當局稀缺、單價高的品項。
seed 41003 對 starter 從 `$86,263` 掉到 `$49,892`，逐件重建收入後的歸因是：

| item | t3 量 → t4 量 | 收入差 |
|---|---|---|
| STRAWBERRY | 120 → 46 | **−18,285** |
| MILK | 108 → 156 | **−14,921**（賣多 44%、收入少 62%） |
| WOOL | 58 → 0 | **−12,394** |
| CARROT | 186 → 336 | +4,867 |
| MELON | 170 → 181 | **+713** |

**MELON 是清白的** —— 期末價 `$43 → $10` 只是 `above_func=sq` 在那個位置太陡的
快照假象，收入其實還多了 $713。真正的洞是 STRAWBERRY 減種、MILK 自我灌爆、
WOOL 整個消失。`max_crop_share` / `max_animal_share` 待跟 `tiles_per_unit` 一起重掃。
