# 基準線：榜上那一版
python -m eval.runner --a gen1 --ladder --games 20 --workers 26

# 市場也走網路那一臂（跟上面同一份權重，唯一差別是 model_market）
KAGGRI_WEIGHTS=model/weights-v5-round0.npz python -m eval.runner \
       --a gen4_demand --ladder --games 20 --workers 26

       KAGGRI_WEIGHTS=model/weights-v5-round0.npz python -m eval.runner \
       --a gen4_demand --ladder --games 20 --workers 10

KAGGRI_WEIGHTS=model/weights-e2e-round2.npz \
  python -m eval.runner --a e2e-restock20 --b gen1 --games 20 --workers 16

KAGGRI_WEIGHTS=model/weights-e2e-round2.npz \
  python -m eval.runner --a e2e-restock20 --ladder config/sweep-hire --games 20 --workers 5

  PYTHONIOENCODING=utf-8 timeout 600 python -m eval.runner --a gen1 --ladder config/sweep-hire.json --games 2 --workers 12 2>&1 | tail -20