# 基準線：榜上那一版
python -m eval.runner --a gen1 --ladder --games 20 --workers 26

# 市場也走網路那一臂（跟上面同一份權重，唯一差別是 model_market）
KAGGRI_WEIGHTS=model/weights-v5-round0.npz python -m eval.runner \
       --a gen4_demand --ladder --games 20 --workers 26

       KAGGRI_WEIGHTS=model/weights-v5-round0.npz python -m eval.runner \
       --a gen4_demand --ladder --games 20 --workers 10