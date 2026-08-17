# === CELL 10: 用 Kaggle 實際的載入方式跑完整對局 —— 只是測試，留在 submission notebook 也安全 ===
# **關鍵：agent 要用「檔案路徑」傳進 env.run()，不要 import 它。**
#
# `env.run([agent_fn, "starter"])` 用的是你 notebook 裡已經 import 好的物件，
# sys.path 是 notebook 的 sys.path —— 那跟評分時的載入路徑不一樣，會漏掉
# 「import 撞名」「少傳一支檔案」這類只在真實載入才炸的問題。
# 傳路徑的話 kaggle_environments 會走 `sys.path.append(exec_dir)` + `exec()`
# 那條路，跟評分時同一條。本機就是這樣才抓到 lux_ai_s3 的 agents.py 撞名。
import os
import time

MAIN = "/kaggle/working/main.py"

if not os.path.isfile(MAIN):
    print(f"{MAIN} 不存在 —— 先跑 cell 7。跳過。")
else:
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _saved = os.dup(1)
    os.dup2(_devnull, 1)              # open_spiel 匯入時會噴幾百行遊戲清單
    try:
        from kaggle_environments import make
    finally:
        os.dup2(_saved, 1)
        os.close(_saved)
        os.close(_devnull)

    SEEDS = (41001, 41002, 41003)
    # 本機 1.32.7 + 預設 configuration 實測 —— 那組設定跟 **ladder 相同**
    # （2026-08-17 由 ladder 對局 93916293 的 JSON 確認）。
    #
    # ⚠️ Kaggle notebook 預裝的是 **1.29.3**，而且 configuration 不一樣
    #    （farmHandCostMult 10、startingMoney 2000、town interval 6/2），
    #    價格模型也不同，所以**這裡一定對不上**。要對得上，notebook 最前面
    #    要先跑 `!pip install --upgrade "kaggle-environments>=1.32.2"` 再 Restart。
    #    這一格的用途是「檔案載入會不會失敗」，不是比強度 —— 分數差很多但
    #    status 是 DONE，就算通過。
    BASELINE = {41001: 90_897, 41002: 88_809, 41003: 78_066}

    results = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        env = make("kaggriculture", configuration={"seed": seed}, debug=True)
        env.run([MAIN, "starter"])
        elapsed = time.perf_counter() - t0
        me, opp = env.state[0], env.state[1]
        results.append((seed, me.reward, opp.reward, me.status, len(env.steps), elapsed))

        base = BASELINE.get(seed)
        delta = f"  本機 {base:>8,.0f}  差 {me.reward - base:+,.0f}" if base else ""
        flag = "  " if me.status == "DONE" else "!! "
        print(f"{flag}seed {seed}  現金 {me.reward:>9,.0f}  對手 {opp.reward:>8,.0f}  "
              f"{me.status}  {len(env.steps)} steps  {elapsed:.1f}s{delta}")
        if me.status != "DONE":
            # 載入失敗時錯誤在這裡，env.run 本身不會拋
            print(f"  info = {me.info}")
            print("  status 不是 DONE：多半是 import 失敗或動作驗證拋錯。"
                  " 用 env.logs 看 agent 的 stderr。")
            for entry in (env.logs or [])[:3]:
                print("  log:", entry)
            break

    ok = [r for r in results if r[3] == "DONE"]
    if len(ok) == len(SEEDS):
        print(f"\nOK {len(ok)} 局全部跑完，平均現金 {sum(r[1] for r in ok)/len(ok):,.0f}")
        print("  跟本機的差異：Kaggle 的 CPU 只影響速度，不影響結果 —— "
              "同 seed 同引擎版本應該完全一致。對不上就代表引擎版本或檔案版本不同。")
    else:
        print(f"\n!! 只有 {len(ok)}/{len(SEEDS)} 局完成")
