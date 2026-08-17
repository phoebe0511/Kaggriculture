# === CELL 20: 送出前的最後檢查 —— 時限餘裕 ===
# cell 10 量的是「整局多久」，那個數字沒有意義：Kaggle 卡的是**單回合**。
#   actTimeout = 1 秒/回合      ← 超過就判這一步失敗
#   runTimeout = 1200 秒/整局
# 720 回合平均 8 ms 也可能有某一回合 900 ms。要看的是尖峰，不是平均。
#
# 這格自己包一層計時再交給 env.run()，所以量到的是 agent 真正的耗時
# （含動作驗證），不含引擎自己的結算時間。
import os
import time

MAIN = "/kaggle/working/main.py"

if not os.path.isfile(MAIN):
    print(f"{MAIN} 不存在 —— 先跑 cell 7。跳過。")
else:
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _saved = os.dup(1)
    os.dup2(_devnull, 1)
    try:
        from kaggle_environments import make
    finally:
        os.dup2(_saved, 1)
        os.close(_saved)
        os.close(_devnull)

    # 用 import 的方式取得 agent 才包得住計時。
    # 警告 這條路徑跟評分不同（見 cell 10 的說明），所以**只拿來量時間**，
    #   「載入會不會成功」以 cell 10 為準。
    import sys
    if "/kaggle/working" not in sys.path:
        sys.path.insert(0, "/kaggle/working")
    for stale in ("main", "gen0", "gen1", "action_validation"):
        sys.modules.pop(stale, None)
    from main import agent as _agent   # noqa: E402

    env = make("kaggriculture", configuration={"seed": 41003}, debug=True)
    cfg = env.configuration
    act_timeout = float(cfg["actTimeout"])
    run_timeout = float(cfg["runTimeout"])

    times = []

    def timed(obs, config):
        t0 = time.perf_counter()
        try:
            return _agent(obs, config)
        finally:
            times.append(time.perf_counter() - t0)

    wall0 = time.perf_counter()
    env.run([timed, "starter"])
    wall = time.perf_counter() - wall0

    times.sort()
    n = len(times)
    peak = times[-1]
    p99 = times[int(n * 0.99)] if n else 0.0
    mean = sum(times) / n if n else 0.0

    print(f"回合數 {n}   現金 {env.state[0].reward:,.0f}   {env.state[0].status}")
    print(f"  平均      {mean*1000:7.1f} ms")
    print(f"  p99       {p99*1000:7.1f} ms")
    print(f"  尖峰      {peak*1000:7.1f} ms      餘裕 {act_timeout/max(peak,1e-9):6.1f}x")
    print(f"  整局      {wall:7.1f} s       餘裕 {run_timeout/max(wall,1e-9):6.1f}x")
    print(f"  最慢的 5 回合 (ms): {[round(t*1000, 1) for t in times[-5:]]}")

    # 門檻用 2x 而不是 1x —— Kaggle 的機器有雜訊，剛好卡在 1.0 秒等於沒有餘裕。
    problems = []
    if peak > act_timeout / 2:
        problems.append(f"單回合尖峰 {peak*1000:.0f} ms 超過 actTimeout 的一半"
                        f"（{act_timeout*500:.0f} ms）")
    if wall > run_timeout / 2:
        problems.append(f"整局 {wall:.0f}s 超過 runTimeout 的一半（{run_timeout/2:.0f}s）")
    if env.state[0].status != "DONE":
        problems.append(f"status = {env.state[0].status}")

    if problems:
        for p in problems:
            print(f"\n!! {p}")
        print("\n時間不夠時，先看 action_validation：它每個 unit 做兩次 deepcopy，"
              "13 個 unit x 720 回合 = 18,720 次。本機實測它讓整局從 2.6s 變 5.8s、"
              "單回合尖峰從 15.7 ms 變 30.6 ms。")
    else:
        print("\nOK 時限餘裕足夠")
