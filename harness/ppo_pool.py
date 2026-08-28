"""多行程 rollout。每個 worker 行程自己跑一組 env，主行程只做 PPO 更新。

    python -m harness.ppo_pool --workers 10 --envs 2      # 量吞吐量

## 為什麼要多行程

單行程 16 個 env、720 步、對手 `cma1-g50-wt`，實測 **2.21 秒/局**
（2026-08-28）。成本拆開來大約是：

    引擎 env.step      49%
    gen0 對手 act      39%
    encode + 網路前向  12%

**網路只佔 12%**，其餘是純 Python/CPU，多行程幾乎線性加速。10^8 步 ≈ 139,000
局：單行程 85 小時，10 行程 8.5 小時。

## 一次迭代搬多少東西

    往 worker   state_dict 1.89 MB × workers（width=64 blocks=4，468,215 參數）
    回主行程    軌跡：spatial float16 是 5.5 MB/局，其餘加起來不到 0.2 MB/局

spatial 存 float16 是 `VecRollout.half_obs` 做的 —— 前向也吃還原後的值，
所以 update 重算的 logprob 跟 rollout 存的位元相同（見 `tests/test_ppo.py`）。

## 🩸 兩個踩過的坑

- **`mp.Pool` 不能在 `silenced()` 裡面建**。Windows 是 spawn，子行程會繼承
  被 dup2 過的 fd，2026-08-27 卡了 30 分鐘才找到。
- **每個 worker 要 `torch.set_num_threads(1)`**。10 個行程各開 20 條執行緒
  會互搶（`worker-count` 那則記憶：這台 20 實體核心，開 24 反而更慢）。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")

#: worker 行程的常駐狀態。每次迭代只換權重，env 和對手不重建。
_W = {}


def _init(cfg):
    """在 worker 行程裡跑一次。import 都留在這裡面，主行程不受影響。"""
    import torch                                          # noqa: PLC0415

    from harness.ppo_rollout import (build_net, load_base_policy,  # noqa: PLC0415
                                     load_league)

    torch.set_num_threads(1)
    _W["cfg"] = cfg
    _W["net"] = build_net(cfg["width"], cfg["blocks"])
    _W["opps"] = load_league(cfg["opponent"])[0] if cfg["opponent"] else None
    _W["base"] = (load_base_policy(cfg["base_policy"])
                  if cfg.get("base_policy") else None)


def _run(job):
    """跑一輪。回傳 `(步數, [(我方現金, 對手現金, league 索引)], 軌跡)`。"""
    import torch                                          # noqa: PLC0415

    from harness.ppo_rollout import VecRollout             # noqa: PLC0415

    state, seed0, game0 = job
    cfg = _W["cfg"]
    _W["net"].load_state_dict(state)
    with torch.no_grad():
        vec = VecRollout(_W["net"], n_envs=cfg["envs"], seed0=seed0,
                         episode_steps=cfg["episode_steps"],
                         opponent=_W["opps"], opp_offset=game0,
                         base_policy=_W["base"])
        steps, cash, trajs = vec.run(collect=True)
    pairs = [(a, b, vec.opp_index(ei))
             for ei, (a, b) in enumerate(vec.our_cash(cash))]
    return steps, pairs, trajs


class RolloutPool:
    """常駐的 worker 池。`collect()` 一次跑 `workers × envs` 局。

    用 `with` 或記得 `close()` —— Pool 沒收掉的話 worker 會留著。
    """

    def __init__(self, workers=10, envs=2, opponent="", width=64, blocks=4,
                 episode_steps=None, base_policy=""):
        self.workers = workers
        self.envs = envs
        # 主行程只留名字，不 import agent 模組 —— 那是 worker 的事。
        self.names = [n.strip() for n in str(opponent).split(",") if n.strip()]
        cfg = {"width": width, "blocks": blocks, "opponent": opponent,
               "envs": envs, "episode_steps": episode_steps,
               "base_policy": base_policy}
        # 🩸 不要包在 silenced() 裡（見模組說明）。
        self.pool = mp.Pool(workers, initializer=_init, initargs=(cfg,))

    def collect(self, net, seed0, game0=0):
        """回傳 `(步數, [(我方, 對手, league 索引)], 軌跡 list)`。

        `game0` 是這一輪第一局的全域編號 —— league 靠它輪轉，不然每個 worker
        只會用到 league 的前 `envs` 支。
        """
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        jobs = [(state, seed0 + i * self.envs, game0 + i * self.envs)
                for i in range(self.workers)]
        steps, pairs, trajs = 0, [], []
        for s, p, t in self.pool.map(_run, jobs):
            steps += s
            pairs.extend(p)
            trajs.extend(t)
        return steps, pairs, trajs

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--envs", type=int, default=2, help="每個 worker 幾個 env")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--episode-steps", type=int, default=0)
    ap.add_argument("--opponent", default="config/params/cma1-g50-wt.json")
    ap.add_argument("--base-policy", default="",
                    help="混合模式的骨幹 spec（工人動作用它，網路只出 market）")
    ap.add_argument("--rounds", type=int, default=2,
                    help="跑幾輪。第一輪含 worker 啟動成本，看第二輪")
    args = ap.parse_args(argv)

    from harness.ppo_rollout import build_net

    net = build_net(args.width, args.blocks)
    games = args.workers * args.envs
    with RolloutPool(args.workers, args.envs, args.opponent, args.width,
                     args.blocks, args.episode_steps or None,
                     args.base_policy) as pool:
        for r in range(args.rounds):
            t0 = time.perf_counter()
            steps, pairs, trajs = pool.collect(net, 5000 + r * games,
                                               game0=r * games)
            dt = time.perf_counter() - t0
            mb = sum(t.nbytes() for t in trajs) / 1e6
            wins = sum(1 for a, b, _ in pairs if a > b)
            print(f"  第 {r} 輪  {games} 局  {dt:>6.1f} 秒  "
                  f"{dt/games:>5.2f} 秒/局  {steps/dt:>7.0f} 個決策/秒  "
                  f"軌跡 {mb:>6.0f} MB  勝 {wins}/{len(pairs)}", flush=True)
            if len(pool.names) > 1:
                for j, name in enumerate(pool.names):
                    sub = [(a, b) for a, b, k in pairs if k == j]
                    if sub:
                        print(f"      {name:<20} {len(sub):>3} 局  "
                              f"勝 {sum(1 for a, b in sub if a > b)}  "
                              f"現金 {sum(a for a, _ in sub)/len(sub):>8,.0f} vs "
                              f"{sum(b for _, b in sub)/len(sub):>8,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
