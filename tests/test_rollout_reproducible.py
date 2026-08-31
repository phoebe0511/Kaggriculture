"""同樣的 seed 一定要跑出同樣的軌跡。

🩸 2026-08-31：動作取樣用 `torch.multinomial` / `torch.rand`，吃的是**全域**
RNG，而 `harness/ppo_pool.py` 的 worker 只呼叫 `torch.set_num_threads(1)`、
沒有 seed torch。`--seed0` 本來只餵給環境（決定地圖），沒餵給動作取樣。

後果：同一個 checkpoint、同樣的參數跑兩次，第 0 輪現金 63,543 vs 60,704、
第 10 輪 greedy 77,628 vs 46,490（差 4.4 個評估標準誤）。**兩次單獨的訓練
沒辦法拿來 A/B** —— run-to-run 的差異比要量的效果還大。

這不會報錯，只會讓每個實驗都在跟雜訊比。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 是開發側依賴")


def _run(seed0, torch_seed=None):
    from harness.ppo_rollout import (VecRollout, build_net, load_base_policy,
                                     load_opponent)

    if torch_seed is not None:          # 故意攪動全域 RNG
        torch.manual_seed(torch_seed)
    spec = "config/params/cma1-g50-wt.json"
    torch.manual_seed(0)
    net = build_net(width=16, blocks=2)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    vec = VecRollout(net, n_envs=2, seed0=seed0, episode_steps=96,
                     opponent=load_opponent(spec),
                     base_policy=load_base_policy(spec))
    with torch.no_grad():
        steps, cash, trajs = vec.run(collect=True)
    return steps, vec.our_cash(cash), trajs


def test_same_seed_gives_identical_cash():
    a_steps, a_cash, _ = _run(5150)
    b_steps, b_cash, _ = _run(5150)
    assert a_steps == b_steps
    assert a_cash == b_cash, f"同 seed 現金不同：{a_cash} vs {b_cash}"


def test_global_rng_state_does_not_leak_in():
    """🩸 這是本檔的重點：取樣必須用自己的 generator，不是全域 RNG。

    全域 RNG 被攪動成不同狀態，結果仍然要一樣。用全域 RNG 的話這裡會掛 ——
    而那正是 pool worker 的實際處境（每個行程的全域種子都不一樣）。
    """
    _s1, c1, _ = _run(5150, torch_seed=1)
    _s2, c2, _ = _run(5150, torch_seed=999_777)
    assert c1 == c2, (
        f"全域 RNG 換了種子就跑出不同結果：{c1} vs {c2} —— "
        f"取樣沒有走 VecRollout 自己的 generator")


def test_different_seeds_actually_differ():
    """反面：seed 不同就該不同，不然是取樣根本沒在隨機。"""
    _s1, c1, _ = _run(5150)
    _s2, c2, _ = _run(8888)
    assert c1 != c2, "換了 seed 結果卻一樣 —— 取樣沒有吃到 seed"


def test_logprobs_are_reproducible_too():
    _s1, _c1, t1 = _run(5150)
    _s2, _c2, t2 = _run(5150)
    assert len(t1) == len(t2)
    for x, y in zip(t1, t2):
        assert np.array_equal(x.logp, y.logp)
        assert np.array_equal(x.mk_present, y.mk_present)
