"""PPO 的 rollout 和 update 必須對同一個動作算出同一個 logprob。

對不上的話 **不會報錯** —— PPO 的 ratio 一開始就不是 1，大部分樣本被 clip
掉，表現成「訓練跑得動但學不起來」。所以直接跑一段短對局再逐 minibatch 比。

用隨機權重、短 episode，L0 預算內跑得完。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 是開發側依賴，submission 不用")


@pytest.fixture(scope="module")
def rollout():
    from harness.ppo_rollout import VecRollout, build_net

    torch.manual_seed(0)
    net = build_net(width=16, blocks=2)
    vec = VecRollout(net, n_envs=2, seed0=777, episode_steps=72)
    steps, cash, trajs = vec.run(collect=True)
    return net, steps, cash, trajs


def test_rollout_collects_two_trajectories_per_env(rollout):
    _net, steps, _cash, trajs = rollout
    # 自對局：一個 env 兩個 player 各一條。
    assert len(trajs) == 4
    assert steps == sum(len(t) for t in trajs)
    for t in trajs:
        # cash 要比步數多一個 —— 最後一個是期末。
        assert t.cash.shape == (len(t) + 1, 2)
        assert len(t.unit_step) == len(t.op_idx) == len(t.tgt_idx)
        assert t.mk_present.shape[0] == len(t)


def test_update_reproduces_rollout_logprob(rollout):
    """這是整條路的地基。"""
    from model.ppo import RolloutBatch, evaluate_actions

    net, _steps, _cash, trajs = rollout
    batch = RolloutBatch(trajs)
    rng = np.random.default_rng(0)
    worst = 0.0
    with torch.no_grad():
        for mb in batch.minibatches(64, rng):
            new_logp, ent, value = evaluate_actions(net, mb)
            worst = max(worst,
                        (new_logp - mb["old_logp"]).abs().max().item())
            assert torch.isfinite(ent).all()
            assert value.shape == mb["ret"].shape
    # float32 的捨入而已。1e-3 是給 unit 數量多的步留的餘裕。
    assert worst < 1e-3, f"重算的 logprob 對不上 rollout 存的：{worst:.3e}"


def test_market_head_is_part_of_the_distribution(rollout):
    """market 不進取樣的話 HIRE 拿不到 gradient，工人數會卡在 1。"""
    _net, _steps, _cash, trajs = rollout
    per_step = sum(len(t.unit_step) for t in trajs) / sum(len(t) for t in trajs)
    assert per_step > 1.5, f"每步只有 {per_step:.2f} 個 unit —— 沒有在雇人"
    assert any(t.mk_present.any() for t in trajs)


def test_gae_and_rewards_are_zero_sum():
    from model.ppo import compute_gae, step_rewards

    a = np.linspace(3000, 90000, 11)
    b = np.linspace(3000, 70000, 11)
    ra, rb = step_rewards(a, b), step_rewards(b, a)
    assert np.allclose(ra, -rb)

    values = np.zeros(11, dtype=np.float32)
    dones = np.zeros(10, dtype=np.float32)
    dones[-1] = 1.0
    adv, ret = compute_gae(ra, values, dones)
    assert adv.shape == (10,) and ret.shape == (10,)
    assert np.isfinite(adv).all()


def test_ragged_minibatch_gather_picks_the_right_units():
    """CSR 那段配錯 unit 不會報錯，只會安靜地學錯 —— 用假資料驗一次。"""
    import contracts as C
    from model.ppo import RolloutBatch, Trajectory

    n_steps, counts = 5, [1, 3, 2, 4, 1]
    total = sum(counts)
    unit_step = np.repeat(np.arange(n_steps), counts).astype(np.int64)
    # op_idx 用「第幾個 unit」當可辨識的印記。
    tr = Trajectory(
        spatial=np.zeros((n_steps, C.N_SPATIAL, 10, 10), np.float32),
        scalar=np.zeros((n_steps, C.N_SCALAR), np.float32),
        value=np.zeros(n_steps, np.float32),
        logp=np.zeros(n_steps, np.float32),
        cash=np.zeros((n_steps + 1, 2), np.float64),
        mk_legal=np.zeros((n_steps, C.N_MARKET_OPS), bool),
        mk_present=np.zeros((n_steps, C.N_MARKET_OPS), bool),
        mk_qty=np.zeros((n_steps, C.N_MARKET_OPS), np.int64),
        unit_step=unit_step,
        unit_pos=np.zeros((total, 2), np.int16),
        unit_feats=np.zeros((total, C.N_UNIT_FEATURES), np.float32),
        op_mask=np.ones((total, C.N_UNIT_OPS), bool),
        tgt_mask=np.ones((total, C.N_TARGET_CELLS), bool),
        op_idx=np.arange(total, dtype=np.int64),
        tgt_idx=np.arange(total, dtype=np.int64),
    )
    batch = RolloutBatch([tr])
    idx = np.array([3, 1])                       # 故意不照順序
    mb = batch._pack(np.sort(idx), device="cpu")
    got = mb["op_idx"].numpy().tolist()
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    want = ([start[1] + i for i in range(counts[1])]
            + [start[3] + i for i in range(counts[3])])
    assert got == want
    assert mb["unit_board"].numpy().tolist() == [0] * counts[1] + [1] * counts[3]
