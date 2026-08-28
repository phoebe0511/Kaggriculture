"""PPO 的 rollout 和 update 必須對同一個動作算出同一個 logprob。

對不上的話 **不會報錯** —— PPO 的 ratio 一開始就不是 1，大部分樣本被 clip
掉，表現成「訓練跑得動但學不起來」。所以直接跑一段短對局再逐 minibatch 比。

用隨機權重、短 episode，L0 預算內跑得完。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 是開發側依賴，submission 不用")


@pytest.mark.parametrize("module", [
    "model.ppo", "model.ppo_train", "harness.ppo_rollout", "harness.ppo_pool",
    "agents.ppo_agent", "agents.gen3_target",
])
def test_entry_points_import(module):
    """🩸 2026-08-28：`model/ppo_train.py` 有語法錯誤，整套 123 項測試照樣全過
    （沒有一項 import 它），過夜的訓練一啟動就死。這裡逐支 import 一次。"""
    __import__(module)


@pytest.mark.parametrize("module", ["model.ppo_train", "harness.ppo_pool"])
def test_argparse_help_renders(module, capsys):
    """🩸 2026-08-28：`--zero-sum` 的 help 裡有「86.4% 的」，argparse 會對 help
    字串做 `% params`，`%的` 不是合法的格式碼 —— `--help` 直接拋
    `ValueError: unsupported format character`。

    正常訓練不受影響（argparse 只在印 help 時展開），所以這個 bug **只會在你想
    先驗一下指令對不對的時候咬你**，正是最不該壞掉的時機。help 裡的 `%` 要寫
    成 `%%`。
    """
    import importlib

    mod = importlib.import_module(module)
    with pytest.raises(SystemExit) as e:
        mod.main(["--help"])
    assert e.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_market_temp_desaturates_only_the_present_head(tmp_path):
    """key 名字改掉的話這個縮放會安靜地變成 no-op，market head 就又收不到梯度。"""
    from harness.ppo_rollout import build_net
    from model.ppo_train import load_init

    torch.manual_seed(1)
    net = build_net(width=16, blocks=1)
    ckpt = tmp_path / "w.pt"
    torch.save({"state_dict": net.state_dict()}, ckpt)
    before = {k: v.clone() for k, v in net.state_dict().items()}

    cold = build_net(width=16, blocks=1)
    taken, skipped = load_init(cold, str(ckpt), market_temp=4.0)
    assert not skipped
    after = cold.state_dict()
    scaled = {"market_present_out.weight", "market_present_out.bias"}
    assert scaled <= set(before), "權重名字改了，縮放會變成 no-op"
    for k, v in before.items():
        want = v / 4.0 if k in scaled else v
        assert torch.allclose(after[k], want), k


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


def test_rewards_default_to_own_cash_only():
    """對手的現金增量佔 86.4% 的變異數而且我們控制不了，所以預設不減它。"""
    from model.ppo import REWARD_SCALE, TERMINAL_BONUS, step_rewards

    a = np.linspace(3000, 90000, 11)
    b = np.linspace(3000, 70000, 11)
    own = step_rewards(a, b)
    zs = step_rewards(a, b, zero_sum=True)
    da, db = np.diff(a), np.diff(b)
    assert np.allclose(own[:-1], da[:-1] / REWARD_SCALE)
    assert np.allclose(zs[:-1], (da - db)[:-1] / REWARD_SCALE)
    # 勝負 bonus 兩種形式都要有 —— 競爭性靠它保留。
    assert own[-1] == pytest.approx(da[-1] / REWARD_SCALE + TERMINAL_BONUS)
    # 輸的一方拿到 −bonus。
    assert step_rewards(b, a)[-1] == pytest.approx(
        np.diff(b)[-1] / REWARD_SCALE - TERMINAL_BONUS)


def test_gae_and_rewards_are_zero_sum():
    from model.ppo import compute_gae, step_rewards

    a = np.linspace(3000, 90000, 11)
    b = np.linspace(3000, 70000, 11)
    ra = step_rewards(a, b, zero_sum=True)
    rb = step_rewards(b, a, zero_sum=True)
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


def test_hybrid_mode_only_credits_the_market_head():
    """混合模式：`gen0.act` 出工人動作，網路只出 market。

    🩸 unit head 的 logprob **不能算進去** —— 那些動作不是網路選的，算進去等於
    在對別人的選擇做 policy gradient，而且 update 重算時會加回來，ratio 就錯了。
    所以軌跡裡的 unit 數必須是 0，logprob 往返照樣要對得上。
    """
    from harness.ppo_rollout import (VecRollout, build_net, load_base_policy,
                                     load_opponent)
    from model.ppo import RolloutBatch, evaluate_actions

    spec = "config/params/cma1-g50-wt.json"
    torch.manual_seed(0)
    net = build_net(width=16, blocks=2)
    vec = VecRollout(net, n_envs=2, seed0=88_000, episode_steps=72,
                     opponent=load_opponent(spec),
                     base_policy=load_base_policy(spec))
    steps, _cash, trajs = vec.run(collect=True)
    assert len(trajs) == 2, "打固定對手時一個 env 只收我方那一條"
    assert sum(len(t.unit_step) for t in trajs) == 0, "混合模式不該存 unit"
    assert steps == sum(len(t) for t in trajs)

    batch = RolloutBatch(trajs)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for mb in batch.minibatches(64, rng):
            lp, ent, _v = evaluate_actions(net, mb)
            assert (lp - mb["old_logp"]).abs().max().item() < 1e-3
            assert torch.isfinite(ent).all()


def test_hybrid_agent_takes_units_from_base_and_market_from_net(tmp_path):
    """混合模式的 agent：工人動作**逐一等於** base policy 的輸出，market 不等於。

    🩸 這兩件事任一個反了都不會報錯 —— 只會表現成「分數跟 gen0 一模一樣」
    （market 沒換到）或「工人亂走」（base 沒接上）。
    """
    import numpy as np

    from agents.ppo_agent import act
    from harness.ppo_rollout import build_net, load_base_policy

    spec = "config/params/cma1-g50-wt.json"
    torch.manual_seed(0)
    net = build_net(width=16, blocks=2)
    ckpt = tmp_path / "h.pt"
    torch.save({"state_dict": net.state_dict(), "width": 16, "blocks": 2}, ckpt)

    import contracts as C
    from kaggle_environments import make

    env = make("kaggriculture", debug=False)
    env.reset(2)
    obs = env.steps[0][0]["observation"]
    cfg = env.configuration

    base = load_base_policy(spec)(obs, cfg)
    got = act(obs, cfg, {"ckpt": str(ckpt), "base": spec, "greedy": True})

    assert got["farmer"] == base["farmer"], "farmer 沒有沿用 base"
    assert got["hands"] == base["hands"], "hands 沒有沿用 base"
    assert set(got) == set(base), "回傳的 key 集合變了"
    # market 來自隨機權重的網路，跟 gen0 自己的訂單重合的機率極低。
    assert got["market"] != base["market"], "market 沒有換成網路的"

    # 不給 base 就是純網路那條，工人動作不該再等於 gen0。
    pure = act(obs, cfg, {"ckpt": str(ckpt), "greedy": True})
    assert set(pure) == set(base)
