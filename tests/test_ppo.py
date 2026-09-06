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


def test_reverse_hybrid_only_credits_the_unit_heads():
    """反向混合：骨幹出 market、網路只出工人動作。

    要守的是 `base_side="units"` 的鏡像：
    - unit 樣本**要**存（那些動作是網路選的）
    - `mk_legal` 整列必須是 False —— `market_logp_entropy` 每一項都乘 `legal`，
      所以 market 對 logprob 的貢獻剛好 0，update 重算時不會把骨幹的訂單
      算進來。少了這一步，ratio 從第一輪就不是 1，樣本會被 clip 掉。
    """
    from harness.ppo_rollout import (VecRollout, build_net, load_base_policy,
                                     load_opponent)
    from model.ppo import RolloutBatch, evaluate_actions

    spec = "config/params/cma1-g50-wt.json"
    torch.manual_seed(0)
    net = build_net(width=16, blocks=2)
    vec = VecRollout(net, n_envs=2, seed0=91_000, episode_steps=72,
                     opponent=load_opponent(spec),
                     base_policy=load_base_policy(spec),
                     base_side="market")
    steps, _cash, trajs = vec.run(collect=True)
    assert len(trajs) == 2
    assert sum(len(t.unit_step) for t in trajs) > 0, "反向混合要存 unit"
    for t in trajs:
        assert not t.mk_legal.any(), "mk_legal 沒清掉，market 會被算進 logprob"
    assert steps == sum(len(t) for t in trajs)

    batch = RolloutBatch(trajs)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for mb in batch.minibatches(64, rng):
            lp, ent, _v = evaluate_actions(net, mb)
            assert (lp - mb["old_logp"]).abs().max().item() < 1e-3
            assert torch.isfinite(ent).all()


def test_base_side_rejects_a_typo():
    """打錯字要當場炸掉，不要安靜地退回預設值。"""
    from harness.ppo_rollout import build_net

    torch.manual_seed(0)
    with pytest.raises(ValueError, match="base_side"):
        from harness.ppo_rollout import VecRollout

        VecRollout(build_net(width=16, blocks=1), n_envs=1,
                   base_side="markets")


def test_reverse_hybrid_agent_takes_market_from_base(tmp_path):
    """agent 側的鏡像：market 逐項等於 base，工人動作不等於。"""
    from agents.ppo_agent import act
    from harness.ppo_rollout import build_net, load_base_policy

    spec = "config/params/cma1-g50-wt.json"
    torch.manual_seed(0)
    net = build_net(width=16, blocks=2)
    ckpt = tmp_path / "r.pt"
    torch.save({"state_dict": net.state_dict(), "width": 16, "blocks": 2}, ckpt)

    from kaggle_environments import make

    env = make("kaggriculture", debug=False)
    env.reset(2)
    obs = env.steps[0][0]["observation"]
    cfg = env.configuration

    base = load_base_policy(spec)(obs, cfg)
    got = act(obs, cfg, {"ckpt": str(ckpt), "base": spec,
                         "base_side": "market", "greedy": True})
    assert got["market"] == base["market"], "market 沒有沿用 base"
    assert set(got) == set(base)
    # 隨機權重的網路跟 gen0 選到完全相同的工人動作，機率極低。
    assert (got["farmer"], got["hands"]) != (base["farmer"], base["hands"])


def test_watch_reads_greedy_from_train_jsonl(tmp_path, capsys):
    """🩸 `greedy_cash` / `greedy_win` 本來寫在 `json.dumps(row)` **之後**，
    所以從來沒進過 train.jsonl，只留在 stdout —— 分析得用 regex 去剖 log，
    而 greedy 才是挑 checkpoint 的依據（§44）。
    """
    import json as _json

    from model.ppo_train import watch

    d = tmp_path / "run"
    d.mkdir()
    rows = [{"iter": 0, "cash_mean": 1000.0, "win_rate": 0.0, "entropy": 2.0,
             "explained_var": 0.5, "epochs_done": 8, "clipfrac": 0.1,
             "kl_last_epoch": 0.031},
            {"iter": 9, "cash_mean": 2000.0, "win_rate": 0.1, "entropy": 1.9,
             "explained_var": 0.6, "epochs_done": 5, "clipfrac": 0.2,
             "kl_last_epoch": 0.033, "greedy_cash": 55555.0, "greedy_win": 0.3}]
    (d / "train.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert watch([str(d)]) == 0
    out = capsys.readouterr().out
    assert "55,555" in out, "greedy 現金沒被讀出來"
    assert "0.033" in out, "逐 epoch KL 沒顯示 —— 看不出護欄逼不逼近門檻"
    assert "epochs 5" in out


def test_watch_survives_a_missing_dir(tmp_path, capsys):
    from model.ppo_train import watch

    assert watch([str(tmp_path / "nope")]) == 1
    assert "train.jsonl" in capsys.readouterr().out


# ------------------------------------------------------------- Φ 接線（§83.3）

@pytest.fixture(scope="module")
def assets_rollout():
    from harness.ppo_rollout import VecRollout, build_net

    torch.manual_seed(0)
    net = build_net(width=16, blocks=2)
    vec = VecRollout(net, n_envs=2, seed0=777, episode_steps=72, phi="assets")
    _steps, _cash, trajs = vec.run(collect=True)
    return trajs


def test_assets_phi_lands_in_the_trajectory_with_a_zero_terminal(assets_rollout):
    """🩸 期末的 Φ 一定要是 0：賣不掉的庫存一分不值，而且 Ng 的定理要它。"""
    for t in assets_rollout:
        assert t.phi is not None
        assert t.phi.shape == (len(t) + 1,)
        assert t.phi[-1] == 0.0
        assert t.phi[0] == 0.0            # 開局身上只有現金
        assert np.isfinite(t.phi).all()
        assert (t.phi >= 0).all()


def test_assets_reward_sums_to_the_final_margin(assets_rollout):
    """γ=1 時整局加總 = 期末 margin（Φ_0 = 0，terminal bonus 關掉）。"""
    from model.ppo import REWARD_SCALE, step_rewards

    for t in assets_rollout:
        r = step_rewards(t.cash[:, 0], t.cash[:, 1], zero_sum=True,
                         phi=t.phi, phi_weight=1 / REWARD_SCALE, gamma=1.0,
                         terminal_bonus=0.0)
        margin = (t.cash[-1, 0] - t.cash[-1, 1]) - (t.cash[0, 0] - t.cash[0, 1])
        assert r.sum() == pytest.approx(margin / REWARD_SCALE, abs=1e-3)


def test_phi_none_leaves_the_trajectory_without_a_potential(rollout):
    _net, _steps, _cash, trajs = rollout
    assert all(t.phi is None for t in trajs)


def test_plants_phi_is_the_difference_between_the_two_sides():
    """舊的 `--plant-weight` 是「我方 − 對手」，換欄位之後要一模一樣。"""
    from harness.ppo_rollout import VecRollout, build_net, count_plants

    torch.manual_seed(0)
    net = build_net(width=16, blocks=2)
    vec = VecRollout(net, n_envs=1, seed0=777, episode_steps=48, phi="plants")
    _steps, _cash, trajs = vec.run(collect=True)
    assert trajs and all(t.phi is not None for t in trajs)
    # 最後一格是「最後一次觀測到的值」，不是 0 —— plants 模式維持舊行為。
    for t in trajs:
        assert t.phi[-1] == t.phi[-2]
    obs = vec.envs[0].steps[0]
    farms = obs[0]["observation"]["farms"]
    want = count_plants(farms[0]) - count_plants(farms[1])
    assert trajs[0].phi[0] == want


# --------------------------------------------------- value 暖身（§88）

def test_value_warmup_moves_only_the_value_head(rollout):
    """🩸 暖身的重點是「policy 一個位元都不動」。軀幹是共用的，所以只能靠
    optimiser 只收 value_head 的參數 + `policy_coef=0` 把 policy loss 關掉。
    這兩件事漏掉任何一件，暖身本身就會把 policy 弄壞。"""
    from model.ppo import RolloutBatch, evaluate_actions, update

    net, _steps, _cash, trajs = rollout
    batch = RolloutBatch(trajs)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}

    idx = np.arange(min(64, batch.n_steps))
    with torch.no_grad():
        lp_before, _e, _v = evaluate_actions(net, batch._pack(idx, "cpu"))

    opt = torch.optim.Adam(net.value_head.parameters(), lr=1e-2)
    update(net, opt, batch, epochs=2, minibatch=batch.n_steps, seed=0,
           device="cpu", policy_coef=0.0, target_kl=0.0)

    after = net.state_dict()
    moved = [k for k in before if not torch.equal(before[k], after[k])]
    assert moved, "value head 完全沒動，暖身沒有作用"
    assert all(k.startswith("value_head.") for k in moved), \
        f"暖身動到了 value head 以外的權重：{[k for k in moved if not k.startswith('value_head.')]}"

    with torch.no_grad():
        lp_after, _e, _v = evaluate_actions(net, batch._pack(idx, "cpu"))
    assert torch.allclose(lp_before, lp_after), "policy 的機率被暖身改掉了"


def test_policy_coef_zero_drops_the_policy_term():
    """`policy_coef=0` 之後總損失只剩 value（entropy 也要一起關掉）。"""
    from model.ppo import ppo_loss

    kw = dict(new_logp=torch.zeros(4), old_logp=torch.zeros(4),
              adv=torch.tensor([1.0, -1.0, 2.0, -2.0]),
              new_value=torch.zeros(4), ret=torch.ones(4),
              entropy=torch.full((4,), 3.0), vf_coef=0.5, ent_coef=0.01)
    full, _p = ppo_loss(**kw)
    only_v, parts = ppo_loss(**kw, policy_coef=0.0)
    assert only_v.item() == pytest.approx(0.5 * 1.0)      # mse(0, 1) = 1
    assert full.item() != pytest.approx(only_v.item())
    assert parts["value"] == pytest.approx(1.0)           # 各項照樣如實回報
    assert parts["entropy"] == pytest.approx(3.0)
