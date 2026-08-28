"""出貨端（numpy）跟開發端（torch）必須挑出完全一樣的動作。

兩邊分歧**不會報錯** —— 只會表現成「本機評估好、上場很爛」。
`tests/test_npz_forward.py` 守的是前向的數值，這一支守的是**解碼**：
遮罩語意、target -> step_toward、market 門檻。

用隨機權重，不需要訓練好的 checkpoint（`*.pt` / `*.npz` 都被 gitignore）。
"""

from __future__ import annotations

import numpy as np
import pytest

import contracts as C

torch = pytest.importorskip("torch", reason="torch 是開發側依賴，submission 不用")


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    """同一份隨機權重，一份 .pt 一份 .npz。"""
    from harness.ppo_rollout import build_net
    from serving.export_npz import export

    tmp = tmp_path_factory.mktemp("gen3")
    torch.manual_seed(3)
    net = build_net(width=32, blocks=2)
    pt = tmp / "ppo.pt"
    torch.save({
        "encoder_version": C.ENCODER_VERSION,
        "state_dict": net.state_dict(),
        "width": 32, "blocks": 2, "labels": "target", "trainer": "ppo",
    }, pt)
    npz = tmp / "ppo.npz"
    export(str(pt), str(npz))
    return str(pt), str(npz)


def _observations(n=12):
    """拿 gen0 對打走出來的盤面 —— 隨機盤面碰不到雇工之後的多 unit 情況。"""
    import os

    os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")
    from tools._quiet import silenced

    with silenced():
        from kaggle_environments import make

        import agents.gen0 as g
        env = make("kaggriculture", configuration={"seed": 31},
                   debug=False)
        env.reset(2)
        out = []
        for step in range(120):
            acts = []
            for p in range(2):
                o = dict(env.steps[-1][p]["observation"])
                o["player"] = p
                acts.append(g.act(o, env.configuration))
            env.step(acts)
            if step % (120 // n) == 0:
                o = dict(env.steps[-1][0]["observation"])
                o["player"] = 0
                out.append((o, env.configuration))
    return out


def test_numpy_and_torch_pick_the_same_action(paired):
    import agents.gen3_target as ship
    import agents.ppo_agent as dev

    pt, npz = paired
    boards = _observations()
    assert len(boards) >= 8
    seen_units = 0
    for i, (obs, cfg) in enumerate(boards):
        a = dev.act(obs, cfg, {"ckpt": pt, "greedy": True})
        b = ship.act(obs, cfg, {"weights": npz})
        assert a["farmer"] == b["farmer"], f"第 {i} 個盤面 farmer 不同"
        assert a["hands"] == b["hands"], f"第 {i} 個盤面 hands 不同"
        assert a["market"] == b["market"], f"第 {i} 個盤面 market 不同"
        seen_units += 1 + len(a["hands"])
    # 只有 farmer 的話等於沒驗到多 unit 的路徑。
    assert seen_units > len(boards), "全部盤面都只有 farmer，沒驗到 hands"


def test_wrong_labels_is_refused(paired, tmp_path):
    """載到 immediate 的權重要直接停，不能安靜地整局 PASS。"""
    import agents.gen3_target as ship

    _pt, npz = paired
    data = dict(np.load(npz, allow_pickle=True))
    data["labels"] = np.asarray(["immediate"])
    bad = tmp_path / "immediate.npz"
    np.savez_compressed(bad, **data)
    ship._POLICY = None
    with pytest.raises(SystemExit, match="op head"):
        ship._policy(str(bad))
    ship._POLICY = None


def test_masked_argmax_matches_the_training_side():
    from agents.gen3_target import masked_argmax
    from model.ppo import masked_log_softmax

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(7, 44)).astype(np.float32)
    mask = rng.random((7, 44)) < 0.3
    mask[3] = False              # 整列全 False 的那一種 unit
    mask[5] = False
    mask[5, 11] = True
    want = masked_log_softmax(torch.as_tensor(logits),
                              torch.as_tensor(mask)).argmax(-1).numpy()
    assert (masked_argmax(logits, mask) == want).all()
