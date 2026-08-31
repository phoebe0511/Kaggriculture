"""`--target-kl` 的提早停止要看**當前 epoch** 的 KL，不是所有 epoch 的累積平均。

🩸 2026-08-31：原本是 `acc["approx_kl"] / n`，也就是所有 epoch 累積起來的
平均。第 0 個 epoch 的 KL 幾乎是 0（實測 0.00136），把平均一路拉低 —— 8 個
epoch 跑完累積平均只到 0.0272，永遠碰不到 `1.5 * 0.02 = 0.03`，而第 7 個
epoch 的實際 KL 已經是 0.0487。

後果是 `--target-kl` 在 `ppo-hybrid` / `ppo-cma1-market` 兩次跑裡一次都沒
觸發（`epochs_done` 300/300 都是 8），信賴區域形同不存在。這不會報錯，只會
表現成「訓練越久越差」。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 是開發側依賴")


def _epoch_kls():
    """實測的逐 epoch KL（kl_probe.py，956 步、lr 3e-5、cma1-round0-tgt）。"""
    return [0.00136, 0.01031, 0.01835, 0.02432,
            0.03177, 0.03842, 0.04434, 0.04865]


def test_cumulative_mean_would_never_fire():
    """守住這個 bug 的形狀本身：累積平均在這組實測值下永遠不觸發。"""
    kl = _epoch_kls()
    thr = 1.5 * 0.02
    run, fired = 0.0, False
    for i, k in enumerate(kl):
        run = (run * i + k) / (i + 1)
        if run > thr:
            fired = True
    assert not fired, "累積平均竟然觸發了 —— 這組對照值失效，要重新量"
    assert max(kl) > thr, "實測值裡沒有超標的 epoch，對照值選錯了"


def test_per_epoch_fires_at_the_right_epoch():
    kl = _epoch_kls()
    thr = 1.5 * 0.02
    first = next(i for i, k in enumerate(kl) if k > thr)
    assert first == 4, first


def _epochs_done_with_profile(profile, target_kl, monkeypatch):
    """用 monkeypatch 餵一組固定的逐 epoch KL，回傳實際跑了幾個 epoch。

    不靠 rollout 的隨機性 —— 這個 bug 的關鍵是 KL 的**形狀**（從接近 0 爬上去），
    真的去跑一段訓練很難穩定重現那個形狀。
    """
    from model import ppo

    state = {"calls": 0, "per_epoch": 1}
    real_loss = ppo.ppo_loss

    def fake_loss(new_logp, old_logp, adv, value, ret, ent, **kw):
        total, parts = real_loss(new_logp, old_logp, adv, value, ret, ent, **kw)
        i = state["calls"] // state["per_epoch"]
        parts["approx_kl"] = profile[min(i, len(profile) - 1)]
        state["calls"] += 1
        return total, parts

    monkeypatch.setattr(ppo, "ppo_loss", fake_loss)

    from harness.ppo_rollout import (VecRollout, build_net, load_base_policy,
                                     load_opponent)

    spec = "config/params/cma1-g50-wt.json"
    torch.manual_seed(0)
    net = build_net(width=16, blocks=2)
    vec = VecRollout(net, n_envs=2, seed0=31_337, episode_steps=72,
                     opponent=load_opponent(spec),
                     base_policy=load_base_policy(spec))
    with torch.no_grad():
        _s, _c, trajs = vec.run(collect=True)
    b = ppo.RolloutBatch(trajs)
    # 每個 epoch 有幾個 minibatch —— 用量的，不要寫死。
    state["per_epoch"] = len(list(b.minibatches(10_000, np.random.default_rng(0))))
    assert state["per_epoch"] >= 1
    state["calls"] = 0
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)
    out = ppo.update(net, opt, b, epochs=8, minibatch=10_000, seed=0,
                     ent_coef=0.0, target_kl=target_kl)
    return int(out["epochs_done"])


def test_early_stop_uses_the_current_epoch_not_the_running_mean(monkeypatch):
    """🩸 這是本檔的重點。用實測的 KL 形狀：

    逐 epoch 判斷 -> 第 4 個 epoch（0.03177 > 0.03）就停，跑 5 個。
    累積平均      -> 8 個跑滿都碰不到 0.03（最高 0.0272）。

    舊寫法會回 8，修好的回 5。
    """
    profile = _epoch_kls()
    got = _epochs_done_with_profile(profile, target_kl=0.02,
                                    monkeypatch=monkeypatch)
    assert got == 5, (
        f"跑了 {got} 個 epoch，應該是 5（第 4 個 epoch KL {profile[4]} 超過 "
        f"0.03 就停）。回 8 代表又變回看累積平均了")


def test_early_stop_off_runs_every_epoch(monkeypatch):
    got = _epochs_done_with_profile(_epoch_kls(), target_kl=0.0,
                                    monkeypatch=monkeypatch)
    assert got == 8, got
