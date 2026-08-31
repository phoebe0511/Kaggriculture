"""分段 reward / 分段 lam / 格子 shaping 的測試（§55.5）。

三件事互相獨立，各自要能單獨開關，而且**都不能改變一整局的總報酬**
（總和變了就是在改目標，不是在改 credit assignment）。
"""
import numpy as np
import pytest

from model.ppo import (REWARD_SCALE, TERMINAL_BONUS, compute_gae, phased_lam,
                       step_rewards)


def _cash(n=30):
    """一步花光、接著趴平、之後賺回來 —— ladder 頂端的形狀（§55.1）。

    回傳兩條 `[n+1]` 的現金，所以 reward 有 n 個。第 0 步 3000 -> 22 是實測
    的形狀（kawashigi 第一個回合就花掉 99.3%），不是攤成好幾步。
    """
    a = np.empty(n + 1)
    a[0] = 3000.0
    a[1:11] = np.linspace(22, 1700, 10)
    a[11:] = np.linspace(1700, 90000, n - 10)
    b = np.empty(n + 1)
    b[0] = 3000.0
    b[1:11] = np.linspace(60, 1500, 10)
    b[11:] = np.linspace(1500, 80000, n - 10)
    return a, b


# ---------------------------------------------------------------- settle_step

def test_settle_step_keeps_the_total_unchanged():
    a, b = _cash()
    plain = step_rewards(a, b, zero_sum=True)
    settled = step_rewards(a, b, zero_sum=True, settle_step=10)
    assert settled.sum() == pytest.approx(plain.sum(), abs=1e-4)


def test_settle_step_zeroes_the_early_steps_and_pays_at_the_boundary():
    a, b = _cash()
    plain = step_rewards(a, b, zero_sum=True)
    settled = step_rewards(a, b, zero_sum=True, settle_step=10)
    assert np.all(settled[:9] == 0.0)
    assert settled[9] == pytest.approx(plain[:10].sum(), abs=1e-4)
    # 邊界之後逐字不動
    assert settled[10:] == pytest.approx(plain[10:], abs=1e-6)


def test_settle_step_removes_the_punishment_for_spending():
    """這是整件事的重點：前 10 步是「正確地把錢花光」，不該被扣分。"""
    a, b = _cash()
    own = step_rewards(a, b)                       # own-cash，沒分段
    assert own[0] < -0.2                           # 花錢當場被扣 0.2 以上
    settled = step_rewards(a, b, settle_step=10)
    assert np.all(settled[:9] == 0.0)


def test_settle_step_zero_is_a_no_op():
    a, b = _cash()
    assert step_rewards(a, b, settle_step=0) == pytest.approx(
        step_rewards(a, b), abs=1e-7)


def test_settle_step_longer_than_the_episode_is_clamped():
    a, b = _cash(12)
    r = step_rewards(a, b, zero_sum=True, settle_step=9999)
    assert len(r) == 12                      # T+1 = 13 個現金點 -> 12 個 reward
    assert np.all(r[:-1] == 0.0)             # 全部被搬到最後一步


# ------------------------------------------------------------------ phased_lam

def test_phased_lam_shape_and_values():
    lam = phased_lam(10, 4, 0.998, 0.95)
    assert lam.shape == (10,)
    assert np.all(lam[:4] == 0.998)
    assert np.all(lam[4:] == 0.95)


def test_phased_lam_with_no_split_is_all_late():
    assert np.all(phased_lam(6, 0, 0.998, 0.95) == 0.95)


def test_phased_lam_split_beyond_T_is_clamped():
    assert np.all(phased_lam(5, 99, 0.998, 0.95) == 0.998)


def test_compute_gae_accepts_an_array_lam():
    rew = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    v = np.zeros(5, dtype=np.float32)
    d = np.array([0, 0, 0, 1], dtype=np.float32)
    scalar, _ = compute_gae(rew, v, d, 0.997, 0.95)
    array, _ = compute_gae(rew, v, d, 0.997, np.full(4, 0.95))
    assert array == pytest.approx(scalar, abs=1e-7)


def test_a_bigger_early_lam_carries_the_late_reward_further_back():
    """唯一的 reward 在最後一步，前面全 0 —— 大 lam 才傳得回第 0 步。"""
    T = 200
    rew = np.zeros(T, dtype=np.float32)
    rew[-1] = 1.0
    v = np.zeros(T + 1, dtype=np.float32)
    d = np.zeros(T, dtype=np.float32)
    d[-1] = 1.0
    small, _ = compute_gae(rew, v, d, 0.997, 0.95)
    big, _ = compute_gae(rew, v, d, 0.997, 0.998)
    assert abs(small[0]) < 1e-4          # 19 步的視窗傳不到 200 步外
    assert big[0] > 0.3                  # 200 步的視窗傳得到
    assert big[0] > small[0] * 100


# --------------------------------------------------------------- plant shaping

def test_plant_shaping_is_potential_based_so_the_total_barely_moves():
    """整條加起來只剩 gamma^T*Φ(s_T) - Φ(s_0)，不是在加一筆新的目標。"""
    a, b = _cash()
    pa = np.linspace(0, 40, len(a))
    pb = np.linspace(0, 45, len(a))
    plain = step_rewards(a, b, zero_sum=True)
    shaped = step_rewards(a, b, zero_sum=True, plants_a=pa, plants_b=pb,
                          plant_weight=0.01, gamma=1.0)
    phi0 = 0.01 * (pa[0] - pb[0])
    phiT = 0.01 * (pa[-1] - pb[-1])
    assert shaped.sum() == pytest.approx(plain.sum() + phiT - phi0, abs=1e-4)


def test_planting_a_tile_pays_immediately():
    """種下去當場就有分，不用等 240 步後收穫 —— 這是加它的理由。"""
    a = np.array([3000.0, 3000.0, 3000.0])
    b = np.array([3000.0, 3000.0, 3000.0])
    flat = step_rewards(a, b, zero_sum=True, plants_a=np.array([0.0, 0.0, 0.0]),
                        plant_weight=0.1)
    grew = step_rewards(a, b, zero_sum=True, plants_a=np.array([0.0, 5.0, 5.0]),
                        plant_weight=0.1)
    assert grew[0] > flat[0]


def test_plant_weight_zero_is_a_no_op():
    a, b = _cash()
    pa = np.linspace(0, 40, len(a))
    assert step_rewards(a, b, plants_a=pa, plant_weight=0.0) == pytest.approx(
        step_rewards(a, b), abs=1e-7)


def test_plants_none_is_a_no_op_even_with_a_weight():
    a, b = _cash()
    assert step_rewards(a, b, plant_weight=0.5) == pytest.approx(
        step_rewards(a, b), abs=1e-7)


def test_plant_shaping_alone_without_an_opponent_count():
    a = np.array([3000.0, 3000.0])
    b = np.array([3000.0, 3000.0])
    r = step_rewards(a, b, zero_sum=True, plants_a=np.array([0.0, 10.0]),
                     plant_weight=0.1, gamma=1.0)
    # 期末 bonus 是 sign(0) = 0，所以只剩 shaping：1.0*1.0 - 0.0
    assert r[0] == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------- 互不干擾

def test_the_three_knobs_compose():
    a, b = _cash()
    pa, pb = np.linspace(0, 40, len(a)), np.linspace(0, 45, len(a))
    r = step_rewards(a, b, zero_sum=True, settle_step=10,
                     plants_a=pa, plants_b=pb, plant_weight=0.01)
    assert np.all(r[:9] == 0.0)
    assert np.isfinite(r).all()


def test_terminal_bonus_survives_every_combination():
    a, b = _cash()
    for kw in ({}, {"settle_step": 10},
               {"plants_a": np.linspace(0, 40, len(a)), "plant_weight": 0.01}):
        r = step_rewards(a, b, zero_sum=True, **kw)
        assert r[-1] > TERMINAL_BONUS / 2, kw     # 我方期末較高 -> +1.0


def test_scale_is_unchanged():
    assert REWARD_SCALE == 10_000.0
