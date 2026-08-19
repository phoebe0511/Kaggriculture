"""比賽端的 numpy 前向要跟訓練端的 torch 算出一樣的數字。

兩邊對不上的話，訓練時看到的準確率跟比賽時的行為就是兩回事，而且**不會報錯**
—— 只會表現成「本機驗證 90%，上場卻很爛」。所以用隨機權重直接比對數值。

用隨機初始化而不是訓練好的 checkpoint，是為了讓這個測試在還沒訓練、或
checkpoint 不在工作目錄時也能跑（`*.pt` / `*.npz` 都被 gitignore）。
"""

from __future__ import annotations

import numpy as np
import pytest

import contracts as C

torch = pytest.importorskip("torch", reason="torch 是開發側依賴，submission 不用")


def _build(tmp_path, width=32, blocks=3):
    from model.net import KaggricultureNet
    from serving.export_npz import export
    from serving.npz_forward import NumpyPolicy

    torch.manual_seed(0)
    model = KaggricultureNet(
        C.N_SPATIAL, C.N_SCALAR, C.N_UNIT_FEATURES, C.N_UNIT_OPS, C.N_QTY,
        width=width, n_blocks=blocks)
    model.eval()

    ckpt = tmp_path / "ckpt.pt"
    torch.save({
        "encoder_version": C.ENCODER_VERSION,
        "state_dict": model.state_dict(),
        "width": width,
        "blocks": blocks,
    }, ckpt)
    npz = export(ckpt, tmp_path / "weights.npz")
    return model, NumpyPolicy(npz)


def test_numpy_forward_matches_torch(tmp_path):
    model, policy = _build(tmp_path)

    rng = np.random.default_rng(0)
    spatial = rng.standard_normal((C.N_SPATIAL, 10, 10)).astype(np.float32)
    scalar = rng.standard_normal(C.N_SCALAR).astype(np.float32)
    n_units = 7
    positions = rng.integers(0, 10, size=(n_units, 2)).astype(np.int16)
    unit_feats = rng.standard_normal((n_units, C.N_UNIT_FEATURES)).astype(np.float32)

    with torch.no_grad():
        t_op, t_qty, t_value = model(
            torch.from_numpy(spatial)[None],
            torch.from_numpy(scalar)[None],
            torch.zeros(n_units, dtype=torch.long),
            torch.from_numpy(positions.astype(np.int64)),
            torch.from_numpy(unit_feats),
        )

    n_op, n_qty, n_value = policy(spatial, scalar, positions, unit_feats)

    assert np.allclose(t_op.numpy(), n_op, atol=1e-4), \
        f"op 最大誤差 {np.abs(t_op.numpy() - n_op).max()}"
    assert np.allclose(t_qty.numpy(), n_qty, atol=1e-4), \
        f"qty 最大誤差 {np.abs(t_qty.numpy() - n_qty).max()}"
    assert abs(float(t_value[0]) - n_value) < 1e-4


def test_argmax_agrees(tmp_path):
    """數值有小誤差沒關係，**選出來的動作**必須一樣。"""
    model, policy = _build(tmp_path)
    rng = np.random.default_rng(1)
    for _ in range(5):
        spatial = rng.standard_normal((C.N_SPATIAL, 10, 10)).astype(np.float32)
        scalar = rng.standard_normal(C.N_SCALAR).astype(np.float32)
        positions = rng.integers(0, 10, size=(13, 2)).astype(np.int16)
        unit_feats = rng.standard_normal((13, C.N_UNIT_FEATURES)).astype(np.float32)

        with torch.no_grad():
            t_op, _, _ = model(
                torch.from_numpy(spatial)[None], torch.from_numpy(scalar)[None],
                torch.zeros(13, dtype=torch.long),
                torch.from_numpy(positions.astype(np.int64)),
                torch.from_numpy(unit_feats))
        n_op, _, _ = policy(spatial, scalar, positions, unit_feats)
        assert (t_op.numpy().argmax(axis=1) == n_op.argmax(axis=1)).all()


def test_exported_weights_reject_version_mismatch(tmp_path):
    """schema 變了就不准載舊權重。"""
    from serving.export_npz import export

    torch.save({
        "encoder_version": C.ENCODER_VERSION + 99,
        "state_dict": {},
        "width": 32,
        "blocks": 3,
    }, tmp_path / "old.pt")
    with pytest.raises(SystemExit):
        export(tmp_path / "old.pt", tmp_path / "old.npz")
