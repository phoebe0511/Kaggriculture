"""`harness/rollout.py` 的檔名標籤。

🩸 2026-08-29：`--policy` 吃得下 `config/params/cma1-g50-wt.json` 這種**路徑**，
但檔名是 `f"{policy_name}-{opponent}-{seed:06d}.npz"` 直接拼的，斜線會被當成
目錄，`np.savez_compressed` 丟 `FileNotFoundError`。而且是**跑完整局之後才丟**
—— 那一局的計算全部白費，1,400 局的批次會全滅。
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("name,want", [
    ("gen1", "gen1"),
    ("e2e", "e2e"),
    ("config/params/cma1-g50-wt.json", "cma1-g50-wt"),
    (r"config\params\cma1-g50-wt.json", "cma1-g50-wt"),
    ("data/x/y.json", "y"),
])
def test_tag_strips_paths_but_leaves_plain_names(name, want):
    from harness.rollout import _tag

    assert _tag(name) == want


def test_tag_output_is_usable_as_a_filename():
    """真正要守的性質：結果不能再含路徑分隔符。"""
    from harness.rollout import _tag

    for name in ("gen1", "config/params/cma1-g50-wt.json",
                 r"config\params\ref-v11.json"):
        got = _tag(name)
        assert "/" not in got and "\\" not in got, got
        assert got, "標籤不能是空的"
