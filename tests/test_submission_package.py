from __future__ import annotations

import tarfile

import pytest

from serving.build_submission import (
    FILE_MAP,
    FILES,
    SIZE_LIMIT,
    VERIFIED_FREE_NAMES,
    WEIGHTS_NAME,
    build,
    copy_files,
)


def test_submission_archive_is_small_and_self_contained(tmp_path):
    """submission **不准 import torch**（載入好幾秒，第一回合就可能超時）。

    ⚠️ 查的是 AST 不是字串。`npz_forward.py` 的 docstring 第一行就寫著
    「不准 import torch」—— 純字串比對會把那句話本身當成違規。
    """
    import ast

    dest = copy_files(tmp_path / "submission")
    output = build(tmp_path / "submission.tar.gz", dest=dest)
    assert output.stat().st_size < SIZE_LIMIT

    with tarfile.open(output, "r:gz") as archive:
        assert archive.getnames() == list(FILES)
        for member in archive.getmembers():
            assert not member.name.startswith("/")
            if not member.isfile() or not member.name.endswith(".py"):
                continue                    # 二進位權重解不成 utf-8
            source = archive.extractfile(member).read().decode("utf-8")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    roots = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    roots = [(node.module or "").split(".")[0]]
                else:
                    continue
                assert "torch" not in roots, f"{member.name}:{node.lineno}"


def test_flattened_names_do_not_shadow_anything_importable():
    """攤平之後每個檔名都變成 top-level 模組，撞名就會抓到別人那一支。

    ⚠️ 這條**只驗得到本機**。Kaggle 的載入環境是另一組 site-packages，
    `serving/build_submission.py` 的 docstring 記過 lux_ai_s3 的 `agents.py`
    就是這樣把 `import agents` 搶走的，而且錯誤訊息跟撞名完全無關。

    所以新增檔案時，除了這條測試還要人工去 Kaggle notebook 確認一次。
    """
    import importlib.util
    import sys

    for flat in FILES:
        if not flat.endswith(".py"):
            continue                      # 權重不是模組，撞不到名字
        name = flat.rsplit(".", 1)[0]
        assert name in VERIFIED_FREE_NAMES, \
            f"{flat} 是新增的，先確認 top-level 名字 {name!r} 在 Kaggle 沒被佔走"
        found = sys.modules.get(name) or importlib.util.find_spec(name)
        if found is None:
            continue
        origin = getattr(found, "origin", getattr(found, "__file__", ""))
        assert "site-packages" not in str(origin), \
            f"{name!r} 在 site-packages 有同名模組：{origin}"


def test_weights_are_packed_and_match_the_current_encoder_version(tmp_path):
    """權重的 schema 版本跟 contracts.py 對不上 = 上場第一回合 SystemExit。"""
    np = pytest.importorskip("numpy")
    import contracts as C

    dest = copy_files(tmp_path / "submission")
    with np.load(dest / WEIGHTS_NAME) as data:
        assert int(data["encoder_version"][0]) == C.ENCODER_VERSION
        assert "demand_out.weight" in data.files, "沒有 v5 的 demand head"


def test_no_lazy_imports_of_packaged_modules(tmp_path):
    """🩸 submission 裡的模組只在 `exec` 那一瞬間 import 得到。

    `kaggle_environments/agent.py:48-58` 是：

        sys.path.append(exec_dir)
        exec(code_object, env)
        sys.path.pop()            # ← 立刻拿掉

    所以任何**延遲到第一回合才做**的 import 都會 `ModuleNotFoundError`，
    而且本機（repo root 在 sys.path 上）完全測不出來。2026-08-20 踩過兩次：
    `gen2_model._policy()` 裡的 `npz_forward`、`gen0.demand_tile_tasks()` 裡的
    `contracts`。上場的後果是每一回合都拋錯、整局 PASS 拿 0 分。

    這條測試掃打包後的原始碼，函式內 import 只放行標準函式庫。
    """
    import ast
    import sys

    dest = copy_files(tmp_path / "submission")
    packaged = {flat.rsplit(".", 1)[0] for flat in FILES}

    offenders = []
    for flat in FILES:
        if not flat.endswith(".py"):
            continue
        tree = ast.parse((dest / flat).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if getattr(node, "col_offset", 0) == 0:
                continue                       # 模組層，安全
            names = ([node.module] if isinstance(node, ast.ImportFrom)
                     else [a.name for a in node.names])
            for name in names:
                root = (name or "").split(".")[0]
                if root in packaged or (
                        root and root not in sys.stdlib_module_names):
                    offenders.append(f"{flat}:{node.lineno} import {root}")

    assert not offenders, (
        "這些 import 在函式裡，上場時 sys.path 已經被 pop 掉了：\n  "
        + "\n  ".join(offenders))


def test_main_entry_matches_what_is_packaged():
    """`main.py` import 的 agent 一定要在 FILE_MAP 裡，否則 submission 少檔案。"""
    text = (FILE_MAP and __import__("pathlib").Path("main.py").read_text(
        encoding="utf-8"))
    packaged = {flat.rsplit(".", 1)[0] for flat in FILES}
    for line in text.splitlines():
        if line.startswith("from agents.") or line.startswith("from serving."):
            module = line.split()[1].split(".")[1]
            assert module in packaged, f"main.py 用了 {module}，但沒有打包進去"
