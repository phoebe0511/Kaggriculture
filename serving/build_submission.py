"""把 submission 需要的檔案**攤平**到 `submission/`。

    python -m serving.build_submission          # 複製 + 改寫 import
    python -m serving.build_submission --tar    # 另外產 submission.tar.gz

預設是複製不壓縮 —— 要看內容、要 diff、要直接編輯都比較方便。

## 為什麼要攤平，不能照 repo 的目錄結構搬

**`agents` 這個 top-level 名字在 Kaggle 的執行環境已經被佔走了。**

`kaggle_environments` 載入 agent 的方式是（`agent.py:51-59`）：

    exec_dir = os.path.dirname(path)
    sys.path.append(exec_dir)        # ← 接在最後面
    exec(compile(raw, path, "exec"), {})

而 `site-packages/kaggle_environments/envs/lux_ai_s3/` 已經在 `sys.path`
更前面的位置（實測是 index 9，我們的目錄是 index 10），那底下有一支
`agents.py`。跨 sys.path entry 是**前面的贏**，所以 `import agents` 會抓到
lux_ai_s3 的那支，然後炸在它自己的相對 import：

    InvalidArgument: Invalid raw Python:
      ImportError('attempted relative import with no known parent package')

⚠️ **`agents/__init__.py` 救不了。** 「正規 package 優先於同名 .py」只在
**同一個 entry 內**成立，跨 entry 不適用。

而且 exec 出來的環境裡 **`__file__` 不存在**（`__name__` 是 `'builtins'`、
`__package__` 是 `None`），所以也沒辦法在 `main.py` 裡靠 `__file__`
把自己的目錄插到 `sys.path` 最前面。

攤平之後就沒有 package，只有四支平面模組，繞開整個問題。
四個名字都確認過在載入環境裡是空的：`main` / `gen0` / `gen1` /
`action_validation`。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = REPO_ROOT / "submission"
OUTPUT = SUBMISSION_DIR / "submission.tar.gz"

# repo 路徑 -> submission 裡的檔名（攤平）
#
# `serving/action_validation.py` **故意不在裡面** —— 理由寫在 `main.py` 底部
# （成本、比賽當下抓到 bug 也沒用、綁死引擎私有 API 導致換版本每回合 TypeError）。
# 它留在開發側，`tests/test_l0_smoke.py` 每回合都會呼叫。
#
# `model/train.py` / `model/net.py` 也不在裡面 —— 那兩支 import torch，
# 而 submission **不准 import torch**（`docs/CLAUDE.md`）。比賽端走
# `npz_forward.py` 的純 numpy 前向。
FILE_MAP = {
    "main.py": "main.py",
    "contracts.py": "contracts.py",
    "agents/gen0.py": "gen0.py",
    "agents/gen2_model.py": "gen2_model.py",
    "agents/gen4_demand.py": "gen4_demand.py",
    "serving/npz_forward.py": "npz_forward.py",
}

#: 權重。二進位，不做 import 改寫，檔名固定成 `weights.npz` ——
#: `agents/gen2_model.py` 沒有 `KAGGRI_WEIGHTS` 時就找自己旁邊這一個。
WEIGHTS_NAME = "weights.npz"
DEFAULT_WEIGHTS = "model/weights-v5-round1.npz"

FILES = tuple(FILE_MAP.values()) + (WEIGHTS_NAME,)

#: 攤平之後這些名字會變成 top-level 模組，撞到 Kaggle 載入環境裡既有的同名
#: 模組就會抓錯（docstring 上面那段 lux_ai_s3 `agents.py` 的事）。
#: 2026-08-20 在 repo 外的目錄用 `importlib.util.find_spec` 逐個確認過都是 None。
#: ⚠️ 那是**本機**的證據，不是 Kaggle runtime 的。新增名字時要重新確認。
VERIFIED_FREE_NAMES = (
    "main", "contracts", "gen0", "gen2_model", "gen4_demand",
    "npz_forward", "action_validation",
)

# 攤平之後 package 前綴要拿掉。用 regex 而不是純字串取代，是為了不去動到
# 註解和文件裡提到的 `agents/gen0.py` 這種路徑寫法。
_REWRITES = (
    (re.compile(r"^(\s*)from agents\.(\w+) import ", re.M), r"\1from \2 import "),
    (re.compile(r"^(\s*)from serving\.(\w+) import ", re.M), r"\1from \2 import "),
    (re.compile(r"^(\s*)import agents\.(\w+)\b", re.M), r"\1import \2"),
    (re.compile(r"^(\s*)import serving\.(\w+)\b", re.M), r"\1import \2"),
)
SIZE_LIMIT = 100 * 2**20


def _flatten(source):
    """讀一支 repo 檔案，回傳攤平後的原始碼與改寫過的行數。"""
    text = source.read_text(encoding="utf-8")
    total = 0
    for pattern, replacement in _REWRITES:
        text, n = pattern.subn(replacement, text)
        total += n
    return text, total


def build(output=OUTPUT, dest=SUBMISSION_DIR):
    """把 `dest` 底下攤平好的檔案打成 tar.gz。**先跑過 `copy_files()`。**"""
    output = Path(output)
    dest = Path(dest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for flat in FILES:
            source = dest / flat
            if not source.is_file():
                raise FileNotFoundError(f"{source}（先跑 copy_files()）")
            archive.add(source, arcname=flat, recursive=False)

    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
    if names != list(FILES):
        raise AssertionError(f"submission 內容不符: {names!r}")
    if output.stat().st_size > SIZE_LIMIT:
        raise AssertionError(f"submission 超過 100 MiB: {output.stat().st_size}")

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    try:
        display = output.relative_to(REPO_ROOT)
    except ValueError:
        display = output
    print(display)
    print(f"files={len(names)} size={output.stat().st_size} sha256={digest}")
    return output


def copy_files(dest=SUBMISSION_DIR, weights=DEFAULT_WEIGHTS):
    """把 `FILE_MAP` 的檔案攤平複製到 `dest`，並改寫 package import。

    整個 `dest` 先砍掉重建 —— 不清的話，上一版的 `agents/` 子目錄會留在
    那裡，而 Kaggle 那邊 import 得到它，等於帶著一份過期的 agent 上場。

    ⚠️ **`dest` 會被 `rmtree`。** 跑之前先看裡面有沒有別人放的備份。
    """
    dest = Path(dest)
    for relative in FILE_MAP:
        if not (REPO_ROOT / relative).is_file():
            raise FileNotFoundError(REPO_ROOT / relative)
    weights_src = REPO_ROOT / weights
    if not weights_src.is_file():
        raise FileNotFoundError(f"{weights_src}（先跑 serving.export_npz）")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    copied = []
    for relative, flat in FILE_MAP.items():
        text, rewrites = _flatten(REPO_ROOT / relative)
        target = dest / flat
        target.write_text(text, encoding="utf-8", newline="\n")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()[:12]
        copied.append((relative, flat, target.stat().st_size, rewrites, digest))

    # 權重是二進位，原樣複製 —— 不能走 `_flatten`（它會用 utf-8 讀）。
    shutil.copyfile(weights_src, dest / WEIGHTS_NAME)
    copied.append((weights, WEIGHTS_NAME, (dest / WEIGHTS_NAME).stat().st_size, 0,
                   hashlib.sha256((dest / WEIGHTS_NAME).read_bytes()).hexdigest()[:12]))

    total = sum(size for _, _, size, _, _ in copied)
    try:
        display = dest.relative_to(REPO_ROOT)
    except ValueError:
        display = dest
    print(display)
    for relative, flat, size, rewrites, digest in copied:
        note = f"  改寫 {rewrites} 處 import" if rewrites else ""
        print(f"  {relative:28s} -> {flat:22s} {size:>8,d} B  "
              f"sha256:{digest}{note}")
    print(f"files={len(copied)} size={total:,d} B")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", action="store_true",
                    help="另外產出 submission.tar.gz")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="要打包的 .npz。ENCODER_VERSION 不符的話 agent 第一回合"
                         "就會 SystemExit —— 所以換 schema 一定要一起換這個")
    args = ap.parse_args(argv)
    copy_files(weights=args.weights)
    if args.tar:
        build()


if __name__ == "__main__":
    main()
