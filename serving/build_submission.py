"""建立 Kaggle 接受的 dist/submission.tar.gz。"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "dist" / "submission.tar.gz"
FILES = (
    "main.py",
    "agents/__init__.py",
    "agents/gen0.py",
    "agents/gen1.py",
    "serving/__init__.py",
    "serving/action_validation.py",
)
SIZE_LIMIT = 100 * 2**20


def build(output=OUTPUT):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative in FILES:
            source = REPO_ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.add(source, arcname=relative, recursive=False)

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


if __name__ == "__main__":
    build()
