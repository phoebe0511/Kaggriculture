from __future__ import annotations

import tarfile

from serving.build_submission import FILES, SIZE_LIMIT, build, copy_files


def test_submission_archive_is_small_and_self_contained(tmp_path):
    dest = copy_files(tmp_path / "submission")
    output = build(tmp_path / "submission.tar.gz", dest=dest)
    assert output.stat().st_size < SIZE_LIMIT

    with tarfile.open(output, "r:gz") as archive:
        assert archive.getnames() == list(FILES)
        for member in archive.getmembers():
            assert not member.name.startswith("/")
            if member.isfile():
                source = archive.extractfile(member).read().decode("utf-8")
                assert "import torch" not in source
