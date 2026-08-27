"""import `kaggle_environments` 的時候把 open_spiel 的 336 行噪音吃掉。

`import kaggle_environments` 會噴兩段

    OpenSpiel exception: Unknown game 'universal_poker'. Available games are:
    2048
    add_noise
    ...

共 336 行到 **stderr**。那是 `kaggle_environments` 掃描 open_spiel 遊戲清單時
的正常輸出，不是錯誤，但會把工具自己的輸出淹掉。

🩸 **`contextlib.redirect_stderr` 攔不到。** open_spiel 是 C++ extension，
直接寫檔案描述子 2，`redirect_stderr` 只換 Python 層的 `sys.stderr`。要在
fd 層級擋。

用法 —— 把**第一次**碰到 `kaggle_environments` 的 import 包起來就好，
之後都是 module cache 命中，不會再噴：

    from tools._quiet import silenced

    with silenced():
        from kaggle_environments import make

`eval/runner.py` 自己已經處理過了，所以 import 它是安靜的；會噴的是那些
直接 import 引擎的工具（`tools/param_space.py:47` 是最常被牽連的一個，
`param_search` / `noise_probe` / `freeze_params` 都靠它）。
"""
from __future__ import annotations

import contextlib
import os
import sys


@contextlib.contextmanager
def silenced():
    """fd 層級靜音 stdout 與 stderr。"""
    with open(os.devnull, "w") as devnull:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
