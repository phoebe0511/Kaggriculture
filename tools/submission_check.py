"""在 repo 外面跑打包好的 submission —— **這才驗得到「少放檔案」**。

    python -m tools.submission_check submission/cma1-g50
    python -m tools.submission_check submission --games 2

## 為什麼需要這一支

2026-08-28 `submission/cma1-g50/` 上 Kaggle 直接死在

    File "/kaggle_simulations/agent/gen0.py", line 48, in <module>
        from contracts import TASK_OPS, target_xy
    ModuleNotFoundError: No module named 'contracts'

而本機驗收 640 局**逐位元組相同、零錯誤**。原因是 `eval/runner.py` 的
builtin 分支在 repo root 底下跑：`contracts.py` 就躺在 root 上，cwd 又在
`sys.path` 裡，所以那個 import 一定成功。Kaggle 上只有
`/kaggle_simulations/agent/` 那個目錄，缺什麼就炸什麼。

**「本機跑得動」對「少放檔案」這一類錯誤完全沒有鑑別力。**

## 做法

把目錄裡的檔案複製到一個 repo 外的暫存目錄，開一個 cwd 在那裡、`sys.path`
不含 repo root 的子行程，用引擎自己的 file-path agent 載入器跑完整局 ——
跟 Kaggle 同一條路徑。

🩸 一定要在**子行程**跑。同一個行程裡 `agents.gen0` / `contracts` 早就在
`sys.modules` 裡了，改 `sys.path` 也擋不掉。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 在暫存目錄裡跑的腳本。獨立成字串，因為它要在乾淨的 sys.path 下執行。
_RUNNER = '''
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path = [p for p in sys.path if os.path.abspath(p) != REPO]
from kaggle_environments import make
ok = True
for seed in SEEDS:
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    env.run([os.path.join(here, "main.py"), "starter"])
    last = env.steps[-1]
    for i, s in enumerate(last):
        if s.get("status") != "DONE":
            ok = False
            print(f"seed {seed} player {i}: {s.get('status')}")
    print(f"seed {seed}: " + " / ".join(f"{s['reward']:,.0f}" for s in last))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
'''


def check(folder, seeds=(41001,)):
    """回傳 (成功與否, 輸出文字)。"""
    src = Path(folder)
    if not (src / "main.py").is_file():
        raise SystemExit(f"{src} 裡沒有 main.py")
    with tempfile.TemporaryDirectory(prefix="subcheck-") as tmp:
        dest = Path(tmp)
        for f in src.iterdir():
            # 只帶 .py —— 權重之類的二進位另外處理，__pycache__ 一定要排除
            if f.is_file() and f.suffix in (".py", ".npz"):
                shutil.copyfile(f, dest / f.name)
        script = ("REPO = %r\nSEEDS = %r\n" % (str(REPO_ROOT), list(seeds))) + _RUNNER
        (dest / "_check.py").write_text(script, encoding="utf-8")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["KAGGRI_LOG_LEVEL"] = "0"      # 不然 agent 每回合噴一行 JSON
        proc = subprocess.run(
            [sys.executable, "_check.py"], cwd=str(dest), env=env,
            capture_output=True, text=True, errors="replace")
    out = proc.stdout + proc.stderr
    return proc.returncode == 0 and "PASS" in proc.stdout, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="打包好的目錄，例：submission/cma1-g50")
    ap.add_argument("--seeds", default="41001", help="逗號分隔")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    ok, out = check(args.folder, seeds)
    # 引擎會噴 open_spiel 的 336 行，只留有用的
    for line in out.splitlines():
        if any(k in line for k in ("seed", "PASS", "FAIL", "Error", "error",
                                   "Traceback", "Module", "line ")):
            print(line)
    print(("OK   " if ok else "壞掉 ") + args.folder)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
