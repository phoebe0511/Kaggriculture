# === CELL 7: 把上傳的 agent 檔案集中到 /kaggle/working ===
# 來源：把本機 repo 的 `submission/` 目錄整包上傳成一個 Kaggle Dataset。
# 那個目錄是 `python -m serving.build_submission` 產生的，已經**攤平**過。
#
# 警告 一定要用 submission/ 那一版，不能直接傳 repo 的 agents/ 和 serving/。
#
#   `agents` 這個 top-level 名字在 Kaggle 的執行環境已經被佔走了 ——
#   site-packages/kaggle_environments/envs/lux_ai_s3/ 底下有一支 agents.py，
#   而 kaggle_environments 載入 agent 的做法是（agent.py:51-59）：
#       exec_dir = os.path.dirname(path)
#       sys.path.append(exec_dir)          # ← 接在最後面
#       exec(compile(raw, path, "exec"), {})
#   我們的目錄排在後面，跨 sys.path entry 是前面的贏，所以 `import agents`
#   會抓到 lux_ai_s3 的那支，然後炸在它自己的相對 import：
#       InvalidArgument: Invalid raw Python:
#         ImportError('attempted relative import with no known parent package')
#   加 agents/__init__.py 救不了 —— 「正規 package 優先於同名 .py」只在
#   同一個 entry 內成立。所以 submission 是四支平面模組，沒有 package。
import glob
import os
import re
import shutil

WORK = "/kaggle/working"
WANT = ("main.py", "gen0.py", "gen1.py", "action_validation.py")


def _find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(
            f"/kaggle/input 底下找不到 {name} —— 確認 submission/ 那個 Dataset 有掛上")
    return sorted(hits, key=os.path.getmtime, reverse=True)[0]      # 取最新的


copied = []
for name in WANT:
    src = _find(name)
    dst = os.path.join(WORK, name)
    shutil.copy(src, dst)
    copied.append((name, os.path.getsize(dst), src))
    print(f"  {name:22s} {os.path.getsize(dst):>8,d} B  <- {src}")

# --- 擋住「傳到未攤平版本」這個錯 -------------------------------------------
# 未攤平的檔案裡會留著 `from agents.gen0 import ...`，在 Kaggle 上一定炸，
# 但錯誤訊息會被 kaggle_environments 包成看不出原因的 InvalidArgument。
# 在這裡就抓出來。
_bad_import = re.compile(r"^\s*(?:from|import)\s+(agents|serving)\b", re.M)
for name, _, _ in copied:
    text = open(os.path.join(WORK, name), encoding="utf-8").read()
    hits = _bad_import.findall(text)
    if hits:
        raise SystemExit(
            f"{name} 還有 package import {sorted(set(hits))} —— "
            f"這是未攤平的版本。在本機跑 `python -m serving.build_submission` "
            f"之後重新上傳 submission/。")

# --- 順手確認兩條硬規則 ------------------------------------------------------
main_src = open(os.path.join(WORK, "main.py"), encoding="utf-8").read()
if 'KAGGRI_LOG_LEVEL"] = "0"' not in main_src and "KAGGRI_LOG_LEVEL'] = '0'" not in main_src:
    print("警告 main.py 沒有把 KAGGRI_LOG_LEVEL 設成 0 —— log 會吃掉每回合的時間")
for name, _, _ in copied:
    if "import torch" in open(os.path.join(WORK, name), encoding="utf-8").read():
        raise SystemExit(f"{name} 有 import torch —— submission 不准（載入就可能超時）")

print(f"\nOK 四支平面模組就緒，沒有 package import、沒有 torch")
print("/kaggle/working:", sorted(f for f in os.listdir(WORK) if not f.startswith(".")))
print("接著跑 cell 10。")
