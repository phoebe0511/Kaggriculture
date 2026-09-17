"""Kaggle 134 條軌跡的逐日「狀態 + 行為」表。

動作直接讀 config/episodes/*.json（不用重播）；狀態讀 temp/curve_episodes.npz
（那是雙方重播錄的，67/67 逐位元重現原局）。

phase = 一天 = 24 步，引擎的 `_end_of_day` 結算週期（configuration.turnsPerDay）。

輸出 temp/kaggle_days.npz：
    act    [traj, 30, N_FIELD]   逐日動作統計（behav_ops 的詞彙）
    st     [traj, 30, 7]         逐日狀態：open/empty/crop/structure/weed 的日均
                                 + 當日最後一刻的 money + 當日 money 變化
    cash   [traj]                期末現金
    margin [traj]                期末 zero-sum 差額
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\_phoebe_priv\Kaggriculture")
os.chdir(r"C:\_phoebe_priv\Kaggriculture")
from tools.behav_ops import tally, N_FIELD, FIELDS          # noqa: E402

DAY, NDAY = 24, 30
ST_FIELDS = ("open", "empty", "crop", "structure", "weed", "money", "d_money")

z = np.load("temp/curve_episodes.npz")
ep_ids = list(z["episode"])
order = {int(e): i for i, e in enumerate(ep_ids)}

acts, sts, cash, margin, keys = [], [], [], [], []
for f in sorted(glob.glob("config/episodes/*.json")):
    d = json.load(open(f, encoding="utf-8"))
    eid = d["episode_id"]
    if eid not in order or d.get("n_steps") != 720:
        continue
    gi = order[eid]
    for pid in (0, 1):
        A = np.zeros((NDAY, N_FIELD))
        for t, step in enumerate(d["actions"][:720]):
            if pid < len(step):
                tally(step[pid], A[t // DAY])
        S = np.zeros((NDAY, len(ST_FIELDS)))
        for k, nm in enumerate(("open", "empty", "crop", "structure", "weed")):
            arr = z[nm][gi, pid]
            S[:, k] = arr.reshape(NDAY, DAY).mean(1)
        mn = z["money"][gi, pid]
        S[:, 5] = mn.reshape(NDAY, DAY)[:, -1]
        S[:, 6] = np.diff(np.concatenate([[mn[0]], S[:, 5]]))
        acts.append(A); sts.append(S)
        c = float(z["cash"][gi, pid]); o = float(z["cash"][gi, 1 - pid])
        cash.append(c); margin.append(c - o)
        keys.append((eid, pid))

acts = np.stack(acts); sts = np.stack(sts)
np.savez_compressed("temp/kaggle_days.npz", act=acts, st=sts,
                    cash=np.array(cash), margin=np.array(margin),
                    key=np.array(keys), fields=np.array(FIELDS),
                    st_fields=np.array(ST_FIELDS))
print(f"{len(cash)} 條軌跡 x {NDAY} 天")
print(f"  act {acts.shape}   st {sts.shape}")
print(f"  期末現金 {min(cash):,.0f} ~ {max(cash):,.0f}")
print("  -> temp/kaggle_days.npz")
