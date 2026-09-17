"""作物 cohort：追蹤每一次 PLANT 到它消失為止。

tile 自己就帶著 cohort 需要的欄位（`planted_day` / `crop` / `yield_units` /
`watered_today` / `consecutive_unwatered` / `fertilized_until_day`），所以不用
把動作歸因到格子 —— 直接看格子的狀態變化。

一筆 cohort =
    traj, tile, crop, plant_step, end_step, end_reason, peak_yield,
    end_yield, n_water_day, n_fert_day, life_steps

end_reason: 0=收穫（變 None）1=變雜草 2=局末仍在場 3=其他
🩸 「收穫」是從 tile 變 None 推斷的（engine-notes §3：HARVEST 後 tile 變回 None）。
   ongoing 作物（TOMATO/STRAWBERRY）中途收成不會清空格子，所以它們的多次產出
   算在同一筆 cohort 裡，用 peak_yield 與 n_harvest_drop 表示。
"""
from __future__ import annotations
import numpy as np

N_COL = 12
COLS = ("traj", "tile", "crop", "plant_step", "end_step", "end_reason",
        "peak_yield", "end_yield", "n_water_day", "n_fert_day",
        "n_yield_drop", "sum_yield_drop")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
CROP_ID = {c: i for i, c in enumerate(CROPS)}


class Tracker:
    """逐步餵 tiles，吐出 cohort。一個 (game, player) 一個 instance。"""

    def __init__(self, traj_id):
        self.traj = traj_id
        self.cur = {}          # tile_idx -> dict
        self.done = []

    def _close(self, k, st, end_step, reason):
        self.done.append((self.traj, k, st["crop"], st["plant_step"], end_step,
                          reason, st["peak"], st["last"], len(st["wdays"]),
                          len(st["fdays"]), st["drops"], st["drop_sum"]))

    def feed(self, tiles, step, day):
        seen = set()
        board = len(tiles)
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                k = y * board + x
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    seen.add(k)
                    yu = float(tile.get("yield_units", 0))
                    st = self.cur.get(k)
                    pd = tile.get("planted_day")
                    if st is None or st["planted_day"] != pd:
                        if st is not None:
                            self._close(k, st, step, 3)
                        st = self.cur[k] = dict(
                            crop=CROP_ID.get(tile.get("crop"), -1),
                            plant_step=step, planted_day=pd, peak=yu, last=yu,
                            wdays=set(), fdays=set(), drops=0, drop_sum=0.0)
                    if yu < st["last"]:
                        st["drops"] += 1
                        st["drop_sum"] += st["last"] - yu
                    st["peak"] = max(st["peak"], yu)
                    st["last"] = yu
                    if tile.get("watered_today"):
                        st["wdays"].add(day)
                    if tile.get("fertilized_until_day", -1) >= day:
                        st["fdays"].add(day)
                elif k in self.cur:
                    reason = 0 if tile is None else (
                        1 if isinstance(tile, dict) and tile.get("kind") == "WEED" else 3)
                    self._close(k, self.cur.pop(k), step, reason)

    def finish(self, step):
        for k, st in list(self.cur.items()):
            self._close(k, st, step, 2)
        self.cur.clear()
        return self.done
