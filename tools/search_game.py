"""Stage 2 gate：**整局套用 search，累積起來贏得了 gen0 嗎？**

    python -m tools.search_game --seeds 0-23 --workers 24

## 為什麼單回合的 oracle 不夠

Stage 1 量的是「在**一個**盤面改一步能好多少」。那是上限，不是可達成的分數，
而且沒有回答真正的問題：**在幾十個決策點各改一次，累積起來會怎樣。**

反過來也成立 —— 就算單回合只有 3%，改 30 次也可能很可觀。這是 policy
improvement 的標準論證，08-21 / 08-24 從來沒量過。

## 做法

跑一整局，在取樣的決策點（預設每天 hour 12）用搜出來的動作取代 `gen0` 的；
其餘回合照常 `gen0`。同一個 seed、同一個對手再跑一局純 `gen0` 當對照。

🩸 **搜的時候 rollout 用 `gen0`，但實際往下走的時候後面的決策點也會被搜。**
所以搜尋當下估的 Q 不等於最後真的拿到的分數 —— 那是這個做法內建的近似，
不是 bug。真正算數的是**最後的期末現金**。

## 平行化放在「局」這一層

一局之內的決策點是**循序**的（後面的狀態取決於前面選了什麼）。同一個決策點的
候選互不相依，但那要把 `env.state` 送進別的行程，而它能不能 pickle 沒驗過。
所以直接一個 seed 一個 worker，局內循序 —— 簡單而且不會踩到未知數。

## 輸出

`temp/search-game-<時間戳>/result.json`，欄位做成 `eval.runner` 的形狀
（`seed` / `a_slot` / `cash_a` / `cash_b` / `status_*` / `error`）。
**`cash_a` 是搜尋版、`cash_b` 是同一個 seed 的純 `gen0`**，已經配對好了，
所以判定直接在這支裡面算（Wilcoxon + MDE），不需要 `tools.paired_stats`
（那支是比兩個不同 run 的 `cash_a`）。

⚠️ **不要看平均差就下結論。** 08-24 §1 踩過：20 局 p=0.123、MDE 8,332，
實測差 5,185 落在測不出的範圍裡，差點被當成「沒有差別」。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_seeds(text):
    """`0-23` 或 `0,3,7` 都吃。"""
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return out


def _cash(v):
    """沒有這個欄位就印破折號 —— 舊的 run 記不到中途現金。"""
    return f"{v:,.0f}" if v is not None else "—"


def watch(run_dir):
    """把 progress/*.jsonl 讀成一張表。跑到一半隨時可以看。"""
    import glob

    prog = Path(run_dir)
    if prog.name != "progress":
        prog = prog / "progress"
    if not prog.is_dir():
        raise SystemExit(f"找不到 {prog} —— 路徑對嗎？")
    # 🩸 一天搜幾個 hour 決定了「決策點總數」。2026-08-26 23:49 使用者抓到
    # 這裡把「決策點數」印成「第幾天」—— `--hours 12` 時兩者剛好相等所以看不
    # 出來，改成一天 4 個之後就差 4 倍。
    per_day = None
    cfg_path = prog.parent / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(io.open(cfg_path, encoding="utf-8").read())
            per_day = len([h for h in str(cfg["hours"]).split(",") if h.strip()])
        except (ValueError, KeyError):
            per_day = None

    files = sorted(glob.glob(str(prog / "*.jsonl")))
    if not files:
        # 開跑後的頭幾分鐘都在算 gen0 對照組，還沒有決策點可以記。
        print(f"{prog} 還沒有 .jsonl —— 十之八九是還在跑 gen0 對照組，等一下再看。")
        return 0

    print(f"（10 局同時跑，一個 seed 一個 worker。"
          f"「第幾天」「決策點」都是那一局**之內**的進度，不是第幾局。"
          f"一天搜幾個 hour 由 --hours 決定，所以決策點總數 = 天數 × 那個數字）")
    print(f"{'seed':>5}{'對手':>11}{'gen0 基準':>11}{'第幾天':>9}{'決策點':>11}"
          f"{'我方現金':>9}{'對手現金':>9}{'目前 Q':>11}"
          f"{'vs gen0':>9}{'改過':>6}{'已花':>8}{'更新':>10}")
    done_rows, running = [], 0
    for f in files:
        seed = int(Path(f).stem[4:])
        base = last = final = base_opp = None
        who = "?"
        total = 30
        for line in open(f, encoding="utf-8"):
            row = json.loads(line)
            if row["phase"] == "baseline":
                base = row["cash"]
                base_opp = row.get("opp")
                who = row.get("opponent", "?")
                total = int(row.get("total_days", 30))
            elif row["phase"] == "search":
                last = row
            elif row["phase"] == "done":
                final = row
        if final:
            done_rows.append((final["cash"], final["baseline"],
                              final.get("opp"), base_opp))
            print(f"{seed:>5}{who:>13}{final['baseline']:>11,.0f}"
                  f"{'完成':>8}"
                  f"{str(last['done'] if last else '?') + ' 個':>11}"
                  f"{_cash(final.get('cash')):>11}"
                  f"{_cash(final.get('opp')):>11}"
                  f"{'':>11}"
                  f"{final['cash'] / final['baseline'] - 1:>8.1%}"
                  f"{last['changed'] if last else 0:>6}"
                  f"{(last['secs'] if last else 0) / 60:>7.0f}分{final['t']:>10}")
        elif last:
            running += 1
            day_cell = f"{last.get('day', 0) + 1}/{total}"
            pts_cell = (f"{last['done']}/{total * per_day}" if per_day
                        else str(last["done"]))
            print(f"{seed:>5}{who:>13}{base:>11,.0f}"
                  f"{day_cell:>9}{pts_cell:>11}"
                  f"{_cash(last.get('me')):>11}"
                  f"{_cash(last.get('opp_now')):>11}"
                  f"{last['q']:>11,.0f}{last['q'] / base - 1:>8.1%}"
                  f"{last['changed']:>6}{last['secs'] / 60:>7.0f}分{last['t']:>10}")
        else:
            running += 1
            print(f"{seed:>5}{who:>13}{(base or 0):>11,.0f}{'還在算對照組':>14}")

    if done_rows:
        diffs = [a - b for a, b, _o, _ob in done_rows]
        print()
        print(f"  整局跑完的有 {len(done_rows)}/{len(files)} 局：配對差平均 "
              f"{statistics.fmean(diffs):>+,.0f}，"
              f"search 較好 {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
        # 🩸 我方變高有兩種可能：自己賺更多，或是壓低對手。市場是共用的，
        # 不分開看的話會把「讓對手的訂單失敗」當成真增益。
        opp_pairs = [(o, ob) for _a, _b, o, ob in done_rows
                     if o is not None and ob is not None]
        if opp_pairs:
            opp_d = statistics.fmean(o - ob for o, ob in opp_pairs)
            print(f"  對手也變高 {opp_d:>+,.0f} —— 市場是共用的，"
                  f"我方的絕對增益有一部分是「餅變大」")
            # 🩸 決策要看的是**現金比**，不是絕對值。絕對值兩邊一起漲的話
            # 名次不會變；現金比才對得上 eval/runner.py 的勝負判定。
            rb = [b / ob for a, b, o, ob in done_rows
                  if o is not None and ob is not None and ob]
            rs = [a / o for a, b, o, ob in done_rows
                  if o is not None and ob is not None and o]
            if rb and rs:
                print(f"  現金比 對照 {statistics.fmean(rb):.3f} -> "
                      f"搜尋 {statistics.fmean(rs):.3f}"
                      f"（差 {statistics.fmean(rs) - statistics.fmean(rb):+.3f}）"
                      f"   變好 "
                      f"{sum(1 for x, y in zip(rs, rb) if x > y)}/{len(rs)}")
    if running:
        print(f"  還在跑 {running} 局（都在同時進行）")
    print()
    print("⚠️ 「目前 Q」是「搜到這一天、之後全用 gen0 打到底」的估值，"
          "不是最後的期末現金。")
    return 0


def _read_resume(prog_dir, seed):
    """把一個 seed 的 progress 讀成續跑用的狀態。沒有檔案就回 None。

    🩸 靠的是「重播動作」而不是序列化 `env` —— 引擎、`agents/gen0.py`、
    `agents/replay.py`、`serving/npz_forward.py` 全部是決定性的
    （2026-08-26 三批不同行程的對照組逐位元組相同，實測過）。
    所以重建 env 之後把記下來的動作依序送進去，就會回到同一個狀態。
    """
    f = Path(prog_dir) / f"seed{seed:04d}.jsonl"
    if not f.is_file():
        return None
    out = {"baseline": None, "actions": [], "changed": 0, "secs": 0,
           "done": None}
    for line in io.open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["phase"] == "baseline":
            out["baseline"] = {"cash": r["cash"], "opp": r["opp"]}
        elif r["phase"] == "search":
            if "action" not in r:
                # 2026-08-26 23:30 之前跑的沒記動作 -> 續不了，整局重跑。
                return {"baseline": out["baseline"]}
            out["actions"].append(r["action"])
            out["changed"] = r.get("changed", 0)
            out["secs"] = r.get("secs", 0)
        elif r["phase"] == "done":
            out["done"] = {
                "a_slot": r.get("a_slot", 0),
                "cash_a": r["cash"], "cash_b": r["baseline"],
                "status_a": "DONE", "status_b": "DONE",
                "opp_cash_searched": r.get("opp"),
                "opp_cash_baseline": r.get("opp_baseline"),
                "decisions": out and len(out["actions"]),
                "changed": out["changed"],
                "elapsed": out["secs"],
            }
    return out


def _play_one(job):
    """跑一個 seed：搜尋版 + 純 gen0 對照。top-level 才 pickle 得動。"""
    (seed, hours, opponent, rng_seed, days, max_cands, a_slot, prog_dir,
     data_dir, policy, resume) = job
    resume = resume or {}

    def _json_safe(o):
        """numpy 的整數/浮點數 `json.dumps` 吞不下，會直接拋。

        🩸 動作裡的數量可能是 `np.int64`（`contracts.decode_*` 產的），
        而 2026-08-26 起 progress 會記下選中的動作。
        """
        if isinstance(o, np.generic):
            return o.item()
        return str(o)

    def note(**row):
        """一個決策點一行。worker 之間各寫各的檔，不會互相蓋。"""
        if not prog_dir:
            return
        row["t"] = time.strftime("%H:%M:%S")
        with open(Path(prog_dir) / f"seed{seed:04d}.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False,
                                default=_json_safe) + "\n")

    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    if resume.get("done"):
        # 已跑完的局：progress 有 done 那一行、npz 也寫過了。把紀錄還原成結果，
        # **不要重跑** —— 不然續跑一次就把整批重算一遍，過夜批永遠跑不完。
        return dict(resume["done"], seed=seed, opponent=opponent, error=None)

    os.environ.pop("KAGGRI_LOG_FILE", None)
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make

    import numpy as np

    from agents.gen0 import act as gen0_act
    from search import candidates as CAND
    from search import rollout_search as RS

    def new_env():
        with contextlib.redirect_stderr(io.StringIO()):
            return make("kaggriculture", configuration={"seed": seed},
                        debug=False)

    t0 = time.perf_counter()
    err = None
    stopped = False
    searched = baseline = opp_searched = opp_baseline = None
    n_searched = n_changed = 0
    try:
        # --- 對照組：純 gen0 ---
        env = new_env()
        cfg = env.configuration

        if opponent == "gen0":
            def act_them(obs):
                return gen0_act(obs, cfg)
        else:
            import inspect

            from eval.runner import build_agent, load_spec
            _opp = build_agent(load_spec(opponent))
            # 🩸 引擎內建的對手（starter / pass / random）`build_agent` 回傳的是
            # **字串**，給 `env.run([...])` 用的。這裡是自己驅動迴圈，要去
            # `env.agents` 把真正的 function 拿出來 —— 而且它們只吃 obs 一個
            # 參數，簽名跟 `agents.*:act` 不一樣。
            if isinstance(_opp, str):
                _opp = env.agents[_opp]
            if len(inspect.signature(_opp).parameters) == 1:
                def act_them(obs):
                    return _opp(obs)
            else:
                def act_them(obs):
                    return _opp(obs, cfg)

        # 🩸 走盤面的 policy 跟 search 內部 rollout 的 policy **必須是同一支**。
        # 盤面由 round6 走、Q 卻假設「之後 gen0 接手」的話，算的是一個不會發生
        # 的未來（2026-08-26 使用者指出）。
        if policy == "net":
            from agents.gen2_model import act as net_act

            def act_me(obs):
                return net_act(obs, cfg)

            base_label = "base"
        else:
            def act_me(obs):
                return gen0_act(obs, cfg)

            base_label = "gen0"

        if resume.get("baseline") is not None:
            # 對照組是決定性的（2026-08-26 三批實測逐位元組相同），紀錄裡有就
            # 直接用 —— 省一次完整 rollout，也保證跟斷掉之前一致。
            baseline = resume["baseline"]["cash"]
            opp_baseline = resume["baseline"]["opp"]
        else:
            baseline = RS.play_to_end(env, act_me, act_them, a_slot, days)
            opp_baseline = RS.final_cash(env, 1 - a_slot)
            note(phase="baseline", cash=baseline, opp=opp_baseline,
                 opponent=opponent, total_days=days)

        # --- 搜尋版 ---
        rec = None
        if data_dir:
            from search.expert import SearchRecorder
            rec = SearchRecorder(a_slot)
        # 續跑時要重播的動作（依序）。用完之後恢復正常搜尋。
        replay = list(resume.get("actions") or [])
        n_changed = resume.get("changed", 0)
        secs_offset = resume.get("secs", 0)
        env = new_env()
        cfg = env.configuration
        while env.state[0].observation["day"] < days and not env.done:
            obs = env.state[a_slot].observation
            # gen0 走盤面而且要錄資料的話，沒搜的那些回合需要 plan
            # （target / demand 標籤）。網路走盤面時沒有 plan 可拿 —— 而且那些
            # 回合的「標籤」就是網路自己的輸出，教它它已經會的事，零資訊，
            # 所以 policy=net 時**只錄搜過的回合**。
            plan = None
            if policy == "net":
                base = act_me(obs)
            elif rec is not None:
                base, plan = gen0_act(obs, cfg, None, return_plan=True)
            else:
                base = gen0_act(obs, cfg)
            if int(obs.get("hour", 0)) in hours and replay:
                # 續跑：這個決策點的答案已經在 progress 裡，重播就好。
                # 那一行也寫過了，不要重寫。
                action = replay.pop(0)
                n_searched += 1
            elif int(obs.get("hour", 0)) in hours:
                if (Path(prog_dir) / "STOP").exists():
                    # 乾淨收工：已寫下的決策點都續得回來，不會弄壞任何東西。
                    stopped = True
                    break
                snap = RS.snapshot(env)
                # 🩸 逐決策點播種。原本整局共用一個 rng 一路往下抽，續跑時前面
                # 的決策點不呼叫 generate，rng 狀態就對不上，多 unit 候選會抽到
                # 不同組 —— 不影響正確性，但續跑就不可重現了。
                rng = np.random.default_rng(rng_seed + seed * 10007 + n_searched)
                cands, _blocked, cinfo = CAND.generate(
                    obs, cfg, base, rng, base_label=base_label)
                if max_cands and len(cands) > max_cands:
                    keep = [c for c in cands if c[0] == base_label]
                    rest = [c for c in cands if c[0] != base_label]
                    idx = rng.choice(len(rest), size=max_cands - 1, replace=False)
                    cands = keep + [rest[int(i)] for i in idx]
                label, action, _q, _scored = RS.search_action(
                    env, snap, cands, act_them, act_me, a_slot, days,
                    base_label=base_label)
                RS.restore(env, snap)
                n_searched += 1
                n_changed += int(label != base_label)
                # 🩸 贏家跟基準線的 Q 差距一定要記。差距很小的話，60 個候選的
                # argmax 是被長 horizon 的混沌決定的 —— 那種標籤本質上學不起來，
                # 加權/換編碼/換對手全都沒用（2026-08-26 journal §8）。
                by = dict(_scored)
                ranked = sorted(by.values(), reverse=True)
                note(phase="search", day=int(obs["day"]),
                     hour=int(obs.get("hour", 0)), label=label,
                     q=round(_q), q_base=round(by[base_label]),
                     gain=round(_q - by[base_label]),
                     runner_up=round(ranked[1]) if len(ranked) > 1 else None,
                     n_cands=len(cands),
                     # 🩸 候選集合的健康度。2026-08-26 seed 4 是 55/56 被擋掉
                     # 之後整局死掉，而在那之前沒有任何紀錄看得出來。
                     blocked=cinfo["blocked"], n_raw=cinfo["n_raw"],
                     base_illegal=cinfo["base_illegal"],
                     dropped_orders=cinfo["dropped_orders"],
                     blocked_why=cinfo["reasons"],
                     me=round(RS.final_cash(env, a_slot)),
                     opp_now=round(RS.final_cash(env, 1 - a_slot)),
                     changed=n_changed, done=n_searched,
                     action=action,          # 🩸 續跑靠這一欄重播
                     secs=round(secs_offset + time.perf_counter() - t0))
            else:
                action = base
            if rec is not None:
                if int(obs.get("hour", 0)) in hours:
                    # search 決定的回合只有 action，沒有 plan。
                    rec.record_search(obs, cfg, action)
                elif policy != "net":
                    rec.record(obs, cfg, action, plan)
            actions = [None, None]
            actions[a_slot] = action
            actions[1 - a_slot] = act_them(env.state[1 - a_slot].observation)
            env.step(actions)
        if stopped:
            # 看到 STOP 就地收工。**不寫 done、不寫 npz** —— 那一局還沒跑完，
            # 寫下去會被當成完成的局，之後 --resume 就跳過它了。
            return {
                "seed": seed, "a_slot": a_slot,
                "cash_a": None, "cash_b": baseline,
                "status_a": "STOPPED", "status_b": "DONE",
                "opp_cash_searched": None, "opp_cash_baseline": opp_baseline,
                "decisions": n_searched, "changed": n_changed,
                "elapsed": time.perf_counter() - t0, "error": None,
            }
        searched = RS.final_cash(env, a_slot)
        # 🩸 對手的期末現金一定要記。市場是兩家共用的，我方分數變高有兩種可能：
        # 自己賺更多，或是壓低對手。不記的話分不出來 —— 而對開迴路 replay 來說
        # 「讓它的 BUY_*/HIRE 失敗」是不會轉移到真實對手身上的假增益。
        opp_searched = RS.final_cash(env, 1 - a_slot)
        if rec is not None:
            rewards = [None, None]
            rewards[a_slot] = searched
            rewards[1 - a_slot] = opp_searched
            payload = rec.finish(rewards, seed)
            np.savez_compressed(
                Path(data_dir) /
                f"{'dagger' if policy == 'net' else 'search'}"
                f"-{opponent}-{seed:06d}.npz", **payload)
        note(phase="done", cash=searched, baseline=baseline,
             opp=opp_searched, opp_baseline=opp_baseline)
    except Exception as exc:                          # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    return {
        "seed": seed,
        "a_slot": a_slot,
        "cash_a": searched,
        "cash_b": baseline,
        "status_a": "DONE" if err is None else "ERROR",
        "status_b": "DONE" if err is None else "ERROR",
        "opp_cash_searched": opp_searched,
        "opp_cash_baseline": opp_baseline,
        "decisions": n_searched,
        "changed": n_changed,
        "elapsed": time.perf_counter() - t0,
        "error": err,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="model/weights-e2e-round6.npz")
    ap.add_argument("--seeds", default="0-23")
    ap.add_argument("--hours", default="12",
                    help="每天的哪幾個 hour 要搜（逗號分隔）。12 = 一天一次")
    # 🩸 預設跟 harness/rollout.py:75 一致（強/中/弱各一，輪流配）。
    # 只跟單一對手打的話期末現金落在很窄的區間，value head 分辨不出盤面好壞
    # —— 那是 harness/rollout.py:48-52 已經實測過的失敗模式，不要重蹈。
    ap.add_argument("--opponent",
                    default=",".join((
                        # 會反應的，強度鋪開（2026-08-21 實測期末現金）：
                        "hire-0", "hire-2", "hire-4", "hire-8",
                        # 真實 ladder 頂端的 replay，8 個不同玩家：
                        "kawashigi", "tetsuya", "utkarsh", "recursion",
                        "thomas", "peikopon", "lucien", "kostiantyn")),
                    help="逗號分隔可以給多支，依 seed 順序輪流配")
    # 🩸 這台是 20 實體核心 / 28 邏輯核心。對局是純 CPU 的 Python 模擬，
    # 行程數超過實體核心就互搶、總時間反而變長，而且整台機器會被吃滿。
    # 2026-08-25 用 24 跑過一次，使用者回報變慢。要開更多之前先問。
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-cands", type=int, default=0,
                    help="每個決策點最多幾個候選，0 = 全部（約 57）")
    ap.add_argument("--policy", default="gen0", choices=("gen0", "net"),
                    help="誰走盤面、以及 search 內部 rollout 用誰。"
                         "net = 真正的 DAgger（網路走、search 貼標籤），"
                         "而且只錄搜過的回合")
    ap.add_argument("--a-slot", type=int, default=0, choices=(0, 1))
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rng-seed", type=int, default=20260825)
    ap.add_argument("--data-out", metavar="DIR",
                    help="把搜尋版那一局錄成訓練 npz 寫進這個目錄（Stage 3）")
    ap.add_argument("--watch", metavar="RUN_DIR",
                    help="只看某一輪的進度，不跑新的")
    ap.add_argument("--resume", metavar="RUN_DIR",
                    help="續跑：接上這個目錄。已完成的局跳過，跑到一半的局從"
                         "最後一個決策點接下去（重播 progress 裡記下的動作）。"
                         "設定一律用該目錄的 config.json，不吃這次的旗標")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.watch:
        return watch(args.watch)

    os.environ["KAGGRI_LOG_LEVEL"] = "0"
    os.environ["KAGGRI_WEIGHTS"] = args.weights

    # 🩸 續跑一定要用原本的設定。`--hours` 之類的變掉的話，重播的動作會落在
    # 不同的回合上，整局的狀態就對不起來 —— 而且不會報錯，只會安靜地產出垃圾。
    CFG_KEYS = ("seeds", "hours", "opponent", "policy", "days", "max_cands",
                "a_slot", "rng_seed", "weights", "data_out")
    if args.resume:
        run_dir = Path(args.resume)
        cfg_path = run_dir / "config.json"
        if not cfg_path.is_file():
            raise SystemExit(f"{cfg_path} 不存在 —— 那一輪是 2026-08-26 23:30 "
                             "之前跑的，沒有存設定，續不了")
        saved = json.loads(io.open(cfg_path, encoding="utf-8").read())
        for k in CFG_KEYS:
            setattr(args, k, saved[k])
        os.environ["KAGGRI_WEIGHTS"] = args.weights
        print(f"續跑 {run_dir}")
        print(f"  設定取自 config.json：{saved}")
    else:
        run_dir = REPO_ROOT / "temp" / f"search-game-{time.strftime('%Y%m%d-%H%M%S')}"

    seeds = _parse_seeds(args.seeds)
    hours = frozenset(int(h) for h in args.hours.split(","))
    prog_dir = run_dir / "progress"
    prog_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        io.open(run_dir / "config.json", "w", encoding="utf-8").write(
            json.dumps({k: getattr(args, k) for k in CFG_KEYS},
                       ensure_ascii=False, indent=2))
    data_dir = Path(args.data_out) if args.data_out else None
    if data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)

    opponents = [o.strip() for o in args.opponent.split(",") if o.strip()]
    resumes = ([_read_resume(prog_dir, s) for s in seeds]
               if args.resume else [None] * len(seeds))
    jobs = [(s, hours, opponents[i % len(opponents)], args.rng_seed, args.days,
             args.max_cands, args.a_slot, str(prog_dir),
             str(data_dir) if data_dir else None, args.policy, r)
            for i, (s, r) in enumerate(zip(seeds, resumes))]
    if args.resume:
        fin = sum(1 for r in resumes if r and r.get("done"))
        part = sum(1 for r in resumes if r and not r.get("done")
                   and r.get("actions"))
        print(f"  已完成 {fin} 局（跳過）、跑到一半 {part} 局（重播接續）、"
              f"從頭跑 {len(seeds) - fin - part} 局")
    if (prog_dir / "STOP").exists():
        raise SystemExit(f"{prog_dir / 'STOP'} 還在 —— 要續跑的話先把它刪掉")

    print(f"{len(seeds)} 個 seed   走盤面 {args.policy}   "
          f"每天搜 {sorted(hours)}   對手 {opponents}   "
          f"權重 {args.weights}   {args.workers} workers")
    print(f"產物 {run_dir}")
    print(f"看進度：python -m tools.search_game --watch {run_dir}")

    t0 = time.perf_counter()
    if args.workers <= 1:
        results = [_play_one(j) for j in jobs]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = []
            for r in pool.imap_unordered(_play_one, jobs):
                results.append(r)
                mark = "⚠️" if r["error"] else "  "
                print(f"  {mark} seed {r['seed']:>4}   搜尋 "
                      f"{(r['cash_a'] or 0):>9,.0f}   gen0 "
                      f"{(r['cash_b'] or 0):>9,.0f}   "
                      f"{((r['cash_a'] or 0) - (r['cash_b'] or 0)):>+9,.0f}   "
                      f"改了 {r['changed']}/{r['decisions']}   "
                      f"{r['elapsed']:.0f}s"
                      + (f"   {r['error']}" if r["error"] else ""), flush=True)
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["error"] is None]
    diffs = [r["cash_a"] - r["cash_b"] for r in ok]
    summary = {
        "a": "search", "b": "gen0",
        "games": len(results), "failures": len(results) - len(ok),
        "mean_cash_a": statistics.fmean(r["cash_a"] for r in ok) if ok else 0,
        "mean_cash_b": statistics.fmean(r["cash_b"] for r in ok) if ok else 0,
        "mean_diff": statistics.fmean(diffs) if diffs else 0,
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
        "draws": sum(1 for d in diffs if d == 0),
        "wall_secs": wall,
        "workers": args.workers,
    }

    payload = {
        "a": {"name": "search", "note": f"gen0 + 離線 search，每天 hour {sorted(hours)}"},
        "b": {"name": "gen0", "note": "同一個 seed、同一個對手的純 gen0"},
        "summary": summary,
        "results": sorted(results, key=lambda r: r["seed"]),
        "argv": vars(args),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n{len(ok)}/{len(results)} 局完成，{wall / 60:.0f} 分鐘")
    if ok:
        print(f"  search 平均 {summary['mean_cash_a']:>10,.0f}")
        print(f"  gen0   平均 {summary['mean_cash_b']:>10,.0f}")
        print(f"  配對差      {summary['mean_diff']:>+10,.0f}   "
              f"（中位 {statistics.median(diffs):>+,.0f}）")
        print(f"  search 較好 {summary['wins']}/{len(ok)}")
        print(f"  平均改掉    "
              f"{statistics.fmean(r['changed'] for r in ok):.1f}/"
              f"{statistics.fmean(r['decisions'] for r in ok):.0f} 個決策點")
    # search 和 gen0 已經配對在同一份 result.json 的 cash_a / cash_b 裡，
    # 所以不需要 tools.paired_stats（那支是比**兩個** run 的 cash_a）。
    # 這裡直接用同一組公式：Wilcoxon + MDE（α=0.05、power=0.8）。
    if len(diffs) > 1:
        try:
            from scipy.stats import wilcoxon
            nz = [d for d in diffs if d]
            if nz:
                p = float(wilcoxon(nz).pvalue)
                sd = statistics.stdev(diffs)
                mde = 2.8016 * sd / len(diffs) ** 0.5
                print(f"\n  Wilcoxon p = {p:.5g}   sd {sd:,.0f}   "
                      f"MDE {mde:,.0f}（α=0.05、power=0.8）")
                if abs(summary["mean_diff"]) < mde:
                    print(f"  ⚪ 判不出來 —— 實測差 {abs(summary['mean_diff']):,.0f} "
                          f"< MDE，要加 seed。")
                elif p < 0.05:
                    print("  ✅ 顯著較強 —— 進 Stage 3。" if summary["mean_diff"] > 0
                          else "  ❌ 顯著較弱。")
                else:
                    print(f"  ⚪ 判不出來（p={p:.3g} ≥ 0.05）。")
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
