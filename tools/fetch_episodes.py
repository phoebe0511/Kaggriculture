"""下載 ladder 對局的 replay JSON。

⚠️ 原本是另一個專案（pokemon-tcg-ai-battle）帶過來的，2026-08-19 改成
Kaggriculture 用。

## 兩條路，權限不同

    --submission <id>   查「我們自己的」submission 打過哪些 episode，再逐一下載。
                        走 ApiListSubmissionEpisodes，**owner-only**。
    --episode <id> ...   直接照 episode id 抓，跳過查詢那一步。

下載那一步（`kaggleusercontent.com/episodes/<id>.json`）是**免認證的公開 CDN**，
兩條路共用。所以只要拿得到 episode id，**不需要是對局參與者、也不需要 API
token** —— 頂端玩家之間的對局就是這樣抓的。

episode id 從 Kaggle 網頁拿：Leaderboard 點進隊伍 → 對局列表 → 每筆一個 id。

## 為什麼要抓別人的對局

2026-08-19 量到 Gen1 對真實 ladder 對手 0 勝 40 負（見
`docs/memory/journal/2026-08-19.md`）。本機對手全是自家血統，量不出東西；
而且拿 Gen1 當 imitation learning 的老師沒有意義 —— 老師本身在輸。

每局 replay 含**雙方**每一步的完整動作與 observation（各自的 `private` 也在），
所以一局就是一組完整的 (obs → action) 訓練對。

## 用法

    python -m tools.fetch_episodes --list                   # 我們自己的 submission
    python -m tools.fetch_episodes --submission 54350256    # 我們自己的對局
    python -m tools.fetch_episodes --episode 93916293       # 任何公開對局
    python -m tools.fetch_episodes --episode 111 222 333    # 一次多筆

下載完用 `tools/extract_episode.py` 抽成小檔（30 MB → 約 220 KB）再進版控。

⚠️ 抽取前先看 `module_version`。舊 episode 可能是 1.29.x / 1.30.x，market
公式和 COW 成本都不同（對照表在 journal 2026-08-17 §A），**混用會污染資料**。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

#: 免認證的 replay CDN。`--episode` 只靠這個，完全不碰 kagglesdk。
CDN = "https://www.kaggleusercontent.com/episodes/{}.json"

#: Kaggle 的內部 episode 查詢端點。**也免認證**（2026-08-19 實測）。
#: 只吃兩種 filter，其他一律 400 "You must specify at least one ID filter"：
#:     {"ids": [<episode_id>, ...]}   查這些對局的 metadata
#:     {"submissionId": <id>}         查這個 submission 打過的所有對局
#: 試過但**不支援**的：teamId、competitionId、episodeIds。
EPISODE_API = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"

#: 每次 API 呼叫之間停一下。這是別人的服務，不要打太兇。
API_SLEEP = 0.5

#: 預設落點。`temp/` 被 gitignore —— 一局 30 MB，不要掉進版控。
DEFAULT_OUT = os.path.join("temp", "episodes")

ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true", help="list my submissions (id/date/score) and exit")
ap.add_argument("--submission", type=int, default=None, help="submission id to fetch episodes for")
ap.add_argument("--episode", type=int, nargs="+", default=None,
                help="直接下載這些 episode id（免認證，可抓別人的對局）")
ap.add_argument("--crawl", type=int, default=None, metavar="SUBMISSION_ID",
                help="從一個 submission id 出發，順著對局圖爬出高分隊伍的對局")
ap.add_argument("--min-score", type=float, default=2800.0,
                help="爬蟲只收「雙方 rating 都 >= 這個值」的對局（預設 2800）")
ap.add_argument("--max-episodes", type=int, default=200,
                help="爬蟲最多收集幾局（預設 200）")
ap.add_argument("--dry-run", action="store_true",
                help="爬蟲只列出找到的 episode id，不下載")
ap.add_argument("--out", default=None, help=f"輸出目錄（預設 {DEFAULT_OUT}）")
ap.add_argument("--comp", default="kaggriculture")
ap.add_argument("--agent-logs", action="store_true",
                help="ALSO download OUR agent's stdout/stderr per episode -> <id>-<idx>.json. "
                     "The replay JSON does NOT carry agent output -- this is the only place "
                     "the CONFIG banner and the MCTS_STATS_EVERY 'stats[N]:' lines survive, "
                     "so it is the only way to confirm WHICH model/code actually ran and "
                     "whether the opponent model engaged (opp_names). Owner-only: asking for "
                     "the other seat's index returns 403.")
args = ap.parse_args()

if not args.list and args.submission is None and not args.episode and args.crawl is None:
    ap.error("需要 --list、--submission <id>、--episode <id> ... 或 --crawl <submission_id>")


def fetch_one(episode_id, out_dir):
    """從免認證 CDN 抓一局。回傳 'got' / 'skip' / 'fail'。

    已經抓過的跳過（可續傳）。門檻用 1000 bytes 而不是「檔案存在」——
    失敗時 CDN 會回一小段錯誤頁，那個也會被 urlretrieve 寫成檔案。
    """
    path = os.path.join(out_dir, f"{episode_id}.json")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return "skip"
    try:
        urllib.request.urlretrieve(CDN.format(episode_id), path)
    except Exception as exc:
        print(f"  FAIL {episode_id}: {exc}", file=sys.stderr)
        return "fail"
    if os.path.getsize(path) <= 1000:
        # 抓到錯誤頁而不是 replay。留著會讓下次重跑誤判成「已下載」。
        print(f"  FAIL {episode_id}: 回應只有 {os.path.getsize(path)} bytes，不是 replay",
              file=sys.stderr)
        os.remove(path)
        return "fail"
    return "got"


def episode_api(body):
    """打 ListEpisodes。回傳 dict，失敗回 None（不中斷整批爬取）。"""
    req = urllib.request.Request(
        EPISODE_API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        print(f"  API {body} -> {exc.code} {exc.read(300).decode('utf-8','replace')}",
              file=sys.stderr)
    except Exception as exc:
        print(f"  API {body} -> {type(exc).__name__}: {exc}", file=sys.stderr)
    return None


def crawl(seed_submission, min_score, max_episodes):
    """從一個 submission id 出發，廣度優先走對局圖。

    每一局的 `agents` 帶著對手的 `submissionId` 和 `updatedScore`，所以從
    任何一個 submission 都能走到跟它對打過的所有隊伍，再往外擴。

    只收「**雙方** rating 都 >= min_score」的對局 —— 強者打弱者的那局，
    強者的動作是在沒有競爭壓力下做的，當訓練資料會學到錯的市場行為
    （2026-08-19 實測：同一份 replay 對弱對手拿 $120,015，對等強對手
    只有 $117,554）。

    回傳 (episode_ids, teams)，teams 是 {teamId: (name, best_score)}。
    """
    queue = [seed_submission]
    seen_submissions = {seed_submission}
    episodes = {}
    teams = {}

    while queue and len(episodes) < max_episodes:
        submission_id = queue.pop(0)
        data = episode_api({"submissionId": submission_id})
        time.sleep(API_SLEEP)
        if not data:
            continue

        for team in data.get("teams", []):
            prev = teams.get(team["id"], ("", 0.0))[1]
            teams[team["id"]] = (team.get("teamName", ""), prev)

        for ep in data.get("episodes", []):
            agents = ep.get("agents", [])
            scores = [a.get("updatedScore") for a in agents]
            if len(agents) != 2 or any(s is None for s in scores):
                continue

            for a in agents:
                tid = a.get("teamId")
                if tid in teams:
                    name, best = teams[tid]
                    teams[tid] = (name, max(best, a.get("updatedScore") or 0.0))

            if str(ep.get("state", "")).endswith("COMPLETED") and min(scores) >= min_score:
                episodes[ep["id"]] = tuple(round(s, 1) for s in scores)
                if len(episodes) >= max_episodes:
                    break

            # 往外擴：跟高分隊伍對打過的 submission 值得再查。
            for a in agents:
                sid = a.get("submissionId")
                if (sid and sid not in seen_submissions
                        and (a.get("updatedScore") or 0.0) >= min_score):
                    seen_submissions.add(sid)
                    queue.append(sid)

        print(f"  查過 {len(seen_submissions) - len(queue)} 個 submission，"
              f"收集 {len(episodes)} 局，待查 {len(queue)}", flush=True)

    return episodes, teams


if args.crawl is not None:
    found, teams = crawl(args.crawl, args.min_score, args.max_episodes)
    print(f"\n找到 {len(found)} 局（雙方 rating 都 >= {args.min_score}）")
    print("\n看到的隊伍（rating 由高到低）：")
    for tid, (name, score) in sorted(teams.items(), key=lambda kv: -kv[1][1])[:20]:
        print(f"  {score:8.1f}  {name}  (teamId {tid})")

    if args.dry_run:
        print("\nepisode id：")
        print(" ".join(str(i) for i in sorted(found)))
        sys.exit(0)

    out = args.out or DEFAULT_OUT
    os.makedirs(out, exist_ok=True)
    tally = {"got": 0, "skip": 0, "fail": 0}
    for i, episode_id in enumerate(sorted(found), 1):
        tally[fetch_one(episode_id, out)] += 1
        if i % 10 == 0:
            print(f"  {i}/{len(found)}  下載 {tally['got']}，已有 {tally['skip']}，"
                  f"失敗 {tally['fail']}", flush=True)
    print(f"done -> {out}   下載 {tally['got']}，已有 {tally['skip']}，失敗 {tally['fail']}")
    sys.exit(0)


# --- 直接照 episode id 抓：不碰 kagglesdk，也不需要任何認證 ---------------
if args.episode:
    out = args.out or DEFAULT_OUT
    os.makedirs(out, exist_ok=True)
    tally = {"got": 0, "skip": 0, "fail": 0}
    for episode_id in args.episode:
        tally[fetch_one(episode_id, out)] += 1
        if tally["got"] and tally["got"] % 10 == 0:
            print(f"  downloaded {tally['got']}...", flush=True)
    print(f"done -> {out}   下載 {tally['got']}，已有 {tally['skip']}，失敗 {tally['fail']}")
    print("接著跑：python -m tools.extract_episode "
          f"{os.path.join(out, str(args.episode[0]))}.json")
    sys.exit(0)


# --- 以下走認證 API：只查得到我們自己的 submission -------------------------
from kagglesdk import KaggleClient                                  # noqa: E402
from kagglesdk.competitions.types.competition_api_service import (  # noqa: E402
    ApiListSubmissionEpisodesRequest,
    ApiListSubmissionsRequest,
)

with KaggleClient() as client:
    api = client.competitions.competition_api_client

    if args.list:
        req = ApiListSubmissionsRequest()
        req.competition_name = args.comp
        subs = api.list_submissions(req).submissions
        print(f"{'id':>10}  {'date':<20}{'score':>8}  description")
        for s in subs:
            date = str(getattr(s, "date", ""))[:19]
            score = getattr(s, "public_score", None) or ""
            print(f"{s.ref:>10}  {date:<20}{score:>8}  {getattr(s, 'description', '')}")
        sys.exit(0)

    req = ApiListSubmissionEpisodesRequest()
    req.submission_id = args.submission
    eps = api.list_submission_episodes(req).episodes

done = [e for e in eps if str(getattr(e, "state", "")).endswith("COMPLETED")]
w = l = d = 0
for e in done:
    mine = [a for a in e.agents if getattr(a, "submission_id", None) == args.submission]
    other = [a for a in e.agents if getattr(a, "submission_id", None) != args.submission]
    if mine and other:
        mr, orr = mine[0].reward or 0, other[0].reward or 0
        w, l, d = w + (mr > orr), l + (mr < orr), d + (mr == orr)
print(f"submission {args.submission}: {len(eps)} episodes ({len(done)} completed)  "
      f"W={w} L={l} D={d}  win%={w / max(w + l, 1):.0%}")

out = args.out or os.path.join(DEFAULT_OUT, f"submission_{args.submission}")
os.makedirs(out, exist_ok=True)
got = skip = fail = 0
for e in done:
    path = os.path.join(out, f"{e.id}.json")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        skip += 1
        continue
    try:
        urllib.request.urlretrieve(f"https://www.kaggleusercontent.com/episodes/{e.id}.json", path)
        got += 1
        if got % 10 == 0:
            print(f"  downloaded {got}...", flush=True)
    except Exception as ex:
        fail += 1
        print(f"  FAIL {e.id}: {ex}", file=sys.stderr)
print(f"done -> {out}\\  downloaded {got}, skipped(existing) {skip}, failed {fail}")

if args.agent_logs:
    # Separate pass, separate client: the replay pull above is an unauthenticated CDN fetch,
    # while agent logs go through the authenticated API (and are visible ONLY to the agent's
    # owner -- the opponent's index 403s, which is also how you can tell you picked the wrong
    # one). The SDK declares the return type as FileDownload, but FileDownload.prepare_from()
    # just hands back the raw requests.Response, so the bytes are on .content.
    from kagglesdk.competitions.types.competition_api_service import ApiGetEpisodeAgentLogsRequest

    lgot = lskip = lfail = 0
    with KaggleClient() as client:
        api = client.competitions.competition_api_client
        for e in done:
            idx = next((i for i, a in enumerate(e.agents)
                        if getattr(a, "submission_id", None) == args.submission), None)
            if idx is None:
                lfail += 1
                print(f"  FAIL {e.id}: our submission not among its agents", file=sys.stderr)
                continue
            path = os.path.join(out, f"{e.id}-{idx}.json")
            if os.path.exists(path) and os.path.getsize(path) > 100:
                lskip += 1
                continue
            try:
                req = ApiGetEpisodeAgentLogsRequest()
                req.episode_id = e.id
                req.agent_index = idx
                with open(path, "wb") as fh:
                    fh.write(api.get_episode_agent_logs(req).content)
                lgot += 1
            except Exception as ex:
                lfail += 1
                print(f"  FAIL agent-log {e.id}-{idx}: {ex}", file=sys.stderr)
    print(f"agent logs -> {out}\\  downloaded {lgot}, skipped(existing) {lskip}, failed {lfail}")
