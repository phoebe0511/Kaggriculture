"""把 Kaggle episode JSON 抽成只含動作序列的小檔，供 `agents/replay.py` 重播。

    python -m tools.extract_episode temp/93916293.json

原始 episode 檔一局約 30 MB（每一步的完整 observation 都在裡面），而且
`temp/` 被 gitignore —— 直接讀它有兩個問題：對戰產線每個 worker 行程都要
parse 一次 30 MB，而且同伴 clone 下來沒有這個檔。

抽出來的只有動作、seed 和驗收用的期末現金，一局約幾百 KB，進版控。

輸出格式（`config/episodes/<episode_id>.json`）：

    {
      "episode_id": 93916293,
      "seed": 372803889,               ← 重現這一局要用
      "module_version": "1.32.7",
      "configuration": {...},          ← 跟本機預設比對，不同就不能拿來當量尺
      "team_names": [...],
      "rewards": [117554.0, 117668.0], ← replay 對 replay 的驗收目標
      "statuses": ["DONE", "DONE"],
      "actions": [[<p0 第 0 步>, <p1 第 0 步>], ...]   ← 720 步
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "config" / "episodes"

#: 動作長這樣就是「什麼都沒做」，省下來的空間不值得少一致性，所以照原樣存。
_PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def extract(episode_path):
    """讀 episode JSON，回傳可以直接寫出去的 dict。"""
    with open(episode_path, encoding="utf-8") as f:
        data = json.load(f)

    steps = data["steps"]
    info = data.get("info") or {}

    actions = []
    for t, entries in enumerate(steps):
        row = []
        for p, entry in enumerate(entries):
            action = entry.get("action")
            # 引擎對非 dict 的 action 一律當成 PASS（kaggriculture.py:914-918），
            # 這裡先正規化，重播時就不用再判斷一次。
            row.append(action if isinstance(action, dict) else dict(_PASS))
        actions.append(row)

    return {
        "episode_id": info.get("EpisodeId"),
        "seed": info.get("seed"),
        "module_version": data.get("module_version"),
        "configuration": data.get("configuration"),
        "team_names": info.get("TeamNames"),
        "rewards": data.get("rewards"),
        "statuses": data.get("statuses"),
        "n_steps": len(steps),
        "actions": actions,
    }


def _sources(paths):
    """把檔案路徑和目錄都攤平成 episode JSON 的清單。"""
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"找不到 {p}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", nargs="+",
                    help="Kaggle 下載的 episode JSON，或裝著它們的目錄")
    ap.add_argument("--out", default=None,
                    help="單檔時的輸出路徑，預設 config/episodes/<id>.json")
    ap.add_argument("--engine", default="1.32.7",
                    help="只收這個引擎版本，設成空字串關掉過濾（預設 1.32.7）")
    args = ap.parse_args(argv)

    sources = _sources(args.episode)
    if args.out and len(sources) > 1:
        raise SystemExit("--out 只能配單一檔案")

    kept = skipped = 0
    versions = {}
    for src in sources:
        out = extract(src)
        versions[out["module_version"]] = versions.get(out["module_version"], 0) + 1

        # 1.29.x / 1.30.x 的 market 公式和 COW 成本都不同（journal 08-17 §A）。
        # 混進訓練資料的話網路會學到錯的價格模型，而且不會報錯。
        if args.engine and out["module_version"] != args.engine:
            print(f"  skip {src.name}：引擎 {out['module_version']} != {args.engine}",
                  file=sys.stderr)
            skipped += 1
            continue

        if out["seed"] is None:
            # 沒有 seed 就重現不了那一局，只能拿來看動作，不能當驗收基準。
            print(f"  ⚠️  {src.name}：info.seed 是 None，重現不了", file=sys.stderr)

        dest = Path(args.out) if args.out else OUT_DIR / f"{out['episode_id']}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        kept += 1

        if len(sources) == 1:
            print(f"{src} ({src.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"  → {dest} ({dest.stat().st_size / 1024:.0f} KB)")
            print(f"  episode {out['episode_id']}  seed {out['seed']}  "
                  f"engine {out['module_version']}")
            print(f"  {out['n_steps']} 步   rewards {out['rewards']}   "
                  f"statuses {out['statuses']}")
            print(f"  teams {out['team_names']}")

    if len(sources) > 1:
        print(f"抽取 {kept} 局 → {OUT_DIR}，跳過 {skipped} 局")
        print(f"引擎版本分布：{versions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
