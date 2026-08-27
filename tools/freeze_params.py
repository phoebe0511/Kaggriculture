"""把 CMA-ES 找到的參數落地成 `config/opponents/<name>.json`。

    python -m tools.freeze_params temp/cma/<時間戳>/best.json --name ref-v12
    python -m tools.freeze_params ... --name ref-v12 --note "CMA-ES 第 180 代"

## 🩸 為什麼不寫回 `DEFAULT_PARAMS`

`ref-v3`…`ref-v11` 各有 19 個 key **沒有展開**，會 fall through 到
`gen0.DEFAULT_PARAMS`。改那些預設值 = 所有凍結量尺同時位移，而且**不會報錯**
（`docs/CLAUDE.md`「凍結量尺的保護沒有真的裝上」）。所以新參數一律開新的
config JSON，51 項完整展開。

## `frozen` 這個 key 是有作用的，不是註記

`eval.runner.build_agent`（`eval/runner.py:135-137`）看到 `frozen` 就會補
`_replace_defaults: True`，讓這組參數**完全不跟今天的 `DEFAULT_PARAMS` 合併**。
沒有它的話，之後有人加一個預設開關，這個對手會在檔案完全沒被改動的情況下
偷偷變成另一個 agent。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools._quiet import silenced

# 🩸 `contracts` / `agents.gen0` 會連帶 import 引擎，而 open_spiel 掃遊戲
# 清單會噴 336 行到 stderr（fd 層級，`redirect_stderr` 攔不到）。
with silenced():
    from agents.gen0 import DEFAULT_PARAMS

REPO_ROOT = Path(__file__).resolve().parents[1]
OPPONENT_DIR = REPO_ROOT / "config" / "opponents"


def build_spec(params, name, note="", frozen=None, source=""):
    """組出 config JSON 的內容，並檢查 51 項有沒有齊。"""
    missing = sorted(set(DEFAULT_PARAMS) - set(params))
    extra = sorted(set(params) - set(DEFAULT_PARAMS) - {"_replace_defaults"})
    if missing:
        raise SystemExit(f"少了 {len(missing)} 個 key，凍結對手必須完整展開：{missing}")
    if extra:
        raise SystemExit(f"多了不認識的 key：{extra}")

    body = {k: v for k, v in params.items() if k != "_replace_defaults"}
    return {
        "name": name,
        "entry": "agents.gen0:act",
        "engine_version": "1.32.7",
        "frozen": frozen or time.strftime("%Y-%m-%d"),
        "source": source,
        "note": note or (
            f"CMA-ES 產出的參數，51 項完整展開。⚠️ 凍結後禁止修改，"
            f"要改請新增下一版。來源：{source}"),
        "params": body,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="把最佳參數落地成凍結對手 config")
    ap.add_argument("best", help="tools.param_search 產出的 best.json，或一份 params JSON")
    ap.add_argument("--name", required=True, help="對手名字，例如 ref-v12")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", help="輸出路徑，預設 config/opponents/<name>.json")
    ap.add_argument("--force", action="store_true", help="覆寫已存在的檔案")
    args = ap.parse_args(argv)

    with open(args.best, encoding="utf-8") as f:
        payload = json.load(f)
    # best.json 是 {"generation":..., "score":..., "x":..., "params": {...}}；
    # 直接給一份 params dict 也吃。
    params = payload.get("params", payload)

    out = Path(args.out) if args.out else OPPONENT_DIR / f"{args.name}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} 已經存在。凍結對手不該被覆寫 —— 要覆寫請加 --force。")

    spec = build_spec(params, args.name, args.note, source=str(args.best))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"寫到 {out}（{len(spec['params'])} 項參數，frozen={spec['frozen']}）")
    print(f"判定指令：\n"
          f"  python -m eval.runner --a {args.name} --b ladder-top-a --games 50 --workers 24\n"
          f"  python -m tools.paired_stats temp/<新的> "
          f"temp/20260824-172742_gen1_vs_ladder-top-a --label-a {args.name} --label-b gen1")


if __name__ == "__main__":
    main()
