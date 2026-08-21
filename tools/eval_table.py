"""把 `temp/*/result.json` 掃成一張表，寫進 `docs/eval-results.md`。

    python -m tools.eval_table            # 重新產生整份
    python -m tools.eval_table --print    # 只印出來不寫檔

⚠️ **`--games N` 是 N 個配對種子，實際跑 2N 局**（`build_jobs(swap=True)`
每個 seed 正反各跑一次）。表裡的「局數」是實際局數，「種子」是 N。
docs/memory/journal 有幾處把兩者寫混了。

⚠️ **不同對手的分數不能互相比較。** 引擎的市場是兩家共用的，所以同一支 agent
對不同對手的期末現金差很多（實測 gen1 對 ladder-top-a 是 66,540、對 starter
是 119,701）。要比就看「相對」那一欄，或只比同一個對手的列。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "eval-results.md"

#: 需要備註才不會被誤讀的 run。key 是 run 目錄名。
NOTES = {
    "20260821-091513_gen2_model_vs_gen1":
        "⚠️ config 錯配：這是 ① 的權重，但走 `gen2_model.json`（參數是為模仿"
        "**老師**調的：n_structures 14、寫死 COW/SHEEP）。對齊版見下面的 `e2e` 那列。",
    "20260820-131642_dagger0_vs_ladder-top-a":
        "round 0 是 BC 不是 DAgger（policy == expert），拿 $0 是預期內的。",
}

#: 哪一條技術路線。`temp/` 只留 agent 名字，對應關係在這裡。
LINES = {
    "規則式": ("main", "main:agent", "gen1", "agents.gen1:act", "agents.gen0:act"),
    "v2 端到端（模仿老師）": ("gen2", "gen2_model", "gen2_capped"),
    "v3/v4 逐 unit target": ("gen3_target", "gen4_market", "dagger", "dagger0"),
    "v5 逐格 demand": ("gen4_demand", "gen4_rulemarket", "submission"),
    "① 端到端（模仿規則式）": ("e2e",),
}


def _line_of(name):
    for line, members in LINES.items():
        if name in members:
            return line
    return "其他"


def collect():
    rows = []
    for path in sorted((REPO_ROOT / "temp").glob("*/result.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        s = raw.get("summary") or {}
        results = raw.get("results") or []
        cash_a = [g["cash_a"] for g in results if "cash_a" in g]
        cash_b = [g["cash_b"] for g in results if "cash_b" in g]
        if not cash_a or not s.get("games"):
            # 整批作廢的 run。`summary["games"]` 是**沒作廢**的局數，
            # 而 `results` 仍然有 6 筆現金 0 的紀錄 —— 只看 cash_a 濾不掉。
            # 實例：20260820-171906 那次每局都 ModuleNotFoundError。
            continue
        argv = raw.get("argv") or []
        seeds = None
        if "--games" in argv:
            seeds = argv[argv.index("--games") + 1]
        # ⚠️ `raw["a"]` 是整個 spec dict（含 note、params、絕對路徑），
        # 名字要從 summary 拿。
        name_a, name_b = s.get("a"), s.get("b")
        if not name_a or not name_b:
            continue
        rows.append({
            "run": path.parent.name,
            "date": path.parent.name[:8],
            "a": name_a, "b": name_b,
            "line": _line_of(name_a),
            "games": len(results), "seeds": seeds,
            "w": s.get("wins"), "d": s.get("draws"), "l": s.get("losses"),
            "mean_a": statistics.mean(cash_a), "min_a": min(cash_a), "max_a": max(cash_a),
            "mean_b": statistics.mean(cash_b), "min_b": min(cash_b), "max_b": max(cash_b),
        })
    return rows


def render(rows):
    out = [
        "# Eval 分數記錄",
        "",
        "**這份檔案由 `python -m tools.eval_table` 產生，不要手改** ——",
        "它掃 `temp/*/result.json`，所以每跑一次 `eval.runner` 就重跑一次即可。",
        "",
        "## 讀這張表之前",
        "",
        "- **`--games N` 是 N 個配對種子，實際跑 2N 局**"
        "（`build_jobs(swap=True)` 每個 seed 正反各一次）。",
        "- 🩸 **不同對手的分數不能互相比較。** 引擎的市場是兩家共用的，",
        "  同一支 agent 對不同對手的期末現金差很多 —— 實測 gen1 對 `ladder-top-a`",
        "  是 66,540、對 `starter` 是 119,701。**要比就看「相對」那一欄，**",
        "  **或只比同一個對手的列。**",
        "- min / max 是平均值藏起來的東西：兩個平均相同的版本，可能一個每局都差不多、",
        "  另一個大部分還行但偶爾整局崩掉拿 0。",
        "- 勝率和現金是兩件事。配對種子下，兩個實力接近的 agent 可能",
        "  「每局都輸一點」= 現金差 4%、勝率 0%。",
        "",
    ]
    for line in list(LINES) + ["其他"]:
        group = [r for r in rows if r["line"] == line]
        if not group:
            continue
        out += [f"## {line}", "",
                "| 日期 | A | 對手 | 局數(種子) | 勝/和/負 | A 平均 | A min | A max"
                " | 對手平均 | 相對 | run |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in group:
            rel = (f"{r['mean_a'] / r['mean_b']:.0%}"
                   if r["mean_b"] else "—")
            seeds = f"({r['seeds']})" if r["seeds"] else ""
            out.append(
                f"| {r['date']} | `{r['a']}` | `{r['b']}` | {r['games']}{seeds} "
                f"| {r['w']}/{r['d']}/{r['l']} | {r['mean_a']:,.0f} "
                f"| {r['min_a']:,.0f} | {r['max_a']:,.0f} | {r['mean_b']:,.0f} "
                f"| **{rel}** | `{r['run']}` |")
        notes = [(r["run"], NOTES[r["run"]]) for r in group if r["run"] in NOTES]
        if notes:
            out.append("")
            for run, note in notes:
                out.append(f"- `{run}`：{note}")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", dest="show")
    args = ap.parse_args(argv)
    text = render(collect())
    if args.show:
        print(text)
        return
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(REPO_ROOT)}  {len(text.splitlines())} 行")


if __name__ == "__main__":
    main()
