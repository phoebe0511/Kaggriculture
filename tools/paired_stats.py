"""兩個 `temp/*/result.json` 的逐局配對統計 —— 判「這個改動真的有差嗎」。

    python -m tools.paired_stats temp/<runA> temp/<runB>
    python -m tools.paired_stats temp/<runA> temp/<runB> --label-a round6 --label-b gen1

repo 裡本來沒有這支，2026-08-24 那批 p 值是臨時算的。Stage 1（量 seed 抽樣誤差）
和 Stage 4（100 局判定）都要用它。

## 配對的鍵是 `(seed, a_slot)`，不是 seed

`eval/runner.py` 的配對種子每個 seed 跑兩局（正手 `a_slot=0`、反手 `a_slot=1`），
所以 `--games 50` 會產生 100 局、**100 個配對單位**。

`cash_a` 永遠是 `--a` 那一方的期末現金 —— `_play()` 回傳的是
`rewards[a_slot]`（`eval/runner.py:288`），對調那一局取的仍是 `--a`。所以兩個 run
只要 `--b` 相同，比 `cash_a` 就是在比「同一個對手、同一個 seed、同一個位置」下
兩個 A 方的表現。

🩸 **兩個 run 的 `--b` 必須是同一個對手**，否則配對沒有意義。這支會檢查並在
不一致時警告，但檢查的是 config 的 `name` 欄位 —— 名字一樣而 params 不同的話
抓不到。

## MDE 的公式

    MDE = (z[1-α/2] + z[1-β]) x sd / sqrt(n)          α=0.05 雙尾, power=0.8
        = 2.8016 x sd / sqrt(n)

意思是「這個樣本數下，真實差距要多大才有 80% 機率被這個檢定抓到」。
**實測差 < MDE = 判不出來，不等於沒有差別。**

⚠️ 2026-08-24 的 journal 也記過 MDE，但沒有寫公式。那邊的數字跟這裡不一定用
同一組 α / power，兩者不要直接對照。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: z[1-α/2] + z[1-β]，α=0.05 雙尾、power=0.8。
_Z_SUM = 1.959964 + 0.841621


def _result_path(target):
    """吃 run 目錄或 result.json 路徑，回傳 result.json 的路徑。"""
    path = Path(target)
    if path.is_dir():
        path = path / "result.json"
    if not path.is_file():
        raise SystemExit(f"找不到 {path}")
    return path


def load_run(target):
    """讀一個 run，回傳 `(cash_by_key, meta)`。

    `cash_by_key` 是 `{(seed, a_slot): cash_a}`，**作廢的局不收**
    （`error`、`cash` 是 None、`status` 不是 DONE）—— 判定標準跟
    `eval.runner.summarise` 那段一致。
    """
    path = _result_path(target)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    cash = {}
    dropped = 0
    for row in payload["results"]:
        bad = (
            row.get("error")
            or row.get("cash_a") is None
            or row.get("cash_b") is None
            or row.get("status_a") != "DONE"
            or row.get("status_b") != "DONE"
        )
        if bad:
            dropped += 1
            continue
        cash[(row["seed"], row["a_slot"])] = float(row["cash_a"])

    meta = {
        "path": str(path),
        "name_a": (payload.get("a") or {}).get("name", "?"),
        "name_b": (payload.get("b") or {}).get("name", "?"),
        "games": len(payload["results"]),
        "dropped": dropped,
    }
    return cash, meta


def wilcoxon_p(diffs):
    """Wilcoxon signed-rank 的雙尾 p。scipy 沒裝就回 None，不要自己實作。"""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 1:
        return None
    return float(wilcoxon(nonzero).pvalue)


def compare(run_a, run_b):
    """兩個 run 的逐局配對統計。A 減 B。"""
    cash_a, meta_a = load_run(run_a)
    cash_b, meta_b = load_run(run_b)

    shared = sorted(set(cash_a) & set(cash_b))
    if not shared:
        raise SystemExit(
            "兩個 run 沒有共同的 (seed, a_slot)。是不是 --seed0 或 --games 不一樣？")

    diffs = [cash_a[k] - cash_b[k] for k in shared]
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    mde = _Z_SUM * sd / math.sqrt(n) if n > 0 else float("inf")
    need = (_Z_SUM * sd / abs(mean)) ** 2 if mean else float("inf")

    return {
        "meta_a": meta_a,
        "meta_b": meta_b,
        "n_pairs": n,
        "only_in_a": len(set(cash_a) - set(cash_b)),
        "only_in_b": len(set(cash_b) - set(cash_a)),
        "mean_a": statistics.fmean(cash_a[k] for k in shared),
        "mean_b": statistics.fmean(cash_b[k] for k in shared),
        "mean_diff": mean,
        "median_diff": statistics.median(diffs),
        "sd_diff": sd,
        "sem": sd / math.sqrt(n) if n else float("inf"),
        "a_better": sum(1 for d in diffs if d > 0),
        "ties": sum(1 for d in diffs if d == 0),
        "b_better": sum(1 for d in diffs if d < 0),
        "wilcoxon_p": wilcoxon_p(diffs),
        "mde": mde,
        "n_for_observed": need,
        "diffs": diffs,
    }


def format_report(s, label_a=None, label_b=None):
    a = label_a or s["meta_a"]["name_a"]
    b = label_b or s["meta_b"]["name_a"]
    p = s["wilcoxon_p"]
    lines = [
        f"A = {a}   ({s['meta_a']['path']})",
        f"B = {b}   ({s['meta_b']['path']})",
        f"對手：A 打 {s['meta_a']['name_b']}、B 打 {s['meta_b']['name_b']}",
        "",
        f"配對局數      {s['n_pairs']}"
        + (f"   ⚠️ 只在 A 有 {s['only_in_a']}、只在 B 有 {s['only_in_b']}"
           if s["only_in_a"] or s["only_in_b"] else ""),
        f"平均現金      A {s['mean_a']:>12,.0f}    B {s['mean_b']:>12,.0f}",
        f"配對差 A−B    平均 {s['mean_diff']:>+12,.0f}   中位 {s['median_diff']:>+12,.0f}",
        f"              sd {s['sd_diff']:>14,.0f}   SEM {s['sem']:>+11,.0f}",
        f"A 較好        {s['a_better']}/{s['n_pairs']}"
        + (f"（平手 {s['ties']}）" if s["ties"] else ""),
        f"Wilcoxon p    {'（scipy 沒裝）' if p is None else f'{p:.5g}'}",
        f"MDE           {s['mde']:>12,.0f}   （α=0.05 雙尾、power=0.8）",
    ]

    drop = s["meta_a"]["dropped"] + s["meta_b"]["dropped"]
    if drop:
        lines.append(f"⚠️ 作廢局      A {s['meta_a']['dropped']}、B {s['meta_b']['dropped']}")
    if s["meta_a"]["name_b"] != s["meta_b"]["name_b"]:
        lines.append("🩸 兩個 run 的對手不同 —— 這樣配對沒有意義。")

    lines.append("")
    if p is None:
        lines.append("判定：scipy 沒裝，只能看敘述統計。")
    elif abs(s["mean_diff"]) < s["mde"]:
        lines.append(
            f"判定：⚪ 判不出來。實測差 {abs(s['mean_diff']):,.0f} < MDE {s['mde']:,.0f}，"
            f"要判到這個差距約需 {s['n_for_observed']:.0f} 個配對局。")
    elif p < 0.05:
        stronger = s["mean_diff"] > 0
        lines.append(f"判定：{'✅' if stronger else '❌'} A 確實"
                     f"{'較強' if stronger else '較弱'}（p={p:.5g}）。")
    else:
        lines.append(f"判定：⚪ 判不出來（p={p:.5g} ≥ 0.05）。")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="兩個 result.json 的逐局配對統計（A 減 B）")
    ap.add_argument("run_a", help="A 方的 run 目錄或 result.json")
    ap.add_argument("run_b", help="B 方的 run 目錄或 result.json")
    ap.add_argument("--label-a", help="報表上 A 的名字，預設取 result.json 的")
    ap.add_argument("--label-b", help="同上")
    ap.add_argument("--json", help="把統計另外寫成 JSON")
    args = ap.parse_args(argv)

    stats = compare(args.run_a, args.run_b)
    print(format_report(stats, args.label_a, args.label_b))

    if args.json:
        payload = {k: v for k, v in stats.items() if k != "diffs"}
        payload["diffs"] = stats["diffs"]
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n統計寫到 {args.json}")


if __name__ == "__main__":
    main()
