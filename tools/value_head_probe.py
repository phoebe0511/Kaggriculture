"""value head 估得準不準 —— 決定「縮短 horizon」這條路走不走得通。

## 為什麼問這個

2026-08-27 量到：離線 search 的 `argmax(我方期末現金)` 挑到的是「這一局走運的
那條 30 天鏈」，不是比較好的動作（journal §6~§8）。跨延續的變異 sd 是 4,572，
而動作本身的差距只有幾百元，所以要 656 個不同的延續才分得出來。
減掉對手也救不了（§9，上限只有 15%）。

剩下的一條路是**縮短 horizon**：不要打完 550 個回合，只打 K 天，剩下用
value head 估。延續能把結果帶偏的空間從 550 回合縮到 K×24 回合。

代價是 value head 會估錯。**它的誤差要小於被砍掉的那段擺盪才划算。**

## 這支量什麼

    預測 = value_head(盤面) × 100_000
    真值 = 從這個盤面**不再搜、用 policy 打到底**的期末現金

殘差的 sd 就是 value head 的誤差尺度，拿去跟 4,572 比。

🩸 **真值一定要用 `--run` 給的 `q_base`，不要用 npz 的 `board_reward`。**
`board_reward`（`harness/rollout.py:185`）是「一路搜到底」的期末現金，而且
一局 120 個盤面共用同一個數字。value head 被訓練成猜「這個 policy 從這裡打
下去會拿多少」—— 用前者當真值會量到一個 −45,018 的假偏差（2026-08-27 踩過），
而且有效樣本數只有局數。`q_base` 是每個決策點各一個，才是對的東西。

🩸 **這量的是絕對誤差，不是相對排序。** search 真正需要的是「兩個很像的盤面
V 排不排得對」。V 可能整體偏 20,000 但相對判斷仍然對。所以這支
**只能否證、不能肯定** —— 殘差小不代表可行，殘差大才是明確的壞消息。

## 2026-08-27 的結論

```
在 search 盤面上（真值 q_base）  殘差 sd 19,674   對照「永遠猜平均」19,808
                                 -> 只解釋掉 1.4% 的變異
在它自己的驗證集上（round6 train.log 最後一行）
                                 value RMSE 0.1614 = 16,140 元（dummy 0.3894）
                                 -> 解釋掉 82.8%
```

**它不是壞掉，但無論如何都不夠**：最好的情況 16,140 元仍是要取代的 4,572 的
3.5 倍。誤差有結構（相鄰決策點殘差相關 +0.869），相減之後仍有 9,975。

## 用法

    python -m tools.value_head_probe data/search/dagger-round7         --run temp/search-game-20260826-233232
"""
from __future__ import annotations

import argparse
import glob
import io
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: 2026-08-27 實測（journal §6）：換一個延續，同一對候選的期末現金差的 sd。
#: value head 的殘差要**明顯小於**它，縮短 horizon 才划算。
CROSS_CONTINUATION_SD = 4572.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dirs", nargs="+")
    ap.add_argument("--weights", default="model/weights-e2e-round6.npz")
    ap.add_argument("--run", metavar="RUN_DIR",
                    help="對應的 tools.search_game 產物。給了就用 progress 裡的"
                         "`q_base` 當真值（每個決策點各一個），那才是 value head "
                         "該猜的東西")
    ap.add_argument("--out", default="temp/_value.txt")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    from serving.npz_forward import NumpyPolicy
    pol = NumpyPolicy(args.weights)

    files = []
    for d in args.data_dirs:
        files += sorted(glob.glob(str(Path(d) / "*.npz")))
    if not files:
        raise SystemExit("那些目錄裡沒有 npz")

    L = [f"權重 {args.weights}   encoder v{pol.encoder_version}"
         f"   width {pol.width} / blocks {pol.blocks}",
         f"檔案 {len(files)} 個"]

    # 🩸 `board_reward` 是「一路搜到底」的期末現金，那**不是** value head 該猜的
    # 東西 —— 它被訓練成猜「這個 policy 從這裡打下去會拿多少」。用前者當真值
    # 會量到一個 −45,018 的假偏差（2026-08-27 踩過）。
    # search 真正需要的估計量是 progress 裡的 `q_base`（從這個盤面不再搜、
    # 用 net policy 打到底的期末現金），而且**每個決策點各一個**，
    # 不像 board_reward 一局共用一個。
    qbase = {}
    if args.run:
        import json
        for f in sorted(glob.glob(str(Path(args.run) / "progress" / "*.jsonl"))):
            seed = int(Path(f).stem[4:])
            for line in io.open(f, encoding="utf-8"):
                r = json.loads(line)
                if r.get("phase") == "search" and r.get("q_base") is not None:
                    key = (seed, int(r["day"]) * 24 + int(r["hour"]))
                    qbase[key] = float(r["q_base"])
        L.append(f"真值來源 {args.run}/progress 的 q_base（{len(qbase)} 個決策點）")
    else:
        L.append("真值來源 npz 的 board_reward（🩸 那是一路搜到底的結果，"
                 "不是 value head 該猜的東西 —— 建議加 --run）")

    pred, truth, days, game = [], [], [], []
    missed = 0
    for gi, f in enumerate(files):
        z = np.load(f)
        if int(z["encoder_version"][0]) != pol.encoder_version:
            L.append(f"  ⚠️ {Path(f).name} 的 encoder v{int(z['encoder_version'][0])}"
                     f" 跟權重的 v{pol.encoder_version} 不合，跳過")
            continue
        sp = z["board_spatial"].astype(np.float32)
        sc = z["board_scalar"].astype(np.float32)
        rw = z["board_reward"].astype(np.float32)
        st = z["board_step"].astype(np.int32)
        seed = int(z["episode_id"][0])
        for i in range(len(rw)):
            if qbase:
                key = (seed, int(st[i]))
                if key not in qbase:
                    missed += 1
                    continue
                y = qbase[key]
            else:
                y = float(rw[i]) * 100_000.0
            feats = pol.trunk(sp[i], sc[i])
            pred.append(pol.value(feats) * 100_000.0)
            truth.append(y)
            days.append(int(st[i]) // 24)
            game.append(gi)
    if missed:
        L.append(f"  ⚠️ {missed} 個盤面在 progress 裡找不到對應的 q_base，已跳過")

    if not pred:
        raise SystemExit("沒有盤面被算到 —— encoder 版本全都不合？")

    pred = np.asarray(pred)
    truth = np.asarray(truth)
    days = np.asarray(days)
    game = np.asarray(game)
    resid = pred - truth
    n_games = len(set(game.tolist()))
    L += ["", f"盤面 {len(pred)} 個，來自 {n_games} 局"]
    if qbase:
        L.append("真值是逐決策點的 q_base，所以有效樣本數就是盤面數。")
    else:
        L.append("🩸 一局裡每個盤面共用同一個真值 —— 有效樣本數是"
                 f"**{n_games} 局**，不是 {len(pred)} 個盤面。")
    L.append("")

    base = truth.mean()
    L += ["## 誤差尺度",
          f"  真值   平均 {truth.mean():>10,.0f}   sd {truth.std(ddof=1):>10,.0f}"
          f"   範圍 {truth.min():,.0f} ~ {truth.max():,.0f}",
          f"  預測   平均 {pred.mean():>10,.0f}   sd {pred.std(ddof=1):>10,.0f}"
          f"   範圍 {pred.min():,.0f} ~ {pred.max():,.0f}",
          "",
          f"  value head 殘差   sd {resid.std(ddof=1):>10,.0f}"
          f"   平均 {resid.mean():>+10,.0f}（偏差）",
          f"  對照：永遠猜平均   sd {(truth - base).std(ddof=1):>10,.0f}",
          f"  -> 解釋掉的變異 1 − (殘差sd/對照sd)² = "
          f"{1 - (resid.std(ddof=1) / max(1e-9, (truth - base).std(ddof=1))) ** 2:>6.1%}"]

    try:
        from scipy.stats import pearsonr, spearmanr
        L += ["", "## 排序能力（真值逐盤面不同時才有意義）",
              f"  預測 vs 真值   Spearman r={spearmanr(pred, truth).statistic:+.3f}"
              f"   Pearson r={pearsonr(pred, truth).statistic:+.3f}"]
    except ImportError:
        pass

    L += ["", "## 判準",
          f"  跨延續的變異 sd（journal §6）= {CROSS_CONTINUATION_SD:,.0f}",
          f"  value head 殘差 sd          = {resid.std(ddof=1):,.0f}"
          f"   （{resid.std(ddof=1) / CROSS_CONTINUATION_SD:.1f} 倍）"]
    if resid.std(ddof=1) > CROSS_CONTINUATION_SD:
        L.append("  🩸 殘差比要取代的擺盪還大 -> 縮短 horizon 換來的誤差更糟，"
                 "這條路不通")
    else:
        L.append("  殘差比擺盪小 -> 值得再做相對排序的測試（這支證不了那件事）")

    L += ["", "## 依 day 分（越接近季末，結果越確定，殘差應該越小）",
          f"  {'day':<10}{'n':>6}{'殘差 sd':>12}{'殘差平均':>12}"]
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)):
        m = (days >= lo) & (days < hi)
        if not m.any():
            continue
        L.append(f"  {f'{lo}~{hi - 1}':<10}{int(m.sum()):>6}"
                 f"{resid[m].std(ddof=1):>12,.0f}{resid[m].mean():>+12,.0f}")
    L.append("  🩸 day 25+ 的殘差若仍然很大，代表 value head 連「快結束了、"
             "錢就這麼多」都學不會。")

    L += ["", "## 逐局（每局的真值只有一個）",
          f"  {'局':<6}{'真值':>12}{'預測平均':>12}{'誤差':>12}{'局內 sd':>12}"]
    order = sorted(set(game.tolist()), key=lambda g: truth[game == g].mean())
    for g in order:
        m = game == g
        L.append(f"  {g:<6}{truth[m].mean():>12,.0f}{pred[m].mean():>12,.0f}"
                 f"{pred[m].mean() - truth[m].mean():>+12,.0f}"
                 f"{pred[m].std(ddof=1):>12,.0f}")
    try:
        from scipy.stats import spearmanr
        gt = np.array([truth[game == g].mean() for g in order])
        gp = np.array([pred[game == g].mean() for g in order])
        rho = spearmanr(gt, gp)
        L.append(f"  逐局排序相關 Spearman r={rho.statistic:+.3f} p={rho.pvalue:.3g}"
                 f"   (n={len(gt)} 局)")
        L.append("    🩸 n 很小，這個 p 值判不了什麼，只當方向參考")
    except ImportError:
        pass

    text = "\n".join(L)
    io.open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
