"""畫一局裡九個產品的市價變化。

資料來源是 agent 的每回合 log（`KAGGRI_LOG_LEVEL >= 2` 才有 `prices` 欄位），
所以跑的時候要開 log：

    python -m eval.runner --a gen0 --b ref-v1 --games 1 --no-swap --log-level 2
    python tools/plot_prices.py                      # 自動抓最新的 run
    python tools/plot_prices.py temp/2026...-xxx/    # 指定 run 目錄
    python tools/plot_prices.py --tag seed0003       # 一個 log 檔裡有多局時挑一局

## 為什麼是九張小圖不是一張九條線

各產品的 base 差 10 倍（WHEAT $25 vs MELON $250），畫在同一組座標軸上
低價的那幾條會被壓成一條直線。而且九條線超過調色盤能安全區分的數量。
每個產品一張圖、各自的 y 軸，就不需要圖例，也沒有辨色問題。

每張圖有一條 base 的虛線 —— 價格高於 base 代表市場缺貨（我們沒供上），
低於 base 代表超供（我們砸了自己的盤）。
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

@contextlib.contextmanager
def _silenced():
    """fd 層級靜音。open_spiel 是 C++ extension，直接寫檔案描述子，
    `redirect_stdout` 只換 Python 層的 `sys.stdout`，攔不到。"""
    with open(os.devnull, "w") as devnull:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


with _silenced():
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        MARKET_PARAMS, PRODUCTS,
    )

# --- 顏色 ---
INK = "#0b0b0b"
INK_SOFT = "#52514e"
LINE = "#2a78d6"
GRID = "#e4e3df"
BASE_LINE = "#9a9992"
SURFACE = "#fcfcfb"

#: **九個產品九個顏色。** 在 OKLCH 空間均勻取九個色相、明度交錯，
#: 掃過 27 組候選挑最差配對最好的一組。
#:
#: ⚠️ 誠實的數字：**沒有任何九色組合能通過全配對驗證**，這組也是。
#: 最差 CVD ΔE 2.2、一般視覺 ΔE 10.9（下限 15）。試過的其他組更差 ——
#: 文件預設的八色 + 青是一般視覺 7.1，手挑的幾組是 8.9~12.3 但 CVD 更低。
#:
#: 所以顏色**不是唯一**的辨識依據：每個產品另外配一組線型，線末有直接標籤，
#: 底下還印統計表。要精確比較兩三個產品用 `--items`。
#: 動物用**牠產物的**顏色 —— 上下對照時同一個顏色講的是同一條供應鏈：
#: 牛的數量變化直接對得上 MILK 的價格。
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}

COLORS = ["#ef774b", "#854800", "#92ac18", "#007136", "#00bac5",
          "#005ea6", "#8d90ff", "#7d318e", "#eb6f9a"]

#: 線型當第二層編碼。實線 / 長虛 / 點 / 點劃 / 短虛 …… 細線上也分得出來。
DASHES = [
    "solid",
    (0, (7, 2.5)),
    (0, (1.5, 1.8)),
    (0, (7, 2, 1.5, 2)),
    (0, (3.5, 1.8)),
    (0, (1.5, 1.5, 6, 1.5)),
    (0, (11, 2.5)),
    (0, (2.5, 1.5)),
    (0, (6, 2, 1.5, 2, 1.5, 2)),
]


def item_style(item):
    """回傳 (顏色, 線型)。用 PRODUCTS 的固定索引 —— 換一組產品畫，
    既有產品的樣式不會跑掉。"""
    i = list(PRODUCTS).index(item)
    return COLORS[i % len(COLORS)], DASHES[i % len(DASHES)]


def find_logs(target=None):
    """找 log 檔。回傳**整個 run 的所有** .jsonl。

    一局落在哪個 worker 檔是排程決定的，檔名看不出來，所以不能只讀一個檔。
    """
    if target:
        p = Path(target)
        if p.is_file():
            return [p]
        found = sorted(p.glob("logs/*.jsonl")) or sorted(p.glob("*.jsonl"))
        if not found:
            raise SystemExit(f"{p} 底下找不到 .jsonl")
        return found

    runs = sorted(glob.glob(str(REPO_ROOT / "temp" / "*" / "logs")),
                  key=os.path.getmtime, reverse=True)
    for run in runs:
        found = sorted(Path(run).glob("*.jsonl"))
        if found:
            return found
    raise SystemExit(
        "temp/ 底下沒有 log。先跑一局並開 log：\n"
        "  python -m eval.runner --a gen0 --b ref-v1 --games 1 "
        "--no-swap --log-level 2")


def read_prices(paths, tag=None, player=0):
    """回傳一局的資料。

    ⚠️ **價格取 `player` 那一邊的（市場共用，兩邊看到的一樣），
    但賣出要兩邊都收。** 價格的崩盤是雙方倒貨相加造成的，只看自己那邊
    會把對手砸的盤算到自己頭上。
    """
    by_tag = {}
    for path in paths:
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            pid = r.get("player")
            prices = r.get("prices")
            if not prices:
                raise SystemExit(
                    "log 裡沒有 prices 欄位 —— 要用 --log-level 2 以上跑才會記。")
            e = by_tag.setdefault(r.get("tag", ""),
                                  {"path": path, "pts": [], "sells": {},
                                   "animals": {}, "players": set()})
            e["players"].add(pid)
            if pid == player:
                e["pts"].append((r["day"] + r["hour"] / 24.0, prices))
                if r.get("animals") is not None and r.get("hour") == 0:
                    e["animals"][r["day"]] = dict(r["animals"])
            # 送出的 SELL 訂單，按天累加，**兩邊都收**。這是「送出的」不是
            # 成交的 —— 引擎會靜默中止（shed 沒貨、超過一回合 10 筆、
            # 對手同時在賣搶走庫存）。
            side = e["sells"].setdefault(pid, {})
            for o in r.get("action", {}).get("market", []):
                if o and o[0] == "SELL" and len(o) >= 3:
                    side.setdefault(o[1], {})
                    side[o[1]][r["day"]] = side[o[1]].get(r["day"], 0) + o[2]

    if not by_tag:
        raise SystemExit("log 裡一筆紀錄都沒有")

    if tag:
        hits = ([x for x in by_tag if x == tag]
                or [x for x in by_tag if x.startswith(tag)])
        if not hits:
            raise SystemExit(
                f"找不到 tag={tag!r}。這個 run 裡有 {len(by_tag)} 局：\n  "
                + "\n  ".join(sorted(by_tag)[:20]))
        if len(hits) > 1:
            raise SystemExit(
                f"tag={tag!r} 對到 {len(hits)} 局，要更精確：\n  "
                + "\n  ".join(sorted(hits)[:20]))
        chosen = hits[0]
    else:
        chosen = sorted(by_tag)[0]
        if len(by_tag) > 1:
            print(f"⚠️ 這個 run 有 {len(by_tag)} 局，預設畫 {chosen}。"
                  f"用 --tag 挑別局，--list 看全部。", file=sys.stderr)

    e = by_tag[chosen]
    if not e["pts"]:
        # ⚠️ 只有 `agents/gen0.py` 會寫 log，`agents/gen2_model.py` 一行都沒有。
        # 所以 `--a e2e --b <gen0 系>` 的 run，檔案裡只有 player 1；預設的
        # `--player 0` 會拿到空的 pts，接著在 `plot_combined` 的 `vals[-1]`
        # 炸成 IndexError（離真正的原因很遠）。這裡先擋掉。
        have = sorted(x for x in e["players"] if x is not None)
        if not have:
            raise SystemExit(f"{chosen} 的 log 裡沒有 player 欄位")
        raise SystemExit(
            f"{chosen} 的 log 裡沒有 player {player} 的紀錄"
            f"（有的是 player {have}）。加 --player {have[0]} 再跑一次。\n"
            "只有 agents/gen0.py 會寫 log，網路版不寫 —— 網路版那一邊本來就沒有資料。")
    e["pts"].sort(key=lambda q: q[0])
    days = [d for d, _ in e["pts"]]
    series = {item: [q[item] for _, q in e["pts"]] for item in PRODUCTS}
    return days, series, e["sells"], e["animals"], chosen, e["path"]


def list_tags(paths, player=0):
    out = {}
    for path in paths:
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("player") == player:
                out.setdefault(r.get("tag", ""), path.name)
    print(f"{len(out)} 局：")
    for t, fn in sorted(out.items()):
        print(f"  {t:<44} {fn}")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # matplotlib 預設的 DejaVu Sans 沒有 CJK，中文會變成方框。
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
                 "Noto Sans CJK TC", "SimHei"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
    return plt


def _tidy(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)


def _label_ends(ax, x, ends, fmt, log=False):
    """在線的右端標名字，並且**把重疊的往下推開**。

    `ends` 是 [(y, 名字), ...]。由上而下走，每個標籤至少跟上一個差
    一個字高；不夠就往下擠。線本身還在原位，標籤只是移開 ——
    所以順序仍然對應線的高低。
    """
    import math
    lo, hi = ax.get_ylim()
    if log:
        # log 軸要在對數空間裡錯開，不然低價那端會被擠成一團
        lo, hi = math.log10(max(lo, 1e-9)), math.log10(hi)
    gap = (hi - lo) * 0.028
    placed = []
    for y, name in sorted(ends, reverse=True):
        pos = math.log10(max(y, 1e-9)) if log else y
        y_lab = pos if not placed else min(pos, placed[-1] - gap)
        placed.append(y_lab)
        y_draw = 10 ** y_lab if log else y_lab
        ax.annotate(fmt(y, name), xy=(x, y_draw), xytext=(8, 0),
                    textcoords="offset points", va="center",
                    color=INK, fontsize=8.5, annotation_clip=False)


def plot_combined(days, series, sells, animals, out_path, title, items=None,
                  player=0, sides=("", "")):
    """一張圖：上面是價格（除以 base 正規化），下面是每天送出的 SELL 數量。

    ## 為什麼要除以 base

    各產品的 base 差 10 倍（WHEAT $25 vs MELON $250），直接畫在同一組座標軸
    上低價的會被壓成一條線。除以 base 之後 1.0 就是「跟開局一樣」，
    **> 1 代表市場缺貨（我們沒供上）、< 1 代表超供（砸了自己的盤）**，
    九個產品可以直接互相比較。

    ## 為什麼下面那格要對齊

    價格的斷崖是自己倒貨造成的。兩格共用 x 軸，一眼就對得起來：
    上面哪天跳水，下面同一天就是那筆 SELL。
    """
    plt = _mpl()
    items = items or list(PRODUCTS)

    n_rows = 3 if animals else 2
    ratios = [1.5, 1, 0.6] if animals else [1.5, 1]
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 13.5 if animals else 12), facecolor=SURFACE,
        sharex=True, gridspec_kw={"height_ratios": ratios, "hspace": 0.07})
    ax_p, ax_s = axes[0], axes[1]
    ax_a = axes[2] if animals else None
    fig.suptitle(title, color=INK, fontsize=13, x=0.052, ha="left", y=0.985)
    if any(sides):
        fig.text(0.052, 0.955, f"下格：往上 = {sides[0]}（我），往下 = {sides[1]}（敵）",
                 color=INK_SOFT, fontsize=9)

    # --- 上：實際市價（元）---
    #     y 軸取 log：價格跨 $1 到 $272，線性軸會把便宜的那幾條壓成一條。
    #     log 軸上「掉一半」不管在哪個價位看起來都一樣長，比較的是變化幅度。
    ends = []
    for item in items:
        color, dash = item_style(item)
        vals = series[item]
        ax_p.plot(days, vals, color=color, lw=2, ls=dash,
                  solid_capstyle="round", zorder=3, label=item)
        ends.append((vals[-1], item))

    ax_p.set_yscale("log")
    ax_p.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 300])
    ax_p.get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _label_ends(ax_p, days[-1], ends, fmt=lambda y, it: f"{it} ${y:,.0f}",
                log=True)

    ax_p.set_ylabel("市價", color=INK_SOFT, fontsize=9)
    ax_p.set_xlim(0, max(days))
    _tidy(ax_p)
    ax_p.legend(ncol=9, fontsize=7.5, frameon=False,
                loc="lower left", bbox_to_anchor=(0, 1.005),
                labelcolor=INK_SOFT, columnspacing=1.1, handlelength=2.2,
                handletextpad=0.5)

    # --- 下：每天送出的 SELL 數量，**上下鏡像** ---
    #     價格是兩邊倒貨相加造成的，只畫自己那邊會把對手砸的盤算到自己頭上。
    #     第一版用半透明畫對手，同色同線型只差透明度，看起來像「沒名字的線」。
    #     改成自己往上、對手往下，中間一條零線 —— 不用靠透明度分。
    mine = sells.get(player, {})
    theirs = sells.get(1 - player, {})
    s_ends = []
    for item in items:
        color, dash = item_style(item)
        for src, sign in ((mine, 1), (theirs, -1)):
            per_day = src.get(item, {})
            if not per_day:
                continue
            xs = sorted(per_day)
            ys = [sign * per_day[d] for d in xs]
            ax_s.plot(xs, ys, color=color, lw=1.7, ls=dash,
                      drawstyle="steps-mid", zorder=3)
            if sign == 1:
                s_ends.append((ys[-1], item))

    ax_s.axhline(0, color=INK_SOFT, lw=1, zorder=4)
    lim = max(abs(v) for v in ax_s.get_ylim())
    ax_s.set_ylim(-lim, lim)
    ax_s.get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):,.0f}"))
    _label_ends(ax_s, max(days), s_ends,
                fmt=lambda y, it: f"{it} 我{sum(mine[it].values()):,}"
                                  f" / 敵{sum(theirs.get(it, {}).values()):,}")

    ax_s.set_ylabel(f"當天送出的 SELL 數量\n↑ {sides[0]}   ↓ {sides[1]}",
                    color=INK_SOFT, fontsize=9)
    ax_s.set_xlabel("day", color=INK_SOFT, fontsize=9)
    _tidy(ax_s)

    # 不用 tight_layout —— 它為了容納右邊的直接標籤會把整張圖往內縮，
    # 左邊和下面就多出一大塊白。右邊留 0.14 給標籤，其餘壓到最小。
    # --- 動物：活著的隻數 ---
    #     動物會餓死、建物種類蓋下去就綁死，光看買入訂單看不出實際養了什麼。
    if ax_a is not None:
        xs = sorted(animals)
        a_ends = []
        for species in sorted({s for d in animals.values() for s in d}):
            ys = [animals[d].get(species, 0) for d in xs]
            color, dash = item_style(ANIMAL_PRODUCT.get(species, species))
            ax_a.plot(xs, ys, color=color, lw=1.9, ls=dash,
                      drawstyle="steps-post", zorder=3)
            a_ends.append((ys[-1], f"{species}→{ANIMAL_PRODUCT[species]}"))
        ax_a.set_ylim(bottom=0)
        _label_ends(ax_a, max(xs), a_ends, fmt=lambda y, n: f"{n} {y:.0f}")
        ax_a.set_ylabel("活著的動物", color=INK_SOFT, fontsize=9)
        _tidy(ax_a)
        ax_a.set_xlabel("day", color=INK_SOFT, fontsize=9)
        ax_s.set_xlabel("")

    fig.subplots_adjust(left=0.052, right=0.86, top=0.905, bottom=0.045)
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot(days, series, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # matplotlib 預設的 DejaVu Sans 沒有 CJK，中文會變成方框。
    # 挑第一個系統裝得到的中文字型；都沒有就退回預設（標題會缺字但圖還是對的）。
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
                 "Noto Sans CJK TC", "SimHei"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), facecolor=SURFACE)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left", y=0.985)

    for ax, item in zip(axes.flat, PRODUCTS):
        base = MARKET_PARAMS[item]["base"]
        vals = series[item]
        ax.set_facecolor(SURFACE)

        ax.axhline(base, color=BASE_LINE, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.plot(days, vals, color=LINE, lw=2, solid_capstyle="round", zorder=3)

        ax.set_title(f"{item}   base ${base}", color=INK, fontsize=10,
                     loc="left", pad=6)
        ax.set_xlim(0, max(days))
        ax.set_ylim(0, max(max(vals), base) * 1.12)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)

        # 直接標在線上：期末價，以及它是 base 的幾倍
        end = vals[-1]
        ax.annotate(f"${end:,.0f}  ({end / base:.2f}× base)",
                    xy=(days[-1], end), xytext=(-4, 6),
                    textcoords="offset points", ha="right",
                    color=INK, fontsize=9)

    for ax in axes.flat[len(PRODUCTS):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("day", color=INK_SOFT, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def summarise(days, series):
    rows = []
    for item in PRODUCTS:
        v = series[item]
        base = MARKET_PARAMS[item]["base"]
        rows.append((item, base, min(v), max(v), v[-1], v[-1] / base))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="畫一局裡九個產品的市價變化")
    ap.add_argument("target", nargs="?", help="run 目錄或 .jsonl 路徑（省略 = temp/ 最新的）")
    ap.add_argument("--tag", help="挑哪一局。完整值長 seed0003_gen0_vs_ref-v1，給前綴也行")
    ap.add_argument("--list", action="store_true", help="列出這個 run 有哪些局、各在哪個檔")
    ap.add_argument("--player", type=int, default=0, help="讀哪一方的 log（價格是共用的）")
    ap.add_argument("--mode", choices=("combined", "facets"), default="combined",
                    help="combined = 一張圖（價格除以 base + 送出的 SELL）；"
                         "facets = 九張小圖各自的 y 軸")
    ap.add_argument("--items", help="只畫這幾個產品，逗號分隔")
    ap.add_argument("-o", "--out", help="輸出的 png 路徑")
    args = ap.parse_args(argv)

    paths = find_logs(args.target)
    if args.list:
        list_tags(paths, args.player)
        return 0

    days, series, sells, animals, tag, path = read_prices(
        paths, args.tag, args.player)
    items = [s.strip().upper() for s in args.items.split(",")] if args.items else None
    if items:
        bad = [i for i in items if i not in PRODUCTS]
        if bad:
            raise SystemExit(f"沒有這些產品：{bad}，可選 {list(PRODUCTS)}")
    out = (Path(args.out) if args.out
           else path.parent.parent / f"prices_{args.mode}_{tag}.png")
    if args.mode == "combined":
        body = tag.split("_", 1)[1]
        names = body.split("_vs_") if "_vs_" in body else [body, "?"]
        sides = (names[args.player], names[1 - args.player])
        plot_combined(days, series, sells, animals, out,
                      f"市價、賣出、動物   {tag}", items, args.player, sides)
    else:
        plot(days, series, out, f"市價變化   {tag}")

    print(f"log      {path}")
    print(f"{'item':12s}{'base':>6}{'最低':>7}{'最高':>7}{'期末':>7}{'期末/base':>10}")
    for item, base, lo, hi, end, ratio in summarise(days, series):
        flag = "  ← 缺貨" if ratio > 1.2 else ("  ← 砸盤" if ratio < 0.5 else "")
        print(f"{item:12s}{base:>6}{lo:>7}{hi:>7}{end:>7}{ratio:>9.2f}x{flag}")
    print(f"\n圖 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
