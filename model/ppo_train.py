"""PPO 的訓練迴圈：rollout -> GAE -> 幾輪 minibatch 更新 -> 存檔。

    python -m model.ppo_train --smoke                 # 兩輪小的，確認接得起來
    python -m model.ppo_train --iters 200 --envs 16 --out model/artifacts/ppo0

rollout 在 `harness/ppo_rollout.py`，損失在 `model/ppo.py`。這支只負責把兩邊
串起來、記數字、存 checkpoint。

## 自對局的兩邊都收

同一份權重下兩邊，所以一個 env 產兩條軌跡，樣本數直接翻倍。reward 是零和的
（`model.ppo.step_rewards`），兩條加起來期望值是 0 —— **所以「平均 reward」
不是進步的指標**，看的是期末現金的平均值（兩邊一起變有錢＝真的在學會種田）
和對 gen0 的勝率。

## 一輪要多久（2026-08-28 實測，RTX 4060 / 20 實體核心）

    16 個 env    3.20 秒/局    -> 一輪 rollout 約 51 秒、23,000 步
    update       epochs=4, minibatch=512

## 🩸 起始權重

`model/weights-e2e-round*.npz` 是監督式學出來的，**value head 預測的是正規化
過的期末現金，不是 PPO 的 return**（零和、除以 `REWARD_SCALE`、有勝負
bonus）。從那裡熱啟動的話 value loss 一開始會很大，前幾輪的 advantage 幾乎
是雜訊。要嘛只載 trunk / policy head，要嘛就接受前幾輪是在重訓 value。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._quiet import silenced                      # noqa: E402

with silenced():
    import torch
    import contracts as C
    from harness.ppo_pool import RolloutPool
    from harness.ppo_rollout import (VecRollout, build_net,
                                     load_base_policy, load_league)
    from model import ppo


def save_checkpoint(path, net, args, meta):
    """跟 `model/train.py` 同一個形狀 —— 那邊的載入程式碼才吃得下。"""
    torch.save({
        "encoder_version": C.ENCODER_VERSION,
        "state_dict": net.state_dict(),
        "width": args.width,
        "blocks": args.blocks,
        "labels": "target",
        "trainer": "ppo",
        "argv": list(sys.argv[1:]),
        **meta,
    }, path)


def load_init(net, path, market_temp=1.0):
    """熱啟動。只吃形狀對得上的張量，對不上的留隨機初始化並回報。

    ## 🩸 `market_temp`：監督式的 market head 飽和到沒有梯度

    `market_present_out` 是 BCE 訓出來的，logit 極端。實測
    （`ckpt-e2e-round6`、seed 909、240 步、只看合法的 op）：

        |logit| 中位數 11.83   p95 69.17
        p < 0.01 或 > 0.99 的比例  81.4%
        sigmoid 斜率 p(1−p) 中位數  0.00001

    斜率 1e-5 等於**這個 head 收不到梯度**。而 market head 決定 HIRE /
    BUY_LAND / BUY_SEED / SELL —— 錢幾乎全在它手上。實測 `ppo-warm2` 跑了
    10 輪之後飽和比例還是 81.2%、斜率 0.00002，**完全沒動**。

    所以熱啟動時把 `market_present_out` 的權重和 bias 除以一個溫度。

    ⚠️ **greedy 的行為完全不變** —— `decode_market_orders` 的門檻是 0，
    除以正數不改變 logit 的正負號。變的只有訓練時取樣的隨機性和梯度大小。
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    src = dict(blob["state_dict"] if "state_dict" in blob else blob)
    if market_temp and market_temp != 1.0:
        for k in ("market_present_out.weight", "market_present_out.bias"):
            if k in src:
                src[k] = src[k] / float(market_temp)
    own = net.state_dict()
    taken, skipped = [], []
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            taken.append(k)
        else:
            skipped.append(k)
    net.load_state_dict(own)
    return taken, skipped


def train(args):
    torch.manual_seed(args.seed)
    net = build_net(args.width, args.blocks)
    if args.init:
        taken, skipped = load_init(net, args.init, args.market_temp)
        print(f"  熱啟動 {args.init}：吃了 {len(taken)} 個張量，"
              f"跳過 {len(skipped)} 個")
    net = net.to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    # 🩸 監督式的 value head 學的是「自己的期末現金 / 100k」（`model/train.py:12`），
    # PPO 的目標是「剩下的 margin / REWARD_SCALE」—— 不同的量，不是換單位。
    # 2026-09-06 實測（round4-tgt、4 局）：預測 +0.25~+1.02、實際 −2.80~−0.56，
    # 相關 −0.48~+0.29。而第 0 輪 `policy -0.036 / value 5.503`，value 大 150 倍，
    # 共用軀幹被它重塑、policy 的頭跟著壞掉（§88）。
    if args.reset_value:
        for m in net.value_head.modules():
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
        print("  value head 重新初始化（--reset-value）")
    # 暖身只更新 value head 自己那兩層，軀幹和 policy 的頭一個位元都不動。
    value_opt = torch.optim.Adam(net.value_head.parameters(),
                                 lr=args.value_lr or args.lr, eps=1e-5)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.jsonl"

    if not args.opponent:
        print("  ⚠️ 自對局：隨機初始的兩邊會一起把 $3000 花光，全程 0 對 0，"
              "reward 幾乎沒有訊號（2026-08-28 實測非零步只有 1.9%）。"
              "先用 --opponent 打 gen0。")

    pool = None
    if args.workers:
        # rollout 的 88% 是引擎和 gen0，純 CPU —— 多行程幾乎線性加速
        # （2026-08-28 實測 2.21 -> 0.38 秒/局）。網路只佔 12%。
        pool = RolloutPool(args.workers, args.envs, args.opponent,
                           args.width, args.blocks, args.episode_steps,
                           args.base_policy, args.base_side,
                           phi=args.phi, recognise=args.recognise)
        games = args.workers * args.envs
    else:
        opp, opp_names = (load_league(args.opponent) if args.opponent
                          else (None, []))
        games = args.envs

    try:
        base = (load_base_policy(args.base_policy)
                if args.base_policy and not args.workers else None)
        # greedy 評估在主行程跑，所以不管有沒有 pool 都要自己載一份。
        eval_opp = (load_league(args.opponent)[0][0] if args.opponent
                    and args.eval_every else None)
        eval_base = (load_base_policy(args.base_policy)
                     if args.base_policy and args.eval_every else None)
        run_loop(args, net, opt, out, log_path, pool, games,
                 None if args.workers else opp,
                 [] if args.workers else opp_names, base,
                 eval_opp, eval_base, value_opt)
    finally:
        if pool is not None:
            pool.close()
    return 0


def greedy_eval(args, net, opp, base, games, seed0):
    """用**上場的解碼**（argmax / logit>0）跑幾局，回傳 (我方現金, 勝率)。

    🩸 訓練曲線看的是取樣版，上場跑的是 greedy 版，**兩者不保證同方向**。
    2026-08-28 實測 `ppo-warm3`：取樣現金 12,999 -> 31,772（升），同一段訓練的
    greedy 是 44,060 -> 30,740（降）。所以 checkpoint 一定要用這個挑。

    在主行程單獨跑（worker pool 沒有 greedy 模式）—— 10 局約 22 秒。
    """
    from harness.ppo_rollout import VecRollout                # noqa: PLC0415

    was_training = net.training
    net.eval()
    vec = VecRollout(net, n_envs=games, seed0=seed0, device=args.device,
                     episode_steps=args.episode_steps, opponent=opp,
                     base_policy=base, greedy=True,
                     base_side=args.base_side)
    _steps, cash, _trajs = vec.run(collect=False)
    pairs = vec.our_cash(cash)
    ours = np.array([a for a, _ in pairs], dtype=np.float64)
    theirs = np.array([b for _, b in pairs], dtype=np.float64)
    if was_training:
        net.train()
    # 🩸 `ours.mean()` 的噪音幾乎全來自地圖難易：`cma1-g50-wt` 自打 40 局實測
    # 單局現金 sd 19,880，同局相減後 sd 只剩 2,567（10 局 SE 6,287 -> 812）。
    # 地圖好兩邊都好，相減就消掉。`margin` 也正好是 `--zero-sum` 的訓練目標
    # （`ppo.step_rewards` 的 `da - db`），所以評估指標跟訓練目標一致。
    # 既有欄位 `greedy_cash` 保留不動 —— ab-zs / ab-lam / ab-settle240 /
    # ab-lamsplit240 都是用它量的，換掉就不能比。
    return (float(ours.mean()), float(np.mean(ours > theirs)),
            float((ours - theirs).mean()))


def watch(dirs):
    """看一或多個訓練目錄的進度，不用等它跑完。

        python -m model.ppo_train --watch model/artifacts/ppo-a model/artifacts/ppo-b

    多給幾個目錄就並排比 —— 這是拿來做 A/B 的。**比之前先確認兩邊的
    `--seed` 一樣**，不然比到的是 run-to-run 的隨機差異（2026-08-31 踩過：
    同設定跑兩次，第 10 輪 greedy 差 31,138，而評估的 SE 只有約 7,000）。

    greedy 才是挑 checkpoint 的依據 —— 取樣現金會跟它反向（§44）。
    """
    rows = {}
    for d in dirs:
        f = Path(d) / "train.jsonl"
        if not f.is_file():
            print("  " + str(d) + "：沒有 train.jsonl")
            continue
        rows[Path(d).name] = [json.loads(l) for l in
                              io.open(f, encoding="utf-8") if l.strip()]
    if not rows:
        return 1
    for name, R in rows.items():
        last = R[-1]
        g = [r for r in R if "greedy_cash" in r]
        print("")
        tail = ("   greedy 評估 %d 次" % len(g)) if g else "   （還沒有 greedy）"
        print("%s   %d 輪%s" % (name, len(R), tail))
        print("  最後一輪  取樣現金 %9s  勝率 %.2f  ent %6.2f  ev %+.3f"
              % (format(last["cash_mean"], ",.0f"), last["win_rate"],
                 last["entropy"], last["explained_var"]))
        kl = last.get("kl_last_epoch")
        klc = ("KL(末 epoch) %.4f  " % kl) if kl is not None else ""
        print("  最後一輪  epochs %d  %sclipfrac %.3f"
              % (int(last["epochs_done"]), klc, last["clipfrac"]))
        if g:
            print("  %5s%13s%7s" % ("輪", "greedy 現金", "勝率"))
            for r in g[-12:]:
                print("  %5d%13s%7.2f" % (r["iter"],
                                          format(r["greedy_cash"], ",.0f"),
                                          r["greedy_win"]))
    if len(rows) > 1:
        names = list(rows)
        print("")
        print("並排（greedy 現金）—— 只有 --seed 相同才比得了")
        print("  " + "輪".rjust(5) + "".join(n[:16].rjust(18) for n in names))
        by = {n: {r["iter"]: r["greedy_cash"]
                  for r in R if "greedy_cash" in r} for n, R in rows.items()}
        for it in sorted({i for d in by.values() for i in d}):
            cells = "".join(
                (format(by[n][it], ",.0f") if it in by[n] else "-").rjust(18)
                for n in names)
            print("  %5d%s" % (it, cells))
    return 0


def run_loop(args, net, opt, out, log_path, pool, games, opp, opp_names,
             base=None, eval_opp=None, eval_base=None, value_opt=None):
    from harness.ppo_rollout import VecRollout                # noqa: PLC0415
    from model import ppo                                     # noqa: PLC0415

    for it in range(args.iters):
        t0 = time.perf_counter()
        # 每一輪換一批 seed —— 固定幾張地圖會學成背地圖。
        seed0 = args.seed0 + it * games
        if pool is not None:
            steps, pairs, trajs = pool.collect(net, seed0, game0=it * games)
            names = pool.names
        else:
            vec = VecRollout(net, n_envs=args.envs, seed0=seed0,
                             device=args.device,
                             episode_steps=args.episode_steps,
                             opponent=opp, opp_offset=it * games,
                             base_policy=base, base_side=args.base_side,
                             phi=args.phi, recognise=args.recognise)
            steps, cash, trajs = vec.run(collect=True)
            pairs = [(a, b, vec.opp_index(ei))
                     for ei, (a, b) in enumerate(vec.our_cash(cash))]
            names = opp_names
        t_roll = time.perf_counter() - t0

        t1 = time.perf_counter()
        # 122 MB 的 concatenate 不是免費的，單獨計時免得算進 rollout 或 update。
        batch = ppo.RolloutBatch(trajs, gamma=args.gamma, lam=args.lam,
                                 zero_sum=args.zero_sum,
                                 settle_step=args.settle_step,
                                 phi_weight=args.phi_weight,
                                 lam_early=args.lam_early,
                                 lam_split=args.lam_split,
                                 terminal_bonus=args.terminal_bonus)
        del trajs
        t_batch = time.perf_counter() - t1
        t1 = time.perf_counter()
        warming = it < args.value_warmup
        parts = ppo.update(net, value_opt if warming else opt, batch,
                           epochs=args.epochs,
                           minibatch=args.minibatch, seed=args.seed + it,
                           device=args.device, clip=args.clip,
                           vf_coef=args.vf_coef, ent_coef=args.ent_coef,
                           max_grad_norm=args.max_grad_norm,
                           target_kl=0.0 if warming else args.target_kl,
                           policy_coef=0.0 if warming else 1.0)
        t_upd = time.perf_counter() - t1

        ours = np.array([a for a, _b, _k in pairs], dtype=np.float64)
        theirs = np.array([b for _a, b, _k in pairs], dtype=np.float64)
        units = len(batch.unit_step) / max(len(batch), 1)
        row = {
            "iter": it,
            "steps": steps,
            "cash_mean": float(ours.mean()),
            "cash_max": float(ours.max()),
            "opp_cash_mean": float(theirs.mean()),
            "win_rate": float(np.mean(ours > theirs)),
            "units_per_step": float(units),
            "adv_std": float(batch.adv.std()),
            "ret_mean": float(batch.ret.mean()),
            "explained_var": round(batch.explained_variance(), 4),
            # league 每支各打了幾局、贏幾局 —— 只看總勝率的話，「克死其中一支、
            # 其餘全輸」跟「平均進步」長得一樣。
            "by_opp": {
                names[k] if k < len(names) else str(k): [
                    sum(1 for a, b, j in pairs if j == k and a > b),
                    sum(1 for _a, _b, j in pairs if j == k)]
                for k in sorted({j for _a, _b, j in pairs})
            } if len(names) > 1 else {},
            "roll_s": round(t_roll, 1),
            "batch_s": round(t_batch, 1),
            "upd_s": round(t_upd, 1),
            **{k: round(v, 5) for k, v in parts.items()},
        }
        print(f"  it {it:>4}  現金 {row['cash_mean']:>9,.0f} vs "
              f"{row['opp_cash_mean']:>9,.0f}  勝率 {row['win_rate']:.2f}  "
              f"unit/步 {units:>5.2f}  "
              f"kl {parts['approx_kl']:+.4f}/{parts['kl_last_epoch']:.4f}  clip {parts['clipfrac']:.3f}  "
              f"ep {int(parts['epochs_done'])}  "
              f"ent {parts['entropy']:>7.2f}  v {parts['value']:.3f}  "
              f"ev {row['explained_var']:+.3f}  "
              f"rollout {t_roll:>5.1f}s  batch {t_batch:>4.1f}s  "
              f"update {t_upd:>5.1f}s", flush=True)

        if args.eval_every and (it + 1) % args.eval_every == 0:
            g_cash, g_win, g_margin = greedy_eval(
                args, net, eval_opp, eval_base,
                args.eval_games, args.eval_seed0)
            row["greedy_cash"] = round(g_cash, 1)
            row["greedy_win"] = round(g_win, 3)
            row["greedy_margin"] = round(g_margin, 1)
            best = getattr(run_loop, "_best", float("-inf"))
            mark = ""
            if g_cash > best:
                run_loop._best = g_cash
                save_checkpoint(out / "best.pt", net, args, row)
                mark = "  <- best.pt"
            print(f"        greedy {args.eval_games} 局  現金 {g_cash:>9,.0f}  "
                  f"差額 {g_margin:>+9,.0f}  勝率 {g_win:.2f}{mark}",
                  flush=True)


        # 🩸 寫檔一定要在 greedy 評估**之後** —— 原本在之前，所以
        # greedy_cash / greedy_win 從來沒進過 train.jsonl，只留在 stdout，
        # 分析時得用 regex 去剖 log。greedy 才是挑 checkpoint 的依據。
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        if args.save_every and (it + 1) % args.save_every == 0:
            save_checkpoint(out / f"ckpt-{it + 1:05d}.pt", net, args, row)
        save_checkpoint(out / "last.pt", net, args, row)
    return 0


def main(argv=None):
    # 🩸 主控台／重導到檔案時 Python 用系統 locale（正體中文是 cp950）。
    # help 字串裡有 U+2212（−）就會讓 `--help` 整支拋 UnicodeEncodeError，
    # 而 `tests/test_ppo.py::test_argparse_help_renders` 抓不到 —— pytest 的
    # stdout 是 UTF-8。2026-09-06 踩到。跟 `tools/action_dist.py:133` 同一個坑。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):     # 不是真的 TextIOWrapper 就算了
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10,
                    help="rollout 行程數。0 = 單行程（慢 5.8 倍，除錯用）")
    ap.add_argument("--envs", type=int, default=2,
                    help="每個 worker 幾個 env。一輪的局數 = workers × envs")
    ap.add_argument("--episode-steps", type=int, default=0,
                    help="0 = 用引擎預設的 720（一整季 30 天）")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--zero-sum", action="store_true",
                    help="reward 減掉對手的現金增量。實測那一項佔 86.4%% 的"
                         "變異數而且我們控制不了，所以預設關掉（§37）")
    ap.add_argument("--settle-step", type=int, default=0,
                    help="前 N 步不逐步發 reward，改成在第 N-1 步一次結算它們"
                         "的和。總和不變。ladder 頂端前 10 天（240 步）現金都"
                         "在 2,000 以下（§55.1），逐步發等於把「正確地把錢花"
                         "光」當扣分。0 = 關掉")
    ap.add_argument("--phi", default="none",
                    choices=("none", "plants", "assets"),
                    help="potential-based shaping 的 Φ 裝什麼。"
                         "plants = 我方作物格數 − 對方（§55.6）；"
                         "assets = 我方非現金資產金額（§83.3）。"
                         "這個形式不改變最佳策略，只是把 credit 提前發")
    ap.add_argument("--phi-weight", type=float, default=0.0,
                    help="Φ 的權重，單位是「scaled reward / Φ 的一單位」。"
                         "--phi plants 用 0.01（一格作物 $100）；"
                         "--phi assets 用 1e-4（資產本來就是錢，"
                         "1/REWARD_SCALE）。0 = 關掉 shaping")
    ap.add_argument("--recognise", default="strict",
                    choices=("strict", "produce"),
                    help="--phi assets 的認列時點（§83.4）。strict = 跟引擎的"
                         "HARVEST 條件一致，收得下來才算成品；"
                         "produce = 澆水長出來就算")
    ap.add_argument("--terminal-bonus", type=float, default=ppo.TERMINAL_BONUS,
                    help="期末勝負的額外 reward（已經是除過 REWARD_SCALE 的"
                         "單位，1.0 = $10,000）。--phi assets 底下 reward "
                         "已經逐步反映資產，這個階梯函數要不要留是未定的"
                         "（§83.7）")
    ap.add_argument("--plant-weight", type=float, default=0.0,
                    help="⚠️ 舊旗標，等於 --phi plants --phi-weight W。"
                         "留著是為了重跑 2026-08-31 之前的命令")
    ap.add_argument("--lam-early", type=float, default=0.0,
                    help="前 --lam-split 步改用這個 lam（後段仍用 --lam）。"
                         "等效視窗是 1/(1-gamma*lam)：0.95 是 19 步、0.998 是"
                         "200 步。收穫在第 240 步，所以前段需要大 lam、後段不"
                         "需要。0 = 整局同一個 lam")
    ap.add_argument("--lam-split", type=int, default=0,
                    help="--lam-early 管到第幾步。建議 240（第 10 天）")
    ap.add_argument("--value-warmup", type=int, default=0,
                    help="前 N 輪只練 value head（軀幹和 policy 的頭凍結）。"
                         "接手監督式的 checkpoint 一定要開 —— 它的 value head "
                         "學的是「期末現金/100k」，跟 PPO 的目標不是同一個量，"
                         "不先校正的話 value loss 會大 150 倍、把共用軀幹"
                         "重塑掉（§88）")
    ap.add_argument("--reset-value", action="store_true",
                    help="開跑前把 value head 的兩層重新初始化。配 "
                         "--value-warmup 用：與其把舊解拉過來，不如重學")
    ap.add_argument("--value-lr", type=float, default=0.0,
                    help="暖身階段的 lr，0 = 跟 --lr 一樣。暖身時沒有信賴區域"
                         "要顧，可以開大一點")
    ap.add_argument("--target-kl", type=float, default=0.0,
                    help="聯合動作的 approx_kl 超過 1.5 倍就停掉這一輪的 "
                         "epoch。0 = 不管")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=100_000,
                    help="env 的 seed 起點。跟評估用的 seed 錯開")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--init", default="", help="熱啟動用的 .pt")
    ap.add_argument("--base-policy", default="",
                    help="混合模式的骨幹 spec（例如 "
                         "config/params/cma1-g50-wt.json）")
    ap.add_argument("--base-side", default="units", choices=("units", "market"),
                    help="骨幹負責哪一半。units＝骨幹出工人、PPO 只練 market "
                         "head；market＝骨幹出 market、PPO 只練 unit head。"
                         "固定一邊才分得出分數變化是哪一個 head 造成的")
    ap.add_argument("--market-temp", type=float, default=1.0,
                    help="熱啟動時把 market present head 的 logit 除以它。"
                         "監督式那份飽和到 sigmoid 斜率 1e-5，收不到梯度")
    ap.add_argument("--opponent", default="config/params/cma1-g50-wt.json",
                    help="固定對手的 spec。空字串 = 自對局（隨機初始時沒訊號）")
    ap.add_argument("--out", default="model/artifacts/ppo0")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=10,
                    help="每幾輪用 greedy 解碼評估一次。0 = 不評估。"
                         "🩸 訓練曲線是取樣版，上場是 greedy 版，要分開看")
    ap.add_argument("--eval-games", type=int, default=10)
    ap.add_argument("--eval-seed0", type=int, default=900_000,
                    help="評估用的 seed 起點，跟訓練的錯開")
    ap.add_argument("--watch", nargs="+", metavar="RUN_DIR",
                    help="看一或多個訓練目錄的進度（不用等跑完）。多給幾個就"
                         "並排比 —— 比之前先確認兩邊 --seed 一樣")
    ap.add_argument("--smoke", action="store_true",
                    help="兩輪、4 個 env、短局 —— 只確認接得起來")
    args = ap.parse_args(argv)
    if getattr(args, "watch", None):
        return watch(args.watch)
    if args.smoke:
        args.iters, args.workers, args.envs = 2, 0, 4
        args.episode_steps = 120
        args.width, args.blocks, args.minibatch = 32, 2, 128
        args.out = args.out + "-smoke"
        args.save_every = 0
        args.eval_every, args.eval_games = 1, 2
    args.episode_steps = args.episode_steps or None
    # 舊旗標 --plant-weight 等於 --phi plants --phi-weight W。
    if args.plant_weight:
        if args.phi != "none":
            raise SystemExit("--plant-weight 和 --phi 不能一起給")
        args.phi, args.phi_weight = "plants", args.plant_weight
    if args.phi != "none" and not args.phi_weight:
        raise SystemExit(f"--phi {args.phi} 要配 --phi-weight（0 = shaping 關掉）")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
