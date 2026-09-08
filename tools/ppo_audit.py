"""PPO 實作稽核：把數學定義逐項對到實際 code，全部用可重現的實測。

    python -m tools.ppo_audit --ckpt model/artifacts/phi-200/ckpt-00200.pt

**這支不改任何 code、不跑訓練。** 每一節印出量到的數字，判讀留給呼叫的人。

節次對應稽核清單：

    1  ACTION / LOGPROB   取樣動作與 argmax 動作的語意、unit 攤平有沒有錯位
    2  MASK               訓練側與上場側的遮罩是不是同一個
    3  RATIO              old_logp / new_logp 是不是同一個 decision
    4  GAE / REWARD       reward[t] 對到 action[t]、邊界、bootstrap
    5  UPDATE             optimizer 真的動到哪些參數、grad 有沒有殘留
    6  INFERENCE          A/B/C/D/E/F 確定性測試（in-memory vs 重載 vs 上場）

🩸 不要在這支裡開 `RolloutPool`（journal §97：臨時腳本開 pool 會卡死）。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KAGGRI_LOG_LEVEL", "0")

from tools._quiet import silenced                       # noqa: E402

with silenced():
    import torch
    from kaggle_environments import make

import contracts as C                                   # noqa: E402
from model import ppo                                   # noqa: E402
from harness.ppo_rollout import (VecRollout, build_net,  # noqa: E402
                                 load_league, masked_sample)


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_ckpt(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(int(blob["width"]), int(blob["blocks"]))
    net.load_state_dict(blob["state_dict"])
    return net.eval(), blob


def collect_obs(seed, opponent, steps):
    """跑一局到指定的步數，把沿途幾個 observation 留下來（雙方都用參數型 agent）。

    要的是**真實的中局盤面**：第 0 步的盤面兩邊都一樣，測不出東西。
    """
    fn, params = _load_agent(opponent)

    def agent(obs, config):
        return fn(obs, config, params)

    with silenced():
        env = make("kaggriculture", configuration={"seed": seed}, debug=False)
        env.reset(2)
    keep = {}
    want = set(steps)
    for t in range(max(steps) + 1):
        if env.done:
            break
        pair = []
        for p in range(2):
            st = env.steps[-1][p]
            obs, cfg = st["observation"], env.configuration
            if t in want and p == 0:
                keep[t] = (dict(obs), dict(cfg))
            with silenced():
                pair.append(agent(obs, cfg))
        with silenced():
            env.step(pair)
    return keep


def _load_agent(spec):
    from harness.rollout import _load                    # noqa: PLC0415
    return _load(spec)


def dist_of(net, obs, cfg):
    """訓練側算分佈的那條路：`C.encode` -> net -> `ppo.masked_log_softmax`。"""
    spatial, scalar = C.encode(obs, cfg)
    pos, feats = C.encode_units(obs, cfg)
    n = len(pos)
    with torch.no_grad():
        op_logits, qty, tgt_logits, mk_p, mk_q, value, _d = net(
            torch.as_tensor(spatial).unsqueeze(0),
            torch.as_tensor(scalar).unsqueeze(0),
            torch.zeros(n, dtype=torch.long),
            torch.as_tensor(np.asarray(pos)),
            torch.as_tensor(feats))
        o_lp = ppo.masked_log_softmax(
            op_logits, torch.as_tensor(C.legal_unit_mask(obs, cfg)))
        t_lp = ppo.masked_log_softmax(
            tgt_logits, torch.as_tensor(C.legal_target_mask(obs, cfg)))
    return {"op_lp": o_lp, "tgt_lp": t_lp, "mk_present": mk_p,
            "mk_qty": mk_q, "value": value, "qty": qty, "pos": pos}


def maxdiff(a, b):
    return float((a - b).abs().max().item())


# --------------------------------------------------------------------------
# 1 / 2 / 3：動作語意、遮罩、ratio
# --------------------------------------------------------------------------

def section_action_logprob(net, ckpt, obs_by_step):
    hr("1-3  ACTION / LOGPROB / MASK / RATIO")

    # 1a. 取樣路徑（rollout）與 argmax 路徑（上場）走的是不是同一組 logits。
    vec = VecRollout(net, n_envs=1, seed0=0, greedy=True)
    for t, (obs, cfg) in sorted(obs_by_step.items()):
        out = vec._policy_batch([(0, 0, obs, cfg)], collect=True)
        (_ei, _p, action, rec) = out[0]
        d = dist_of(net, obs, cfg)
        op_arg = d["op_lp"].argmax(-1).numpy()
        tgt_arg = d["tgt_lp"].argmax(-1).numpy()
        same_op = np.array_equal(rec["op_idx"], op_arg)
        same_tgt = np.array_equal(rec["tgt_idx"], tgt_arg)
        # 送進引擎的動作，能不能被 `encode_unit_action` 認回同一個 op
        emitted = [action["farmer"]] + list(action["hands"])
        pos = d["pos"]
        arrived, ok_round, mismatch = 0, 0, []
        for i, act in enumerate(emitted):
            tx, ty = C.target_xy(int(tgt_arg[i]), C.BOARD_SIZE)
            if (int(pos[i][0]), int(pos[i][1])) == (tx, ty):
                arrived += 1
                enc = C.encode_unit_action(act)
                if enc is not None and enc[0] == int(op_arg[i]):
                    ok_round += 1
                else:
                    mismatch.append((i, act, int(op_arg[i]), enc))
        print(f"  step {t:>3}  units {len(pos):>2}  "
              f"rollout==argmax  op {same_op}  target {same_tgt}   "
              f"已到目標格 {arrived:>2}，其中 encode_unit_action 對得回去 "
              f"{ok_round}")
        if mismatch:
            print(f"    對不回去：{mismatch[:3]}")

    # 1b. unit 攤平／切 chunk 有沒有錯位：同一批用不同 chunk 重算 logprob。
    #     `_pack` 的偏移量算錯的話，單獨打包一步跟包在大 chunk 裡會不一樣。
    opp, _ = load_league("config/params/cma5-g175.json")
    roll = VecRollout(net, n_envs=2, seed0=4242, episode_steps=96,
                      opponent=opp, phi="assets")
    _steps, _cash, trajs = roll.run(collect=True)
    batch = ppo.RolloutBatch(trajs, gamma=1.0, lam=0.998, zero_sum=True,
                             phi_weight=1e-4)
    lp1 = ppo.all_logp(net, batch, chunk=1)
    lp64 = ppo.all_logp(net, batch, chunk=64)
    lp512 = ppo.all_logp(net, batch, chunk=512)
    print(f"\n  攤平／切塊一致性（{batch.n_steps} 步、"
          f"{len(batch.unit_step)} 個 unit）")
    print(f"    chunk 1 vs 64    max |Δlogp| {np.abs(lp1 - lp64).max():.3e}")
    print(f"    chunk 1 vs 512   max |Δlogp| {np.abs(lp1 - lp512).max():.3e}")
    print(f"    unit_step 遞增    {bool(np.all(np.diff(batch.unit_step) >= 0))}")
    print(f"    unit_count 總和   {int(batch.unit_count.sum())} == "
          f"{len(batch.unit_step)}")

    # 1c. rollout 存的 old_logp 跟 update 重算的 new_logp 是不是同一個決策。
    #     網路一個位元都沒動，兩者必須相等（差只來自 float16 觀測）。
    d = np.abs(lp512 - batch.old_logp)
    print(f"\n  old_logp vs 重算（網路未更新）"
          f"  max {d.max():.3e}  mean {d.mean():.3e}")
    print(f"    ratio = exp(Δ) 的最大偏離 1： {np.expm1(d.max()):.3e}")

    # 3b. 🩸 真正的訓練是 **worker 在 CPU 上 rollout、主行程在 CUDA 上 update**
    #     （`--workers 8 --device cuda`）。old_logp 來自 CPU、new_logp 來自
    #     CUDA，兩邊的浮點不會逐位元相同 —— 第一個 epoch 的 ratio 就不是 1。
    if torch.cuda.is_available():
        lp_gpu = ppo.all_logp(net.to("cuda"), batch, device="cuda")
        net.to("cpu")
        d2 = np.abs(lp_gpu - batch.old_logp)
        r = np.exp(lp_gpu - batch.old_logp)
        print(f"\n  CPU rollout vs CUDA 重算（同一份權重、同一個動作）")
        print(f"    max |Δlogp| {d2.max():.3e}   mean {d2.mean():.3e}")
        print(f"    ratio 的範圍 {r.min():.6f} ~ {r.max():.6f}   "
              f"偏離 0.2 的 clip 邊界？ {bool((np.abs(r - 1) > 0.2).any())}")
        print(f"    |ratio-1| > 0.01 的比例 "
              f"{100 * (np.abs(r - 1) > 0.01).mean():.2f}%")

    # 2. 遮罩：訓練側與上場側是不是同一個函式的同一個結果。
    hr("2  MASK")
    for t, (obs, cfg) in sorted(obs_by_step.items()):
        om = C.legal_unit_mask(obs, cfg)
        tm = C.legal_target_mask(obs, cfg)
        lp = ppo.masked_log_softmax(
            torch.zeros(om.shape), torch.as_tensor(om))
        p = lp.exp().numpy()
        leak = float(p[~om].sum(-1).max()) if (~om).any() else 0.0
        allfalse = int((~om.any(axis=-1)).sum())
        print(f"  step {t:>3}  op 合法數 {om.sum(-1).tolist()}  "
              f"target 合法數 {int(tm[0].sum())}  "
              f"非法動作的機率總和最大 {leak:.3e}  全 False 的 unit {allfalse}")
    print("  （非法動作的機率是 softmax(-1e9) 的殘值，不是 0 但是 1e-40 量級）")
    return batch


# --------------------------------------------------------------------------
# 4：reward / GAE 對齊
# --------------------------------------------------------------------------

def section_gae(batch, trajs_cash=None):
    hr("4  GAE / REWARD ALIGNMENT")
    tr_n = len(batch.traj_start)
    starts = batch.traj_start
    ends = np.append(starts[1:], batch.n_steps)
    print(f"  軌跡 {tr_n} 條，步 {batch.n_steps}，邊界 {starts.tolist()}")

    # reward 是不是 cash 的一階差：閉式解 —— 一條軌跡的 Σrew 應該等於
    # (期末差額 - 起始差額)/REWARD_SCALE + Φ 項 + 勝負 bonus。
    for k, (s, e) in enumerate(zip(starts, ends)):
        rew = batch.rew[s:e]
        print(f"    軌跡 {k}: T={e - s}  Σrew {rew.sum():+.4f}  "
              f"最後一步 {rew[-1]:+.4f}（含 terminal bonus ±1）  "
              f"期末差額 {batch.traj_cash[k, 0] - batch.traj_cash[k, 1]:+,.0f}")

    # GAE 的邊界：最後一步的 done=1，bootstrap 必須是 0。
    # 用同一支 compute_gae 重算一條，跟 batch 的 adv 比。
    s, e = starts[0], ends[0]
    T = e - s
    rew = batch.rew[s:e]
    v = np.concatenate([batch.old_value[s:e], np.zeros(1, np.float32)])
    dones = np.zeros(T, np.float32)
    dones[-1] = 1.0
    adv, ret = ppo.compute_gae(rew, v, dones, gamma=1.0, lam=0.998)
    print(f"\n  重算軌跡 0 的 GAE：max |Δadv| "
          f"{np.abs(adv - batch.adv[s:e]).max():.3e}   "
          f"max |Δret| {np.abs(ret - batch.ret[s:e]).max():.3e}")
    # 最後一步的 advantage 必須等於 r_T - V(s_T)（bootstrap = 0）
    last = float(rew[-1] - batch.old_value[e - 1])
    print(f"  最後一步 adv = r_T - V(s_T)：{batch.adv[e - 1]:+.5f} vs "
          f"{last:+.5f}   差 {abs(batch.adv[e - 1] - last):.2e}")
    # 跨軌跡有沒有滲漏：把第 0 條的最後一步 advantage 用第 1 條的第一步算
    # 會差很多；這裡只確認邊界處 done=1 有生效（上面那一項就是證據）。
    print(f"  ret == adv + old_value：max "
          f"{np.abs(batch.ret - (batch.adv + batch.old_value)).max():.3e}")


# --------------------------------------------------------------------------
# 5：optimizer 真的動到什麼
# --------------------------------------------------------------------------

def section_update(batch, real_net):
    hr("5  POLICY UPDATE（value warmup 期間的 grad 殘留）")
    import copy                                          # noqa: PLC0415
    # 用**真正那個 checkpoint 的複本**，梯度量級才是實際訓練時的量級。
    net = copy.deepcopy(real_net)
    trunk = [p for n, p in net.named_parameters()
             if not n.startswith("value_head")]
    head = [p for n, p in net.named_parameters() if n.startswith("value_head")]
    value_opt = torch.optim.Adam(net.value_head.parameters(), lr=1e-4)

    def gnorm(ps):
        tot = 0.0
        for p in ps:
            if p.grad is not None:
                tot += float(p.grad.norm() ** 2)
        return tot ** 0.5

    # 逐個 minibatch 看真正的 update 裡 grad 的狀態。
    # 🩸 監看的方式是暫時包住 `torch.nn.utils.clip_grad_norm_` —— `model/ppo.py`
    # 是在呼叫當下才查這個屬性，所以包得到**真正跑的那條路**，不是複製一份。
    import torch.nn.utils as U                          # noqa: PLC0415
    orig = U.clip_grad_norm_
    seen = []

    def spy(params, max_norm, *a, **kw):
        params = list(params)
        seen.append((gnorm(trunk), gnorm(head)))
        return orig(params, max_norm, *a, **kw)

    print(f"  update 前   trunk grad {gnorm(trunk):.4e}   "
          f"value_head grad {gnorm(head):.4e}")
    U.clip_grad_norm_ = spy
    try:
        for k in range(3):
            seen.clear()
            parts = ppo.update(net, value_opt, batch, epochs=1, minibatch=64,
                               seed=k, policy_coef=0.0, target_kl=0.0)
            trace = "  ".join(f"{t:.3e}" for t, _h in seen)
            print(f"  warmup 第 {k + 1} 次 update（{len(seen)} 個 minibatch）")
            print(f"    每個 minibatch clip 之前的 trunk grad： {trace}")
            print(f"    結束時 trunk {gnorm(trunk):.4e}  "
                  f"value_head {gnorm(head):.4e}  "
                  f"log 的 grad_norm {parts['grad_norm']:.4f}")
    finally:
        U.clip_grad_norm_ = orig
    print("  -> `ppo.update` 只呼叫傳進去的 optimizer 的 zero_grad"
          "（model/ppo.py:594）。")
    print("    value warmup 傳的是只含 value_head 的 optimizer，"
          "所以 trunk 的 .grad 會一路累積；")
    print("    而 `clip_grad_norm_(net.parameters(), ...)` "
          "（model/ppo.py:597）算的是**全部參數**的範數。")

    # optimizer 真的動到哪些頭？逐個 head 看「更新後參數變了多少」與
    # 「有沒有拿到梯度」。PPO 的 logprob 只用到 op / target / market 兩個頭
    # （`ppo.evaluate_actions`），unit 的 qty head 和 demand head 不在裡面。
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    full = torch.optim.Adam(net.parameters(), lr=1e-4)
    parts = ppo.update(net, full, batch, epochs=1, minibatch=64, seed=9,
                       policy_coef=1.0, target_kl=0.0)
    groups = {}
    for name, p in net.named_parameters():
        head = name.split(".")[0]
        d = float((p.detach() - before[name]).abs().max())
        g = 0.0 if p.grad is None else float(p.grad.abs().max())
        cur = groups.setdefault(head, [0.0, 0.0, 0])
        cur[0] = max(cur[0], d)
        cur[1] = max(cur[1], g)
        cur[2] += p.numel()
    print("\n  一次 update 之後，各個 head 的變化（max |Δ參數| / max |grad|）")
    for head, (d, g, n) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        print(f"    {head:<24}{n:>9,} 個參數   Δ {d:.3e}   grad {g:.3e}")
    print(f"\n  換成全參數 optimizer 之後  trunk grad {gnorm(trunk):.4e}"
          f"   log 的 grad_norm {parts['grad_norm']:.4f}"
          "   （zero_grad 會清掉殘留，所以不會有髒的 step）")


# --------------------------------------------------------------------------
# 6：A/B/C/D/E/F 確定性測試
# --------------------------------------------------------------------------

def section_inference(net, blob, obs_by_step, ckpt_path):
    hr("6  INFERENCE 確定性測試（A/B/C/D/E/F）")
    from model import ppo_train                          # noqa: PLC0415
    import agents.ppo_agent as PA                        # noqa: PLC0415

    tmp = Path(tempfile.mkdtemp(prefix="ppo-audit-")) / "roundtrip.pt"
    # B：用**訓練那支**的存檔函式存（`save_checkpoint(path, net, args, meta)`）
    ppo_train.save_checkpoint(
        tmp, net,
        argparse.Namespace(width=blob["width"], blocks=blob["blocks"]),
        {"iter": -1})
    # C：用**上場那支**的載入函式載
    PA._CACHE.clear()
    net_re = PA._load(str(tmp))

    print(f"  存到 {tmp}")
    for t, (obs, cfg) in sorted(obs_by_step.items()):
        a = dist_of(net, obs, cfg)                     # A：記憶體裡的網路
        d = dist_of(net_re, obs, cfg)                  # D：重載之後
        train_mode = net.training
        net.train()
        a_tr = dist_of(net, obs, cfg)                  # 順便測 train/eval 模式
        net.train(train_mode)
        print(f"  step {t:>3}  A vs D  op {maxdiff(a['op_lp'], d['op_lp']):.3e}"
              f"  target {maxdiff(a['tgt_lp'], d['tgt_lp']):.3e}"
              f"  market {maxdiff(a['mk_present'], d['mk_present']):.3e}"
              f"  value {maxdiff(a['value'], d['value']):.3e}"
              f"   | train() vs eval() op "
              f"{maxdiff(a['op_lp'], a_tr['op_lp']):.3e}")

    # E / F：三條解碼路徑送出的動作要一模一樣
    print("\n  E/F  三條路徑的 greedy 動作")
    vec = VecRollout(net, n_envs=1, seed0=0, greedy=True)
    for t, (obs, cfg) in sorted(obs_by_step.items()):
        roll_act = vec._policy_batch([(0, 0, obs, cfg)], collect=False)[0][2]
        agent_act = PA.act(obs, cfg, {"ckpt": str(tmp), "greedy": True})
        agent_act2 = PA.act(obs, cfg, {"ckpt": ckpt_path, "greedy": True})
        same_units = (roll_act["farmer"] == agent_act["farmer"]
                      and roll_act["hands"] == agent_act["hands"])
        same_mkt = roll_act["market"] == agent_act["market"]
        same_ckpt = (agent_act["farmer"] == agent_act2["farmer"]
                     and agent_act["hands"] == agent_act2["hands"]
                     and agent_act["market"] == agent_act2["market"])
        # F：送進引擎的 unit 動作，`encode_unit_action` 認不認得
        emitted = [roll_act["farmer"]] + list(roll_act["hands"])
        unknown = [a for a in emitted if C.encode_unit_action(a) is None]
        print(f"  step {t:>3}  rollout==ppo_agent  units {same_units}  "
              f"market {same_mkt}   存檔前後 {same_ckpt}   "
              f"引擎編碼器認不得的動作 {len(unknown)}   "
              f"訂單 {len(roll_act['market'])} 筆")
        if not same_units:
            print(f"    rollout : {emitted[:4]}")
            print(f"    ppo_agent: {[agent_act['farmer']] + list(agent_act['hands'])[:3]}")

    # 動作語意：PICKUP / PLACE 的數量
    hr("附註  PICKUP / PLACE 的數量在 PPO 的動作空間裡是什麼")
    print("  `contracts.decode_unit(op_index, qty_index=None)` -> qty 固定 1")
    print("  呼叫端：harness/ppo_rollout.py:_policy_batch 與 "
          "agents/ppo_agent.py:act 都傳 None")
    print(f"  引擎：PICKUP 的第三個元素就是數量"
          f"（kaggriculture.py:364 `n = int(action[2]) if len(action) >= 3`）")


class PlantProbe(VecRollout):
    """每一步比對「送出去的 PLANT 需求」和「手上的種子」。

    引擎的原子規則（`kaggriculture.py:920-931`）：**同一回合某作物的 PLANT
    需求總數超過種子數，該作物的 PLANT 全部作廢**（不是只砍超出的那幾筆）。

    `contracts.legal_unit_mask` 的 docstring 明說這條表達不了 —— 它是逐 unit
    的遮罩，而這是跨 unit 的約束。PPO 的每個 unit 是**獨立取樣**的，所以沒有
    任何機制阻止 5 個 unit 同時要種只剩 3 顆種子的作物。

    同時用同一個盤面問參照 agent（它的規劃器是全域配對），當對照組。
    """

    def __init__(self, *args, ref=None, **kw):
        super().__init__(*args, **kw)
        self.ref = ref
        self.stat = {"steps": 0, "plant_steps": 0, "blocked_steps": 0,
                     "plant_units": 0, "blocked_units": 0,
                     "ref_plant_steps": 0, "ref_blocked_steps": 0,
                     "ref_plant_units": 0, "ref_blocked_units": 0}

    @staticmethod
    def _demand(units):
        d = {}
        for a in units:
            if isinstance(a, (list, tuple)) and len(a) >= 2 and a[0] == "PLANT":
                d[a[1]] = d.get(a[1], 0) + 1
        return d

    def _tally(self, units, seeds, prefix):
        d = self._demand(units)
        n = sum(d.values())
        if not n:
            return
        blocked = {c: k for c, k in d.items() if k > int(seeds.get(c, 0))}
        self.stat[f"{prefix}plant_steps"] += 1
        self.stat[f"{prefix}plant_units"] += n
        if blocked:
            self.stat[f"{prefix}blocked_steps"] += 1
            self.stat[f"{prefix}blocked_units"] += sum(blocked.values())

    def _policy_batch(self, items, collect=False):
        out = super()._policy_batch(items, collect=collect)
        for (ei, p, obs, cfg), (_e, _p, action, _rec) in zip(items, out):
            seeds = obs["private"].get("seeds", {}) or {}
            self.stat["steps"] += 1
            self._tally([action["farmer"]] + list(action["hands"] or []),
                        seeds, "")
            if self.ref is not None:
                with silenced():
                    ra = self.ref(obs, cfg)
                self._tally([ra["farmer"]] + list(ra["hands"] or []),
                            seeds, "ref_")
        return out


def section_atomic_plant(net, opponent, games, seed0):
    hr("附註二  PLANT 的跨 unit 原子規則（引擎會整批作廢）")
    opp, _ = load_league(opponent)
    fn, params = _load_agent(opponent)

    def ref(obs, cfg):
        return fn(obs, cfg, params)

    vec = PlantProbe(net, n_envs=games, seed0=seed0, opponent=opp,
                     greedy=True, ref=ref)
    vec.run(collect=False)
    s = vec.stat
    print(f"  {games} 局 greedy、{s['steps']:,} 個決策步")
    for tag, pre in (("網路（greedy）", ""), ("參照 agent（同一批盤面）", "ref_")):
        ps, bs = s[f"{pre}plant_steps"], s[f"{pre}blocked_steps"]
        pu, bu = s[f"{pre}plant_units"], s[f"{pre}blocked_units"]
        print(f"  {tag}")
        print(f"    有送 PLANT 的步 {ps:,}   其中被原子規則整批作廢的 "
              f"{bs:,}（{100 * bs / max(ps, 1):.2f}%）")
        print(f"    PLANT 的 unit 動作 {pu:,}   被作廢的 "
              f"{bu:,}（{100 * bu / max(pu, 1):.2f}%）")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default="model/artifacts/phi-200/ckpt-00200.pt")
    ap.add_argument("--opponent", default="config/params/cma5-g175.json")
    ap.add_argument("--seed", type=int, default=950_000)
    ap.add_argument("--steps", default="1,120,360,600")
    ap.add_argument("--plant-games", type=int, default=4,
                    help="附註二那節跑幾局")
    args = ap.parse_args(argv)

    steps = [int(s) for s in args.steps.split(",")]
    net, blob = load_ckpt(args.ckpt)
    print(f"  checkpoint {args.ckpt}（it {int(blob.get('iter', -1)) + 1}）")
    print(f"  取樣盤面：seed {args.seed}、步 {steps}、雙方 {args.opponent}")
    obs_by_step = collect_obs(args.seed, args.opponent, steps)
    print(f"  拿到 {len(obs_by_step)} 個 observation")

    batch = section_action_logprob(net, blob, obs_by_step)
    section_gae(batch)
    section_update(batch, net)
    section_inference(net, blob, obs_by_step, args.ckpt)
    section_atomic_plant(net, args.opponent, args.plant_games, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
