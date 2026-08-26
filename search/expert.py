"""把 search 的答案錄成訓練資料 —— Stage 3。

## 為什麼要一個新的 Recorder

`harness/rollout.py` 的 `Recorder.record()` 要 `(action, plan)` 兩樣東西：

    action  這一步實際送出什麼      -> unit_op / unit_qty / market_*
    plan    規劃器的**意圖**        -> unit_target / unit_term_* / demand

`gen0` 兩樣都給得出來（`act(..., return_plan=True)`）。
**search 只給得出 action** —— 它挑的是「哪個聯合動作打到最後最好」，
沒有「這個 unit 要走去哪、到了做什麼」這種意圖。

🩸 **不能把 `plan=None` 丟給父類別。** `harness/rollout.py:147` 在
`plan["plan"][i] is None` 時會寫出「原地 PASS」這個**看起來正常的錯標籤**，
不是跳過 —— 網路會學到「這個 unit 應該站著不動」。

## 缺的標籤怎麼處理（三個都不用改 A 的檔案）

| 欄位 | 寫什麼 | 為什麼安全 |
|---|---|---|
| `unit_target` | −1 | `train.py:481` 的 `cross_entropy(..., ignore_index=-1)` |
| `unit_term_op` / `unit_term_qty` | −1 | `--labels immediate` 下根本不讀（`train.py:156-162`） |
| `demand` / `demand_legal` | 全 0 | demand loss 是 `(bce * legal).sum() / legal.sum()`，legal 全 0 的盤面分子分母都不貢獻（`train.py:511-515`） |

`gen2_model.act` 實際出手只用 op / qty / market 三組 head
（`agents/gen2_model.py:395-400`），而那三組 search 全部給得出來。

## 一局裡有兩種回合

搜過的回合用 `record_search()`（只有 action），沒搜的回合仍然是 `gen0` 在下，
所以照樣用父類別的 `record()` 拿完整標籤。兩種混在同一個 npz 裡，
`model/train.py` 不用改。

**沒搜的那些回合仍然有價值** —— 它們的**盤面**來自被 search 改良過的軌跡，
那是網路沒看過的分布。DAgger 的重點本來就是「在 policy 會走到的局面上貼標籤」。
"""
from __future__ import annotations

import numpy as np

import contracts as C
from harness.rollout import Recorder


class SearchRecorder(Recorder):
    """`Recorder` + 一條「只有 action」的錄製路徑。

    `record()`（父類別，要 plan）跟 `record_search()`（只要 action）可以混用，
    產出的 npz 欄位跟 `harness/rollout.py` 逐項相同。
    """

    def __init__(self, player):
        super().__init__(player)
        #: 這一局有幾個回合是 search 決定的，寫進 npz 當紀錄。
        self.searched_turns = 0

    def record_search(self, obs, config, action):
        """search 決定的回合：只有 action，沒有 plan。"""
        spatial, scalar = C.encode(obs, config)
        positions, feats = C.encode_units(obs, config)

        index = len(self.spatial)
        self.spatial.append(spatial.astype(np.float16))
        self.scalar.append(scalar)
        self.step.append(int(obs.get("step", 0)))
        self.searched_turns += 1

        # demand 那一半沒有標籤。legal 全 0 -> 那個 head 在這個盤面上不算 loss。
        # 🩸 形狀要跟父類別一致（packbits 之後是 [N_TASK_OPS, ceil(cells/8)]），
        # 不然 np.concatenate 會在 model/train.py 裡炸掉。
        blank = np.packbits(
            np.zeros((C.N_TASK_OPS, C.N_TARGET_CELLS), dtype=bool), axis=-1)
        self.demand.append(blank)
        self.demand_legal.append(blank)

        emitted = [action.get("farmer")] + list(action.get("hands") or [])
        for i in range(len(positions)):
            now = C.encode_unit_action(emitted[i]) if i < len(emitted) else None
            if now is None:
                # 認不得的動作。父類別也是跳過並計數，這裡照做 —— 靜默當成
                # PASS 會教網路一個它從來沒做過的動作。
                self.skipped += 1
                continue
            self.unit_board.append(index)
            self.unit_pos.append(positions[i])
            self.unit_feat.append(feats[i])
            self.unit_op.append(now[0])
            self.unit_qty.append(-1 if now[1] is None else now[1])
            # 意圖類的標籤 search 給不出來 -> −1，訓練時被 ignore_index 跳過。
            self.unit_target.append(-1)
            self.unit_term_op.append(-1)
            self.unit_term_qty.append(-1)

        for order in action.get("market") or []:
            decoded = C.encode_market_action(order)
            if decoded is None:
                continue
            self.market_board.append(index)
            self.market_op.append(decoded[0])
            self.market_qty.append(decoded[1])

    def finish(self, rewards, episode_id):
        payload = super().finish(rewards, episode_id)
        payload["searched_turns"] = np.asarray([self.searched_turns], dtype=np.int32)
        return payload
