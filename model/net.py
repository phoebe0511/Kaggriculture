"""模仿 ladder 頂端玩家的網路。

## 形狀

    spatial   [B, 38, 10, 10]   盤面
    scalar    [B, 66]           現金、時間、市價……

    trunk：CNN 整盤跑一次 → [B, 64, 10, 10]
    scalar 那路先過 MLP，再**廣播到每一格**加進 trunk ——
    「現在是第 25 天、草莓 $220」這種資訊對每一格都成立。

    per-unit head（每個 unit 一次，但很便宜）：
        取 trunk 在該 unit 格子的 64 維  +  該 unit 自己的 18 維
        → MLP → op_logits [44]、qty_logits [12] 與 target_logits [100]

    value head：trunk 全域池化（mean + max）→ 期末現金（正規化過）
    market head：同一個池化 → present [21] 與 qty [21, 18]

## v4：市場也在裡面

市場是逐盤面的決定，而且是**集合**不是單選 —— 一回合可以同時下 0~10 筆訂單。
所以 present 走 multi-label（BCE），數量走逐 op 的分類（18 個桶）。

為什麼要學：2026-08-20 量到 gen0 的市場整季只買 3 隻 COW、13 個 WHEAT，
老師是 14 隻動物、553 個 WHEAT。unit 的動作是在老師那種農場上學的，
放到 gen0 蓋出來的農場上就沒事可做（PASS 佔 36%）。

## v3：op head 的語意是「終點動作」，不是「這一步」

`target_logits` 指出這個 unit 這一段要走去哪一格，`op_logits` 是**到了那裡要做
什麼**。走路不經過網路 —— 用 `agents/gen0.step_toward`，實測對 60 局 replay 有
93.02% 的命中率（19,750 個 MOVE 裡對 18,372 個）。

v2 是每一步直接輸出動作，結果每一步 96.7% 猜對、整局打不出來（MOVE 佔 75.6%，
老師 51.8%）。細節見 `contracts.py` 的 `ENCODER_VERSION` 註解。

## 為什麼 unit 要獨立的 head

實測 60 局 replay：11.6% 的 unit-turn 是「兩個以上 unit 站同一格、但動作不同」。
純 per-cell 的 `[44, 10, 10]` 在那些情況只能給一個答案。同格的 unit 靠自己的
inventory 區分 —— `FEED` 是從 unit 自己的 inventory 扣 WHEAT，手上沒飼料的
只能先回倉庫。細節見 `contracts.py` docstring。

## 成本

一回合 1 次 trunk + n 次 head。head 是 (64+18) -> 256 -> 44，約 3.3 萬次乘法，
比 trunk 小兩個數量級 —— 13 個 unit 的總成本仍由 trunk 主導。

⚠️ **submission 端不會用這個檔**。比賽端不准 import torch，要走
`serving/` 的 numpy 前向 + `.npz` 權重。這裡只負責訓練。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KaggricultureNet(nn.Module):
    def __init__(self, n_spatial, n_scalar, n_unit_features,
                 n_ops, n_qty, n_targets, n_market_ops, n_market_qty,
                 width=64, unit_hidden=256, n_blocks=4):
        super().__init__()
        self.width = width
        self.n_market_ops = n_market_ops
        self.n_market_qty = n_market_qty

        self.stem = nn.Conv2d(n_spatial, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Conv2d(width, width, 3, padding=1) for _ in range(n_blocks))
        self.norms = nn.ModuleList(
            nn.GroupNorm(8, width) for _ in range(n_blocks))

        # scalar 走自己的 MLP，輸出寬度跟 trunk 一樣，才能直接相加廣播
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalar, 128), nn.ReLU(),
            nn.Linear(128, width),
        )

        self.unit_head = nn.Sequential(
            nn.Linear(width + n_unit_features, unit_hidden), nn.ReLU(),
            nn.Linear(unit_hidden, unit_hidden), nn.ReLU(),
        )
        self.op_out = nn.Linear(unit_hidden, n_ops)
        self.qty_out = nn.Linear(unit_hidden, n_qty)
        # 目標格。trunk 有 4 層以上 3×3 的殘差塊，receptive field 已經蓋滿
        # 10×10，所以 unit 自己那格的特徵看得到整個盤面。
        self.target_out = nn.Linear(unit_hidden, n_targets)

        self.value_head = nn.Sequential(
            nn.Linear(width * 2, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

        # 市場是**逐盤面**的決定（不是逐 unit），所以掛在同一路全域池化上。
        # present 是 multi-label 而不是 44 選 1 —— 一回合可以同時下好幾筆，
        # 實測 51.7% 的回合 0 筆、p95 是 5 筆、最多 10 筆。
        self.market_head = nn.Sequential(
            nn.Linear(width * 2, 256), nn.ReLU(),
        )
        self.market_present_out = nn.Linear(256, n_market_ops)
        self.market_qty_out = nn.Linear(256, n_market_ops * n_market_qty)

    def trunk(self, spatial, scalar):
        """回傳 [B, width, H, W]。一個 batch 的盤面只跑一次。"""
        x = F.relu(self.stem(spatial))
        # scalar -> [B, width, 1, 1]，加到每一格
        x = x + self.scalar_mlp(scalar)[:, :, None, None]
        for conv, norm in zip(self.blocks, self.norms):
            x = F.relu(norm(conv(x))) + x      # 殘差，讓層數多也好訓
        return x

    def unit_logits(self, features, unit_board, unit_pos, unit_features):
        """在每個 unit 的格子上取特徵，接上它自己的特徵，出動作分數。

        `unit_board` 指回這個 unit 屬於 batch 裡第幾個盤面 —— 一個盤面有
        好幾個 unit，所以 unit 的數量跟 batch 大小不一樣。
        """
        # features[unit_board, :, y, x]
        picked = features[unit_board, :, unit_pos[:, 1].long(), unit_pos[:, 0].long()]
        hidden = self.unit_head(torch.cat([picked, unit_features], dim=1))
        return self.op_out(hidden), self.qty_out(hidden), self.target_out(hidden)

    def pooled(self, features):
        """整盤面壓成一個向量。value 與 market 共用。"""
        return torch.cat([
            features.mean(dim=(2, 3)),
            features.amax(dim=(2, 3)),
        ], dim=1)

    def value(self, features):
        """scalar 不用再傳一次 —— `trunk` 已經把它廣播加進每一格了。"""
        return self.value_head(self.pooled(features)).squeeze(-1)

    def market_logits(self, features):
        """回傳 `(present [B, ops], qty [B, ops, buckets])`。"""
        hidden = self.market_head(self.pooled(features))
        qty = self.market_qty_out(hidden).view(
            -1, self.n_market_ops, self.n_market_qty)
        return self.market_present_out(hidden), qty

    def forward(self, spatial, scalar, unit_board, unit_pos, unit_features):
        features = self.trunk(spatial, scalar)
        op_logits, qty_logits, target_logits = self.unit_logits(
            features, unit_board, unit_pos, unit_features)
        market_present, market_qty = self.market_logits(features)
        return (op_logits, qty_logits, target_logits,
                market_present, market_qty, self.value(features))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
