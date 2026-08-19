"""模仿 ladder 頂端玩家的網路。

## 形狀

    spatial   [B, 38, 10, 10]   盤面
    scalar    [B, 66]           現金、時間、市價……

    trunk：CNN 整盤跑一次 → [B, 64, 10, 10]
    scalar 那路先過 MLP，再**廣播到每一格**加進 trunk ——
    「現在是第 25 天、草莓 $220」這種資訊對每一格都成立。

    per-unit head（每個 unit 一次，但很便宜）：
        取 trunk 在該 unit 格子的 64 維  +  該 unit 自己的 18 維
        → MLP → op_logits [44] 與 qty_logits [12]

    value head：trunk 全域池化 + scalar → 期末現金（正規化過）

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
                 n_ops, n_qty, width=64, unit_hidden=256, n_blocks=4):
        super().__init__()
        self.width = width

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

        self.value_head = nn.Sequential(
            nn.Linear(width * 2, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

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
        return self.op_out(hidden), self.qty_out(hidden)

    def value(self, features, scalar):
        pooled = torch.cat([
            features.mean(dim=(2, 3)),
            features.amax(dim=(2, 3)),
        ], dim=1)
        return self.value_head(pooled).squeeze(-1)

    def forward(self, spatial, scalar, unit_board, unit_pos, unit_features):
        features = self.trunk(spatial, scalar)
        op_logits, qty_logits = self.unit_logits(
            features, unit_board, unit_pos, unit_features)
        return op_logits, qty_logits, self.value(features, scalar)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
