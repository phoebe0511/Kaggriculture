"""純 numpy 的前向，比賽端用。**不准 import torch。**

torch 載入要好幾秒，第一回合就可能超時（`workflow.md` §4）。所以訓練好的權重
匯出成 `.npz`，比賽端用 numpy 手刻前向。

`serving/export_npz.py` 負責匯出，`tests/test_contracts.py` 之外另有一項測試
比對兩邊的數值 —— 差超過 1e-4 就是這裡跟 `model/net.py` 對不上。

## 形狀

    spatial [C, 10, 10]、scalar [D]     一個盤面（比賽端一次只算一個）
    trunk  -> [width, 10, 10]
    每個 unit：trunk 在它格子的 width 維 + 它自己的 18 維
               -> op / qty / target 分數

## 為什麼 conv 用 im2col

10×10 的盤面小，直接展開成矩陣乘法比迴圈快得多，而且 numpy 的 `@` 會走
BLAS。padding 1 的 3×3 就是把 10×10 補成 12×12 再取 9 個位移。
"""

from __future__ import annotations

import numpy as np


def _im2col_3x3(x):
    """`[C, H, W]` -> `[C*9, H*W]`，對應 padding=1 的 3×3 卷積。"""
    channels, height, width = x.shape
    padded = np.zeros((channels, height + 2, width + 2), dtype=x.dtype)
    padded[:, 1:height + 1, 1:width + 1] = x
    cols = np.empty((channels * 9, height * width), dtype=x.dtype)
    k = 0
    for dy in range(3):
        for dx in range(3):
            patch = padded[:, dy:dy + height, dx:dx + width]
            cols[k * channels:(k + 1) * channels] = patch.reshape(channels, -1)
            k += 1
    return cols


def _conv3x3(x, weight, bias):
    """`weight` 是 torch 的 `[out, in, 3, 3]`。"""
    out_channels, in_channels = weight.shape[0], weight.shape[1]
    height, width = x.shape[1], x.shape[2]
    # torch 的排列是 [out, in, kh, kw]；im2col 疊的順序是 (kh, kw) 外層、in 內層
    flat = weight.transpose(2, 3, 1, 0).reshape(9 * in_channels, out_channels)
    out = flat.T @ _im2col_3x3(x)
    return (out + bias[:, None]).reshape(out_channels, height, width)


def _group_norm(x, weight, bias, groups=8, eps=1e-5):
    channels = x.shape[0]
    grouped = x.reshape(groups, channels // groups, -1)
    mean = grouped.mean(axis=(1, 2), keepdims=True)
    var = grouped.var(axis=(1, 2), keepdims=True)
    normed = (grouped - mean) / np.sqrt(var + eps)
    normed = normed.reshape(x.shape)
    return normed * weight[:, None, None] + bias[:, None, None]


def _relu(x):
    return np.maximum(x, 0.0)


def _linear(x, weight, bias):
    """`weight` 是 torch 的 `[out, in]`。"""
    return x @ weight.T + bias


class NumpyPolicy:
    """從 `.npz` 載入權重，提供跟 `model/net.py` 等價的前向。"""

    def __init__(self, path):
        data = np.load(path)
        self.w = {k: data[k].astype(np.float32) for k in data.files
                  if data[k].dtype != np.int32 or k.endswith("_")}
        self.meta = {k: int(data[k][0]) for k in ("encoder_version", "width", "blocks")
                     if k in data.files}
        self.width = self.meta["width"]
        self.blocks = self.meta["blocks"]
        self.encoder_version = self.meta["encoder_version"]

    def trunk(self, spatial, scalar):
        w = self.w
        x = _relu(_conv3x3(spatial, w["stem.weight"], w["stem.bias"]))

        h = _relu(_linear(scalar, w["scalar_mlp.0.weight"], w["scalar_mlp.0.bias"]))
        h = _linear(h, w["scalar_mlp.2.weight"], w["scalar_mlp.2.bias"])
        x = x + h[:, None, None]

        for i in range(self.blocks):
            y = _conv3x3(x, w[f"blocks.{i}.weight"], w[f"blocks.{i}.bias"])
            y = _group_norm(y, w[f"norms.{i}.weight"], w[f"norms.{i}.bias"])
            x = _relu(y) + x
        return x

    def unit_logits(self, features, positions, unit_features):
        """`positions` 是 `[n, 2]` 的 (x, y)。

        回傳 `(op [n, ops], qty [n, qty], target [n, cells])`。v3 起 `op` 是
        **到了目標格要做什麼**，不是這一步做什麼。
        """
        w = self.w
        picked = features[:, positions[:, 1], positions[:, 0]].T   # [n, width]
        h = np.concatenate([picked, unit_features], axis=1)
        h = _relu(_linear(h, w["unit_head.0.weight"], w["unit_head.0.bias"]))
        h = _relu(_linear(h, w["unit_head.2.weight"], w["unit_head.2.bias"]))
        return (_linear(h, w["op_out.weight"], w["op_out.bias"]),
                _linear(h, w["qty_out.weight"], w["qty_out.bias"]),
                _linear(h, w["target_out.weight"], w["target_out.bias"]))

    def pooled(self, features):
        """整盤面壓成一個向量。value 與 market 共用（`model/net.py` 同名方法）。"""
        return np.concatenate([features.mean(axis=(1, 2)), features.max(axis=(1, 2))])

    def value(self, features):
        w = self.w
        h = _relu(_linear(self.pooled(features),
                          w["value_head.0.weight"], w["value_head.0.bias"]))
        return float(_linear(h, w["value_head.2.weight"], w["value_head.2.bias"])[0])

    def market_logits(self, features):
        """回傳 `(present [ops], qty [ops, buckets])`。"""
        w = self.w
        h = _relu(_linear(self.pooled(features),
                          w["market_head.0.weight"], w["market_head.0.bias"]))
        present = _linear(h, w["market_present_out.weight"],
                          w["market_present_out.bias"])
        qty = _linear(h, w["market_qty_out.weight"], w["market_qty_out.bias"])
        return present, qty.reshape(len(present), -1)

    def __call__(self, spatial, scalar, positions, unit_features):
        features = self.trunk(spatial, scalar)
        op_logits, qty_logits, target_logits = self.unit_logits(
            features, positions, unit_features)
        market_present, market_qty = self.market_logits(features)
        return (op_logits, qty_logits, target_logits,
                market_present, market_qty, self.value(features))
