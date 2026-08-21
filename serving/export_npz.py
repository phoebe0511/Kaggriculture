"""把訓練好的 torch checkpoint 轉成 `.npz`，給比賽端的 numpy 前向用。

    python -m serving.export_npz --ckpt model/checkpoints/best.pt --out model/weights.npz

⚠️ 這支**會** import torch（它跑在開發側）。比賽端載入的是產出的 `.npz`，
走 `serving/npz_forward.py`，那支不碰 torch。

`ENCODER_VERSION` 一起寫進檔案。載入時版本不符要直接拋錯，不要試圖相容 ——
schema 變了，舊權重讀到的輸入語意就不一樣（`contracts.py` docstring）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contracts as C  # noqa: E402


def export(ckpt_path, out_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    version = int(ckpt.get("encoder_version", -1))
    if version != C.ENCODER_VERSION:
        raise SystemExit(
            f"checkpoint 是 ENCODER_VERSION {version}，現在是 {C.ENCODER_VERSION}"
            " —— 重新訓練，不要硬轉")

    arrays = {k: v.detach().cpu().numpy().astype(np.float32)
              for k, v in ckpt["state_dict"].items()}
    arrays["encoder_version"] = np.asarray([C.ENCODER_VERSION], dtype=np.int32)
    arrays["width"] = np.asarray([int(ckpt["width"])], dtype=np.int32)
    # 🩸 `labels` 一定要存。它決定 op head 預測的是「當下這一步」(immediate)
    # 還是「段落終點動作」(target) —— 兩者的 ENCODER_VERSION 一樣，所以版本
    # 檢查放行，但把 target 的權重餵給 `agents/gen2_model.py` 會**整局 PASS
    # 拿 0 分而且完全不報錯**（2026-08-21 踩到：MOVE 0.0%、PASS 80.8%）。
    arrays["labels"] = np.asarray(
        [str(ckpt.get("labels", "target"))], dtype="U16")
    arrays["blocks"] = np.asarray([int(ckpt["blocks"])], dtype=np.int32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    size = out_path.stat().st_size
    print(f"{ckpt_path} -> {out_path}")
    print(f"  張量 {len(ckpt['state_dict'])}  參數 "
          f"{sum(v.size for k, v in arrays.items() if v.dtype == np.float32):,}")
    print(f"  width {int(ckpt['width'])}  blocks {int(ckpt['blocks'])}  "
          f"ENCODER_VERSION {C.ENCODER_VERSION}")
    print(f"  檔案大小 {size / 2**20:.2f} MB  "
          f"（submission 上限 100 MiB）")
    if "val_op_acc" in ckpt:
        print(f"  訓練時的驗證準確率 {float(ckpt['val_op_acc']):.4f}")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="model/checkpoints/best.pt")
    ap.add_argument("--out", default="model/weights.npz")
    args = ap.parse_args(argv)
    export(args.ckpt, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
