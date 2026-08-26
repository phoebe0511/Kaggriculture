"""把 cp950 寫出來的訓練 log 濾掉 OpenSpiel 雜訊、轉成 UTF-8。

用法：python tools/_viewlog.py temp/round7-train.log [輸出檔]
"""
import io, sys
raw = open(sys.argv[1], "rb").read()
for enc in ("utf-8", "cp950", "big5"):
    try:
        t = raw.decode(enc)
        if "訓練" in t or "epoch" in t:
            break
    except Exception:
        pass
else:
    t = raw.decode("utf-8", "replace")
lines = t.split("\n")
start = next((i for i, l in enumerate(lines) if "訓練" in l or l.startswith("epoch")), 0)
out = [l.rstrip() for l in lines[start:] if l.strip()]
dst = sys.argv[2] if len(sys.argv) > 2 else "temp/_view.txt"
io.open(dst, "w", encoding="utf-8").write("\n".join(out))
print(f"{len(out)} 行 -> {dst}")
