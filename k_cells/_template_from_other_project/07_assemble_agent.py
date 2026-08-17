# === CELL 7: assemble the MCTS agent into /kaggle/working (for notebook testing) ===
# main.py resolves its files from the cwd first (a notebook's cwd is /kaggle/working), so
# gather the scattered uploaded files (code dataset + deck + model) into /kaggle/working,
# next to where you run main.py. (On the real submission the grader assembles the zip into
# /kaggle_simulations/agent — this cell is only for notebook testing.)
import glob, os, shutil

WORK = "/kaggle/working"


def _find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)   # sources live under /kaggle/input
    assert hits, f"{name} not found in /kaggle/input — add the dataset/model that has it"
    return sorted(hits, key=os.path.getmtime, reverse=True)[0]     # newest


# the files main.py resolves from cwd. NOTE: main.py is NOT copied here — paste it directly
# into a notebook cell yourself (run this cell first, then your main.py cell).
# opp_recognize.py: the shared opponent-deck recognizer BOTH nn_agent.py (mode feature)
# and nn_agent_mcts.py (sandbox opp-model) import -- omitting it is an ImportError at run.
for f in ("nn_agent.py", "nn_agent_mcts.py", "opp_recognize.py", "deckak.csv", "akconfig.json"):
    src = _find(f)
    shutil.copy(src, os.path.join(WORK, f))
    print(f"  {f:18s} <- {src}")

# Step-2A/B (crus5_mcts+): the opponent-model DECK LIBRARY (decks/*.csv) and the advisor
# BRAINS (rules/*.py). main.py fails LOUD if these folders are missing.
for sub in ("decks", "rules"):
    hits = glob.glob(f"/kaggle/input/**/{sub}/*.*", recursive=True)
    assert hits, f"{sub}/ not found in /kaggle/input — upload the package's {sub} folder"
    dst = os.path.join(WORK, sub)
    os.makedirs(dst, exist_ok=True)
    src_dir = os.path.dirname(sorted(hits, key=os.path.getmtime, reverse=True)[0])
    for p in glob.glob(os.path.join(src_dir, "*.*")):
        shutil.copy(p, dst)
    print(f"  {sub + '/':18s} <- {src_dir} ({len(os.listdir(dst))} files)")

# the model: main.py expects it named model_mcts.pth
try:
    mp = _find("model_mcts.pth")
except AssertionError:
    mp = _find("*.pth")   # fall back to whatever single .pth is uploaded
shutil.copy(mp, os.path.join(WORK, "model_mcts.pth"))
print(f"  model_mcts.pth     <- {mp}")

# the FULL cg engine (api.py + libcg.so) from the cg-lib dataset -> /kaggle/working/cg so
# `import cg` resolves the complete package (the cabt env's cg has game.py only, no api.py).
_cg_hits = glob.glob("/kaggle/input/**/cg/api.py", recursive=True)
assert _cg_hits, "cg-lib dataset (with cg/api.py + cg/libcg.so) not attached"
_cg_src = os.path.dirname(sorted(_cg_hits, key=os.path.getmtime, reverse=True)[0])
_cg_dst = os.path.join(WORK, "cg")
shutil.rmtree(_cg_dst, ignore_errors=True)
shutil.copytree(_cg_src, _cg_dst)
print(f"  cg/ (full engine)  <- {_cg_src}")

print("\nassembled in /kaggle/working:", sorted(f for f in os.listdir(WORK) if not f.startswith(".")))
print("Now run cell 10 (it clears any stale cg import and uses /kaggle/working/cg).")
