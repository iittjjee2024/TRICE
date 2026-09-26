# =====================================================================================
# TRICE — single-cell Kaggle runner
# =====================================================================================
# Paste this ENTIRE block into ONE Kaggle cell and run it. It is fully self-contained: it
# clones the repo, finds the dataset however it was uploaded, wires paths via env vars,
# and runs every stage. Nothing depends on any other cell, so a stale notebook cannot
# cause the errors you have been hitting.
#
# Before running: attach the challenge dataset via 'Add Input', Internet = On,
# Accelerator = None.
# For a fast smoke test set SUBSET = 4000; for the real submission set SUBSET = None.
# =====================================================================================
import glob
import os
import subprocess
import sys
import time

SUBSET = 4000                      # entities/country for training; None = full run
REPO = "https://github.com/iittjjee2024/TRICE.git"
CODE = "/kaggle/working/TRICE"
WORK = "/kaggle/working/trice_run"
OUT = "/kaggle/working/output"

# ---- 1. clone / update the code ------------------------------------------------------
if os.path.isdir(CODE):
    subprocess.run(["git", "-C", CODE, "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, CODE], check=True)
head = subprocess.run(["git", "-C", CODE, "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print("repo at commit", head)

# ---- 2. dependencies (rest ship with Kaggle) -----------------------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz", "Unidecode"],
               check=True)

# ---- 3. locate the dataset regardless of how it was uploaded -------------------------
REQUIRED = [("train", "train_source1.tsv"), ("train", "train_source2.tsv"),
            ("train", "train_source3.tsv"), ("train", "train_ground_truth.tsv"),
            ("test", "test_source1.tsv"), ("test", "test_source2.tsv"),
            ("test", "test_source3.tsv")]


def _is_junk(path):
    """Skip macOS AppleDouble sidecars ('._name') and __MACOSX entries.

    A dataset zipped on a Mac carries a tiny '._train_source1.tsv' resource-fork file
    (~254 bytes of 'Mac OS X ... com.apple.quarantine' metadata) next to each real file.
    Matching one of those instead of the ~200 MB data file is the failure this guards.
    """
    base = os.path.basename(path)
    return base.startswith("._") or "__MACOSX" in path.replace("\\", "/").split("/")


def _norm(name):
    # strip only leading punctuation that is NOT the AppleDouble marker (handled above)
    return os.path.basename(name).lower().lstrip("-_ ")


all_tsv = [os.path.join(r, f) for r, _d, fs in os.walk("/kaggle/input")
           for f in fs if f.lower().endswith(".tsv") and not _is_junk(os.path.join(r, f))]
found = {}
for split, fname in REQUIRED:
    # prefer an EXACT basename match; fall back to the normalised stem
    exact = [p for p in all_tsv if os.path.basename(p).lower() == fname.lower()]
    hits = exact or [p for p in all_tsv if _norm(p) == fname.lower()]
    # a real source file is megabytes; never accept a tiny sidecar that slipped through
    hits = [p for p in hits if os.path.getsize(p) > 1024] or hits
    if hits:
        found[(split, fname)] = sorted(hits, key=os.path.getsize, reverse=True)[0]
missing = [f"{s}/{f}" for (s, f) in REQUIRED if (s, f) not in found]
if missing:
    print("MISSING:", missing)
    print("tsv files seen under /kaggle/input (excluding macOS sidecars):")
    for p in all_tsv:
        print("  ", p, os.path.getsize(p), "bytes")
    raise SystemExit("Attach the dataset via 'Add Input' (it must contain the 7 TSVs).")
print("selected files:")
for (s, f), p in sorted(found.items()):
    print(f"  {s}/{f}: {os.path.getsize(p)/1e6:.1f} MB  <- {p}")

os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.join(WORK, "dataset")
for (split, fname), src in found.items():
    d = os.path.join(DATA_DIR, split)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, fname)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)
print("dataset wired ->", DATA_DIR)

# ---- 4. path env vars for the pipeline scripts ---------------------------------------
ART = os.path.join(WORK, "artifacts")
os.makedirs(ART, exist_ok=True)
os.environ["TRICE_DATA_DIR"] = DATA_DIR
os.environ["TRICE_ARTIFACTS_DIR"] = ART
os.environ["TRICE_OUTPUT_DIR"] = OUT

# ---- 5. preflight: tolerant read straight from the cloned code -----------------------
sys.path.insert(0, os.path.join(CODE, "src"))
import importlib
import trice.paths
importlib.reload(trice.paths)
from trice.paths import open_text
assert "errors" in __import__("inspect").getsource(open_text), \
    "cloned code is stale — the pull did not update src/trice/paths.py"
gt = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
with open_text(gt) as fh:
    print("ground truth header:", next(fh).rstrip().split("\t"))
print("PREFLIGHT OK\n")

# ---- 6. run every stage, streaming output; reprint the tail of a failing stage -------
import collections

TRAIN_ENTITIES = 70000 if SUBSET is None else SUBSET
STAGES = [
    (["scripts/02_mine_variants.py", "--sample", "250000"], "mine variants"),
    (["scripts/03_prepare.py", "--workers", "3"], "prepare record store"),
    (["scripts/05_train.py", "--entities", str(TRAIN_ENTITIES), "--run-id", "kaggle"],
     "train + validate"),
    (["scripts/07_tune_decision.py", "--run-id", "kaggle"], "tune decision"),
    (["scripts/06_infer.py", "--run-id", "kaggle", "--query-batch", "120000"],
     "full test inference"),
]


def run(args, title):
    print("=" * 78, f"\n{title}\n" + "=" * 78, flush=True)
    t0 = time.time()
    tail = collections.deque(maxlen=40)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen([sys.executable, *args], cwd=CODE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for line in p.stdout:
        print(line, end="")
        tail.append(line)
    p.wait()
    print(f"\n[{title}] exit={p.returncode}  {time.time() - t0:.0f}s", flush=True)
    if p.returncode != 0:
        print("\n----- last lines of failing stage -----\n" + "".join(tail))
        raise RuntimeError(f"{title} failed (exit {p.returncode})")


for args, title in STAGES:
    run(args, title)

# ---- 7. validate + surface the submittable TSV ---------------------------------------
sub = subprocess.run(
    [sys.executable, os.path.join(CODE, "student_resource/utils/validate_submission.py"),
     "--matching", os.path.join(OUT, "matching_results.tsv"),
     "--candidate", os.path.join(OUT, "candidate_pairs.tsv"),
     "--test-dir", os.path.join(DATA_DIR, "test")],
    cwd=CODE, capture_output=True, text=True)
print(sub.stdout, sub.stderr)

import shutil
for f in ("matching_results.tsv", "candidate_pairs.tsv"):
    src = os.path.join(OUT, f)
    if os.path.isfile(src):
        shutil.copy(src, os.path.join("/kaggle/working", f))
        print(f"{f}: {os.path.getsize(src) / 1e6:.1f} MB -> /kaggle/working/{f}")
print("\nDONE — download matching_results.tsv from the Output tab.")
