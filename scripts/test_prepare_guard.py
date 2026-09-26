"""Verify 03_prepare fails loudly on empty/sidecar inputs and works on real ones."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def make_dataset(base, rows_per_file):
    for split in ("train", "test"):
        d = os.path.join(base, split)
        os.makedirs(d, exist_ok=True)
        srcs = [f"{split}_source{i}.tsv" for i in (1, 2, 3)]
        if split == "train":
            srcs.append("train_ground_truth.tsv")
        for fn in srcs:
            with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
                if "ground_truth" in fn:
                    f.write("source1_entity_id\tmatched_entity_ids\n")
                    for i in range(rows_per_file):
                        f.write(f"S1-{i}\tS2-{i}\n")
                else:
                    f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                    for i in range(rows_per_file):
                        src = fn[len(split) + 1:len(split) + 2] if False else fn.split("_source")[1][0] if "_source" in fn else "1"
                        f.write(f"S{src}-{i}\tAcme {i} Ltd\t{i} Main St, Austin, TX\tUS\n")


def run_prepare(base, art, force=True):
    env = {**os.environ, "TRICE_DATA_DIR": base, "TRICE_ARTIFACTS_DIR": art,
           "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    args = [PY, "scripts/03_prepare.py", "--workers", "1"]
    if force:
        args.append("--force")
    return subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)


# 1. a macOS sidecar masquerading as the source: 254 bytes of junk -> must fail loudly
d1 = tempfile.mkdtemp(prefix="ds_junk_")
os.makedirs(os.path.join(d1, "train"))
os.makedirs(os.path.join(d1, "test"))
JUNK = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X" + b"\x00" * 200
for split in ("train", "test"):
    for fn in ([f"{split}_source{i}.tsv" for i in (1, 2, 3)] +
               (["train_ground_truth.tsv"] if split == "train" else [])):
        with open(os.path.join(d1, split, fn), "wb") as f:
            f.write(JUNK)
art1 = tempfile.mkdtemp(prefix="art_junk_")
r = run_prepare(d1, art1)
print("=== junk-input prepare ===  exit", r.returncode)
tail = (r.stdout + r.stderr)[-400:]
assert r.returncode != 0, "should have failed on junk input"
assert "0 rows" in (r.stdout + r.stderr) or "produced 0 rows" in (r.stdout + r.stderr), tail
print("  correctly failed with a clear message:")
print("  ", [ln for ln in (r.stdout + r.stderr).splitlines() if "0 rows" in ln][:1])

# 2. a real (tiny) dataset -> must succeed and produce non-empty stores
d2 = tempfile.mkdtemp(prefix="ds_ok_")
make_dataset(d2, 500)
art2 = tempfile.mkdtemp(prefix="art_ok_")
r = run_prepare(d2, art2)
print("\n=== valid-input prepare ===  exit", r.returncode)
print((r.stdout + r.stderr)[-500:])
assert r.returncode == 0, "valid dataset should prepare cleanly"

import pyarrow.parquet as pq  # noqa: E402
store = os.path.join(art2, "store")
for fn in os.listdir(store):
    n = pq.ParquetFile(os.path.join(store, fn)).metadata.num_rows
    print(f"  {fn}: {n} rows")
    assert n > 0

import shutil  # noqa: E402
for p in (d1, art1, d2, art2):
    shutil.rmtree(p, ignore_errors=True)
print("\nPREPARE GUARD OK")
