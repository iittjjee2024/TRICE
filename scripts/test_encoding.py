"""Confirm the pipeline tolerates non-UTF-8 bytes in the source TSVs.

Reproduces the Kaggle failure (`UnicodeDecodeError: 0xcc`) by writing a tiny dataset with a
raw 0xCC byte in a name and address, then runs the read paths that previously crashed.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from trice.evaluate import parse_ground_truth        # noqa: E402
from trice.export import read_entity_ids             # noqa: E402
from trice.paths import open_text                    # noqa: E402
from trice.records import iter_blocks, normalize_file_to_parquet  # noqa: E402

d = tempfile.mkdtemp(prefix="enc_test_")

# a source1 file with a stray 0xCC byte (lone Latin-1 'Ì' lead byte) in name + address
s1 = os.path.join(d, "train_source1.tsv")
with open(s1, "wb") as f:
    f.write(b"entity_id\tbusiness_name\tbusiness_address\tcountry\n")
    f.write(b"S1-1\tCaf\xcc Ecole\t12 Rue Cl\xcc\tFrance\n")
    f.write(b"S1-2\tNormal Corp\t5 Main St, Austin, TX\tUS\n")

gt = os.path.join(d, "train_ground_truth.tsv")
with open(gt, "wb") as f:
    f.write(b"source1_entity_id\tmatched_entity_ids\n")
    f.write(b"S1-1\tS2-9\n")
    f.write(b"S1-2\t\n")

print("wrote test files with raw 0xCC bytes")

# 1. strict utf-8 read must fail (this is the bug we are fixing)
try:
    with open(s1, encoding="utf-8") as fh:
        fh.readlines()
    print("  UNEXPECTED: strict utf-8 did not raise")
except UnicodeDecodeError as e:
    print(f"  strict utf-8 raises as expected: {e}")

# 2. tolerant read via open_text must succeed
with open_text(s1) as fh:
    lines = fh.readlines()
print(f"  open_text read {len(lines)} lines OK")

# 3. iter_blocks (used by 03_prepare) must succeed
blocks = list(iter_blocks(s1, block_lines=10))
print(f"  iter_blocks read {sum(len(b) for b in blocks)} data lines OK")

# 4. ground-truth + entity-id readers
gtd = parse_ground_truth(gt)
print(f"  parse_ground_truth: {gtd}")
ids = read_entity_ids(s1)
print(f"  read_entity_ids: {ids}")

# 5. full normalize -> parquet path
out = os.path.join(d, "out.parquet")
n = normalize_file_to_parquet(s1, 1, out, {}, {}, block_lines=10)
print(f"  normalize_file_to_parquet wrote {n} rows to parquet OK")

import pandas as pd  # noqa: E402
df = pd.read_parquet(out)
print(df[["num", "name_core", "addr_alpha", "country"]].to_string(index=False))

import shutil  # noqa: E402
shutil.rmtree(d, ignore_errors=True)
print("\nALL ENCODING CHECKS PASSED")
