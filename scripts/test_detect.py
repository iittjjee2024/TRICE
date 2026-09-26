"""Verify the dataset detector skips macOS AppleDouble sidecars and picks the real file."""
import os
import tempfile

REQUIRED = [("train", "train_source1.tsv"), ("train", "train_source2.tsv"),
            ("train", "train_source3.tsv"), ("train", "train_ground_truth.tsv"),
            ("test", "test_source1.tsv"), ("test", "test_source2.tsv"),
            ("test", "test_source3.tsv")]


def _is_junk(path):
    base = os.path.basename(path)
    return base.startswith("._") or "__MACOSX" in path.replace("\\", "/").split("/")


def _norm(name):
    return os.path.basename(name).lower().lstrip("-_ ")


def locate(root):
    all_tsv = []
    for r, _d, fs in os.walk(root):
        for f in fs:
            p = os.path.join(r, f)
            if f.lower().endswith(".tsv") and not _is_junk(p):
                all_tsv.append(p)
    found = {}
    for split, fname in REQUIRED:
        exact = [p for p in all_tsv if os.path.basename(p).lower() == fname.lower()]
        hits = exact or [p for p in all_tsv if _norm(p) == fname.lower()]
        hits = [p for p in hits if os.path.getsize(p) > 1024] or hits
        if hits:
            found[(split, fname)] = sorted(hits, key=os.path.getsize, reverse=True)[0]
    return found


d = tempfile.mkdtemp(prefix="mac_ds_")
base = os.path.join(d, "datasets", "host45467", "dataset1")
os.makedirs(base)
os.makedirs(os.path.join(d, "__MACOSX", "dataset1"))

APPLEDOUBLE = (b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        "
               b"com.apple.quarantine\x00")

for _split, fname in REQUIRED:
    # the real ~2 MB data file
    real = os.path.join(base, fname)
    with open(real, "w", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for i in range(40000):
            f.write(f"S1-{i}\tAcme {i} Ltd\t{i} Main St, Austin, TX\tUS\n")
    # the macOS AppleDouble sidecar (254 bytes of junk) right next to it
    with open(os.path.join(base, "._" + fname), "wb") as f:
        f.write(APPLEDOUBLE)
    # and one inside __MACOSX for good measure
    with open(os.path.join(d, "__MACOSX", "dataset1", "._" + fname), "wb") as f:
        f.write(APPLEDOUBLE)

found = locate(d)
missing = [f"{s}/{f}" for (s, f) in REQUIRED if (s, f) not in found]
print("matched", len(found), "of 7; missing:", missing)
for (s, f), p in sorted(found.items()):
    size = os.path.getsize(p)
    print(f"  {s}/{f}: {size/1e6:.2f} MB  base={os.path.basename(p)}")
    assert not os.path.basename(p).startswith("._"), "picked an AppleDouble sidecar!"
    assert size > 1024, "picked a tiny junk file!"
assert not missing, "failed to resolve all 7"

import shutil  # noqa: E402
shutil.rmtree(d, ignore_errors=True)
print("\nDETECTOR OK: skips ._ sidecars and __MACOSX, picks the real data file")
