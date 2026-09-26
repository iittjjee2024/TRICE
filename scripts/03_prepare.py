"""
Normalise every source TSV into the compact Parquet record store.

Usage:
    python scripts/03_prepare.py                 # both splits, all sources
    python scripts/03_prepare.py --split test
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trice.paths import artifacts_dir, dataset_dir  # noqa: E402

DATA = dataset_dir(ROOT)
ART = artifacts_dir(ROOT)
STORE = os.path.join(ART, "store")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--block-lines", type=int, default=50_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from trice.records import normalize_file_to_parquet, store_path  # noqa: E402

    os.makedirs(STORE, exist_ok=True)

    vpath = os.path.join(ART, "variants.json")
    addr_map, name_map = {}, {}
    if os.path.isfile(vpath):
        with open(vpath, encoding="utf-8") as fh:
            v = json.load(fh)
        addr_map = v.get("token_map", {})
        name_map = v.get("name_map", {})
        log(f"loaded mined variants: {len(addr_map)} address, {len(name_map)} name")
    else:
        log("WARNING: artifacts/variants.json missing - run 02_mine_variants.py first")

    splits = ["train", "test"] if args.split == "both" else [args.split]
    log(f"using {args.workers} worker processes")

    import pyarrow.parquet as pq

    def existing_rows(path: str) -> int:
        """Row count of an existing store file, or -1 if absent/unreadable."""
        if not os.path.isfile(path):
            return -1
        try:
            return pq.ParquetFile(path).metadata.num_rows
        except Exception:
            return -1

    with mp.Pool(args.workers) as pool:
        for split in splits:
            for src in (1, 2, 3):
                out = store_path(STORE, split, src)
                raw = os.path.join(DATA, split, f"{split}_source{src}.tsv")

                if not os.path.isfile(raw):
                    raise SystemExit(
                        f"source file not found: {raw}\n"
                        f"  DATA (TRICE_DATA_DIR) resolves to: {DATA}\n"
                        f"  expected the dataset under {DATA}/{split}/.")

                # Regenerate unless a NON-EMPTY store already exists. A previous broken run
                # (e.g. reading a 254-byte macOS sidecar) could have written an empty
                # Parquet; skipping it silently is what caused 'train countries = []'.
                have = existing_rows(out)
                if have > 0 and not args.force:
                    log(f"skip {split}/source{src} (exists, {have:,} rows)")
                    continue
                if have == 0:
                    log(f"  {split}/source{src}: existing store is EMPTY - regenerating")

                t0 = time.time()
                last = [0.0]

                def progress(done: int) -> None:
                    now = time.time()
                    if now - last[0] > 15:
                        last[0] = now
                        log(f"    {split}/source{src}: {done:,} rows "
                            f"({done / (now - t0):,.0f}/s)")

                n = normalize_file_to_parquet(
                    raw, src, out, addr_map, name_map, pool=pool,
                    block_lines=args.block_lines, progress=progress)
                size = os.path.getsize(out) / 1e6
                log(f"  {split}/source{src}: {n:,} rows -> {out} "
                    f"({size:,.0f} MB) in {time.time() - t0:.0f}s")
                if n == 0:
                    raise SystemExit(
                        f"{split}/source{src} produced 0 rows from {raw} "
                        f"({os.path.getsize(raw):,} bytes). The source file is not the "
                        f"real dataset (likely a macOS '._' sidecar or an empty upload).")

    log("done")


if __name__ == "__main__":
    mp.freeze_support()
    main()
