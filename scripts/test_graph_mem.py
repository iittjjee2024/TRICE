"""Measure transient RAM of build_graph_features with out= (memmap) at scale.

Builds a synthetic candidate set of N pairs over E entities, writes graph features into a
memmap via out=, and reports peak process RSS growth. Extrapolates per-pair transient cost
so we can predict the India (~58 M pair) peak on Kaggle's 30 GB budget.

Usage: python scripts/test_graph_mem.py [n_pairs] [n_entities]
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trice.graph import GraphConfig, N_GRAPH_FEATURES, build_graph_features  # noqa


def rss_gb() -> float:
    try:
        import psutil  # type: ignore
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        # Windows fallback via ctypes
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 1e9


def peak_gb() -> float:
    import ctypes
    import ctypes.wintypes as wt

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 1e9


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    rng = np.random.default_rng(0)

    # synthetic candidate arrays, same dtypes the real pipeline uses
    q_row = np.sort(rng.integers(0, e, size=n)).astype(np.int32)
    c_row = rng.integers(0, max(e * 6, 2), size=n).astype(np.int32)
    p1 = rng.random(n).astype(np.float32)
    cand_src = rng.integers(2, 4, size=n).astype(np.int8)
    postal = rng.integers(0, 500_000, size=n).astype(np.int64)
    digit = rng.integers(0, 500_000, size=n).astype(np.int64)
    skel = rng.integers(0, 500_000, size=n).astype(np.int64)
    cfg = GraphConfig()

    base = rss_gb()
    print(f"n_pairs={n:,} n_entities={e:,}")
    print(f"  input arrays resident, RSS={base:.2f} GB")

    tmp = tempfile.mkdtemp()
    g_path = os.path.join(tmp, "G.f32")
    G_mm = np.memmap(g_path, dtype=np.float32, mode="w+", shape=(n, N_GRAPH_FEATURES))

    # sample RSS in a background thread to capture the true transient peak
    import threading
    peak = {"v": base}
    stop = {"go": True}

    def sampler() -> None:
        while stop["go"]:
            r = rss_gb()
            if r > peak["v"]:
                peak["v"] = r
            time.sleep(0.02)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    t0 = time.time()
    # precompute per-candidate lookups once, as 06_infer.py now does
    g_q = q_row.astype(np.int64)
    g_c = c_row.astype(np.int64)
    build_graph_features(g_q, g_c, p1, cand_src, postal, digit, skel, cfg, out=G_mm)
    G_mm.flush()
    dt = time.time() - t0

    stop["go"] = False
    th.join(timeout=1.0)
    pk = peak["v"]
    transient = pk - base
    print(f"  built in {dt:.1f}s, peak RSS={pk:.2f} GB")
    print(f"  transient growth above resident = {transient:.2f} GB")
    print(f"  => {transient / n * 58_000_000:.2f} GB projected for 58 M India pairs")
    del G_mm
    try:
        os.remove(g_path)
        os.rmdir(tmp)
    except OSError:
        pass


if __name__ == "__main__":
    main()
