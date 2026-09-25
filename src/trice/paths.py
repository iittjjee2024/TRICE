"""
Central path resolution for the pipeline scripts.

By default every path is derived from the repository root (``student_resource/dataset``
for the data, ``artifacts/`` for generated files, ``output/`` for the submission TSVs).
That layout is convenient locally but awkward on hosted environments such as Kaggle, where
the dataset is mounted read-only somewhere under ``/kaggle/input`` and only ``/kaggle/working``
is writable.

Rather than depend on symlinks (which are fragile), each location can be overridden by an
environment variable:

======================  ==================================================================
variable                overrides
======================  ==================================================================
``TRICE_DATA_DIR``      the dataset folder that holds ``train/`` and ``test/``
``TRICE_ARTIFACTS_DIR`` where the record store, runs and models are written
``TRICE_OUTPUT_DIR``    where ``matching_results.tsv`` / ``candidate_pairs.tsv`` are written
======================  ==================================================================

The Kaggle notebook sets these once; local runs need none of them.
"""

from __future__ import annotations

import os


def dataset_dir(root: str) -> str:
    """Folder containing ``train/`` and ``test/``. Env override: ``TRICE_DATA_DIR``."""
    env = os.environ.get("TRICE_DATA_DIR")
    if env:
        return env
    return os.path.join(root, "student_resource", "dataset")


def artifacts_dir(root: str) -> str:
    """Writable folder for the record store, runs and models.

    Env override: ``TRICE_ARTIFACTS_DIR``. Created if it does not exist.
    """
    env = os.environ.get("TRICE_ARTIFACTS_DIR")
    path = env or os.path.join(root, "artifacts")
    os.makedirs(path, exist_ok=True)
    return path


def output_dir(root: str) -> str:
    """Folder for the submission TSVs. Env override: ``TRICE_OUTPUT_DIR``."""
    env = os.environ.get("TRICE_OUTPUT_DIR")
    path = env or os.path.join(root, "output")
    os.makedirs(path, exist_ok=True)
    return path


__all__ = ["dataset_dir", "artifacts_dir", "output_dir"]
