"""
Submission writers and a self-contained format validator.

Produces the two files the challenge asks for:

* ``matching_results.tsv`` -- ``source1_entity_id \\t matched_entity_ids``; the only file
  scored on the leaderboard.
* ``candidate_pairs.tsv``  -- ``source1_entity_id \\t candidate_entity_ids``; the **final**
  candidate set the matcher ran inference over, not an earlier blocking pass.

Every rule from the problem statement is enforced at write time rather than hoped for:

1. exactly one row per Source 1 test entity, **including** entities with no candidates
2. empty list for singletons (trailing tab, nothing after it)
3. no duplicate ids inside a list, no duplicate ``source1_entity_id`` rows
4. S2-/S3- ids only, never a self-match to Source 1
5. tab-separated, comma-joined ids, no quoting, UTF-8

:func:`validate_submission` re-reads the written files and re-checks all of the above plus
the matches-subset-of-candidates relation, so a green result here means the official
``utils/validate_submission.py`` will also pass.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"


def _write_id_list_file(path: str, header: str, entity_ids: Sequence[str],
                        lists: Iterable[Sequence[str]]) -> Tuple[int, int]:
    """Write one results-style TSV. Returns ``(n_rows, n_nonempty)``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n_rows = 0
    n_nonempty = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        for eid, ids in zip(entity_ids, lists):
            if ids:
                # dedupe while preserving order, and drop anything not S2-/S3-
                seen: Set[str] = set()
                clean: List[str] = []
                for i in ids:
                    if i in seen:
                        continue
                    if not (i.startswith("S2-") or i.startswith("S3-")):
                        continue
                    seen.add(i)
                    clean.append(i)
                if clean:
                    fh.write(f"{eid}\t{','.join(clean)}\n")
                    n_nonempty += 1
                else:
                    fh.write(f"{eid}\t\n")
            else:
                fh.write(f"{eid}\t\n")
            n_rows += 1
    return n_rows, n_nonempty


def write_matching_results(path: str, entity_ids: Sequence[str],
                           matches: Iterable[Sequence[str]]) -> Tuple[int, int]:
    return _write_id_list_file(path, MATCHING_HEADER, entity_ids, matches)


def write_candidate_pairs(path: str, entity_ids: Sequence[str],
                          candidates: Iterable[Sequence[str]]) -> Tuple[int, int]:
    return _write_id_list_file(path, CANDIDATE_HEADER, entity_ids, candidates)


def ids_from_codes(src: np.ndarray, num: np.ndarray) -> List[str]:
    """Rebuild ``S{src}-{num}`` strings from the compact encoding."""
    return [f"S{s}-{n}" for s, n in zip(src.tolist(), num.tolist())]


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------

def _read_id_list_file(path: str, expected_header: List[str]
                       ) -> Tuple[Dict[str, List[str]], List[str]]:
    issues: List[str] = []
    mapping: Dict[str, List[str]] = {}
    name = os.path.basename(path)
    if not os.path.isfile(path):
        return mapping, [f"{name}: file not found"]
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
        cols = [c.strip().lower() for c in header.rstrip("\n").split("\t")]
        if cols != expected_header:
            issues.append(f"{name}: header is {cols}, expected {expected_header}")
            return mapping, issues
        dup_rows: Set[str] = set()
        for line_no, line in enumerate(fh, start=2):
            s1, tab, rest = line.partition("\t")
            if not tab:
                if s1.strip():
                    issues.append(f"{name}: no tab on line {line_no}")
                continue
            if s1 in mapping:
                dup_rows.add(s1)
            rest = rest.rstrip("\n")
            mapping[s1] = rest.split(",") if rest else []
        if dup_rows:
            issues.append(f"{name}: {len(dup_rows)} duplicate source1_entity_id rows")
    return mapping, issues


def validate_submission(matching_path: str, candidate_path: str | None,
                        required_s1: Set[str],
                        valid_targets: Set[str] | None = None) -> dict:
    """Check both output files against every rule in the problem statement."""
    issues: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, bool] = {}

    matched, iss = _read_id_list_file(matching_path,
                                      ["source1_entity_id", "matched_entity_ids"])
    issues += iss
    if iss:
        return {"ok": False, "issues": issues, "warnings": warnings, "checks": checks}

    checks["one_row_per_s1"] = len(matched) == len(set(matched))
    missing = required_s1 - set(matched)
    extra = set(matched) - required_s1
    checks["all_test_s1_present"] = not missing
    checks["no_unknown_s1_rows"] = not extra
    if missing:
        issues.append(f"matching_results.tsv: {len(missing)} required S1 entities missing "
                      f"(e.g. {sorted(missing)[:3]})")
    if extra:
        issues.append(f"matching_results.tsv: {len(extra)} rows for S1 ids not in the "
                      f"test set (e.g. {sorted(extra)[:3]})")

    intra_dupes = 0
    self_match = 0
    bad_prefix = 0
    unknown = 0
    n_nonempty = 0
    n_ids = 0
    for s1, ids in matched.items():
        if not ids:
            continue
        n_nonempty += 1
        n_ids += len(ids)
        if len(ids) != len(set(ids)):
            intra_dupes += 1
        for i in ids:
            if i.startswith("S1-"):
                self_match += 1
            elif not (i.startswith("S2-") or i.startswith("S3-")):
                bad_prefix += 1
            elif valid_targets is not None and i not in valid_targets:
                unknown += 1

    checks["no_duplicate_ids_within_list"] = intra_dupes == 0
    checks["no_self_matches"] = self_match == 0
    checks["only_s2_s3_ids"] = bad_prefix == 0 and self_match == 0
    if intra_dupes:
        issues.append(f"matching_results.tsv: {intra_dupes} rows repeat an id in the list")
    if self_match:
        issues.append(f"matching_results.tsv: {self_match} S1 self-matches")
    if bad_prefix:
        issues.append(f"matching_results.tsv: {bad_prefix} ids without an S2-/S3- prefix")
    if valid_targets is not None:
        checks["all_ids_exist_in_test"] = unknown == 0
        if unknown:
            issues.append(f"matching_results.tsv: {unknown} ids absent from the test "
                          f"Source-2/3 files")
    else:
        warnings.append("id-existence check skipped (no target id set supplied)")

    cand_stats = None
    if candidate_path and os.path.isfile(candidate_path):
        cands, iss2 = _read_id_list_file(
            candidate_path, ["source1_entity_id", "candidate_entity_ids"])
        issues += iss2
        if not iss2:
            cmissing = required_s1 - set(cands)
            checks["candidates_cover_all_s1"] = not cmissing
            if cmissing:
                issues.append(f"candidate_pairs.tsv: {len(cmissing)} required S1 entities "
                              f"missing")
            offenders = [s1 for s1, ids in matched.items()
                         if set(ids) - set(cands.get(s1, []))]
            checks["matches_subset_of_candidates"] = not offenders
            if offenders:
                warnings.append(
                    f"{len(offenders)} entities have matched ids absent from "
                    f"candidate_pairs.tsv (e.g. {offenders[:3]}) - pipeline bug")
            cand_stats = {
                "rows": len(cands),
                "nonempty": sum(1 for v in cands.values() if v),
                "total_ids": sum(len(v) for v in cands.values()),
            }
    elif candidate_path:
        warnings.append(f"{candidate_path} not found - required in the submission zip")

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
        "matching": {
            "rows": len(matched),
            "nonempty": n_nonempty,
            "empty": len(matched) - n_nonempty,
            "total_ids": n_ids,
            "mean_ids_per_nonempty": round(n_ids / n_nonempty, 4) if n_nonempty else 0.0,
        },
        "candidates": cand_stats,
    }


def read_entity_ids(tsv_path: str) -> List[str]:
    """First-column ids of a source TSV, in file order."""
    out: List[str] = []
    with open(tsv_path, encoding="utf-8", errors="replace") as fh:
        next(fh)
        for line in fh:
            if line.strip():
                out.append(line.split("\t", 1)[0].strip())
    return out


__all__ = ["write_matching_results", "write_candidate_pairs", "validate_submission",
           "ids_from_codes", "read_entity_ids", "MATCHING_HEADER", "CANDIDATE_HEADER"]
