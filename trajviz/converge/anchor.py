"""Anchor file classification and per-class write metrics."""

from __future__ import annotations

import fnmatch
from typing import Any

from .canonical import CanonicalAction, _normalize_target, _targets_match


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

# Ordered list of (pattern, category) — first match wins.
# Priority: fixture > test > generated > spec > source
DEFAULT_FILE_CLASS_RULES: list[tuple[str, str]] = [
    # Fixtures / test data
    ("**/testdata/**", "fixture"),
    ("**/fixtures/**", "fixture"),
    ("**/*after_roundtrip*", "fixture"),
    # Tests
    ("**/*_test.go", "test"),
    ("**/*_test.py", "test"),
    ("**/test_*.py", "test"),
    ("**/*_spec.ts", "test"),
    ("**/*_spec.js", "test"),
    ("**/__tests__/**", "test"),
    ("**/test/e2e*/**", "test"),
    ("**/test/integration/**", "test"),
    # Generated code
    ("**/zz_generated.*", "generated"),
    ("**/*.pb.go", "generated"),
    ("**/generated.pb.go", "generated"),
    ("**/generated.protomessage.pb.go", "generated"),
    ("**/zz_generated.model_name.go", "generated"),
    # API specs / proto definitions
    ("**/openapi-spec/**", "spec"),
    ("**/*swagger*", "spec"),
    ("**/*.proto", "spec"),
    # Compatibility / lifecycle
    ("**/compatibility_lifecycle/**", "fixture"),
]


def classify_file(
    path: str,
    custom_rules: list[tuple[str, str]] | None = None,
) -> str:
    """Classify a single file path into a category.

    Applies custom_rules first, then DEFAULT_FILE_CLASS_RULES. First match wins.
    Returns one of: 'source', 'generated', 'test', 'fixture', 'spec'.
    """
    # Normalize for matching
    norm = path.replace("\\", "/")

    all_rules = (custom_rules or []) + DEFAULT_FILE_CLASS_RULES
    for pattern, category in all_rules:
        if fnmatch.fnmatch(norm, pattern):
            return category

    return "source"


def classify_anchor_files(
    anchor_files: set[str],
) -> tuple[dict[str, str], dict[str, int]]:
    """Classify all anchor files.

    No custom-rules parameter (C6): nothing could reach it — no entry point,
    CLI flag or UI control ever threaded one through compute_anchor_analysis,
    so it was permanently None. :func:`classify_file` keeps its own hook, which
    is the documented first-match-wins extension point.

    Returns:
        (file_to_class, class_counts) where:
        - file_to_class maps each file path to its category
        - class_counts maps each category to its count
    """
    file_to_class: dict[str, str] = {}
    class_counts: dict[str, int] = {
        "source": 0, "generated": 0, "test": 0, "fixture": 0, "spec": 0,
    }

    for f in sorted(anchor_files):
        cat = classify_file(f)
        file_to_class[f] = cat
        class_counts[cat] = class_counts.get(cat, 0) + 1

    return file_to_class, class_counts


# ---------------------------------------------------------------------------
# Anchor write metrics
# ---------------------------------------------------------------------------

def _match_anchor(target: str, anchor_files: set[str]) -> str | None:
    """Find which anchor file a target matches, if any."""
    for af in anchor_files:
        if _targets_match(target, _normalize_target(af)):
            return af
    return None


def compute_anchor_metrics(
    actions: list[CanonicalAction],
    anchor_files: set[str],
    file_to_class: dict[str, str],
) -> dict[str, Any]:
    """Compute anchor metrics for one trajectory.

    Returns dict with: write_precision, write_recall, write_recall_by_class,
    off_patch_write_ratio, time_to_first_anchor_read, time_to_first_anchor_write.
    """
    files_written: set[str] = set()
    anchor_files_written: set[str] = set()
    first_anchor_read: int | None = None
    first_anchor_write: int | None = None

    for a in actions:
        if a.action_type == "REASON":
            continue

        if a.action_type == "FILE_WRITE":
            files_written.add(a.target)
            matched = _match_anchor(a.target, anchor_files)
            if matched:
                anchor_files_written.add(matched)
                if first_anchor_write is None:
                    first_anchor_write = a.step_index

        if a.action_type == "FILE_READ":
            matched = _match_anchor(a.target, anchor_files)
            if matched and first_anchor_read is None:
                first_anchor_read = a.step_index

    total_written = len(files_written)
    anchor_written = len(anchor_files_written)
    total_anchor = len(anchor_files)

    # Precision: of files written, how many were in the anchor?
    write_precision = anchor_written / total_written if total_written > 0 else None

    # Recall: of anchor files, how many were written?
    write_recall = anchor_written / total_anchor if total_anchor > 0 else None

    # Per-class recall
    class_totals: dict[str, int] = {}
    class_written: dict[str, int] = {}
    for cat in file_to_class.values():
        class_totals[cat] = class_totals.get(cat, 0) + 1
    for af in anchor_files_written:
        cat = file_to_class.get(af, "source")
        class_written[cat] = class_written.get(cat, 0) + 1

    write_recall_by_class: dict[str, float | None] = {}
    for cat in sorted(set(class_totals.keys()) | set(class_written.keys())):
        ct = class_totals.get(cat, 0)
        cw = class_written.get(cat, 0)
        write_recall_by_class[cat] = round(cw / ct, 4) if ct > 0 else None

    # Off-patch ratio
    off_patch = total_written - anchor_written
    off_patch_write_ratio = off_patch / total_written if total_written > 0 else None

    return {
        "write_precision": round(write_precision, 4) if write_precision is not None else None,
        "write_recall": round(write_recall, 4) if write_recall is not None else None,
        "write_recall_by_class": write_recall_by_class,
        "off_patch_write_ratio": round(off_patch_write_ratio, 4) if off_patch_write_ratio is not None else None,
        "files_written": total_written,
        "anchor_files_written": anchor_written,
        "time_to_first_anchor_read": first_anchor_read,
        "time_to_first_anchor_write": first_anchor_write,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_anchor_analysis(
    ref_actions: list[CanonicalAction],
    cmp_actions: list[CanonicalAction],
    anchor_files: set[str],
) -> dict[str, Any] | None:
    """Compute full anchor analysis for both trajectories.

    Returns None if anchor_files is empty or None.
    """
    if not anchor_files:
        return None

    file_to_class, class_counts = classify_anchor_files(anchor_files)

    ref_metrics = compute_anchor_metrics(ref_actions, anchor_files, file_to_class)
    cmp_metrics = compute_anchor_metrics(cmp_actions, anchor_files, file_to_class)

    return {
        "total_anchor_files": len(anchor_files),
        "file_classes": class_counts,
        "reference": ref_metrics,
        "compared": cmp_metrics,
    }
