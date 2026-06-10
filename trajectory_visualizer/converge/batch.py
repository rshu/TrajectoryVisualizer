"""Batch comparison: manifest handling, execution, aggregation, and pattern promotion."""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

from .alignment import build_comparison_report


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    task_id: str
    reference: str
    compared: str
    anchor: str | None = None


def parse_manifest(manifest_path: str, *, confine: bool = True) -> list[ManifestEntry]:
    """Parse and validate a batch manifest JSON file.

    The manifest is a JSON array of objects with required fields task_id,
    reference, compared, and optional anchor. Paths are resolved relative
    to the manifest file's directory.

    Security: when ``confine`` is True (the default, used for UNTRUSTED manifests
    such as those uploaded through the web UI), referenced paths must be relative
    and must stay within the manifest's directory — absolute paths and ``..``
    escapes are rejected. This prevents a hostile manifest from reading arbitrary
    files (e.g. ``/etc/passwd``, ``~/.ssh/id_rsa``) into the rendered report.
    Trusted callers (the local CLI) may pass ``confine=False`` to allow absolute
    paths to a trajectory store outside the manifest directory.
    """
    with open(manifest_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON array")

    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    entries: list[ManifestEntry] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry {i} must be an object")

        for field_name in ("task_id", "reference", "compared"):
            if field_name not in item:
                raise ValueError(
                    f"Manifest entry {i} (task_id={item.get('task_id', '?')}) "
                    f"missing required field '{field_name}'"
                )

        def _resolve(p: str, field_label: str) -> str:
            if not confine:
                # Trusted (local CLI) mode: legacy behaviour.
                if os.path.isabs(p):
                    return p
                return os.path.normpath(os.path.join(base_dir, p))
            # Untrusted mode: relative + contained within base_dir only.
            if os.path.isabs(p):
                raise ValueError(
                    f"Manifest entry {i} (task_id={item.get('task_id', '?')}): "
                    f"absolute paths are not allowed for '{field_label}' ({p!r}); "
                    f"use a path relative to the manifest directory."
                )
            resolved = os.path.normpath(os.path.join(base_dir, p))
            if os.path.commonpath([base_dir, resolved]) != base_dir:
                raise ValueError(
                    f"Manifest entry {i} (task_id={item.get('task_id', '?')}): "
                    f"'{field_label}' path escapes the manifest directory ({p!r})."
                )
            return resolved

        ref_path = _resolve(item["reference"], "reference")
        cmp_path = _resolve(item["compared"], "compared")
        anchor_path = _resolve(item["anchor"], "anchor") if item.get("anchor") else None

        # Validate files exist
        for label, path in [("reference", ref_path), ("compared", cmp_path)]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Manifest entry {i} (task_id={item['task_id']}): "
                    f"{label} file not found: {path}"
                )
        if anchor_path and not os.path.isfile(anchor_path):
            raise FileNotFoundError(
                f"Manifest entry {i} (task_id={item['task_id']}): "
                f"anchor file not found: {anchor_path}"
            )

        entries.append(ManifestEntry(
            task_id=item["task_id"],
            reference=ref_path,
            compared=cmp_path,
            anchor=anchor_path,
        ))

    return entries


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    task_id: str
    report: dict | None = None
    error: str | None = None


def run_batch(
    entries: list[ManifestEntry],
    token_rate: float = 50.0,
    fuzzy_commands: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[BatchResult]:
    """Run pairwise comparison for each manifest entry.

    Error isolation: a failure on one task does not abort the batch.
    """
    results: list[BatchResult] = []
    total = len(entries)

    for i, entry in enumerate(entries):
        try:
            report = build_comparison_report(
                ref_file=entry.reference,
                cmp_file=entry.compared,
                token_rate=token_rate,
                fuzzy_commands=fuzzy_commands,
                anchor_patch=entry.anchor,
                task_id=entry.task_id,
            )
            results.append(BatchResult(task_id=entry.task_id, report=report))
        except Exception as e:
            results.append(BatchResult(task_id=entry.task_id, error=str(e)))

        if progress_callback:
            progress_callback(i + 1, total)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _safe_stat(values: list[float], func, default=None):
    """Compute a statistic, returning default if values is empty."""
    if not values:
        return default
    return round(func(values), 4)


def _percentile(values: list[float], q: float) -> float:
    """Compute percentile using nearest-rank."""
    if not values:
        return 0.0
    vals = sorted(values)
    idx = max(0, min(len(vals) - 1, int((len(vals) - 1) * q)))
    return vals[idx]


def aggregate_reports(results: list[BatchResult]) -> dict[str, Any]:
    """Compute aggregate statistics across successful task reports.

    Returns dict with per-metric stats (count, mean, median, p5, p95, min, max, stdev).
    """
    successful = [r for r in results if r.report is not None]
    if not successful:
        return {"task_count": 0, "success_count": 0, "failure_count": len(results), "metrics": {}}

    # Define which metrics to aggregate and where to find them
    metric_paths = {
        "reference_recall": lambda r: r["alignment"].get("reference_recall"),
        "behavioral_precision": lambda r: r["alignment"].get("behavioral_precision"),
        "alignment_f1": lambda r: r["alignment"].get("alignment_f1"),
        "overhead_ratio": lambda r: r["alignment"].get("overhead_ratio"),
        "harmful_ratio": lambda r: r["alignment"].get("harmful_ratio"),
    }

    # Anchor metrics (only from anchored tasks)
    anchor_metric_paths = {
        "anchor_write_precision_ref": lambda r: (r.get("anchor_analysis") or {}).get("reference", {}).get("write_precision"),
        "anchor_write_precision_cmp": lambda r: (r.get("anchor_analysis") or {}).get("compared", {}).get("write_precision"),
        "anchor_write_recall_ref": lambda r: (r.get("anchor_analysis") or {}).get("reference", {}).get("write_recall"),
        "anchor_write_recall_cmp": lambda r: (r.get("anchor_analysis") or {}).get("compared", {}).get("write_recall"),
        "off_patch_write_ratio_ref": lambda r: (r.get("anchor_analysis") or {}).get("reference", {}).get("off_patch_write_ratio"),
        "off_patch_write_ratio_cmp": lambda r: (r.get("anchor_analysis") or {}).get("compared", {}).get("off_patch_write_ratio"),
    }

    metrics: dict[str, dict] = {}

    # Aggregate standard metrics
    for name, extractor in metric_paths.items():
        values = []
        for r in successful:
            v = extractor(r.report)
            if v is not None:
                values.append(float(v))
        if values:
            metrics[name] = {
                "count": len(values),
                "mean": _safe_stat(values, statistics.mean),
                "median": _safe_stat(values, statistics.median),
                "p5": round(_percentile(values, 0.05), 4),
                "p95": round(_percentile(values, 0.95), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "stdev": _safe_stat(values, statistics.stdev) if len(values) >= 2 else 0.0,
            }

    # Aggregate anchor metrics (only from anchored reports)
    anchored_count = sum(1 for r in successful if r.report.get("anchor_analysis") is not None)
    for name, extractor in anchor_metric_paths.items():
        values = []
        for r in successful:
            if r.report.get("anchor_analysis") is None:
                continue
            v = extractor(r.report)
            if v is not None:
                values.append(float(v))
        if values:
            metrics[name] = {
                "count": len(values),
                "mean": _safe_stat(values, statistics.mean),
                "median": _safe_stat(values, statistics.median),
                "p5": round(_percentile(values, 0.05), 4),
                "p95": round(_percentile(values, 0.95), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "stdev": _safe_stat(values, statistics.stdev) if len(values) >= 2 else 0.0,
                "note": f"from {len(values)} anchored tasks of {len(successful)} total",
            }

    return {
        "task_count": len(results),
        "success_count": len(successful),
        "failure_count": len(results) - len(successful),
        "anchored_count": anchored_count,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Pattern frequency and promotion
# ---------------------------------------------------------------------------

def compute_pattern_frequency(results: list[BatchResult]) -> dict[str, dict]:
    """Count how many tasks each divergence pattern type appears in.

    Returns dict mapping pattern type to {count, prevalence, tasks}.
    """
    successful = [r for r in results if r.report is not None]
    total = len(successful)
    if total == 0:
        return {}

    type_tasks: dict[str, list[str]] = {}
    for r in successful:
        seen_types: set[str] = set()
        for p in r.report.get("patterns", []):
            ptype = p.get("type", "unknown")
            if ptype not in seen_types:
                type_tasks.setdefault(ptype, []).append(r.task_id)
                seen_types.add(ptype)

    frequency: dict[str, dict] = {}
    for ptype, tasks in sorted(type_tasks.items(), key=lambda x: -len(x[1])):
        frequency[ptype] = {
            "count": len(tasks),
            "prevalence": round(len(tasks) / total, 4),
            "tasks": tasks,
        }

    return frequency


def promote_patterns(
    frequency: dict[str, dict],
    min_tasks: int = 3,
    min_prevalence: float = 0.5,
) -> dict[str, str]:
    """Determine evidence level for each pattern type.

    Returns dict mapping pattern type to evidence_level.
    """
    levels: dict[str, str] = {}
    for ptype, data in frequency.items():
        if data["count"] >= min_tasks and data["prevalence"] >= min_prevalence:
            levels[ptype] = "supported_finding"
        else:
            levels[ptype] = "single_pair_hypothesis"
    return levels


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------

def compute_consistency(aggregate: dict) -> dict[str, float | None]:
    """Compute coefficient of variation (stdev/mean) per metric.

    Lower CV = more consistent across tasks.
    """
    consistency: dict[str, float | None] = {}
    for name, stats in aggregate.get("metrics", {}).items():
        mean = stats.get("mean", 0)
        stdev = stats.get("stdev", 0)
        if mean and mean != 0 and stdev is not None:
            consistency[name] = round(stdev / abs(mean), 4)
        else:
            consistency[name] = None
    return consistency


# ---------------------------------------------------------------------------
# Batch report
# ---------------------------------------------------------------------------

def build_batch_report(
    manifest_path: str,
    results: list[BatchResult],
    aggregate: dict,
    pattern_frequency: dict,
    promoted: dict,
    consistency: dict,
) -> dict[str, Any]:
    """Assemble the full aggregate batch report."""
    per_task = []
    for r in results:
        entry: dict[str, Any] = {"task_id": r.task_id}
        if r.error:
            entry["status"] = "failed"
            entry["error"] = r.error
        else:
            entry["status"] = "success"
            outcome = r.report.get("outcome", {})
            alignment = r.report.get("alignment", {})
            entry["outcome"] = {
                "reference_success": outcome.get("reference_success"),
                "compared_success": outcome.get("compared_success"),
            }
            entry["alignment_summary"] = {
                "recall": alignment.get("reference_recall"),
                "precision": alignment.get("behavioral_precision"),
                "f1": alignment.get("alignment_f1"),
                "harmful_ratio": alignment.get("harmful_ratio"),
            }
            entry["anchor_mode"] = r.report.get("anchor_mode")
            entry["pattern_count"] = len(r.report.get("patterns", []))
        per_task.append(entry)

    # Pattern frequency with promotion
    pattern_table = {}
    for ptype, data in pattern_frequency.items():
        pattern_table[ptype] = {
            **data,
            "evidence_level": promoted.get(ptype, "single_pair_hypothesis"),
        }

    return {
        "manifest": manifest_path,
        "summary": aggregate,
        "consistency": consistency,
        "pattern_frequency": pattern_table,
        "per_task": per_task,
    }
