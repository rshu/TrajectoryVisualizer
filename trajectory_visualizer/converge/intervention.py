"""Before/after intervention comparison: paired deltas, statistical testing, guardrail detection."""

from __future__ import annotations

import logging
import math
from typing import Any

from .batch import BatchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric direction definitions
# ---------------------------------------------------------------------------

# "higher" = higher is better; "lower" = lower is better
METRIC_DIRECTIONS: dict[str, str] = {
    "reference_recall": "higher",
    "behavioral_precision": "higher",
    "alignment_f1": "higher",
    "overhead_ratio": "lower",
    "harmful_ratio": "lower",
    "anchor_write_precision_ref": "higher",
    "anchor_write_precision_cmp": "higher",
    "anchor_write_recall_ref": "higher",
    "anchor_write_recall_cmp": "higher",
    "off_patch_write_ratio_ref": "lower",
    "off_patch_write_ratio_cmp": "lower",
}

# Pattern count metrics — lower count is better
PATTERN_DIRECTIONS: dict[str, str] = {
    "write_retry": "lower",
    "reverted_and_rewritten": "lower",
    "iterative_refinement": "lower",
    "broad_exploration": "lower",
    "error_recovery_overhead": "lower",
    "redundant_search": "lower",
    "dead_end_branch": "lower",
    "ordering_inefficiency": "lower",
    "premature_validation": "lower",
}


# ---------------------------------------------------------------------------
# Task pairing
# ---------------------------------------------------------------------------

def pair_tasks(
    before_results: list[BatchResult],
    after_results: list[BatchResult],
) -> tuple[list[tuple[BatchResult, BatchResult]], list[BatchResult], list[BatchResult]]:
    """Match tasks between before and after batches by task_id.

    Returns (paired, before_only, after_only).
    """
    before_map = {r.task_id: r for r in before_results if r.report is not None}
    after_map = {r.task_id: r for r in after_results if r.report is not None}

    all_ids = set(before_map.keys()) | set(after_map.keys())
    paired_ids = set(before_map.keys()) & set(after_map.keys())

    paired = [(before_map[tid], after_map[tid]) for tid in sorted(paired_ids)]
    before_only = [before_map[tid] for tid in sorted(set(before_map.keys()) - paired_ids)]
    after_only = [after_map[tid] for tid in sorted(set(after_map.keys()) - paired_ids)]

    return paired, before_only, after_only


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_metric(report: dict, metric_name: str) -> float | None:
    """Extract a metric value from a comparison report."""
    # Standard alignment metrics
    if metric_name in ("reference_recall", "behavioral_precision", "alignment_f1",
                       "overhead_ratio", "harmful_ratio"):
        return report.get("alignment", {}).get(metric_name)

    # Anchor metrics: "anchor_write_precision_ref" → strip "_ref" and "anchor_" prefix
    # to get "write_precision" which matches anchor.py output keys
    aa = report.get("anchor_analysis")
    if aa is None:
        return None
    if metric_name.endswith("_ref"):
        base = metric_name[:-4]  # strip "_ref"
        if base.startswith("anchor_"):
            base = base[7:]  # strip "anchor_" → "write_precision"
        return aa.get("reference", {}).get(base)
    if metric_name.endswith("_cmp"):
        base = metric_name[:-4]  # strip "_cmp"
        if base.startswith("anchor_"):
            base = base[7:]  # strip "anchor_" → "write_precision"
        return aa.get("compared", {}).get(base)

    return None


def _count_pattern(report: dict, pattern_type: str) -> int:
    """Count occurrences of a pattern type in a report."""
    return sum(1 for p in report.get("patterns", []) if p.get("type") == pattern_type)


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_metric_deltas(
    paired: list[tuple[BatchResult, BatchResult]],
) -> dict[str, dict[str, Any]]:
    """Compute before/after deltas for each metric.

    Returns dict mapping metric name to {before_mean, after_mean, delta, direction,
    before_values, after_values}.
    """
    deltas: dict[str, dict[str, Any]] = {}

    for metric_name, good_dir in METRIC_DIRECTIONS.items():
        before_vals = []
        after_vals = []
        for br, ar in paired:
            bv = _extract_metric(br.report, metric_name)
            av = _extract_metric(ar.report, metric_name)
            if bv is not None and av is not None:
                before_vals.append(bv)
                after_vals.append(av)

        if not before_vals:
            continue

        before_mean = sum(before_vals) / len(before_vals)
        after_mean = sum(after_vals) / len(after_vals)
        delta = after_mean - before_mean

        if good_dir == "higher":
            direction = "improved" if delta > 0.01 else ("regressed" if delta < -0.01 else "unchanged")
        else:
            direction = "improved" if delta < -0.01 else ("regressed" if delta > 0.01 else "unchanged")

        deltas[metric_name] = {
            "before_mean": round(before_mean, 4),
            "after_mean": round(after_mean, 4),
            "delta": round(delta, 4),
            "direction": direction,
            "before_values": before_vals,
            "after_values": after_vals,
        }

    return deltas


def compute_pattern_deltas(
    paired: list[tuple[BatchResult, BatchResult]],
) -> dict[str, dict[str, Any]]:
    """Compute pattern frequency deltas between before and after batches."""
    all_types: set[str] = set()
    for br, ar in paired:
        for p in br.report.get("patterns", []):
            all_types.add(p.get("type", ""))
        for p in ar.report.get("patterns", []):
            all_types.add(p.get("type", ""))

    deltas: dict[str, dict] = {}
    total = len(paired)

    for ptype in sorted(all_types):
        if not ptype:
            continue
        before_count = sum(1 for br, _ in paired if _count_pattern(br.report, ptype) > 0)
        after_count = sum(1 for _, ar in paired if _count_pattern(ar.report, ptype) > 0)
        before_prev = before_count / total if total > 0 else 0
        after_prev = after_count / total if total > 0 else 0

        good_dir = PATTERN_DIRECTIONS.get(ptype, "lower")
        delta = after_prev - before_prev
        if good_dir == "lower":
            direction = "improved" if delta < -0.01 else ("regressed" if delta > 0.01 else "unchanged")
        else:
            direction = "improved" if delta > 0.01 else ("regressed" if delta < -0.01 else "unchanged")

        deltas[ptype] = {
            "before_frequency": round(before_prev, 4),
            "after_frequency": round(after_prev, 4),
            "delta": round(delta, 4),
            "direction": direction,
        }

    return deltas


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def test_significance(
    before_values: list[float],
    after_values: list[float],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Test whether the delta is statistically significant.

    Uses Wilcoxon signed-rank (scipy) with fallback to sign test.
    """
    n = len(before_values)
    if n < 2:
        return {"p_value": None, "significant": False, "test_method": "none",
                "warning": "fewer than 2 paired observations"}
    if n < 6:
        warning = "fewer than 6 paired observations — low statistical power"
    else:
        warning = None

    # Try Wilcoxon signed-rank
    try:
        from scipy.stats import wilcoxon
        diffs = [a - b for a, b in zip(after_values, before_values)]
        # wilcoxon requires at least one non-zero difference
        if all(d == 0 for d in diffs):
            return {"p_value": 1.0, "significant": False, "test_method": "wilcoxon",
                    "warning": warning}
        stat, p_value = wilcoxon(diffs)
        return {
            "p_value": round(p_value, 6),
            "significant": p_value < threshold,
            "test_method": "wilcoxon",
            "warning": warning,
        }
    except ImportError:
        # scipy is a declared dependency; reaching here means a broken/partial
        # install. Degrade to the weaker sign test, but make it loud + visible
        # rather than silently changing the reported p-values.
        logger.warning(
            "scipy not available — falling back to the binomial sign test "
            "(weaker than Wilcoxon). Reinstall with scipy to restore it."
        )
        warning = "; ".join(
            w for w in (warning, "scipy unavailable: used sign test (weaker than Wilcoxon)") if w
        )

    # Fallback: sign test (binomial)
    diffs = [a - b for a, b in zip(after_values, before_values)]
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    n_nonzero = n_pos + n_neg
    if n_nonzero == 0:
        return {"p_value": 1.0, "significant": False, "test_method": "sign_test",
                "warning": warning}

    # Two-sided binomial p-value
    from math import comb
    k = min(n_pos, n_neg)
    p_value = 0.0
    for i in range(k + 1):
        p_value += comb(n_nonzero, i) * (0.5 ** n_nonzero)
    p_value *= 2  # two-sided
    p_value = min(p_value, 1.0)

    return {
        "p_value": round(p_value, 6),
        "significant": p_value < threshold,
        "test_method": "sign_test",
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Guardrail detection
# ---------------------------------------------------------------------------

def detect_guardrail_regressions(
    metric_deltas: dict[str, dict],
    relative_threshold: float = 0.10,
    significance_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect when improving one metric causes regression in others.

    For each improved metric, check all others for regression beyond threshold.
    """
    # Find improved metrics
    improved = [name for name, d in metric_deltas.items() if d["direction"] == "improved"]

    if not improved:
        return []

    warnings: list[dict] = []
    seen: set[str] = set()

    for regressed_name, d in metric_deltas.items():
        if d["direction"] != "regressed":
            continue
        if regressed_name in seen:
            continue

        before_mean = d["before_mean"]
        delta = abs(d["delta"])

        # Check relative threshold
        relative_regression = delta / abs(before_mean) if before_mean != 0 else 0
        exceeds_relative = relative_regression >= relative_threshold

        # Check statistical significance
        sig = test_significance(
            d.get("before_values", []),
            d.get("after_values", []),
            significance_threshold,
        )
        exceeds_significance = sig.get("significant", False)

        if exceeds_relative or exceeds_significance:
            seen.add(regressed_name)
            warnings.append({
                "metric": regressed_name,
                "before_mean": d["before_mean"],
                "after_mean": d["after_mean"],
                "delta": d["delta"],
                "direction": "regressed",
                "relative_regression": round(relative_regression, 4),
                "p_value": sig.get("p_value"),
                "significant": sig.get("significant", False),
                "warning": (
                    f"{regressed_name} regressed from {d['before_mean']:.4f} to "
                    f"{d['after_mean']:.4f} ({relative_regression*100:.1f}% relative"
                    f"{', p=' + str(sig['p_value']) if sig.get('p_value') is not None else ''}"
                    f")"
                ),
            })

    return warnings


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def generate_recommendation(
    metric_deltas: dict[str, dict],
    guardrails: list[dict],
) -> str:
    """Generate a one-line recommendation based on intervention effect."""
    improved = [n for n, d in metric_deltas.items() if d["direction"] == "improved"]
    regressed = [n for n, d in metric_deltas.items() if d["direction"] == "regressed"]

    if not improved and not regressed:
        return "No significant changes detected."
    if improved and not guardrails:
        return f"Intervention improved {', '.join(improved[:3])} without regressions."
    if improved and guardrails:
        regressed_names = [g["metric"] for g in guardrails]
        return (
            f"Intervention improved {', '.join(improved[:2])} but regressed "
            f"{', '.join(regressed_names[:2])}. Review tradeoffs."
        )
    if regressed and not improved:
        return f"Intervention regressed {', '.join(regressed[:3])} without improvements."
    return "Mixed results — review per-metric deltas."


# ---------------------------------------------------------------------------
# Intervention report
# ---------------------------------------------------------------------------

def build_intervention_report(
    before_manifest: str,
    after_manifest: str,
    paired: list[tuple[BatchResult, BatchResult]],
    before_only: list[BatchResult],
    after_only: list[BatchResult],
    metric_deltas: dict[str, dict],
    pattern_deltas: dict[str, dict],
    guardrails: list[dict],
) -> dict[str, Any]:
    """Assemble the full intervention comparison report."""
    clean_deltas = {}
    sig_cache: dict[str, dict] = {}
    for name, d in metric_deltas.items():
        if name not in sig_cache:
            sig_cache[name] = test_significance(
                d.get("before_values", []), d.get("after_values", []))
        sig = sig_cache[name]
        clean_deltas[name] = {
            "before_mean": d["before_mean"],
            "after_mean": d["after_mean"],
            "delta": d["delta"],
            "direction": d["direction"],
            "p_value": sig.get("p_value"),
            "significant": sig.get("significant", False),
            "significance_warning": sig.get("warning"),
        }

    recommendation = generate_recommendation(metric_deltas, guardrails)

    return {
        "before_manifest": before_manifest,
        "after_manifest": after_manifest,
        "paired_tasks": len(paired),
        "unpaired_before": len(before_only),
        "unpaired_after": len(after_only),
        "intervention_effect": clean_deltas,
        "guardrail_warnings": guardrails,
        "pattern_deltas": pattern_deltas,
        "recommendation": recommendation,
    }
