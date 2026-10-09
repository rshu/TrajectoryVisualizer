"""Three-layer evaluation metrics: file selection, edit stability, completion quality."""

from __future__ import annotations

from typing import Any

from .divergence import count_patterns


# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------

LAYER_DEFINITIONS = {
    "file_selection": {
        "description": "Is the agent working on the right files?",
        "intervention_targets": "System prompt, task decomposition, file discovery heuristics",
        "metrics": {
            "anchor_write_precision": {"source": "anchor", "key": "compared.write_precision", "good_dir": "higher"},
            "off_patch_write_ratio": {"source": "anchor", "key": "compared.off_patch_write_ratio", "good_dir": "lower"},
            "broad_exploration_count": {"source": "patterns", "pattern_type": "broad_exploration", "good_dir": "lower"},
        },
    },
    "edit_stability": {
        "description": "Is the agent's editing reliable?",
        "intervention_targets": "Model quality, context window, edit verification steps",
        "metrics": {
            "reverted_and_rewritten_count": {"source": "patterns", "pattern_type": "reverted_and_rewritten", "good_dir": "lower"},
            "iterative_refinement_count": {"source": "patterns", "pattern_type": "iterative_refinement", "good_dir": "lower"},
            "error_recovery_count": {"source": "patterns", "pattern_type": "error_recovery_overhead", "good_dir": "lower"},
        },
    },
    "completion_quality": {
        "description": "Did the agent finish the job?",
        "intervention_targets": "Multi-agent architecture, checklist prompting, task decomposition",
        "metrics": {
            "anchor_write_recall": {"source": "anchor", "key": "compared.write_recall", "good_dir": "higher"},
            "dead_end_branch_count": {"source": "patterns", "pattern_type": "dead_end_branch", "good_dir": "lower"},
        },
    },
}

# Default verdict thresholds (per metric). Users can override via
# ``compute_eval_layers(..., verdict_thresholds={...})``.
DEFAULT_VERDICT_THRESHOLDS: dict[str, dict[str, float | int]] = {
    # anchor metrics
    "anchor_write_precision": {"good": 0.75, "warn": 0.50},
    "off_patch_write_ratio": {"good": 0.25, "warn": 0.50},
    "anchor_write_recall": {"good": 0.50, "warn": 0.25},
    # pattern counts
    "broad_exploration_count": {"good": 3, "warn": 8},
    "reverted_and_rewritten_count": {"good": 2, "warn": 5},
    "iterative_refinement_count": {"good": 3, "warn": 8},
    "error_recovery_count": {"good": 1, "warn": 3},
    "dead_end_branch_count": {"good": 1, "warn": 3},
}


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_layer_metric(
    metric_def: dict,
    patterns: list[dict],
    anchor_analysis: dict | None,
) -> float | int | None:
    """Extract a single metric value from pattern/anchor data."""
    source = metric_def.get("source")

    if source == "patterns":
        return count_patterns(patterns, metric_def["pattern_type"])

    if source == "anchor" and anchor_analysis:
        key = metric_def["key"]
        parts = key.split(".")
        obj = anchor_analysis
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None
        return obj

    return None


# ---------------------------------------------------------------------------
# Layer computation
# ---------------------------------------------------------------------------

def compute_eval_layers(
    patterns: list[dict],
    anchor_analysis: dict | None = None,
    verdict_thresholds: dict[str, dict[str, float | int]] | None = None,
) -> dict[str, dict]:
    """Compute three evaluation layers from divergence patterns and anchor data.

    Args:
        verdict_thresholds: Optional per-metric thresholds overriding
            ``DEFAULT_VERDICT_THRESHOLDS``. Keys are metric names, values
            are dicts with ``good`` and ``warn`` keys.

    Returns dict mapping layer name to {description, metrics, verdict, intervention_targets}.
    """
    effective_thresholds = dict(DEFAULT_VERDICT_THRESHOLDS)
    if verdict_thresholds:
        effective_thresholds.update(verdict_thresholds)

    layers: dict[str, dict] = {}

    for layer_name, layer_def in LAYER_DEFINITIONS.items():
        layer_metrics: dict[str, Any] = {}

        for metric_name, metric_def in layer_def["metrics"].items():
            value = _extract_layer_metric(metric_def, patterns, anchor_analysis)
            layer_metrics[metric_name] = value

        verdict = _compute_verdict(layer_metrics, effective_thresholds)

        layers[layer_name] = {
            "description": layer_def["description"],
            "intervention_targets": layer_def["intervention_targets"],
            "metrics": layer_metrics,
            "verdict": verdict,
        }

    return layers


def _compute_verdict(
    metrics: dict[str, Any],
    thresholds: dict[str, dict[str, float | int]] | None = None,
) -> str:
    """Classify layer as strong/moderate/weak based on metric thresholds."""
    effective = thresholds or DEFAULT_VERDICT_THRESHOLDS
    has_bad = False
    has_warn = False
    has_any = False

    for metric_name, value in metrics.items():
        if value is None:
            continue
        has_any = True

        metric_thresholds = effective.get(metric_name)
        if not metric_thresholds:
            continue

        good = metric_thresholds["good"]
        warn = metric_thresholds["warn"]

        good_dir = None
        for layer_def in LAYER_DEFINITIONS.values():
            if metric_name in layer_def["metrics"]:
                good_dir = layer_def["metrics"][metric_name].get("good_dir", "higher")
                break

        if good_dir == "lower":
            if value > warn:
                has_bad = True
            elif value > good:
                has_warn = True
        else:
            if value < warn:
                has_bad = True
            elif value < good:
                has_warn = True

    if not has_any:
        return "n/a"
    if has_bad:
        return "weak"
    if has_warn:
        return "moderate"
    return "strong"
