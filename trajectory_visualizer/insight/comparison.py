"""Bridge module: orchestrates Converge's comparison pipeline from Insight's UI.

Calls Converge's lower-level functions directly (not build_comparison_report)
so we can pass already-loaded data instead of file paths.
"""

import logging
import os
import re

import plotly.graph_objects as go

from trajectory_visualizer.insight.loaders import load_trajectory
from trajectory_visualizer.insight.parser import parse_steps
from trajectory_visualizer.insight.charts import _apply_dark

from trajectory_visualizer.converge.canonical import canonicalize_steps, assign_effect_labels
from trajectory_visualizer.converge.alignment import (
    align_trajectories, compute_alignment_metrics,
    compute_harmful_divergence, DEFAULT_TOKEN_RATE,
)
from trajectory_visualizer.converge.milestones import (
    extract_milestones, compute_milestone_deltas,
    segment_by_milestones, compare_segments,
)
from trajectory_visualizer.converge.divergence import classify_divergences, compute_pattern_costs
from trajectory_visualizer.converge.charts import (
    build_milestone_timeline_chart,
    build_segment_cost_chart,
    build_divergence_waterfall_chart,
    build_anchor_class_chart,
)
from trajectory_visualizer.converge.rendering import build_comparison_report_html

logger = logging.getLogger(__name__)


def _empty_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=300)
    return fig


def _detect_success(steps: list[dict]) -> bool:
    """Heuristic: run succeeded if last assistant step has finish=stop/end_turn."""
    for s in reversed(steps):
        if s.get("role") == "assistant":
            return s.get("finish") in ("stop", "end_turn")
    return False


def run_comparison(
    ref_raw: dict,
    cmp_raw: dict,
    anchor_path: str | None = None,
    token_rate: float = DEFAULT_TOKEN_RATE,
    fuzzy: bool = False,
    dark: bool = False,
) -> dict:
    """Run Converge's full comparison pipeline.

    Parameters
    ----------
    ref_raw : dict
        Raw trajectory dict for the **reference / baseline** trajectory —
        the one the user uploaded on the Comparison tab (left of the anchor
        patch slot). Already loaded via ``load_trajectory``.
    cmp_raw : dict
        Raw trajectory dict for the **compared** trajectory — the one the
        user loaded on the Overview tab. Already loaded via ``load_trajectory``.
    anchor_path : str or None
        Optional path to a .patch/.diff file for anchor-grounded comparison.
    token_rate : float
        Token rate for cost computation (default 50.0).
    fuzzy : bool
        Enable fuzzy command matching.
    dark : bool
        Apply dark mode template to charts.

    Returns
    -------
    dict with keys: report_html, milestone_fig, segment_fig, waterfall_fig, anchor_fig
    """
    empty = {
        "report_html": "",
        "milestone_fig": _empty_fig(),
        "segment_fig": _empty_fig(),
        "waterfall_fig": _empty_fig(),
        "anchor_fig": _empty_fig(),
    }

    try:
        if "_error" in ref_raw:
            empty["report_html"] = (
                f"<div style='color:var(--ov-bad);padding:1em;'>"
                f"Error loading reference trajectory: {ref_raw['_error']}</div>"
            )
            return empty
        if "_error" in cmp_raw:
            empty["report_html"] = (
                f"<div style='color:var(--ov-bad);padding:1em;'>"
                f"Error loading compared trajectory: {cmp_raw['_error']}</div>"
            )
            return empty

        ref_steps = parse_steps(ref_raw)
        cmp_steps = parse_steps(cmp_raw)

        # Anchor files from patch
        anchor_files = None
        if anchor_path and os.path.isfile(anchor_path):
            try:
                with open(anchor_path) as f:
                    content = f.read()
                anchor_files = set(
                    re.findall(r'^[+-]{3}\s+[ab]/(.+)$', content, re.MULTILINE)
                )
            except Exception:
                anchor_files = None

        # Canonicalize
        ref_actions = canonicalize_steps(ref_steps)
        cmp_actions = canonicalize_steps(cmp_steps)
        assign_effect_labels(ref_actions, ref_steps, anchor_files)
        assign_effect_labels(cmp_actions, cmp_steps, anchor_files)

        # Outcome
        ref_success = _detect_success(ref_steps)
        cmp_success = _detect_success(cmp_steps)
        ref_tokens = sum(s["tokens"]["total"] for s in ref_steps)
        cmp_tokens = sum(s["tokens"]["total"] for s in cmp_steps)
        from trajectory_visualizer.converge.alignment import _describe_format, _session_duration_s
        ref_path = ref_raw.get("_source_path", "") if isinstance(ref_raw, dict) else ""
        cmp_path = cmp_raw.get("_source_path", "") if isinstance(cmp_raw, dict) else ""
        outcome = {
            "reference_success": ref_success,
            "compared_success": cmp_success,
            "reference_steps": len(ref_steps),
            "compared_steps": len(cmp_steps),
            "reference_tokens": ref_tokens,
            "compared_tokens": cmp_tokens,
            "reference_filename": os.path.basename(ref_path) if ref_path else "",
            "compared_filename": os.path.basename(cmp_path) if cmp_path else "",
            "reference_format": _describe_format(ref_raw),
            "compared_format": _describe_format(cmp_raw),
            "reference_duration_s": _session_duration_s(ref_raw),
            "compared_duration_s": _session_duration_s(cmp_raw),
            "reference_tool_calls": sum(len(s.get("tool_calls", [])) for s in ref_steps),
            "compared_tool_calls": sum(len(s.get("tool_calls", [])) for s in cmp_steps),
            "success_detection": "heuristic (finish marker, not task correctness)",
        }

        # Alignment
        alignment = align_trajectories(ref_actions, cmp_actions, fuzzy)
        metrics = compute_alignment_metrics(alignment, ref_actions, cmp_actions, token_rate)

        # Target files for milestones
        from trajectory_visualizer.insight.diagnostics import identify_target_files
        _norm = lambda p: os.path.normpath(p) if p else p
        if anchor_files:
            milestone_targets = {_norm(f) for f in anchor_files}
        else:
            ref_targets = {_norm(f) for f in identify_target_files(ref_steps)}
            cmp_targets = {_norm(f) for f in identify_target_files(cmp_steps)}
            milestone_targets = ref_targets | cmp_targets if (ref_targets or cmp_targets) else None

        # Milestones
        ref_milestones = extract_milestones(ref_actions, target_files=milestone_targets)
        cmp_milestones = extract_milestones(cmp_actions, target_files=milestone_targets)
        milestone_deltas = compute_milestone_deltas(ref_milestones, cmp_milestones)

        ref_segments = segment_by_milestones(ref_actions, ref_milestones)
        cmp_segments = segment_by_milestones(cmp_actions, cmp_milestones)
        segment_result = compare_segments(
            ref_segments, cmp_segments, ref_milestones, cmp_milestones,
            ref_actions, cmp_actions, token_rate,
        )

        # Divergence patterns
        extra_actions = [cmp_actions[j] for j in alignment["extra"] if j < len(cmp_actions)]
        matched_actions = [cmp_actions[j] for _, j in alignment["matched_pairs"] if j < len(cmp_actions)]
        patterns = classify_divergences(
            extra_actions, matched_actions, cmp_actions,
            matched_pairs=alignment["matched_pairs"],
            anchor_files=anchor_files,
        )
        compute_pattern_costs(patterns, token_rate)

        # Harmful divergence
        dead_end_steps: set[int] = set()
        for p in patterns:
            if p.get("type") == "dead_end_branch":
                dead_end_steps.update(p.get("steps", []))
        harmful = compute_harmful_divergence(
            alignment["extra"], cmp_actions, token_rate, dead_end_steps)

        # Anchor analysis
        anchor_analysis = None
        anchor_mode = "external" if anchor_files else "self"
        if anchor_mode == "external" and anchor_files:
            from trajectory_visualizer.converge.anchor import compute_anchor_analysis
            anchor_analysis = compute_anchor_analysis(ref_actions, cmp_actions, anchor_files)

        # Evaluation layers
        from trajectory_visualizer.converge.eval_layers import compute_eval_layers
        eval_layers = compute_eval_layers(
            {"alignment": {**metrics, **harmful}},
            patterns,
            anchor_analysis,
        )

        # Notes
        notes = []
        if anchor_mode == "self" and (ref_success != cmp_success):
            notes.append(
                "Warning: runs produced different outcomes. Unanchored metrics are "
                "informational (behavioral similarity), not evaluative."
            )
        notes.append("This comparison is observational for one task pair.")

        # Agent names
        ref_agent = ref_raw.get("metadata", {}).get("generator_name", "reference") if isinstance(ref_raw.get("metadata"), dict) else "reference"
        cmp_agent = cmp_raw.get("metadata", {}).get("generator_name", "compared") if isinstance(cmp_raw.get("metadata"), dict) else "compared"

        # Confidence
        confidence = {
            "alignment": "informational" if (anchor_mode == "self" and ref_success != cmp_success) else "heuristic",
            "milestones": "anchored" if anchor_files else "heuristic",
            "segments": "heuristic",
            "divergence": "heuristic",
            "outcome": "heuristic",
        }

        # Build report dict (same structure as build_comparison_report output)
        report = {
            "task_id": "",
            "reference_agent": ref_agent,
            "compared_agent": cmp_agent,
            "outcome": outcome,
            "alignment": {**metrics, **harmful},
            "milestones": milestone_deltas,
            "ref_milestones": ref_milestones,
            "cmp_milestones": cmp_milestones,
            "segments": segment_result,
            "patterns": patterns,
            "anchor_mode": anchor_mode,
            "anchor_analysis": anchor_analysis,
            "eval_layers": eval_layers,
            "confidence": confidence,
            "evidence_level": "single_pair_hypothesis",
            "notes": notes,
        }

        # Render HTML report. The Insight UI's Comparison tab suppresses the
        # divergence-patterns section (heading, glossary, table) — the patterns
        # are still computed for downstream consumers but are hidden from the
        # rendered report. The standalone Converge app (converge/app.py) goes
        # through build_comparison_report directly and is unaffected.
        report_html = build_comparison_report_html({**report, "patterns": []})

        # Build charts with dark mode
        milestone_fig = build_milestone_timeline_chart(ref_milestones, cmp_milestones)
        _apply_dark(milestone_fig, dark)

        segment_fig = build_segment_cost_chart(
            segment_result, segment_result.get("milestone_order_matches", False))
        _apply_dark(segment_fig, dark)

        waterfall_fig = build_divergence_waterfall_chart(patterns)
        _apply_dark(waterfall_fig, dark)

        anchor_fig = build_anchor_class_chart(anchor_analysis)
        _apply_dark(anchor_fig, dark)

        return {
            "report_html": report_html,
            "milestone_fig": milestone_fig,
            "segment_fig": segment_fig,
            "waterfall_fig": waterfall_fig,
            "anchor_fig": anchor_fig,
        }

    except Exception:
        # Log the full traceback server-side; never echo it (or the raw exception
        # text) to the browser — it was previously rendered UNESCAPED, leaking
        # paths/versions and allowing HTML injection from trajectory-derived text.
        logger.exception("Insight comparison failed")
        empty["report_html"] = (
            "<div style='color:var(--ov-bad);padding:1em;'>"
            "<strong>Comparison failed.</strong> Check the reference and compared "
            "trajectories; full details were written to the server log.</div>"
        )
        return empty
