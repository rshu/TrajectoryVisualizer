"""Gradio UI for Insight."""

import base64
import html
import json
import logging
import os
import re

import gradio as gr
import plotly.graph_objects as go

from .parser import (
    load_trajectory, parse_steps, build_message_metrics, compute_metrics,
    _build_hotspots_md,
    _build_per_message_md, compute_health_verdict, validate_token_integrity,
    format_session_md, format_performance_md,
    format_behavioral_md, format_output_md,
    extract_agent_info,
    wall_clock_fmt, format_banner_html,
    compute_agent_summary,
    effective_agent,
)
from .analytics import compute_step_analytics, detect_phases
from .diagnostics import (
    extract_file_interactions,
    identify_target_files,
    compute_file_targeting_metrics,
    detect_failure_chains,
    classify_chain_steps,
    link_chains_to_agents,
    compute_failure_chain_metrics,
    cluster_errors,
    annotate_clusters_with_agents,
    compute_bottleneck_explanations,
)
from .charts import (
    build_token_chart, build_duration_chart, build_tool_chart,
    build_agent_swimlane_chart,
    build_agent_token_chart,
    build_tool_outcome_timeline,
    build_label_phase_count_chart,
    build_label_action_count_chart,
    build_label_phase_duration_chart,
    build_label_action_duration_chart,
    build_label_timeline_chart,
    build_file_interaction_chart,

    build_plan_timeline_chart,
    build_error_classification_chart,
    build_context_growth_chart,
    build_context_growth_ca_chart,
)

from .comparison import run_comparison
from .patterns import (
    detect_tool_sequences, detect_failure_patterns,
    extract_plan_history, compute_plan_metrics,
    extract_subagent_sessions, compute_subagent_metrics,
    detect_fruitless_streaks, compute_autonomy_ratio,
    detect_tool_selection_antipatterns,
)
from .help import HELP_TEXT
from .parser import load_labeled_json, aggregate_labels
from .training_labels import aggregate_training_labels, load_training_labeled_json
from .rendering import (
    render_workflow_html, render_toc_sidebar, render_filter_chips,
    format_step_detail, render_agent_summary_cards,
    build_root_cause_html,
    build_antipattern_summary_html,
)
from .styles import APP_CSS

logger = logging.getLogger(__name__)

_DETAIL_PLACEHOLDER = (
    "<div id='wf-detail-content'>"
    "<div style='padding:2em 1em;text-align:center;color:#9ca3af;'>"
    "<p style='font-size:15px;margin-bottom:0.5em;'>Select a step to inspect</p>"
    "<p style='font-size:12px;'>Click any card on the left, or press <kbd>j</kbd>/<kbd>k</kbd> to navigate</p>"
    "</div></div>"
)
_FILTER_CHIPS_DEFAULT = ["Assistant", "User", "Tool Calls", "Errors", "Reasoning"]

# Maximum steps to process — keeps rendering, metrics, and charts bounded.
_MAX_STEPS = 2000


def _prerender_step_details(steps: list[dict]) -> str:
    """Pre-render all step details as HTML and return a base64-encoded JSON blob.

    ``format_step_detail()`` now returns styled HTML directly, so no
    markdown-it pass is needed.
    """
    details = {}
    for step in steps:
        details[str(step['index'])] = format_step_detail(step)
    b64 = base64.b64encode(json.dumps(details).encode()).decode()
    return f'<div data-b64="{b64}" style="display:none"></div>'


def _compute_anomalies(metrics: dict, message_rows: list[dict]) -> list[dict]:
    """Return a list of anomaly dicts (type, step_idx, value_str) from metrics."""
    anomalies: list[dict] = []
    if not message_rows:
        return anomalies

    # Longest step by duration
    with_dur = [r for r in message_rows if r.get("duration") is not None]
    if with_dur:
        longest = max(with_dur, key=lambda r: r["duration"])
        anomalies.append({
            "type": "Slowest",
            "step_idx": longest["index"],
            "value": f"{longest['duration']:.1f}s",
        })

    # Highest token step
    if message_rows:
        highest_tok = max(message_rows, key=lambda r: r["tokens_total"])
        if highest_tok["tokens_total"] > 0:
            anomalies.append({
                "type": "Most Tokens",
                "step_idx": highest_tok["index"],
                "value": f"{highest_tok['tokens_total']:,} tok",
            })

    # Lowest cache ratio (assistant steps with tokens)
    asst_with_tok = [r for r in message_rows
                     if r.get("role") == "assistant" and r["tokens_total"] > 0]
    if asst_with_tok:
        lowest_cache = min(asst_with_tok, key=lambda r: r["cache_ratio"])
        anomalies.append({
            "type": "Lowest Cache",
            "step_idx": lowest_cache["index"],
            "value": f"{lowest_cache['cache_ratio'] * 100:.1f}%",
        })

    # Most tool calls
    with_tools = [r for r in message_rows if r["tool_calls"] > 0]
    if with_tools:
        most_tools = max(with_tools, key=lambda r: r["tool_calls"])
        anomalies.append({
            "type": "Most Tools",
            "step_idx": most_tools["index"],
            "value": f"{most_tools['tool_calls']} calls",
        })

    # Error steps
    error_steps = [r for r in message_rows if r.get("error_count", 0) > 0]
    if error_steps:
        anomalies.append({
            "type": "Errors",
            "step_idx": error_steps[0]["index"],
            "value": f"{len(error_steps)} step(s)",
        })

    return anomalies[:5]


def _build_card_jump_onclick(idx) -> str:
    """Return a JS onclick string that switches to the Workflow tab,
    scrolls step card *idx* into view, and clicks it."""
    return (
        f"(function(){{"
        f"var tabs=document.querySelectorAll('.tab-nav button');"
        f"if(tabs.length>1)tabs[1].click();"
        f"setTimeout(function(){{"
        f"var c=document.getElementById('wf-card-{idx}');"
        f"if(c){{c.scrollIntoView({{behavior:'smooth',block:'center'}});c.click();}}"
        f"}},200);"
        f"}})()"
    )


def _build_anomaly_strip_html(anomalies: list[dict]) -> str:
    """Render clickable anomaly badges with data-step-idx attributes."""
    if not anomalies:
        return ""
    badges = []
    for a in anomalies:
        idx = a["step_idx"]
        onclick = _build_card_jump_onclick(idx)
        badges.append(
            f"<span class='anomaly-badge' data-step-idx='{idx}'"
            f" onclick=\"{onclick}\" style='cursor:pointer;'>"
            f"{html.escape(a['type'])}: #{idx} ({html.escape(a['value'])})"
            f"</span>"
        )
    return "<div class='anomaly-strip'>" + "".join(badges) + "</div>"


def _build_sparkline_svg(values: list[float], width: int = 100, height: int = 20) -> str:
    """Generate a minimal inline SVG sparkline from a list of values."""
    if not values or len(values) < 2:
        return ""
    max_v = max(values) or 1
    min_v = min(values)
    range_v = max_v - min_v or 1
    points = []
    for i, v in enumerate(values):
        x = round(i / (len(values) - 1) * width, 1)
        y = round(height - (v - min_v) / range_v * (height - 2) - 1, 1)
        points.append(f"{x},{y}")
    polyline = " ".join(points)
    return (
        f"<div class='ov-kpi-sparkline'>"
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<polyline points='{polyline}' fill='none' stroke='#3b82f6' stroke-width='1.5' "
        f"stroke-linecap='round' stroke-linejoin='round'/>"
        f"</svg></div>"
    )


def _build_session_detail_html(
    metrics: dict, wall_fmt: str, timing: dict, metadata: dict,
    raw: dict,
    *, model_id: str = "", agent_id: str = "",
) -> str:
    """Build Session Details panel as a chip grid of session environment fields.

    Uses the same inline chip style as ``format_session_md`` — small cards
    with uppercase label and value, wrapped in a flex grid.
    """
    md = metadata
    started = timing.get("started_at", "N/A")
    finished = timing.get("finished_at", "N/A")
    if isinstance(started, str) and len(started) > 19:
        started = started[:19].replace("T", " ")
    if isinstance(finished, str) and len(finished) > 19:
        finished = finished[:19].replace("T", " ")

    fields = [
        ("Model", model_id or md.get("model") or "N/A"),
        ("Agent", agent_id or md.get("agent", "N/A")),
        ("Start", str(started)),
        ("End", str(finished)),
        ("Session", (md.get("session_id") or "N/A")[:16]),
        ("Branch", md.get("branch") or "N/A"),
        ("Directory", md.get("directory_name") or "N/A"),
        ("Platform", (md.get("platform") or "N/A")[:24]),
    ]
    if md.get("server_version"):
        fields.insert(2, ("Version", md["server_version"]))

    chips = "".join(
        f"<div style='display:inline-flex;flex-direction:column;"
        f"background:var(--ov-card);border:1px solid var(--ov-border);"
        f"border-radius:8px;padding:6px 10px;min-width:100px;'>"
        f"<span style='font-size:10px;color:var(--ov-muted);text-transform:uppercase;'>"
        f"{html.escape(label)}</span>"
        f"<span style='font-size:13px;font-weight:500;'>"
        f"{html.escape(str(val))}</span></div>"
        for label, val in fields
    )
    return f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>{chips}</div>"


def _build_overview_kpi_html(metrics: dict, wall_fmt: str,
                             verdicts: list[dict] | None = None,
                             message_rows: list[dict] | None = None) -> str:
    """Build at-a-glance KPI card strip for Overview tab.

    When *verdicts* is provided, matching KPI cards get a colored left border
    and a tooltip with the verdict detail string.
    """
    # Build verdict lookup: metric label -> (status, detail)
    _verdict_map: dict[str, tuple[str, str]] = {}
    if verdicts:
        _metric_to_kpi = {
            "Tool Success": "Tool Success",
            "Throughput": "Tokens",
            "Token Efficiency": "Tokens",
            "Errors": "Steps",
        }
        for v in verdicts:
            kpi_label = _metric_to_kpi.get(v["metric"], "")
            if kpi_label:
                _verdict_map[kpi_label] = (v["status"], v["detail"])

    _status_colors = {
        "good": "#059669",
        "warn": "#d97706",
        "bad": "#dc2626",
    }

    # Build sparkline data from message_rows
    sparkline_data: dict[str, list[float]] = {}
    if message_rows:
        sparkline_data["Tokens"] = [r.get("tokens_total", 0) for r in message_rows]
        sparkline_data["Wall-Clock"] = [r.get("duration", 0) or 0 for r in message_rows]

    # Help text keys for each KPI label
    _label_to_help_key = {
        "Steps": "steps",
        "Wall-Clock": "wall_clock",
        "Tokens": "tokens",
        "Tool Success": "tool_success",
    }

    user_steps = metrics.get('user_steps', 0)
    cards = [
        ("Steps", f"{metrics.get('total_steps', 0):,}",
         f"{metrics.get('assistant_steps', 0)} assistant, {user_steps} user"),
        ("Wall-Clock", wall_fmt,
         f"P95 {metrics.get('p95_duration', 0)}s"),
        ("Tokens", f"{metrics.get('tokens', {}).get('total', 0):,}",
         f"{metrics.get('tokens_per_second', 0):,} tok/s"),
        ("Tool Success", f"{metrics.get('tool_success_rate', 0)}%",
         f"{metrics.get('tool_call_count', 0):,} calls"),
    ]
    card_html = []
    for label, value, sub in cards:
        verdict_info = _verdict_map.get(label)
        extra_style = ""
        title_attr = ""
        data_attr = ""
        if verdict_info:
            status, detail = verdict_info
            border_color = _status_colors.get(status, "#6b7280")
            extra_style = f" style='border-left:4px solid {border_color};'"
            title_attr = f" title='{html.escape(detail)}'"
            data_attr = f" data-status='{html.escape(status)}'"
        # Help tooltip
        help_key = _label_to_help_key.get(label, "")
        help_attr = ""
        if help_key and help_key in HELP_TEXT:
            help_attr = f" data-help='{html.escape(HELP_TEXT[help_key])}'"
        # Sparkline
        sparkline = ""
        if label in sparkline_data:
            sparkline = _build_sparkline_svg(sparkline_data[label])
        # Verdict detail as visible subtitle
        verdict_sub = ""
        if verdict_info:
            status, detail = verdict_info
            vcolor = _status_colors.get(status, "#6b7280")
            verdict_sub = f"<div style='font-size:11px;color:{vcolor};margin-top:2px;'>{html.escape(detail)}</div>"
        card_html.append(
            f"<div class='ov-kpi-card'{extra_style}{title_attr}{data_attr}>"
            f"<div class='ov-kpi-label'{help_attr}>{html.escape(str(label))}</div>"
            f"<div class='ov-kpi-value'>{html.escape(str(value))}</div>"
            f"<div class='ov-kpi-sub'>{html.escape(str(sub))}</div>"
            f"{verdict_sub}"
            f"{sparkline}"
            "</div>"
        )
    return (
        "<div class='ov-kpi-grid'>"
        + "".join(card_html)
        + "</div>"
    )


def _build_overview_outputs(
    steps: list[dict], raw: dict, metrics: dict, message_rows: list[dict],
    verdicts: list[dict], wfmt: str, file_path: str,
    trajectory_format: str | None = None,
) -> dict:
    """Compute overview-related outputs: banner, KPI, metadata, and metrics markdown."""
    if raw.get("_analysis_profile") == "training":
        return _build_training_overview_outputs(steps, raw, metrics, message_rows, file_path)

    _d = lambda k: raw.get(k, {}) if isinstance(raw.get(k), dict) else {}
    md, timing, outp = _d("metadata"), _d("timing"), _d("output")
    session_raw, retry = _d("session_raw"), _d("retry")
    model_id, provider_id, agent_id = extract_agent_info(steps)

    generator = md.get("generator_name", "")
    banner = format_banner_html(os.path.basename(file_path), metrics, wfmt,
                                generator=generator,
                                trajectory_format=trajectory_format)
    kpi_html = _build_overview_kpi_html(metrics, wfmt, verdicts=verdicts,
                                        message_rows=message_rows)
    metrics_text = format_performance_md(metrics, wfmt)
    from .metrics import compute_diagnostic_metrics
    traj = raw.get("trajectory", [])
    diag_metrics = compute_diagnostic_metrics(steps, traj) if traj else None
    behavior_text = format_behavioral_md(metrics, diag_metrics=diag_metrics)
    hotspots_text = _build_hotspots_md(message_rows)
    per_message_text = _build_per_message_md(message_rows)

    anomalies = _compute_anomalies(metrics, message_rows)
    anomaly_html = _build_anomaly_strip_html(anomalies)

    return {
        "banner": banner,
        "anomaly_html": anomaly_html,
        "kpi_html": kpi_html,
        "metrics_text": metrics_text,
        "behavior_text": behavior_text,
        "hotspots_text": hotspots_text,
        "per_message_text": per_message_text,
    }


def _build_training_overview_outputs(
    steps: list[dict], raw: dict, metrics: dict, message_rows: list[dict], file_path: str,
) -> dict:
    """Overview copy for training conversations shown as basic trajectories."""
    scaffold = raw.get("_training_scaffold", {}) if isinstance(raw.get("_training_scaffold"), dict) else {}
    scaffold_label = scaffold.get("label", "Unknown")
    scaffold_conf = scaffold.get("confidence", "unknown")
    assistant_steps = [s for s in steps if s.get("role") == "assistant"]
    user_steps = [s for s in steps if s.get("role") == "user"]
    reasoning_steps = [s for s in assistant_steps if s.get("has_reasoning")]
    est_total = sum((s.get("tokens", {}) or {}).get("estimated_total", 0) or 0 for s in steps)
    est_reasoning = sum((s.get("tokens", {}) or {}).get("estimated_reasoning_tokens", 0) or 0 for s in steps)
    est_content = sum((s.get("tokens", {}) or {}).get("estimated_content_tokens", 0) or 0 for s in steps)

    banner = format_banner_html(
        os.path.basename(file_path),
        metrics,
        "n/a",
        generator="",
        trajectory_format=trajectory_format_label("training_conversation"),
    )
    kpi_html = (
        "<div class='ov-kpi-grid'>"
        f"<div class='ov-kpi-card'><div class='ov-kpi-label'>Turns</div><div class='ov-kpi-value'>{len(steps)}</div>"
        f"<div class='ov-kpi-sub'>{len(user_steps)} user, {len(assistant_steps)} assistant</div></div>"
        f"<div class='ov-kpi-card'><div class='ov-kpi-label'>Estimated Tokens</div><div class='ov-kpi-value'>{est_total:,}</div>"
        f"<div class='ov-kpi-sub'>{est_content:,} content, {est_reasoning:,} reasoning</div></div>"
        f"<div class='ov-kpi-card'><div class='ov-kpi-label'>Reasoning Coverage</div><div class='ov-kpi-value'>{len(reasoning_steps)}</div>"
        f"<div class='ov-kpi-sub'>assistant turns with reasoning content</div></div>"
        f"<div class='ov-kpi-card'><div class='ov-kpi-label'>Scaffold</div><div class='ov-kpi-value'>{html.escape(str(scaffold_label))}</div>"
        f"<div class='ov-kpi-sub'>{html.escape(str(scaffold_conf))} confidence</div></div>"
        f"<div class='ov-kpi-card'><div class='ov-kpi-label'>Profile</div><div class='ov-kpi-value'>Training</div>"
        "<div class='ov-kpi-sub'>basic conversation trajectory view</div></div>"
        "</div>"
    )
    metrics_text = (
        "### Training conversation\n\n"
        f"- Scaffold: `{scaffold_label}`\n"
        f"- Total turns: `{len(steps)}`\n"
        f"- User turns: `{len(user_steps)}`\n"
        f"- Assistant turns: `{len(assistant_steps)}`\n"
        f"- Assistant turns with reasoning content: `{len(reasoning_steps)}`\n"
        f"- Estimated total tokens: `{est_total}`\n"
        f"- Estimated content tokens: `{est_content}`\n"
        f"- Estimated reasoning tokens: `{est_reasoning}`\n\n"
        "Timing, tool runtime, and recorder-derived efficiency metrics are not available for this profile."
    )
    behavior_text = (
        "### Training behavior\n\n"
        f"- This dataset is treated as a base conversation trajectory.\n"
        f"- Inferred scaffold: `{scaffold_label}` ({scaffold_conf} confidence).\n"
        f"- `{len(assistant_steps)}` assistant turns are available for response-shape inspection.\n"
        f"- `{len(reasoning_steps)}` assistant turns include explicit reasoning content.\n"
        "- No tool-call or agent-runtime behavior is assumed in v1."
    )
    hotspots_text = (
        "### Turn hotspots\n\n"
        "Use this section to inspect longer assistant turns, reasoning-heavy turns, and role transitions."
    )
    per_message_lines = []
    for row in message_rows:
        if row.get("role") != "assistant":
            continue
        step = next((s for s in steps if s.get("index") == row.get("index")), None)
        tok = (step or {}).get("tokens", {})
        per_message_lines.append(
            f"- Step `{row.get('index')}`: estimated `{tok.get('estimated_total', 0)}` tokens, "
            f"`{tok.get('estimated_reasoning_tokens', 0)}` reasoning, preview: "
            f"`{((step or {}).get('text_preview', '')[:80])}`"
        )
    if not per_message_lines:
        per_message_lines.append("- No assistant turns available.")
    return {
        "banner": banner,
        "anomaly_html": "",
        "kpi_html": kpi_html,
        "metrics_text": metrics_text,
        "behavior_text": behavior_text,
        "hotspots_text": hotspots_text,
        "per_message_text": "### Assistant turn details\n\n" + "\n".join(per_message_lines),
    }


def _build_chart_outputs(
    steps: list[dict], message_rows: list[dict], phases: list[dict],
    step_analytics: list[dict], agent_summaries: list[dict],
    dark: bool = False,
    trajectory: list[dict] | None = None,
    trajectory_format: str | None = None,
) -> dict:
    """Build all chart figures and analytics markdown."""
    if trajectory_format == "training_conversation":
        return _build_training_chart_outputs(steps, dark=dark)

    traj = trajectory or []

    # Compute diagnostic data for charts
    plan_history = extract_plan_history(steps)
    plan_metrics = compute_plan_metrics(plan_history)
    subagent_sessions = extract_subagent_sessions(steps, traj)
    subagent_metrics = compute_subagent_metrics(subagent_sessions, steps)
    fruitless_streaks = detect_fruitless_streaks(steps, traj)
    tool_selection = detect_tool_selection_antipatterns(steps)

    # Detect compression steps
    compression_steps = []
    for i, entry in enumerate(traj):
        if entry.get("info", {}).get("compressMessage"):
            compression_steps.append(steps[i].get("index", i) if i < len(steps) else i)

    # Core charts
    tok_fig = build_token_chart(steps, dark=dark, format=trajectory_format)
    dur_fig = build_duration_chart(steps, dark=dark,
                                    compression_steps=compression_steps or None)
    tl_fig = build_tool_chart(steps, dark=dark)

    tool_outcome_fig = build_tool_outcome_timeline(steps, dark=dark)

    agent_tok_fig = build_agent_token_chart(agent_summaries, dark=dark)
    swimlane_fig = build_agent_swimlane_chart(steps, dark=dark)

    # New diagnostic charts
    plan_timeline_fig = build_plan_timeline_chart(plan_history, plan_metrics, dark=dark)
    error_class_fig = build_error_classification_chart(steps, dark=dark)

    # Context growth chart. CodeArts re-emits cumulative context size per step,
    # so it gets the format-specific builder with compression-event markers.
    # Other formats (OpenCode, Claude Code) only expose per-step deltas, so we
    # fall back to the format-agnostic cumulative-input view.
    context_limit = None
    is_codearts = any(
        isinstance(e.get("_codearts_raw"), dict) for e in traj
    )
    if is_codearts:
        context_limit = 192_000
        context_growth_fig = build_context_growth_ca_chart(
            steps, traj, context_limit=context_limit, dark=dark)
    else:
        # Skip Boot/Steady/Closeout phase overlays here — the heuristic
        # bands clutter the cumulative-token view without adding signal.
        context_growth_fig = build_context_growth_chart(
            message_rows, phases=None, dark=dark)

    # New panels
    error_count = sum(1 for s in steps for tc in s.get("tool_calls", []) if tc.get("error_type"))
    antipattern_html = build_antipattern_summary_html(
        fruitless_streaks, tool_selection, plan_metrics, error_count=error_count,
    )

    return {
        "tok_fig": tok_fig,
        "dur_fig": dur_fig,
        "tl_fig": tl_fig,
        "tool_outcome_fig": tool_outcome_fig,
        "agent_cards_html": render_agent_summary_cards(agent_summaries),
        "agent_tok_fig": agent_tok_fig,
        "swimlane_fig": swimlane_fig,
        "plan_timeline_fig": plan_timeline_fig,
        "error_class_fig": error_class_fig,
        "antipattern_html": antipattern_html,
        "context_growth_fig": context_growth_fig,
    }


def trajectory_format_label(fmt: str | None) -> str:
    """Return a human-readable trajectory format label."""
    labels = {
        "training_conversation": "Training Conversation",
        "ccsession": "Claude Code",
        "codearts": "CodeArts",
        "opencode": "OpenCode",
    }
    return labels.get(fmt or "", fmt or "Unknown")


def _training_role_count_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    counts: dict[str, int] = {}
    for step in steps:
        role = step.get("role", "unknown")
        counts[role] = counts.get(role, 0) + 1
    fig = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=["#2563eb", "#16a34a", "#7c3aed", "#6b7280"][: len(counts)],
        text=list(counts.values()),
        textposition="outside",
    ))
    fig.update_layout(template="plotly_white", height=380, title="Conversation Turns by Role")
    return fig


def _training_estimated_tokens_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    xs = [s.get("index", 0) for s in steps]
    content = [(s.get("tokens", {}) or {}).get("estimated_content_tokens", 0) or 0 for s in steps]
    reasoning = [(s.get("tokens", {}) or {}).get("estimated_reasoning_tokens", 0) or 0 for s in steps]
    fig = go.Figure()
    fig.add_bar(x=xs, y=content, name="Content", marker_color="#2563eb")
    fig.add_bar(x=xs, y=reasoning, name="Reasoning", marker_color="#7c3aed")
    fig.update_layout(template="plotly_white", height=380, barmode="stack", title="Estimated Tokens by Turn")
    return fig


def _training_reasoning_ratio_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    assistant_steps = [s for s in steps if s.get("role") == "assistant"]
    xs = [s.get("index", 0) for s in assistant_steps]
    vals = []
    for step in assistant_steps:
        tok = step.get("tokens", {}) or {}
        total = tok.get("estimated_total", 0) or 0
        reasoning = tok.get("estimated_reasoning_tokens", 0) or 0
        vals.append(round((reasoning / total) * 100, 1) if total else 0)
    fig = go.Figure(go.Scatter(x=xs, y=vals, mode="lines+markers", marker_color="#7c3aed"))
    fig.update_layout(template="plotly_white", height=380, title="Reasoning Share by Assistant Turn", yaxis_title="%")
    return fig


def _build_training_chart_outputs(steps: list[dict], dark: bool = False) -> dict:
    """Chart outputs that reinterpret work sections for training conversations."""
    assistant_steps = [s for s in steps if s.get("role") == "assistant"]
    reasoning_steps = sum(1 for s in assistant_steps if s.get("has_reasoning"))
    has_tools = any(s.get("tool_call_count", 0) > 0 for s in steps)
    return {
        "tok_fig": _training_estimated_tokens_chart(steps, dark=dark),
        "dur_fig": _training_role_count_chart(steps, dark=dark),
        "tl_fig": build_tool_chart(steps, dark=dark) if has_tools else _training_role_count_chart(steps, dark=dark),
        "tool_outcome_fig": build_tool_outcome_timeline(steps, dark=dark) if has_tools else _training_reasoning_ratio_chart(steps, dark=dark),
        "agent_cards_html": (
            "<div class='overview-card'>"
            "<div style='font-weight:700;'>Training profile</div>"
            f"<div style='font-size:12px;color:var(--ov-muted);'>{len(assistant_steps)} assistant turns, "
            f"{reasoning_steps} with reasoning content. Agent/runtime identity is not assumed in v1.</div>"
            "</div>"
        ),
        "agent_tok_fig": _training_estimated_tokens_chart(steps, dark=dark),
        "swimlane_fig": _training_role_count_chart(steps, dark=dark),
        "plan_timeline_fig": _training_reasoning_ratio_chart(steps, dark=dark),
        "error_class_fig": _training_role_count_chart(steps, dark=dark),
        "antipattern_html": (
            "<div class='overview-card'>"
            "<div style='font-weight:700;'>Training interpretation</div>"
            "<div style='font-size:12px;color:var(--ov-muted);'>"
            "Diagnostics are adapted for conversation structure in v1. Focus on role balance, "
            "assistant coverage, and reasoning density rather than tool/runtime failures."
            "</div></div>"
        ),
        "context_growth_fig": _training_estimated_tokens_chart(steps, dark=dark),
    }


def _build_diagnostics_outputs(
    steps: list[dict], step_analytics: list[dict],
    agent_summaries: list[dict], dark: bool = False,
    trajectory_format: str | None = None,
) -> dict:
    """Build all diagnostics outputs: file interactions, failure chains, root causes, bottlenecks."""
    _empty_fig = go.Figure()
    _empty_fig.update_layout(template="plotly_white", height=380)

    # File interaction analysis
    interactions = extract_file_interactions(steps)
    target_files = identify_target_files(steps)
    file_targeting = compute_file_targeting_metrics(
        interactions, target_files, len(steps))
    file_chart = build_file_interaction_chart(interactions, target_files, dark=dark) if interactions else _empty_fig

    # Failure chain analysis (metrics feed the summary line; UI strip removed)
    chains = detect_failure_chains(steps)
    chains = link_chains_to_agents(chains, steps, agent_summaries)
    chain_metrics = compute_failure_chain_metrics(
        chains, sum(1 for s in steps if s.get("role") == "assistant"))

    # Root-cause attribution. The findings panel is suppressed for OpenCode
    # because its tool-error reporting is noisy (every non-zero bash exit code
    # surfaces as a "failure") and the cluster summaries become misleading.
    # Counts are still computed so the summary line stays accurate.
    clusters = cluster_errors(steps)
    clusters = annotate_clusters_with_agents(clusters, steps, agent_summaries)
    if trajectory_format == "opencode":
        rootcause_html = ""
    else:
        rootcause_html = build_root_cause_html(clusters)

    # Bottleneck explanation (explanations feed the summary line; UI cards removed)
    bottleneck_explanations = compute_bottleneck_explanations(steps, step_analytics)

    # Summary line
    issue_count = chain_metrics["total_chains"] + len(bottleneck_explanations)
    parts = []
    if chain_metrics["total_chains"]:
        parts.append(f"{chain_metrics['total_chains']} failure chain(s)")
    if clusters:
        parts.append(f"{len(clusters)} root cause(s)")
    if bottleneck_explanations:
        parts.append(f"{len(bottleneck_explanations)} hotspot(s)")
    if interactions:
        unique_files = len({i["path"] for i in interactions})
        parts.append(f"{unique_files} file(s) touched · {len(target_files)} edited")
    summary = " &middot; ".join(parts) if parts else "No diagnostic issues detected."
    summary_html = (
        f"<div style='font-size:12px;color:var(--ov-muted);margin-bottom:8px;'>"
        f"{summary}</div>"
    )

    return {
        "file_targeting": file_targeting,
        "chain_metrics": chain_metrics,
        "error_clusters": clusters,
        "failure_chains": chains,
        "bottleneck_explanations": bottleneck_explanations,
        "diag_summary_html": summary_html,
        "diag_file_chart": file_chart,
        "diag_rootcause_html": rootcause_html,
    }


def _build_workflow_outputs(steps: list[dict], agent_summaries: list[dict]) -> dict:
    """Build workflow HTML, TOC, filter chips, and detail store."""
    wf_html = render_workflow_html(steps)
    wf_count = f"<div class='wf-count'>Showing {len(steps)} of {len(steps)} steps</div>"
    toc_html_val = render_toc_sidebar(steps)
    detail_store_val = _prerender_step_details(steps)

    agent_label_list = [{"label": a["label"], "agent_id": a["agent_id"]}
                        for a in agent_summaries]
    wf_chips = render_filter_chips(agent_labels=agent_label_list)
    default_filters = list(_FILTER_CHIPS_DEFAULT)
    for al in agent_label_list:
        default_filters.append(f"agent:{al['agent_id']}")
    wf_filter_val = ",".join(default_filters)

    return {
        "wf_chips": wf_chips,
        "wf_filter_val": wf_filter_val,
        "wf_count": wf_count,
        "toc_html_val": toc_html_val,
        "wf_html": wf_html,
        "detail_store_val": detail_store_val,
    }


def _render_tool_sequences_html(sequences: list[dict]) -> str:
    """Render tool sequence patterns as an HTML table."""
    if not sequences:
        return "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>No recurring tool sequences detected (minimum frequency: 3).</div>"
    rows = []
    for s in sequences[:20]:
        seq_str = " → ".join(html.escape(t) for t in s["sequence"])
        indices = ", ".join(str(i) for i in s["step_indices"][:10])
        if len(s["step_indices"]) > 10:
            indices += f" … (+{len(s['step_indices']) - 10} more)"
        rows.append(
            f"<tr style='border-bottom:1px solid var(--ov-border);'>"
            f"<td style='padding:6px 10px;font-family:monospace;font-size:12px;'>{seq_str}</td>"
            f"<td style='padding:6px 10px;text-align:center;font-weight:700;'>{s['frequency']}</td>"
            f"<td style='padding:6px 10px;font-size:11px;color:var(--ov-muted);'>{indices}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<tr style='background:var(--ov-table-header-bg);'>"
        "<th style='text-align:left;padding:6px 10px;'>Sequence</th>"
        "<th style='text-align:center;padding:6px 10px;'>Count</th>"
        "<th style='text-align:left;padding:6px 10px;'>Step Indices</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _count_bar_chart(counts: dict[str, int], title: str, dark: bool = False) -> go.Figure:
    """Build a small categorical count chart for label summaries."""
    fig = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        text=list(counts.values()),
        textposition="outside",
        marker_color="#2563eb",
    ))
    fig.update_layout(template="plotly_dark" if dark else "plotly_white", height=380, title=title)
    return fig


def _training_label_compact_count_chart(
    counts: dict[str, int],
    title: str,
    *,
    orientation: str = "v",
    color_map: dict[str, str] | None = None,
    dark: bool = False,
) -> go.Figure:
    """Compact charts for training-label summaries inside the Labels panel."""
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    labels = [key for key, _ in items]
    values = [value for _, value in items]
    colors = [
        (color_map or {}).get(label, "#2563eb")
        for label in labels
    ]
    if orientation == "h":
        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            text=[str(v) for v in values],
            textposition="inside",
            marker_color=colors,
            cliponaxis=True,
            hovertemplate="%{y}: %{x}<extra></extra>",
        ))
        fig.update_yaxes(automargin=True)
        xaxis_title = "Count"
        yaxis_title = ""
    else:
        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            text=[str(v) for v in values],
            textposition="inside",
            marker_color=colors,
            cliponaxis=True,
            hovertemplate="%{x}: %{y}<extra></extra>",
        ))
        xaxis_title = ""
        yaxis_title = "Count"
    max_val = max(values or [0])
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        height=280,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        margin=dict(l=80 if orientation == "h" else 42, r=24, t=48, b=54),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=False,
    )
    if max_val > 0:
        if orientation == "h":
            fig.update_xaxes(range=[0, max_val * 1.12])
        else:
            fig.update_yaxes(range=[0, max_val * 1.12])
    return fig


def _build_training_label_token_timeline_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Training-label timeline using assistant turn token length instead of duration."""
    if not steps:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark" if dark else "plotly_white", height=380)
        return fig

    y_pos = list(range(len(steps)))
    step_indices = [s.get("index", i) for i, s in enumerate(steps)]
    tokens = [s.get("tokens_total") or 0 for s in steps]
    phases = [s.get("phase", "") for s in steps]
    actions = [s.get("action", "") for s in steps]
    decisions = [(s.get("decision", {}) or {}).get("label", "") for s in steps]
    values = [(s.get("value", {}) or {}).get("tier", "") for s in steps]
    qualities = [(s.get("quality", {}) or {}).get("verdict", "") for s in steps]
    value_tags = [
        (s.get("value", {}) or {}).get("tags") or []
        for s in steps
    ]
    row_labels = [
        f"{step_indices[i]} · {actions[i]}"
        for i in range(len(steps))
    ]

    from .palette import LABEL_PHASE_COLORS
    colors = [LABEL_PHASE_COLORS.get(phase, "#6b7280") for phase in phases]
    hover = [
        f"<b>Assistant step {step_indices[i]}</b><br>"
        f"Phase: {phases[i]}<br>"
        f"Action: {actions[i]}<br>"
        f"Quality: {qualities[i]}<br>"
        f"Value: {values[i]}<br>"
        f"Value tags: {', '.join(value_tags[i]) or 'none'}<br>"
        f"Decision: {decisions[i]}<br>"
        f"Estimated tokens: {tokens[i]:,}"
        for i in range(len(steps))
    ]

    fig = go.Figure(go.Bar(
        x=tokens,
        y=y_pos,
        orientation="h",
        marker_color=colors,
        text=[f"{tokens[i]:,} tok" for i in range(len(steps))],
        textposition="inside",
        insidetextanchor="start",
        textfont=dict(size=10, color="white"),
        cliponaxis=True,
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    ))
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_pos,
        ticktext=row_labels,
        autorange="reversed",
        automargin=True,
    )
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        title="Assistant Token Length Timeline (colored by phase)",
        xaxis_title="Estimated tokens",
        yaxis_title="Assistant step",
        height=max(320, 24 * len(steps)),
        margin=dict(l=140, r=28, t=56, b=42),
    )
    max_tokens = max(tokens or [0])
    if max_tokens > 0:
        fig.update_xaxes(range=[0, max_tokens * 1.08])
    return fig


def _counts_inline(counts: dict[str, int], *, limit: int | None = None) -> str:
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        items = items[:limit]
    if not items:
        return "none"
    return ", ".join(f"{html.escape(str(key))}: {value}" for key, value in items)


def _build_training_label_summary_html(agg: dict) -> str:
    return (
        "<div style='padding:0.75em 1em;border:1px solid var(--ov-border);"
        "border-radius:8px;background:var(--ov-card);font-size:13px;'>"
        "<div style='font-weight:700;margin-bottom:6px;'>Training label summary</div>"
        f"<div><b>Quality</b>: {_counts_inline(agg.get('quality_verdict_counts', {}))}</div>"
        f"<div><b>Defects</b>: {_counts_inline(agg.get('quality_flag_counts', {}))}</div>"
        f"<div><b>Value</b>: {_counts_inline(agg.get('value_tier_counts', {}))}</div>"
        f"<div><b>Decision</b>: {_counts_inline(agg.get('decision_counts', {}))}</div>"
        f"<div><b>Top value tags</b>: {_counts_inline(agg.get('value_tag_counts', {}), limit=5)}</div>"
        "</div>"
    )


def attach_training_labels_to_steps(steps: list[dict], label_data: dict) -> list[dict]:
    """Return steps with matching training v2 labels attached to assistant turns."""
    labels_by_index = {
        step.get("index"): step
        for step in label_data.get("steps", [])
        if isinstance(step, dict)
    }
    enriched = []
    for step in steps:
        copied = dict(step)
        label = labels_by_index.get(step.get("index"))
        if label and step.get("role") == "assistant":
            copied["training_label"] = {
                "phase": label.get("phase", ""),
                "action": label.get("action", ""),
                "quality": label.get("quality", {}),
                "value": label.get("value", {}),
                "decision": label.get("decision", {}),
            }
        enriched.append(copied)
    return enriched


def _load_label_json_kind(file_path: str) -> str:
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "unknown"
    if isinstance(data, dict) and data.get("schema_version") == "trajectory_labels.v2":
        return "training_v2"
    return "legacy"


def build_label_ui_payload(file_path: str) -> dict:
    """Build UI-facing label payload for legacy and training v2 label files."""
    kind = _load_label_json_kind(file_path)
    if kind == "training_v2":
        data = load_training_labeled_json(file_path)
        agg = aggregate_training_labels(data)
        from .palette import LABEL_PHASE_COLORS
        action_to_phase = {
            step.get("action", "unknown"): step.get("phase", "unknown")
            for step in agg.get("steps", [])
        }
        action_colors = {
            action: LABEL_PHASE_COLORS.get(action_to_phase.get(action, ""), "#2563eb")
            for action in agg["action_counts"]
        }
        phase_fig = _training_label_compact_count_chart(
            agg["phase_counts"],
            "Phase Counts",
            color_map=LABEL_PHASE_COLORS,
        )
        action_fig = _training_label_compact_count_chart(
            agg["action_counts"],
            "Action Counts",
            orientation="h",
            color_map=action_colors,
        )
        quality_fig = _training_label_compact_count_chart(
            agg["quality_verdict_counts"],
            "Quality Verdicts",
        )
        value_fig = _training_label_compact_count_chart(
            agg["value_tier_counts"],
            "Value Tiers",
        )
        timeline_fig = _build_training_label_token_timeline_chart(agg["steps"])
        n_steps = len(agg.get("steps", []))
        decisions = _counts_inline(agg.get("decision_counts", {}))
        badge = (
            "<div style='display:flex;flex-direction:column;gap:6px;margin:6px 0;'>"
            "<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
            "<span style='background:#2563eb;color:white;"
            "padding:4px 12px;border-radius:12px;font-size:13px;'>"
            f"Training labels loaded — {n_steps} assistant turns, decisions: {decisions}</span>"
            "<a href='#' onclick=\"document.querySelector('[class*=per-message-acc]:last-child button')?.click();"
            "return false;\" style='font-size:12px;color:#2563eb;text-decoration:underline;cursor:pointer;'>"
            "Jump to Labels</a>"
            "</div></div>"
        )
        return {
            "kind": "training_v2",
            "badge_html": badge,
            "status_html": _build_training_label_summary_html(agg),
            "phase_count_fig": phase_fig,
            "action_count_fig": action_fig,
            "phase_duration_fig": quality_fig,
            "action_duration_fig": value_fig,
            "timeline_fig": timeline_fig,
        }

    data = load_labeled_json(file_path)
    agg = aggregate_labels(data)
    pc_fig = build_label_phase_count_chart(agg["phase_counts"])
    ac_fig = build_label_action_count_chart(agg["action_counts"], agg["action_to_phase"])
    pd_fig = build_label_phase_duration_chart(agg["phase_durations"])
    ad_fig = build_label_action_duration_chart(agg["action_durations"], agg["action_to_phase"])
    tl_fig = build_label_timeline_chart(agg["steps"])

    n_steps = len(agg.get("steps", []))
    phase_counts = agg.get("phase_counts", {})
    n_phases = len(phase_counts)

    from .palette import LABEL_PHASE_COLORS
    bar_segments = "".join(
        f"<span style='flex:{count};background:{LABEL_PHASE_COLORS.get(phase, '#6b7280')};height:8px;'"
        f" title='{phase}: {count}'></span>"
        for phase, count in phase_counts.items() if count > 0
    )
    phase_bar = (
        "<div style='display:flex;border-radius:4px;overflow:hidden;width:200px;margin-top:4px;'>"
        f"{bar_segments}</div>"
    ) if bar_segments else ""

    phase_chips = "".join(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;"
        f"background:{LABEL_PHASE_COLORS.get(phase, '#6b7280')};color:white;'>"
        f"{phase}: {count}</span>"
        for phase, count in phase_counts.items() if count > 0
    )

    badge = (
        "<div style='display:flex;flex-direction:column;gap:6px;margin:6px 0;'>"
        "<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
        "<span style='background:#059669;color:white;"
        "padding:4px 12px;border-radius:12px;font-size:13px;'>"
        f"Labels loaded — {n_steps} steps, {n_phases} phases</span>"
        "<a href='#' onclick=\"document.querySelector('[class*=per-message-acc]:last-child button')?.click();"
        "return false;\" style='font-size:12px;color:#059669;text-decoration:underline;cursor:pointer;'>"
        "Jump to Labels</a>"
        "</div>"
        f"<div style='display:flex;gap:4px;flex-wrap:wrap;'>{phase_chips}</div>"
        f"{phase_bar}"
        "</div>"
    )
    return {
        "kind": "legacy",
        "badge_html": badge,
        "status_html": "",
        "phase_count_fig": pc_fig,
        "action_count_fig": ac_fig,
        "phase_duration_fig": pd_fig,
        "action_duration_fig": ad_fig,
        "timeline_fig": tl_fig,
    }


def _render_failure_patterns_html(patterns: list[dict]) -> str:
    """Render failure pattern clusters as expandable cards."""
    if not patterns:
        return "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>No errors detected — no failure patterns to analyze.</div>"
    cards = []
    for p in patterns:
        label = html.escape(p.get("cluster_label", "Unknown"))
        count = p.get("count", 0)
        example = html.escape(str(p.get("example_error", ""))[:200])
        recovery = p.get("recovery_path")
        recovery_html = (
            " → ".join(html.escape(t) for t in recovery)
            if recovery else "<em>No recovery path found</em>"
        )
        step_ids = p.get("steps") or []
        steps_html = (
            "".join(
                f"<span style='display:inline-block;padding:1px 6px;margin:0 4px 2px 0;"
                f"border-radius:8px;background:var(--ov-table-header-bg);"
                f"font-size:11px;font-variant-numeric:tabular-nums;'>#{int(idx)}</span>"
                for idx in step_ids
            )
            if step_ids else "<em>—</em>"
        )
        cards.append(
            f"<div class='overview-card' style='margin-bottom:8px;'>"
            f"<div style='font-weight:700;font-size:13px;color:var(--ov-text);'>{label} "
            f"<span style='font-size:11px;color:var(--ov-muted);font-weight:400;'>({count} occurrences)</span></div>"
            f"<div style='font-size:12px;color:var(--ov-muted);margin:4px 0;'>{example}</div>"
            f"<div style='font-size:12px;margin-bottom:4px;'><strong>Steps:</strong> {steps_html}</div>"
            f"<div style='font-size:12px;'><strong>Recovery path:</strong> {recovery_html}</div>"
            f"</div>"
        )
    return "".join(cards)


def build_ui() -> gr.Blocks:
    """Build the full Gradio Blocks UI."""

    with gr.Blocks(title="Insight", elem_classes=["trajectory-viz"]) as app:
        # Per-session state via gr.State
        state_steps = gr.State([])
        state_dark = gr.State(False)
        state_raw = gr.State({})

        gr.HTML(
            "<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:4px;'>"
            "<span style='font-size:20px;font-weight:700;letter-spacing:-0.02em;'>Insight</span>"
            "<span style='font-size:12px;color:var(--ov-muted);'>Trajectory analysis dashboard</span>"
            "</div>"
        )

        # -- Upload area (collapses after load) --
        with gr.Column(elem_classes=["upload-row"]) as upload_accordion:
            with gr.Row(equal_height=True):
                format_selector = gr.Dropdown(
                    label="Format",
                    choices=[
                        ("Claude Code", "ccsession"),
                        ("CodeArts", "codearts"),
                        ("OpenCode", "opencode"),
                        ("Training Conversation", "training_conversation"),
                    ],
                    value="ccsession",
                    interactive=True,
                    scale=1,
                    min_width=140,
                )
                with gr.Column(scale=2, min_width=200):
                    file_upload = gr.File(
                        label="Trajectory (.json)",
                        file_types=[".json"],
                        height=110,
                    )
                    load_btn = gr.Button("Load Trajectory", variant="primary",
                                         size="sm", min_width=120)
                with gr.Column(scale=2, min_width=200):
                    label_file_upload = gr.File(
                        label="Labels (optional)",
                        file_types=[".json"],
                        height=110,
                    )
                    label_load_btn = gr.Button("Load Labels", variant="secondary",
                                               size="sm", min_width=120)

        # Summary banner + anomaly strip (hidden until load)
        with gr.Column(visible=False) as summary_area:
            summary_banner = gr.HTML("", elem_classes=["summary-banner"])
            label_badge_html = gr.HTML("")
            anomaly_strip_html = gr.HTML("")

        # -- Tabs (hidden until trajectory loaded) --
        with gr.Tabs(visible=False) as main_tabs:
            # ===== Overview Tab (unified — includes former Analytics content) =====
            with gr.TabItem("Overview"):
                _toc_js = (
                    "function(label){"
                    "var els=document.querySelectorAll('.per-message-acc button span');"
                    "for(var e of els){if(e.textContent.includes(label)){"
                    "e.closest('.per-message-acc').scrollIntoView({behavior:'smooth',block:'start'});"
                    "var btn=e.closest('button');"
                    "if(btn&&btn.getAttribute('aria-expanded')==='false')btn.click();"
                    "break;}}}"
                )
                _toc_sections = [
                    "Performance", "Efficiency", "Tools", "Agents",
                    "Diagnostics", "Deep Dive", "Labels",
                ]
                _toc_links = "".join(
                    f"<a class='toc-link' onclick=\"({_toc_js})('{s.replace('&amp;', '&')}')\">{s}</a>"
                    for s in _toc_sections
                )
                gr.HTML(
                    f"<nav class='insight-toc'>"
                    f"<span class='toc-nav-title'>Contents</span>"
                    f"{_toc_links}"
                    f"</nav>",
                    elem_classes=["sidebar-toc"],
                )
                session_detail_html = gr.HTML("")
                overview_kpi_html = gr.HTML("", elem_classes=["overview-kpi-strip"])

                with gr.Accordion("Performance", open=True, elem_classes=["per-message-acc"]):
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_performance'])}</div>")
                    metrics_md = gr.Markdown("")
                    with gr.Row(equal_height=True):
                        token_chart = gr.Plot(show_label=False, label="Token Usage")
                        duration_chart = gr.Plot(show_label=False, label="Step Duration")

                with gr.Accordion("Efficiency", open=False, elem_classes=["per-message-acc"]):
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_efficiency'])}</div>")
                    context_growth_chart = gr.Plot(show_label=False, label="Context Growth")

                with gr.Accordion("Tools", open=False, elem_classes=["per-message-acc"]):
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_tools'])}</div>")
                    behavior_md = gr.Markdown("")
                    with gr.Row(equal_height=True):
                        tool_chart = gr.Plot(show_label=False, label="Tool Call Frequency")
                        gr.Column(scale=1)  # reserved for future chart
                    with gr.Row(equal_height=True):
                        tool_outcome_chart = gr.Plot(show_label=False, label="Tool Outcome Timeline")

                with gr.Accordion("Agents", open=False, elem_classes=["per-message-acc"]):
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_agents'])}</div>")
                    agent_summary_html = gr.HTML("")
                    with gr.Row(equal_height=True):
                        agent_token_chart = gr.Plot(show_label=False, label="Token Breakdown by Agent")
                        gr.Column(scale=1)  # reserved for future chart
                    with gr.Row(equal_height=True):
                        agent_swimlane_chart = gr.Plot(show_label=False, label="Agent Swimlane")

                with gr.Accordion("Diagnostics", open=False, elem_classes=["per-message-acc"]) as diag_accordion:
                    diag_summary_html = gr.HTML("")
                    diag_file_chart = gr.Plot(show_label=False, label="File Interaction Timeline",
                                              elem_classes=["resizable-chart"])
                    diag_rootcause_html = gr.HTML("")
                    with gr.Row(equal_height=True):
                        error_class_chart = gr.Plot(show_label=False, label="Tool Error Classification")
                        plan_timeline_chart = gr.Plot(show_label=False, label="Plan Progress Timeline")

                with gr.Accordion("Per-Step Deep Dive", open=False, elem_classes=["per-message-acc"]):
                    hotspots_md = gr.Markdown("")
                    per_message_md = gr.Markdown("")

                with gr.Accordion("Labels — Phase and action classification from labeled JSON",
                                 open=False, elem_classes=["per-message-acc"]) as labels_accordion:
                    label_status_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                        "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>"
                    )
                    with gr.Row(equal_height=True, visible=False) as label_charts_row1:
                        label_phase_count_chart = gr.Plot(show_label=False, label="Phase Count Distribution")
                        label_action_count_chart = gr.Plot(show_label=False, label="Action Count Distribution")
                    with gr.Row(equal_height=True, visible=False) as label_charts_row2:
                        label_phase_dur_chart = gr.Plot(show_label=False, label="Phase Duration Distribution")
                        label_action_dur_chart = gr.Plot(show_label=False, label="Action Duration Distribution")
                    with gr.Row(equal_height=True, visible=False) as label_timeline_row:
                        label_timeline_chart = gr.Plot(show_label=False, label="Step Timeline")

            # ===== Patterns Tab =====
            with gr.TabItem("Patterns"):
                gr.HTML("<div class='section-subtitle'>Recurring patterns detected in the trajectory — tool sequences, failure clusters, and phase transition anomalies.</div>")
                with gr.Accordion("Tool Sequence Patterns", open=True, elem_classes=["per-message-acc"]):
                    patterns_tool_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>Load a trajectory to detect tool sequence patterns.</div>"
                    )
                with gr.Accordion("Failure Patterns", open=True, elem_classes=["per-message-acc"]):
                    patterns_failure_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>Load a trajectory to detect failure patterns.</div>"
                    )
                with gr.Accordion("Anti-Pattern Summary", open=True, elem_classes=["per-message-acc"]):
                    antipattern_summary_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>Load a trajectory to detect anti-patterns.</div>"
                    )

            # ===== Comparison Tab (Converge embedded) =====
            with gr.TabItem("Comparison"):
                _cmp_placeholder = (
                    "<div style='padding:2em;color:var(--ov-muted);text-align:center;font-size:14px;'>"
                    "Load a trajectory in the Overview tab first &mdash; it will become the "
                    "<b>compared</b> trajectory. Upload the <b>reference</b> (baseline) below.</div>"
                )
                cmp_status_html = gr.HTML(_cmp_placeholder)
                with gr.Accordion("Upload & Run", open=True, elem_classes=["per-message-acc"]):
                    with gr.Row(equal_height=True):
                        cmp_format_selector = gr.Dropdown(
                            label="Format",
                            choices=[
                                ("Claude Code", "ccsession"),
                                ("CodeArts", "codearts"),
                                ("OpenCode", "opencode"),
                            ],
                            value="ccsession",
                            interactive=True,
                            scale=1,
                            min_width=140,
                        )
                        cmp_file_upload = gr.File(
                            label="Reference Trajectory (.json)",
                            file_types=[".json"],
                            scale=2,
                        )
                        cmp_anchor_upload = gr.File(
                            label="Anchor Patch (optional, .patch/.diff)",
                            file_types=[".patch", ".diff"],
                            scale=1,
                        )
                    with gr.Row(equal_height=True):
                        cmp_ref_labels_upload = gr.File(
                            label="Reference Labels (optional, *_labeled.json)",
                            file_types=[".json"],
                            scale=4,
                        )
                    with gr.Row(equal_height=False):
                        cmp_is_baseline = gr.Checkbox(
                            label="Reference trajectory is the baseline",
                            info="This trajectory is what we compared to, i.e., the baseline.",
                            value=True, scale=4,
                        )
                        cmp_run_btn = gr.Button(
                            "Run Comparison", variant="primary",
                            size="sm", scale=1, min_width=140,
                        )
                cmp_report_html = gr.HTML("")
                with gr.Row(equal_height=True):
                    cmp_phase_count_chart = gr.Plot(show_label=False, label="Step Count by Phase — Reference vs Compared")
                    cmp_phase_duration_chart = gr.Plot(show_label=False, label="Duration by Phase — Reference vs Compared")

            # ===== Workflow Tab =====
            with gr.TabItem("Workflow"):
                with gr.Row(equal_height=True):
                    wf_toc_toggle = gr.Button("TOC", variant="secondary", scale=0, min_width=50)
                    wf_filter_chips_html = gr.HTML(
                        render_filter_chips(),
                        elem_id="wf-filter-chips",
                    )
                    wf_search = gr.Textbox(
                        label="Search", placeholder="Filter by keyword...",
                        scale=1,
                    )
                # Hidden textbox that JS writes active filters into (comma-separated)
                wf_filter_hidden = gr.Textbox(
                    value="Assistant,User,Tool Calls,Errors,Reasoning",
                    visible=False,
                    elem_id="wf-filter-hidden",
                )
                wf_count_html = gr.HTML("")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=0, min_width=150):
                        toc_html = gr.HTML("", elem_id="wf-toc-container")
                    with gr.Column(scale=3, min_width=400):
                        workflow_html = gr.HTML(
                            "<div style='padding:3em;color:#9ca3af;text-align:center;"
                            "font-size:15px;'>Load a trajectory to see the step flow.</div>",
                            js_on_load="""
                            /* Filter chip click handler (delegated, survives re-renders) */
                            if (!window.__wfChipHandlerAttached) {
                                window.__wfChipHandlerAttached = true;
                                document.addEventListener('click', function(e) {
                                    var chip = e.target.closest('.filter-chip');
                                    if (!chip) return;
                                    chip.classList.toggle('chip-active');
                                    var bar = chip.closest('.filter-bar') || document.getElementById('wf-filter-chips');
                                    if (!bar) return;
                                    var active = Array.from(bar.querySelectorAll('.filter-chip.chip-active'))
                                        .map(function(c) { return c.dataset.filter; });
                                    var hiddenEl = document.querySelector('#wf-filter-hidden textarea, #wf-filter-hidden input');
                                    if (hiddenEl) {
                                        var nativeSet = Object.getOwnPropertyDescriptor(
                                            window.HTMLInputElement.prototype, 'value'
                                        ) || Object.getOwnPropertyDescriptor(
                                            window.HTMLTextAreaElement.prototype, 'value'
                                        );
                                        if (nativeSet && nativeSet.set) {
                                            nativeSet.set.call(hiddenEl, active.join(','));
                                        } else {
                                            hiddenEl.value = active.join(',');
                                        }
                                        hiddenEl.dispatchEvent(new Event('input', {bubbles: true}));
                                    }
                                });
                            }

                            /* Detail tab click handler (delegated, survives detail HTML replacement) */
                            if (!window.__dpTabHandlerAttached) {
                                window.__dpTabHandlerAttached = true;
                                document.addEventListener('click', function(e) {
                                    var tab = e.target.closest('.dp-tab');
                                    if (!tab) return;
                                    var panel = tab.closest('.dp-panel');
                                    if (!panel) return;
                                    panel.querySelectorAll('.dp-tab').forEach(function(x) {
                                        x.classList.remove('dp-tab-active');
                                    });
                                    panel.querySelectorAll('.dp-tab-content').forEach(function(x) {
                                        x.classList.remove('dp-tab-visible');
                                    });
                                    tab.classList.add('dp-tab-active');
                                    var content = panel.querySelector(
                                        '[data-tab-content="' + tab.dataset.tab + '"]'
                                    );
                                    if (content) content.classList.add('dp-tab-visible');
                                });
                            }

                            /* Card click handler */
                            function selectCard(card) {
                                if (!card) return;
                                element.querySelectorAll('.wf-card').forEach(function(c) {
                                    c.classList.remove('wf-active');
                                });
                                card.classList.add('wf-active');
                                var idx = card.dataset.stepIdx;
                                /* URL deep linking */
                                if (idx != null) {
                                    history.replaceState(null, '', '#step-' + idx);
                                }
                                var storeEl = document.querySelector('#wf-detail-store [data-b64]');
                                var target = document.getElementById('wf-detail-content');
                                if (!storeEl || !target) return;
                                try {
                                    var details = JSON.parse(atob(storeEl.dataset.b64));
                                    if (details[idx] != null) {
                                        target.innerHTML = details[idx];
                                        var detailPanel = target.closest('.detail-panel');
                                        if (detailPanel) detailPanel.scrollTop = 0;
                                    }
                                } catch(ex) { console.error('wf-click:', ex); }
                            }
                            element.addEventListener('click', function(e) {
                                selectCard(e.target.closest('.wf-card'));
                            });

                            /* Keyboard navigation: j/k for next/prev step */
                            document.addEventListener('keydown', function(e) {
                                var tag = (e.target.tagName || '').toLowerCase();
                                if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
                                if (e.key !== 'j' && e.key !== 'k') return;
                                var cards = Array.from(element.querySelectorAll('.wf-card'));
                                if (!cards.length) return;
                                var activeIdx = cards.findIndex(function(c) { return c.classList.contains('wf-active'); });
                                var nextIdx;
                                if (e.key === 'j') {
                                    nextIdx = activeIdx < 0 ? 0 : Math.min(activeIdx + 1, cards.length - 1);
                                } else {
                                    nextIdx = activeIdx < 0 ? 0 : Math.max(activeIdx - 1, 0);
                                }
                                cards[nextIdx].scrollIntoView({behavior:'smooth', block:'center'});
                                selectCard(cards[nextIdx]);
                            });

                            /* Deep link: on load, scroll to step from URL hash */
                            setTimeout(function() {
                                var hash = window.location.hash;
                                var m = hash && hash.match(/^#step-(\\d+)$/);
                                if (m) {
                                    var card = document.getElementById('wf-card-' + m[1]);
                                    if (card) {
                                        card.scrollIntoView({behavior:'smooth', block:'center'});
                                        selectCard(card);
                                    }
                                }
                            }, 500);

                            /* Auto-select first assistant card if no hash link */
                            setTimeout(function() {
                                if (window.location.hash) return;
                                var cards = element.querySelectorAll('.wf-card');
                                for (var i = 0; i < cards.length; i++) {
                                    var badges = cards[i].querySelectorAll('.wf-badge');
                                    for (var j = 0; j < badges.length; j++) {
                                        if (badges[j].textContent.trim() === 'Assistant') {
                                            selectCard(cards[i]);
                                            return;
                                        }
                                    }
                                }
                                if (cards.length) selectCard(cards[0]);
                            }, 600);

                            /* ===== Chart Interactivity Bridge ===== */
                            /* Navigate to a step in the workflow tab */
                            window.__navigateToStep = function(stepIdx) {
                                var tabs = document.querySelectorAll('.tab-nav button');
                                if (tabs.length > 1) tabs[1].click();
                                setTimeout(function() {
                                    var card = document.getElementById('wf-card-' + stepIdx);
                                    if (card) {
                                        card.scrollIntoView({behavior: 'smooth', block: 'center'});
                                        card.click();
                                    }
                                }, 300);
                            };

                            /* Attach click handlers to Plotly charts for phase regions */
                            if (!window.__plotlyClickAttached) {
                                window.__plotlyClickAttached = true;
                                document.addEventListener('click', function(e) {
                                    /* Phase region annotations are rendered as text elements */
                                    var annot = e.target.closest('.annotation-text');
                                    if (!annot) return;
                                    var text = annot.textContent || '';
                                    /* Check if it looks like a phase annotation */
                                    if (['Boot', 'Steady', 'Closeout'].indexOf(text.trim()) === -1) return;
                                    /* Find the phase data from the chart's vrects */
                                    var plotDiv = annot.closest('.js-plotly-plot');
                                    if (!plotDiv || !plotDiv.layout) return;
                                    var shapes = plotDiv.layout.shapes || [];
                                    var annotations = plotDiv.layout.annotations || [];
                                    /* Find matching annotation index */
                                    for (var i = 0; i < annotations.length; i++) {
                                        if (annotations[i].text === text.trim() && shapes[i]) {
                                            var startStep = Math.round(shapes[i].x0 + 0.5);
                                            window.__navigateToStep(startStep);
                                            break;
                                        }
                                    }
                                });

                                /* Outlier annotation click → navigate to step */
                                document.addEventListener('click', function(e) {
                                    var annot = e.target.closest('.annotation-text');
                                    if (!annot) return;
                                    var text = annot.textContent || '';
                                    var m = text.match(/spike:\\s*([\\d,]+)/);
                                    if (!m) return;
                                    /* Outlier annotations have x coordinate = step index */
                                    var plotDiv = annot.closest('.js-plotly-plot');
                                    if (!plotDiv || !plotDiv.layout) return;
                                    var annotations = plotDiv.layout.annotations || [];
                                    for (var i = 0; i < annotations.length; i++) {
                                        if (annotations[i].text && annotations[i].text.indexOf(text.trim()) >= 0) {
                                            var stepIdx = Math.round(annotations[i].x);
                                            window.__navigateToStep(stepIdx);
                                            break;
                                        }
                                    }
                                });
                            }
                            """,
                        )
                    with gr.Column(scale=2, min_width=300, elem_classes=["detail-panel"]):
                        detail_html = gr.HTML(
                            _DETAIL_PLACEHOLDER,
                            elem_id="wf-detail-panel",
                        )
                detail_store = gr.HTML("", elem_id="wf-detail-store")

            # ===== Raw Data Tab =====
            with gr.TabItem("Raw Data"):
                raw_json = gr.Code(
                    label="Full trajectory JSON",
                    language="json",
                    value="",
                    max_lines=50,
                )

        # -- Callbacks --

        _empty_fig = go.Figure()
        _empty_fig.update_layout(template="plotly_white", height=380)

        def _empty_result(banner="", detail="*No data*"):
            """Return the empty outputs tuple for error states."""
            f = _empty_fig
            return (
                gr.update(visible=False),  # main_tabs
                gr.update(visible=False),  # summary_area
                gr.update(),               # upload_accordion (no change)
                [],              # state_steps
                banner,          # summary_banner
                "",              # anomaly_strip_html
                "",              # overview_kpi_html
                "",              # session_detail_html
                "",              # metrics_md
                f, f,            # token_chart, duration_chart
                f,               # context_growth_chart
                "",              # behavior_md
                f,               # tool_chart
                f,               # tool_outcome_chart
                "",              # agent_summary_html
                f,               # agent_token_chart
                f,               # agent_swimlane_chart
                "",              # diag_summary_html
                f,               # diag_file_chart
                "",              # diag_rootcause_html
                # New diagnostic charts
                f,               # error_class_chart
                f,               # plan_timeline_chart
                "",              # hotspots_md
                "",              # per_message_md
                render_filter_chips(),  # wf_filter_chips_html
                ",".join(_FILTER_CHIPS_DEFAULT),  # wf_filter_hidden
                "",              # wf_count_html
                "",              # toc_html
                "<div></div>",   # workflow_html
                "",              # detail_store
                _DETAIL_PLACEHOLDER,  # detail_html
                "",              # raw_json
                # Patterns
                "",              # patterns_tool_html
                "",              # patterns_failure_html
                "",              # antipattern_summary_html
                {},              # state_raw
            )

        def do_load(upload_obj, dark=False, selected_format="ccsession"):
            """Load a trajectory, converting any unexpected failure into a
            friendly banner (logged server-side) instead of an uncaught crash.

            A valid-JSON-but-wrong-shape file (e.g. a top-level list/scalar — a
            common heterogeneous-export shape) otherwise throws uncaught inside
            ``detect_format``/``parse_steps`` and surfaces only as a generic
            Gradio error with nothing recorded.
            """
            try:
                return _do_load_impl(upload_obj, dark, selected_format)
            except Exception:
                logger.exception(
                    "do_load failed (file=%r, format=%r)",
                    getattr(upload_obj, "name", upload_obj), selected_format,
                )
                banner = (
                    "<p style='color:#dc2626;'>Could not load this trajectory. "
                    "It may be malformed or an unsupported shape for the selected "
                    "format. Check the file, or try a different format. "
                    "(Details were written to the server log.)</p>"
                )
                return _empty_result(banner=banner, detail="*Could not load trajectory.*")

        def _do_load_impl(upload_obj, dark=False, selected_format="ccsession"):
            """Load trajectory from uploaded file."""
            from .loaders import detect_format

            file_path = None
            if upload_obj is not None:
                file_path = upload_obj if isinstance(upload_obj, str) else upload_obj.name

            if not file_path or not os.path.isfile(file_path):
                return _empty_result(detail="*No file selected or file not found.*")

            raw = load_trajectory(file_path)
            if "_error" in raw:
                err_banner = f"<p style='color:#dc2626;'>Error: {html.escape(raw['_error'])}</p>"
                return _empty_result(banner=err_banner, detail="*Error loading file.*")

            # Validate format matches user selection
            _FORMAT_LABELS = {"ccsession": "Claude Code", "codearts": "CodeArts",
                              "opencode": "OpenCode",
                              "training_conversation": "Training Conversation",
                              "unknown": "Unknown"}
            detected = detect_format(raw)
            if detected != selected_format and detected != "unknown":
                err_msg = (
                    f"Format mismatch: selected "
                    f"<b>{html.escape(_FORMAT_LABELS.get(selected_format, selected_format))}</b>"
                    f" but file detected as "
                    f"<b>{html.escape(_FORMAT_LABELS.get(detected, detected))}</b>."
                )
                return _empty_result(
                    banner=f"<p style='color:#dc2626;'>{err_msg}</p>",
                    detail="*Please select the correct format and try again.*",
                )

            steps = parse_steps(raw)
            steps_total = len(steps)
            load_warnings = ""
            if steps_total > _MAX_STEPS:
                steps = steps[:_MAX_STEPS]
                load_warnings = (
                    f"<p style='color:#d97706;font-size:13px;margin:0 0 4px;'>"
                    f"&#9888; Showing first {_MAX_STEPS:,} of {steps_total:,} steps "
                    f"({steps_total - _MAX_STEPS:,} truncated).</p>"
                )
            token_warnings = validate_token_integrity(steps)
            for tw in token_warnings:
                load_warnings += (
                    f"<p style='color:#d97706;font-size:13px;margin:0 0 4px;'>"
                    f"&#9888; {html.escape(tw)}</p>"
                )
            message_rows = build_message_metrics(steps)
            metrics = compute_metrics(steps, raw, message_rows=message_rows)
            _, wfmt = wall_clock_fmt(metrics)

            step_analytics = compute_step_analytics(steps)
            phases = detect_phases(step_analytics)
            verdicts = compute_health_verdict(metrics, step_analytics if steps else [])
            agent_summaries = compute_agent_summary(steps, raw)

            # Delegate to focused builders
            dg = _build_diagnostics_outputs(steps, step_analytics, agent_summaries, dark=dark,
                                            trajectory_format=detected)
            ov = _build_overview_outputs(steps, raw, metrics, message_rows,
                                         verdicts, wfmt, file_path,
                                         trajectory_format=detected)

            _d = lambda k: raw.get(k, {}) if isinstance(raw.get(k), dict) else {}
            _md, _timing = _d("metadata"), _d("timing")
            _model_id, _, _agent_id = extract_agent_info(steps)
            session_detail = _build_session_detail_html(
                metrics, wfmt, _timing, _md, raw,
                model_id=_model_id, agent_id=_agent_id,
            )

            ch = _build_chart_outputs(steps, message_rows, phases,
                                      step_analytics, agent_summaries, dark=dark,
                                      trajectory=raw.get("trajectory", []),
                                      trajectory_format=detected)
            wf = _build_workflow_outputs(steps, agent_summaries)

            # Pattern detection
            tool_seqs = detect_tool_sequences(steps)
            fail_pats = detect_failure_patterns(steps)
            pat_tool_html = _render_tool_sequences_html(tool_seqs)
            pat_fail_html = _render_failure_patterns_html(fail_pats)

            # Raw data
            raw_str = json.dumps(raw, indent=2, ensure_ascii=False, default=str)
            if len(raw_str) > 500_000:
                raw_str = raw_str[:500_000] + "\n\n... (truncated at 500KB)"

            return (
                gr.update(visible=True),      # main_tabs
                gr.update(visible=True),      # summary_area
                gr.update(),                  # upload_accordion (keep visible)
                steps,                        # state_steps
                load_warnings + ov["banner"],  # summary_banner
                ov["anomaly_html"],           # anomaly_strip_html
                ov["kpi_html"],               # overview_kpi_html
                session_detail,               # session_detail_html
                ov["metrics_text"],           # metrics_md
                ch["tok_fig"],                # token_chart
                ch["dur_fig"],                # duration_chart
                ch["context_growth_fig"],     # context_growth_chart
                ov["behavior_text"],          # behavior_md
                ch["tl_fig"],                 # tool_chart
                ch["tool_outcome_fig"],       # tool_outcome_chart
                ch["agent_cards_html"],       # agent_summary_html
                ch["agent_tok_fig"],          # agent_token_chart
                ch["swimlane_fig"],           # agent_swimlane_chart
                dg["diag_summary_html"],      # diag_summary_html
                dg["diag_file_chart"],        # diag_file_chart
                dg["diag_rootcause_html"],    # diag_rootcause_html
                # New diagnostic charts
                ch["error_class_fig"],        # error_class_chart
                ch["plan_timeline_fig"],      # plan_timeline_chart
                ov["hotspots_text"],          # hotspots_md
                ov["per_message_text"],       # per_message_md
                wf["wf_chips"],               # wf_filter_chips_html
                wf["wf_filter_val"],          # wf_filter_hidden
                wf["wf_count"],               # wf_count_html
                wf["toc_html_val"],           # toc_html
                wf["wf_html"],                # workflow_html
                wf["detail_store_val"],       # detail_store
                _DETAIL_PLACEHOLDER,          # detail_html
                raw_str,                      # raw_json
                # Patterns
                pat_tool_html,                # patterns_tool_html
                pat_fail_html,                # patterns_failure_html
                ch["antipattern_html"],       # antipattern_summary_html
                raw,                          # state_raw
            )

        all_outputs = [
            main_tabs,
            summary_area,
            upload_accordion,
            state_steps,
            summary_banner,
            anomaly_strip_html,
            overview_kpi_html,
            session_detail_html,
            metrics_md,
            token_chart,
            duration_chart,
            context_growth_chart,
            behavior_md,
            tool_chart,
            tool_outcome_chart,
            agent_summary_html,
            agent_token_chart,
            agent_swimlane_chart,
            diag_summary_html,
            diag_file_chart,
            diag_rootcause_html,
            # New diagnostic outputs
            error_class_chart,
            plan_timeline_chart,
            hotspots_md,
            per_message_md,
            wf_filter_chips_html,
            wf_filter_hidden,
            wf_count_html,
            toc_html,
            workflow_html,
            detail_store,
            detail_html,
            raw_json,
            # Patterns
            patterns_tool_html,
            patterns_failure_html,
            antipattern_summary_html,
            state_raw,
        ]

        load_btn.click(
            fn=do_load,
            inputs=[file_upload, state_dark, format_selector],
            outputs=all_outputs,
        )
        # Auto-load when file is uploaded (no separate click needed)
        file_upload.change(
            fn=do_load,
            inputs=[file_upload, state_dark, format_selector],
            outputs=all_outputs,
        )

        # -- Comparison callback --
        # Semantics: the trajectory uploaded in this tab's file slot is the
        # **reference/baseline**; the trajectory loaded on the Overview tab
        # is the **compared** trajectory.
        def on_run_comparison(ref_file, anchor_file, ref_format,
                              ref_labels_file, cmp_labels_file,
                              overview_raw, dark):
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_white", height=380)

            if not overview_raw:
                return (
                    "<div style='color:var(--ov-warn);padding:1em;text-align:center;'>"
                    "Load a trajectory in the Overview tab first — it will be the compared trajectory.</div>",
                    empty_fig, empty_fig,
                    "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
                    "Load a trajectory in the Overview tab first.</div>",
                )

            if ref_file is None:
                return (
                    "<div style='color:var(--ov-warn);padding:1em;text-align:center;'>"
                    "Upload a reference trajectory first.</div>",
                    empty_fig, empty_fig,
                    "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
                    "Upload a reference trajectory.</div>",
                )

            ref_path = ref_file if isinstance(ref_file, str) else ref_file.name
            anchor_path = None
            if anchor_file is not None:
                anchor_path = anchor_file if isinstance(anchor_file, str) else anchor_file.name

            from .loaders import load_trajectory as _load_traj
            ref_raw = _load_traj(ref_path, format_hint=ref_format)

            result = run_comparison(
                ref_raw=ref_raw,
                cmp_raw=overview_raw,
                anchor_path=anchor_path,
                token_rate=50.0,
                fuzzy=False,
                dark=bool(dark),
            )

            # Per-phase comparison charts — only populated when BOTH trajectories
            # have a labeled JSON uploaded. Compared labels come from the
            # Overview tab's label upload; reference labels come from this tab.
            phase_count_fig = empty_fig
            phase_duration_fig = empty_fig
            ref_labels_path = (ref_labels_file if isinstance(ref_labels_file, str)
                               else (ref_labels_file.name if ref_labels_file else None))
            cmp_labels_path = (cmp_labels_file if isinstance(cmp_labels_file, str)
                               else (cmp_labels_file.name if cmp_labels_file else None))
            if ref_labels_path and cmp_labels_path:
                try:
                    from .labels import load_labeled_json, aggregate_labels
                    from .charts import (
                        build_phase_count_comparison_chart,
                        build_phase_duration_comparison_chart,
                    )
                    ref_agg = aggregate_labels(load_labeled_json(ref_labels_path))
                    cmp_agg = aggregate_labels(load_labeled_json(cmp_labels_path))
                    ref_label_name = os.path.basename(ref_path) if ref_path else "reference"
                    cmp_label_name = (
                        os.path.basename(overview_raw.get("_source_path", ""))
                        if isinstance(overview_raw, dict) else "compared"
                    ) or "compared"
                    phase_count_fig = build_phase_count_comparison_chart(
                        ref_agg["phase_counts"], cmp_agg["phase_counts"],
                        ref_label=ref_label_name, cmp_label=cmp_label_name,
                        dark=bool(dark),
                    )
                    phase_duration_fig = build_phase_duration_comparison_chart(
                        ref_agg["phase_durations"], cmp_agg["phase_durations"],
                        ref_label=ref_label_name, cmp_label=cmp_label_name,
                        dark=bool(dark),
                    )
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).debug(
                        "Phase comparison chart build failed: %s", exc)

            status = (
                "<div style='color:var(--ov-success);padding:0.5em;font-size:13px;'>"
                "Comparison complete.</div>"
            )

            return (result["report_html"], phase_count_fig, phase_duration_fig, status)

        cmp_run_btn.click(
            fn=on_run_comparison,
            inputs=[cmp_file_upload, cmp_anchor_upload, cmp_format_selector,
                    cmp_ref_labels_upload, label_file_upload,
                    state_raw, state_dark],
            outputs=[cmp_report_html, cmp_phase_count_chart,
                     cmp_phase_duration_chart, cmp_status_html],
        )

        # -- Workflow filter callback --
        def _filter_workflow_steps(steps, active_filters, keyword):
            """Return indices of steps matching filters and keyword."""
            if not steps:
                return []
            keyword = (keyword or "").strip().lower()
            active = set(active_filters)
            # Separate role filters (gate) from content filters (narrow)
            role_filters = active & {"Assistant", "User"}
            content_filters = active & {"Tool Calls", "Errors", "Reasoning"}
            # Agent filters: "agent:<id>" entries
            agent_filters = {f[6:] for f in active if f.startswith("agent:")}
            filtered = []
            for i, s in enumerate(steps):
                # Agent gate: if agent chips are active, step must be from one of them
                if agent_filters:
                    step_agent = effective_agent(s)
                    if step_agent not in agent_filters:
                        continue
                role = s["role"]
                # Role gate: step must match a checked role
                role_ok = (
                    ("Assistant" in role_filters and role == "assistant")
                    or ("User" in role_filters and role == "user")
                )
                if not role_ok:
                    continue
                # Content gate: only applies to steps that have filterable content
                if content_filters:
                    has_content = (s["tool_call_count"] > 0
                                   or s["error_count"] > 0
                                   or s["has_reasoning"])
                    if has_content:
                        content_ok = (
                            ("Tool Calls" in content_filters and s["tool_call_count"] > 0)
                            or ("Errors" in content_filters and s["error_count"] > 0)
                            or ("Reasoning" in content_filters and s["has_reasoning"])
                        )
                        if not content_ok:
                            continue
                # Keyword match
                if keyword:
                    text = (s.get("text_preview") or "").lower()
                    tool_names = " ".join(tc["tool_name"] for tc in s.get("tool_calls", [])).lower()
                    tool_args = " ".join(
                        str(tc.get("input", "")) for tc in s.get("tool_calls", [])
                    ).lower()
                    if keyword not in text and keyword not in tool_names and keyword not in tool_args:
                        continue
                filtered.append(i)
            return filtered

        def do_filter_workflow(steps, filter_csv, keyword):
            """Re-render workflow HTML with filters applied."""
            if not steps:
                return (
                    "<div style='padding:3em;color:#9ca3af;text-align:center;"
                    "font-size:15px;'>Load a trajectory to see the step flow.</div>",
                    "",
                )
            active_filters = [f.strip() for f in (filter_csv or "").split(",") if f.strip()]
            if not active_filters:
                return (
                    "<div style='padding:2em;color:#9ca3af;text-align:center;'>"
                    "No filters selected &mdash; click a chip to see steps.</div>",
                    "<div class='wf-count'>Showing 0 of "
                    f"{len(steps)} steps</div>",
                )
            indices = _filter_workflow_steps(steps, active_filters, keyword)
            filtered_steps = [steps[i] for i in indices]
            wf_html = render_workflow_html(filtered_steps)
            count_html = (
                f"<div class='wf-count'>Showing {len(filtered_steps)} of "
                f"{len(steps)} steps</div>"
            )
            return wf_html, count_html

        # -- TOC toggle callback --
        def on_toc_toggle(current_toc):
            """Toggle TOC sidebar visibility via CSS class."""
            if not current_toc:
                return current_toc
            if "toc-hidden" in current_toc:
                return current_toc.replace("toc-hidden", "").strip()
            return current_toc.replace("wf-toc-sidebar", "wf-toc-sidebar toc-hidden")

        wf_toc_toggle.click(
            fn=on_toc_toggle,
            inputs=[toc_html],
            outputs=[toc_html],
        )

        wf_filter_hidden.change(
            fn=do_filter_workflow,
            inputs=[state_steps, wf_filter_hidden, wf_search],
            outputs=[workflow_html, wf_count_html],
        )
        wf_search.change(
            fn=do_filter_workflow,
            inputs=[state_steps, wf_filter_hidden, wf_search],
            outputs=[workflow_html, wf_count_html],
        )

        # -- Label file upload callback --
        _empty_label_fig = go.Figure()
        _empty_label_fig.update_layout(template="plotly_white", height=380)

        def do_load_labels(upload_obj, steps_state):
            """Load labeled JSON and update all label UI components.

            Returns values matching the output list below:
              0: label_badge_html
              1: labels_accordion
              2: label_status_html
              3: label_charts_row1 (Row visibility)
              4: label_phase_count_chart (Plot)
              5: label_action_count_chart (Plot)
              6: label_charts_row2 (Row visibility)
              7: label_phase_dur_chart (Plot)
              8: label_action_dur_chart (Plot)
             9: label_timeline_row (Row visibility)
             10: label_timeline_chart (Plot)
             11: state_steps
             12: wf_count_html
             13: workflow_html
             14: detail_store
             15: detail_html
            """
            file_path = None
            if upload_obj is not None:
                file_path = upload_obj if isinstance(upload_obj, str) else upload_obj.name

            empty = _empty_label_fig

            if not file_path or not os.path.isfile(file_path):
                return (
                    "",  # label_badge_html
                    gr.update(),  # labels_accordion (no change)
                    "<div style='padding:1em;color:#9ca3af;text-align:center;'>"
                    "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>",
                    gr.update(visible=False), empty, empty,
                    gr.update(visible=False), empty, empty,
                    gr.update(visible=False), empty,
                    steps_state, gr.update(), gr.update(), gr.update(), gr.update(),
                )

            try:
                payload = build_label_ui_payload(file_path)
                updated_steps = steps_state
                wf_count_update = gr.update()
                workflow_update = gr.update()
                detail_store_update = gr.update()
                detail_html_update = gr.update()
                if payload.get("kind") == "training_v2" and steps_state:
                    label_data = load_training_labeled_json(file_path)
                    updated_steps = attach_training_labels_to_steps(steps_state, label_data)
                    wf_count_update = (
                        f"<div class='wf-count'>Showing {len(updated_steps)} of {len(updated_steps)} steps</div>"
                    )
                    workflow_update = render_workflow_html(updated_steps)
                    detail_store_update = _prerender_step_details(updated_steps)
                    detail_html_update = _DETAIL_PLACEHOLDER
            except Exception as exc:
                return (
                    "",  # label_badge_html
                    gr.update(),  # labels_accordion (no change)
                    f"<div style='padding:1em;color:#dc2626;text-align:center;'>"
                    f"Error: {html.escape(str(exc))}</div>",
                    gr.update(visible=False), empty, empty,
                    gr.update(visible=False), empty, empty,
                    gr.update(visible=False), empty,
                    steps_state, gr.update(), gr.update(), gr.update(), gr.update(),
                )

            return (
                payload["badge_html"],  # label_badge_html
                gr.update(open=True),  # labels_accordion (auto-open)
                payload["status_html"],
                gr.update(visible=True), payload["phase_count_fig"], payload["action_count_fig"],
                gr.update(visible=True), payload["phase_duration_fig"], payload["action_duration_fig"],
                gr.update(visible=True), payload["timeline_fig"],
                updated_steps, wf_count_update, workflow_update, detail_store_update, detail_html_update,
            )

        label_outputs = [
            label_badge_html,
            labels_accordion,
            label_status_html,
            label_charts_row1, label_phase_count_chart,
            label_action_count_chart,
            label_charts_row2, label_phase_dur_chart,
            label_action_dur_chart,
            label_timeline_row, label_timeline_chart,
            state_steps,
            wf_count_html,
            workflow_html,
            detail_store,
            detail_html,
        ]
        label_load_btn.click(
            fn=do_load_labels,
            inputs=[label_file_upload, state_steps],
            outputs=label_outputs,
        )
        # Auto-load labels on file upload
        label_file_upload.change(
            fn=do_load_labels,
            inputs=[label_file_upload, state_steps],
            outputs=label_outputs,
        )

        # Reset label UI whenever a new trajectory is loaded. Without this, stale
        # label state from a prior *_labeled.json upload lingers and the Labels
        # accordion renders data for the wrong trajectory.
        def _reset_labels():
            empty = _empty_label_fig
            return (
                "",                                       # label_badge_html
                gr.update(open=False),                    # labels_accordion
                ("<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                 "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>"),  # label_status_html
                gr.update(visible=False), empty, empty,   # row1 + two charts
                gr.update(visible=False), empty, empty,   # row2 + two charts
                gr.update(visible=False), empty,          # timeline row + chart
                gr.update(),                              # state_steps
                gr.update(),                              # wf_count_html
                gr.update(),                              # workflow_html
                gr.update(),                              # detail_store
                gr.update(),                              # detail_html
                gr.update(value=None),                    # clear the file picker
            )

        _reset_outputs = label_outputs + [label_file_upload]
        load_btn.click(fn=_reset_labels, inputs=None, outputs=_reset_outputs)
        file_upload.change(fn=_reset_labels, inputs=None, outputs=_reset_outputs)

        # Detect browser dark mode at page load and store in state_dark.
        app.load(
            fn=lambda dark: dark,
            inputs=[state_dark],
            outputs=[state_dark],
            js="() => [window.matchMedia('(prefers-color-scheme: dark)').matches]",
        )

    return app
