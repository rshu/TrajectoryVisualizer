"""Overview tab HTML and Plotly: summary banner, KPIs, charts, and diagnostics."""

from __future__ import annotations

import html
import os
from collections import Counter

import plotly.graph_objects as go

from ..charts import (
    build_agent_swimlane_chart,
    build_agent_token_chart,
    build_context_pressure_chart,
    build_duration_chart,
    build_file_interaction_chart,
    build_plan_timeline_chart,
    build_skill_agent_chart,
    build_token_chart,
    build_tool_chart,
    build_tool_duration_chart,
    build_tool_outcome_timeline,
)
from ..context_usage import PRESSURE_ALL_AGENTS
from ..formatting import (
    format_banner_html,
    format_behavioral_md,
    format_context_pressure_html,
    format_performance_md,
    _build_hotspots_md,
    _build_per_message_md,
)
from ..help import HELP_TEXT
from ..loaders import FORMAT_LABELS
from ..metrics import extract_agent_info
from ..palette import AGENT_COLORS
from ..rendering import (
    build_root_cause_html,
    render_agent_summary_cards,
)
from .issues import (
    ISSUE_KIND_COLORS,
    OverviewIssue,
    build_overview_issues_html,
    collect_overview_issues,
    rank_issues,
)

from ..session import MAX_STEPS, LoadedSession


def trajectory_format_label(fmt: str | None) -> str:
    """Return a human-readable trajectory format label."""
    return FORMAT_LABELS.get(fmt or "", fmt or "Unknown")


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
    timing: dict,
    metadata: dict,
    *,
    model_id: str = "",
    agent_id: str = "",
) -> str:
    """Build Session Details panel as a chip grid of session environment fields."""
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
    return (
        "<details class='session-detail' style='margin:0 0 10px;'>"
        "<summary style='cursor:pointer;font-size:12px;color:var(--ov-muted);"
        "user-select:none;'>Session details</summary>"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;'>{chips}</div>"
        "</details>"
    )


_ISSUE_KIND_LABELS = (
    ("error", "error"),
    ("antipattern", "pattern"),
    ("bottleneck", "bottleneck"),
)


def _kpi_breakdown_html(
    items: list[tuple[str, int, str]],
    *,
    title: str,
) -> str:
    """Color-coded swatch list used by Steps (agents) and Issues (kinds)."""
    if not items:
        return ""
    rows: list[str] = []
    for name, count, color in items:
        safe = html.escape(name)
        rows.append(
            "<div class='ov-kpi-breakdown-row'>"
            f"<span class='ov-kpi-breakdown-swatch' style='background:{color};'></span>"
            f"<span class='ov-kpi-breakdown-name' style='color:{color};' "
            f"title='{safe}'>{safe}</span>"
            f"<span class='ov-kpi-breakdown-count'>{count}</span>"
            "</div>"
        )
    return (
        f"<div class='ov-kpi-breakdown' title='{html.escape(title)}'>"
        + "".join(rows)
        + "</div>"
    )


def _issues_kpi_parts(issues: list[OverviewIssue]) -> tuple[str, str, str, str, str]:
    """Return (value, sub, status, detail, breakdown_html) for the Issues KPI card."""
    issue_count = len(issues)
    value = f"{issue_count:,}"
    if issue_count <= 0:
        return (
            value,
            "none detected",
            "good",
            "No major workflow issues detected",
            "",
        )
    kinds = Counter(issue.kind for issue in issues)
    items = [
        (
            label if kinds[key] == 1 else f"{label}s",
            kinds[key],
            ISSUE_KIND_COLORS[key],
        )
        for key, label in _ISSUE_KIND_LABELS
        if kinds[key]
    ]
    if issue_count >= 3:
        status, detail = "bad", f"{issue_count} issues — review Overview Issues"
    else:
        status, detail = "warn", f"{issue_count} issue{'s' if issue_count != 1 else ''} — review Overview Issues"
    return value, "", status, detail, _kpi_breakdown_html(items, title="Issues by kind")


def _agent_steps_breakdown(agent_summaries: list[dict] | None) -> str:
    """Color-coded agent step list for the Steps KPI card."""
    summaries = agent_summaries or []
    if not summaries:
        return ""
    items = [
        (
            str(agent.get("label") or agent.get("agent_id") or "agent").strip() or "agent",
            int(agent.get("step_count") or 0),
            AGENT_COLORS[idx % len(AGENT_COLORS)],
        )
        for idx, agent in enumerate(summaries)
    ]
    return _kpi_breakdown_html(items, title="Assistant steps by agent")


def build_overview_kpi_html(
    metrics: dict,
    wall_fmt: str,
    verdicts: list[dict] | None = None,
    message_rows: list[dict] | None = None,
    *,
    issues: list[OverviewIssue] | None = None,
    agent_summaries: list[dict] | None = None,
) -> str:
    """Build at-a-glance KPI card strip (global strip above the main Tabs).

    When *verdicts* is provided, matching KPI cards get a colored left border
    and a tooltip with the verdict detail string. *issues* drives the Issues
    card after Tokens. Steps shows an agent/subagent breakdown instead of the
    Errors health verdict (that signal lives on Tool Success / Issues).
    """
    _verdict_map: dict[str, tuple[str, str]] = {}
    if verdicts:
        _metric_to_kpi = {
            "Tool Success": "Tool Success",
            "Throughput": "Tokens",
            "Token Efficiency": "Tokens",
        }
        for v in verdicts:
            kpi_label = _metric_to_kpi.get(v["metric"], "")
            if kpi_label:
                _verdict_map[kpi_label] = (v["status"], v["detail"])

    issue_value, issue_sub, issue_status, issue_detail, issue_breakdown = _issues_kpi_parts(
        issues or [],
    )
    # Border/tooltip only — kind breakdown is rendered like the Steps agent list.
    _verdict_map["Issues"] = (issue_status, issue_detail)

    _status_colors = {
        "good": "#059669",
        "warn": "#d97706",
        "bad": "#dc2626",
    }

    sparkline_data: dict[str, list[float]] = {}
    if message_rows:
        sparkline_data["Tokens"] = [r.get("tokens_total", 0) for r in message_rows]
        sparkline_data["Wall-Clock"] = [r.get("duration", 0) or 0 for r in message_rows]

    _label_to_help_key = {
        "Steps": "steps",
        "Wall-Clock": "wall_clock",
        "Tokens": "tokens",
        "Issues": "issues",
        "Tool Success": "tool_success",
    }

    output_rate = metrics.get("output_tokens_per_sec")
    if isinstance(output_rate, (int, float)) and not isinstance(output_rate, bool):
        throughput_sub = f"{output_rate:,} gen tok/s"
        if (metrics.get("output_throughput_tool_wait_seconds") or 0) > 0:
            throughput_sub += " excl. tools"
    else:
        throughput_sub = "Generation throughput: N/A"
    timed_steps = metrics.get("output_throughput_timed_steps")
    throughput_steps = metrics.get("output_throughput_total_steps")
    if (
        metrics.get("output_throughput_incomplete")
        and isinstance(timed_steps, int)
        and isinstance(throughput_steps, int)
        and throughput_steps > 0
    ):
        throughput_sub += f" · {timed_steps}/{throughput_steps} timed"

    user_steps = metrics.get("user_steps", 0)
    agent_breakdown = _agent_steps_breakdown(agent_summaries)
    cards = [
        (
            "Steps",
            f"{metrics.get('total_steps', 0):,}",
            f"{metrics.get('assistant_steps', 0)} assistant, {user_steps} user",
        ),
        ("Wall-Clock", wall_fmt, f"P95 {metrics.get('p95_duration', 0)}s"),
        ("Tokens", f"{metrics.get('tokens', {}).get('total', 0):,}", throughput_sub),
        ("Issues", issue_value, issue_sub),
        ("Tool Success", f"{metrics.get('tool_success_rate', 0)}%", f"{metrics.get('tool_call_count', 0):,} calls"),
    ]
    card_html = []
    for label, value, sub in cards:
        verdict_info = _verdict_map.get(label)
        extra_style = ""
        title_attr = ""
        data_attr = ""
        jump_attr = ""
        extra_class = ""
        if verdict_info:
            status, detail = verdict_info
            border_color = _status_colors.get(status, "#6b7280")
            extra_style = f" style='border-left:4px solid {border_color};'"
            title_attr = f" title='{html.escape(detail)}'"
            data_attr = f" data-status='{html.escape(status)}'"
        if label == "Issues":
            extra_class = " ov-kpi-card--issues"
            jump_attr = (
                " role='link' tabindex='0'"
                " onclick=\"if(window.tvClickMainTab){window.tvClickMainTab('Overview');}"
                "var el=document.getElementById('overview-issues');"
                "if(el){el.scrollIntoView({behavior:'smooth',block:'start'});"
                "if(!el.open){el.open=true;}}\""
            )
        help_key = _label_to_help_key.get(label, "")
        help_attr = ""
        if help_key and help_key in HELP_TEXT:
            help_attr = f" data-help='{html.escape(HELP_TEXT[help_key])}'"
        sparkline = ""
        if label in sparkline_data:
            sparkline = _build_sparkline_svg(sparkline_data[label])
        sub_block = (
            f"<div class='ov-kpi-sub'>{html.escape(str(sub))}</div>" if sub else ""
        )
        extra_line = ""
        if label == "Steps" and agent_breakdown:
            extra_line = agent_breakdown
        elif label == "Issues" and issue_breakdown:
            extra_line = issue_breakdown
        elif verdict_info and label != "Issues":
            status, detail = verdict_info
            vcolor = _status_colors.get(status, "#6b7280")
            extra_line = (
                f"<div style='font-size:11px;color:{vcolor};margin-top:2px;'>"
                f"{html.escape(detail)}</div>"
            )
        card_html.append(
            f"<div class='ov-kpi-card{extra_class}'{extra_style}{title_attr}{data_attr}{jump_attr}>"
            f"<div class='ov-kpi-label'{help_attr}>{html.escape(str(label))}</div>"
            f"<div class='ov-kpi-value'>{html.escape(str(value))}</div>"
            f"{sub_block}"
            f"{extra_line}"
            f"{sparkline}"
            "</div>"
        )
    return "<div class='ov-kpi-grid'>" + "".join(card_html) + "</div>"


def empty_plotly_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def build_summary_outputs(session: LoadedSession) -> dict:
    """Legacy helper: stats one-liner (kept for tests; not shown in the live UI)."""
    return {
        "banner": format_banner_html(
            os.path.basename(session.path),
            session.metrics,
            session.wall_clock,
        ),
    }


def build_overview_outputs(session: LoadedSession) -> dict:
    """Compute overview-related outputs: banner, KPI, metadata, and metrics markdown."""
    steps = session.steps
    raw = session.raw
    metrics = session.metrics
    message_rows = session.message_rows
    verdicts = session.verdicts
    wfmt = session.wall_clock

    summary = build_summary_outputs(session)
    ranked_issues = rank_issues(collect_overview_issues(session))
    kpi_html = build_overview_kpi_html(
        metrics,
        wfmt,
        verdicts=verdicts,
        message_rows=message_rows,
        issues=ranked_issues,
        agent_summaries=session.agent_summaries,
    )
    issues_html = build_overview_issues_html(session, issues=ranked_issues)
    metrics_text = format_performance_md(metrics, wfmt)

    behavior_text = format_behavioral_md(metrics, diag_metrics=session.diagnostic_metrics)
    hotspots_text = _build_hotspots_md(message_rows)
    per_message_text = _build_per_message_md(message_rows)

    def _d(k):
        return raw.get(k, {}) if isinstance(raw.get(k), dict) else {}

    _md, _timing = _d("metadata"), _d("timing")
    _model_id, _, _agent_id = extract_agent_info(steps)
    session_detail = _build_session_detail_html(
        _timing,
        _md,
        model_id=_model_id,
        agent_id=_agent_id,
    )

    return {
        **summary,
        "kpi_html": kpi_html,
        "issues_html": issues_html,
        "session_detail": session_detail,
        "metrics_text": metrics_text,
        "behavior_text": behavior_text,
        "hotspots_text": hotspots_text,
        "per_message_text": per_message_text,
    }


def build_chart_outputs(session: LoadedSession, dark: bool = False) -> dict:
    """Build Overview chart figures."""
    steps = session.steps
    agent_summaries = session.agent_summaries
    trajectory_format = session.format

    return {
        "tok_fig": build_token_chart(steps, dark=dark, format=trajectory_format),
        "dur_fig": build_duration_chart(steps, dark=dark),
        "tl_fig": build_tool_chart(steps, dark=dark),
        "tool_dur_fig": build_tool_duration_chart(steps, dark=dark),
        "skill_fig": build_skill_agent_chart(steps, dark=dark),
        "tool_outcome_fig": build_tool_outcome_timeline(steps, dark=dark),
        "agent_cards_html": render_agent_summary_cards(agent_summaries),
        "agent_tok_fig": build_agent_token_chart(agent_summaries, dark=dark),
        "swimlane_fig": build_agent_swimlane_chart(steps, dark=dark),
        "plan_timeline_fig": build_plan_timeline_chart(session.plan_history, session.plan_metrics, dark=dark),
    }


def build_diagnostics_outputs(session: LoadedSession, dark: bool = False) -> dict:
    """Build diagnostics outputs from precomputed session domain data."""
    empty_fig = empty_plotly_fig()
    interactions = session.file_interactions
    target_files = session.target_files
    file_chart = (
        build_file_interaction_chart(
            interactions,
            target_files,
            dark=dark,
            steps=session.steps,
        )
        if interactions
        else empty_fig
    )

    rootcause_html = build_root_cause_html(session.clusters) if session.show_root_cause else ""

    pressure_fig = build_context_pressure_chart(
        session.steps,
        agent_key=PRESSURE_ALL_AGENTS,
        raw=session.raw,
        dark=dark,
    )
    pressure_html = format_context_pressure_html(
        session.pressure_series,
        steps=session.steps,
        raw=session.raw,
        agent_key=PRESSURE_ALL_AGENTS,
    )
    pressure_choices = session.pressure_choices
    pressure_dropdown = {
        "choices": pressure_choices or [("All agents", PRESSURE_ALL_AGENTS)],
        "value": PRESSURE_ALL_AGENTS,
        "visible": len(pressure_choices) > 2,
    }

    parts = []
    chain_metrics = session.chain_metrics
    if chain_metrics["total_chains"]:
        parts.append(f"{chain_metrics['total_chains']} failure chain(s)")
    if session.clusters:
        parts.append(f"{len(session.clusters)} root cause(s)")
    if session.performance_bottlenecks:
        parts.append(f"{len(session.performance_bottlenecks)} bottleneck(s)")
    if interactions:
        unique_files = len({i["path"] for i in interactions})
        parts.append(f"{unique_files} file(s) touched · {len(target_files)} edited")
    compaction_count = len(session.pressure_series.get("events") or [])
    if compaction_count:
        parts.append(f"{compaction_count} compaction event(s)")
    summary = " &middot; ".join(parts) if parts else "No diagnostic issues detected."
    summary_html = f"<div style='font-size:12px;color:var(--ov-muted);margin-bottom:8px;'>{summary}</div>"

    return {
        "diag_summary_html": summary_html,
        "diag_pressure_html": pressure_html,
        "diag_pressure_dropdown": pressure_dropdown,
        "diag_pressure_chart": pressure_fig,
        "diag_file_chart": file_chart,
        "diag_rootcause_html": rootcause_html,
    }


def load_warnings_html(session: LoadedSession) -> str:
    """HTML warning strip for truncation (when the run exceeds MAX_STEPS)."""
    if not session.truncated:
        return ""
    extra = session.steps_total - MAX_STEPS
    return (
        f"<p style='color:#d97706;font-size:13px;margin:0 0 4px;'>"
        f"&#9888; Showing first {MAX_STEPS:,} of {session.steps_total:,} steps "
        f"({extra:,} truncated).</p>"
    )
