"""Patterns tab HTML: tool sequences, failure clusters, and anti-patterns."""

from __future__ import annotations

import html

from ..rendering import _step_link_chip, build_antipattern_summary_html
from ..session import LoadedSession
from ..tool_failure import tool_call_failed


def count_tool_errors(steps: list[dict]) -> tuple[int, list[int]]:
    """Return ``(failed_call_count, step_indices)`` using :func:`tool_call_failed`."""
    error_steps: list[int] = []
    error_count = 0
    for step in steps:
        failed = [tc for tc in (step.get("tool_calls") or []) if tool_call_failed(tc)]
        if not failed:
            continue
        error_count += len(failed)
        error_steps.append(int(step.get("index", 0)))
    return error_count, error_steps


def build_antipattern_html(session: LoadedSession) -> str:
    """Anti-pattern summary HTML for Overview and Patterns."""
    error_count, error_steps = count_tool_errors(session.steps)
    return build_antipattern_summary_html(
        session.fruitless_streaks,
        session.tool_selection,
        session.plan_metrics,
        error_count=error_count,
        error_steps=error_steps,
    )


def render_tool_sequences_html(sequences: list[dict]) -> str:
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
        "<th style='text-align:left;padding:6px 10px;'>Step Indices</th></tr>" + "".join(rows) + "</table>"
    )


def render_failure_patterns_html(patterns: list[dict]) -> str:
    """Render failure pattern clusters as cards with Workflow jump chips."""
    if not patterns:
        return "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>No errors detected — no failure patterns to analyze.</div>"
    cards = []
    for p in patterns:
        label = html.escape(p.get("cluster_label", "Unknown"))
        count = p.get("count", 0)
        example = html.escape(str(p.get("example_error", ""))[:200])
        recovery = p.get("recovery_path")
        recovery_html = " → ".join(html.escape(t) for t in recovery) if recovery else "<em>No recovery path found</em>"
        step_ids = p.get("steps") or []
        steps_html = (
            "".join(_step_link_chip(int(idx)) for idx in step_ids)
            if step_ids
            else "<em>—</em>"
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
