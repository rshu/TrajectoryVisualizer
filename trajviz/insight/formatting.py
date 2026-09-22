"""Markdown and HTML display string generation."""

import html


_VERDICT_STYLES = {
    "good": ("var(--ov-success, #16a34a)", "#f0fdf4", "#bbf7d0"),
    "warn": ("var(--ov-warn, #d97706)", "#fffbeb", "#fde68a"),
    "bad":  ("var(--ov-bad, #dc2626)",  "#fef2f2", "#fecaca"),
}


def _metric_chip(label: str, value: str, *, wide: bool = False,
                 verdict: str | None = None, hint: str = "") -> str:
    """Render a single metric as a compact card chip (HTML).

    *verdict* adds a colored left border (good/warn/bad).
    *hint* adds a subtitle line below the value.
    """
    # label/value can be untrusted (tool, model, agent names); escape for HTML.
    label = html.escape(str(label))
    value = html.escape(str(value))
    hint = html.escape(str(hint)) if hint else ""
    min_w = "140px" if wide else "100px"
    border_left = ""
    bg = "#f8fafc"
    border_color = "#e2e8f0"
    if verdict and verdict in _VERDICT_STYLES:
        color, bg, border_color = _VERDICT_STYLES[verdict]
        border_left = f"border-left:3px solid {color};"
    hint_html = (f"<span style='font-size:9px;color:#94a3b8;margin-top:1px;'>{hint}</span>"
                 if hint else "")
    return (
        f"<div style='display:inline-flex;flex-direction:column;background:{bg};"
        f"border:1px solid {border_color};border-radius:8px;padding:6px 10px;"
        f"min-width:{min_w};{border_left}'>"
        f"<span style='font-size:10px;color:#64748b;text-transform:uppercase;'>{label}</span>"
        f"<span style='font-size:13px;color:#1e293b;font-weight:500;'>{value}</span>"
        f"{hint_html}"
        f"</div>"
    )


def _metric_grid(chips: list[str], title: str = "") -> str:
    """Wrap chips in a flex-wrap grid with optional title."""
    header = f"### {title}\n\n" if title else ""
    return header + "<div style='display:flex;flex-wrap:wrap;gap:6px;'>" + "".join(chips) + "</div>\n"


_FINISH_LABELS = {
    "tool-calls": "Tool Call",
    "stop": "Completed",
    "end_turn": "End Turn",
}


def _friendly_finish(raw: str | None) -> str:
    """Map internal finish-reason enums to user-friendly labels."""
    if not raw:
        return ""
    return _FINISH_LABELS.get(raw, raw.replace("-", " ").replace("_", " ").title())






def _build_hotspots_md(rows: list[dict]) -> str:
    """Build markdown tables for top latency/token/cache-miss hotspots."""
    with_dur = [r for r in rows if r.get("duration") is not None]
    top_d = sorted(with_dur, key=lambda r: r["duration"], reverse=True)[:5]
    top_t = sorted(rows, key=lambda r: r["tokens_total"], reverse=True)[:5]

    def fmt_table(items: list[dict], value_field: str, value_header: str,
                  value_fmt: str, extra_cols: list[tuple[str, str, str]] | None = None) -> str:
        if not items:
            return "*No data*"
        extra = extra_cols or [("tokens_total", "Tokens", ","), ("tool_calls", "Tool Calls", "")]
        hdr = " | ".join(h for _, h, _ in extra)
        lines = [
            f"| Step | Role | {value_header} | {hdr} |",
            "|---:|---|" + "---:|" * (1 + len(extra)),
        ]
        for r in items:
            v = r[value_field]
            v_str = format(v, value_fmt) if isinstance(v, (int, float)) else str(v)
            extras = " | ".join(
                format(r[f], ef) if isinstance(r[f], (int, float)) and ef else str(r[f])
                for f, _, ef in extra
            )
            lines.append(f"| {r['index']} | `{r['role']}` | {v_str} | {extras} |")
        return "\n".join(lines)

    sections = [
        "### Message Hotspots\n\n"
        "**Top latency steps**\n\n"
        + fmt_table(top_d, "duration", "Duration (s)", ".2f")
        + "\n\n**Top token-load steps**\n\n"
        + fmt_table(top_t, "tokens_total", "Tokens", ",",
                    extra_cols=[("tool_calls", "Tool Calls", ""),
                                ("duration", "Duration (s)", ".2f")])
    ]

    # Lowest cache ratio (assistant steps with tokens, excluding 0-token steps).
    # Skip the table entirely if no step reports any cache read — otherwise the
    # table is just five rows of 0.0% (uninformative) for trajectories whose
    # provider doesn't emit cache metrics (e.g., some Codex sessions).
    asst_with_tok = [r for r in rows
                     if r.get("role") == "assistant" and r["tokens_total"] > 0]
    if asst_with_tok and any(r["cache_ratio"] > 0 for r in asst_with_tok):
        low_cache = sorted(asst_with_tok, key=lambda r: r["cache_ratio"])[:5]
        lines = [
            "| Step | Role | Cache Read % | Fresh Input | Tokens |",
            "|---:|---|---:|---:|---:|",
        ]
        for r in low_cache:
            lines.append(
                f"| {r['index']} | `{r['role']}` | {r['cache_ratio'] * 100:.1f}% | "
                f"{r['non_cache_tokens']:,} | {r['tokens_total']:,} |"
            )
        sections.append(
            "\n\n**Lowest cache read steps** (optimization targets)\n\n"
            + "\n".join(lines)
        )

    return "".join(sections)


def _build_per_message_md(rows: list[dict], limit: int = 80) -> str:
    """Build a compact per-message diagnostics table."""
    if not rows:
        return "*No messages parsed.*"

    has_agent = any(r.get("agent") for r in rows)

    lines = ["### Per-Message Diagnostics", ""]
    if has_agent:
        lines.append(
            "| Step | Role | Agent | Finish | Duration (s) | Tokens | Tok/s | Cache Read % | Fresh Input | Tool Calls |"
        )
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append(
            "| Step | Role | Finish | Duration (s) | Tokens | Tok/s | Cache Read % | Fresh Input | Tool Calls |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows[:limit]:
        dur = "N/A" if r["duration"] is None else f"{r['duration']:.2f}"
        tokps = "N/A" if r["tokens_per_sec"] is None else f"{r['tokens_per_sec']:.1f}"
        finish = _friendly_finish(r['finish']) or '-'
        if has_agent:
            agent = r.get("agent", "") or "Main agent"
            lines.append(
                f"| {r['index']} | `{r['role']}` | `{agent}` | `{finish}` | {dur} | "
                f"{r['tokens_total']:,} | {tokps} | {r['cache_ratio'] * 100:.1f}% | "
                f"{r['non_cache_tokens']:,} | {r['tool_calls']} |"
            )
        else:
            lines.append(
                f"| {r['index']} | `{r['role']}` | `{finish}` | {dur} | "
                f"{r['tokens_total']:,} | {tokps} | {r['cache_ratio'] * 100:.1f}% | "
                f"{r['non_cache_tokens']:,} | {r['tool_calls']} |"
            )
    if len(rows) > limit:
        lines.append(f"\n*Showing first {limit} / {len(rows)} messages.*")
    return "\n".join(lines)


def format_performance_md(metrics: dict, wall_fmt: str) -> str:
    """Format performance & token metrics as a markdown table."""
    tok = metrics["tokens"]
    timing_chips = [
        _metric_chip("Steps", str(metrics["total_steps"])),
        _metric_chip("Wall-clock", wall_fmt),
        _metric_chip("Avg duration", f"{metrics['avg_duration']}s"),
        _metric_chip("Med / P95", f"{metrics['median_duration']}s / {metrics['p95_duration']}s", wide=True),
        _metric_chip("Max duration", f"{metrics['max_duration']}s"),
    ]
    # Token breakdown: show N/A when format doesn't provide subcategories
    has_breakdown = tok['input'] > 0 or tok['output'] > 0 or tok['cache_read'] > 0
    def _tok_val(v: int) -> str:
        return f"{v:,}" if has_breakdown else "N/A"
    reasoning_val = (
        f"{tok['reasoning']:,}"
        if has_breakdown and metrics.get("reasoning_tokens_reported")
        else "N/A"
    )

    token_chips = [
        _metric_chip("Total tokens", f"{tok['total']:,}"),
        _metric_chip("Input", _tok_val(tok['input'])),
        _metric_chip("Output", _tok_val(tok['output'])),
        _metric_chip("Reasoning", reasoning_val),
        _metric_chip("Cache read", _tok_val(tok['cache_read'])),
        _metric_chip("Cache write", _tok_val(tok['cache_write'])),
        _metric_chip("Fresh input",
                     f"{metrics['non_cache_tokens']:,} ({metrics['non_cache_ratio']}%)"
                     if has_breakdown else "N/A"),
    ]
    eff_chips = [
        _metric_chip("Avg tok/step", f"{metrics['avg_tokens_per_step']:,}"),
        _metric_chip("Total processed tok/sec", f"{metrics['tokens_per_second']:,}"),
        _metric_chip("Median processed tok/sec", f"{metrics['median_tokens_per_second']:,}"),
        _metric_chip("Out/In ratio", str(metrics["output_input_ratio"]) if has_breakdown else "N/A"),
        _metric_chip("Tok/tool call", f"{metrics['tokens_per_tool']:,}"),
    ]

    tool_chips = [
        _metric_chip(k, str(v))
        for k, v in sorted(metrics["tool_breakdown"].items(), key=lambda x: -x[1])
    ]
    # Build agent chips with main agent included and sub-agents labeled
    agent_bd = metrics.get("agent_breakdown", {})
    agent_labels = metrics.get("agent_labels") or {}
    sub_agent_steps = sum(agent_bd.values())
    main_agent_steps = metrics.get("assistant_steps", 0) - sub_agent_steps
    agent_chips = []
    if agent_bd:
        agent_chips.append(_metric_chip("main agent", f"{main_agent_steps} steps", wide=True))
        for k, v in sorted(agent_bd.items(), key=lambda x: -x[1]):
            label = agent_labels.get(k) or f"sub-agent {k[:12]}"
            agent_chips.append(_metric_chip(label, f"{v} steps", wide=True))
    model_chips = [
        _metric_chip(k, str(v))
        for k, v in sorted(metrics.get("model_breakdown", {}).items(), key=lambda x: -x[1])
    ]

    sections = [
        _metric_grid(timing_chips, "Timing"),
        _metric_grid(token_chips, "Tokens"),
        _metric_grid(eff_chips, "Efficiency"),
    ]
    if tool_chips:
        sections.append(
            f"\n**Tool calls** ({metrics['tool_call_count']} total, {metrics['tool_success_rate']}% success)\n\n"
            + _metric_grid(tool_chips)
        )
    if agent_chips:
        n_sub = len(agent_bd)
        sections.append(
            f"\n**Agent breakdown** (1 main + {n_sub} sub-agent{'s' if n_sub > 1 else ''})\n\n"
            + _metric_grid(agent_chips)
        )
    if model_chips:
        sections.append("\n**Model breakdown**\n\n" + _metric_grid(model_chips))

    return "\n".join(sections)


def _cache_verdict(avg_cache: float) -> str | None:
    if avg_cache >= 60:
        return "good"
    if avg_cache >= 30:
        return "warn"
    return "bad" if avg_cache > 0 else None


def _tool_wait_verdict(wait_share: float) -> str | None:
    if wait_share <= 30:
        return "good"
    if wait_share <= 60:
        return "warn"
    return "bad"


def format_behavioral_md(metrics: dict, diag_metrics: dict | None = None) -> str:
    """Format behavioral diagnostics as card grid with verdict badges."""
    avg_cache = metrics.get("avg_cache_ratio", 0)
    tool_wait = metrics.get("tool_wait_share", 0)
    has_cache_data = metrics.get("cache_read_tokens", 0) > 0 or avg_cache > 0
    has_tool_timing = metrics.get("tool_time_total", 0) > 0

    chips = [
        _metric_chip("Asst steps", str(metrics["assistant_steps"])),
        _metric_chip("Multi-tool", str(metrics["multi_tool_steps"])),
        _metric_chip("No-tool", str(metrics["no_tool_assistant_steps"])),
        _metric_chip("Med tok/step", f"{metrics['median_step_tokens']:,}"),
        _metric_chip("P95 tok/step", f"{metrics['p95_step_tokens']:,}"),
    ]
    # Cache metrics — N/A when format doesn't provide cache breakdown
    if has_cache_data:
        chips.append(_metric_chip("Avg cache %", f"{avg_cache}%",
                     verdict=_cache_verdict(avg_cache),
                     hint="≥60% good" if avg_cache < 60 else ""))
        chips.append(_metric_chip("Cache-dom", str(metrics["cache_dominant_steps"])))
    else:
        chips.append(_metric_chip("Avg cache %", "N/A", hint="not available for this format"))

    # Tool timing — N/A when format doesn't provide per-tool timestamps
    if has_tool_timing:
        chips.append(_metric_chip("Tool time", f"{metrics['tool_time_total']}s"))
        chips.append(_metric_chip("Tool-wait %", f"{tool_wait}%",
                     verdict=_tool_wait_verdict(tool_wait),
                     hint="≤30% good" if tool_wait > 30 else ""))
        chips.append(_metric_chip("Tool dur avg", f"{metrics['avg_tool_duration']}s"))
        chips.append(_metric_chip("Tool dur P95", f"{metrics['p95_tool_duration']}s"))
        chips.append(_metric_chip("Tool dur max", f"{metrics['max_tool_duration']}s"))
    else:
        chips.append(_metric_chip("Tool timing", "N/A", hint="not available for this format"))

    # Diagnostic metrics
    dm = diag_metrics or {}

    # Sub-agents
    if dm.get("subagent_session_count", 0) > 0:
        chips.append(_metric_chip("Sub-agents",
                                  f"{dm['subagent_session_count']} ({dm.get('subagent_total_steps', 0)} steps)"))

    # Tool errors
    if dm.get("classified_error_count", 0) > 0:
        chips.append(_metric_chip("Tool errors", str(dm['classified_error_count']),
                                  verdict="bad"))

    return _metric_grid(chips, "Behavioral Diagnostics")


def wall_clock_fmt(metrics: dict) -> tuple[float, str]:
    """Return (wall_seconds, formatted_string) for wall-clock time."""
    wall = metrics["wall_clock"] if isinstance(metrics["wall_clock"], (int, float)) else metrics["total_duration"]
    fmt = f"{wall:.0f}s" if wall < 3600 else f"{wall / 60:.1f}m"
    return wall, fmt


def format_banner_html(filename: str, metrics: dict, wall_fmt: str) -> str:
    """Build the one-line HTML summary banner for the loaded trajectory."""
    import html as _html
    parts = [
        f"<strong>{_html.escape(filename)}</strong> &nbsp;&mdash;&nbsp; ",
        f"{metrics['total_steps']} steps &middot; ",
        f"{metrics['tool_call_count']} tool calls ({metrics['tool_success_rate']}% success) &middot; ",
    ]
    # Only show token metrics if the format provides them
    total_tokens = metrics.get("tokens", {}).get("total", 0)
    if total_tokens > 0:
        parts.append(f"{total_tokens:,} tokens &middot; ")
        output_rate = metrics.get("output_tokens_per_sec")
        if isinstance(output_rate, (int, float)) and not isinstance(output_rate, bool):
            parts.append(f"{output_rate} gen tok/s &middot; ")
        else:
            parts.append(f"{metrics['tokens_per_second']} total processed tok/s &middot; ")
    parts.append(f"{wall_fmt} wall-clock")
    if metrics.get("reasoning_parts", 0) > 0:
        parts.append(f" &middot; {metrics['reasoning_parts']} reasoning")
    return "".join(parts)


def format_context_pressure_html(
    series: dict,
    *,
    steps: list[dict] | None = None,
    raw: dict | None = None,
    agent_key: str | None = None,
    snapshot_step: int | None = None,
) -> str:
    """Stats strip and context-usage breakdown for the Diagnostics chart."""
    from .context_usage import context_usage_breakdown, format_context_usage_html
    from .context_usage import context_pressure_stats

    stats = context_pressure_stats(series)
    peak = stats.get("peak_occupancy") or 0
    chips = [_metric_chip("Peak occupancy", f"{peak:,}")]
    peak_pct = stats.get("peak_pct")
    if isinstance(peak_pct, (int, float)):
        if peak_pct >= 90:
            verdict = "bad"
        elif peak_pct >= 70:
            verdict = "warn"
        else:
            verdict = "good"
        chips.append(_metric_chip("Peak pressure", f"{peak_pct:g}%", verdict=verdict))
    chips.append(_metric_chip("Compactions", str(stats.get("compaction_count") or 0)))
    drop = stats.get("largest_drop") or 0
    chips.append(_metric_chip("Largest drop", f"-{drop:,}" if drop else "0"))
    usage_html = ""
    if steps:
        usage_html = format_context_usage_html(
            context_usage_breakdown(
                steps,
                raw=raw,
                agent_key=agent_key,
                window_limit=series.get("window_limit"),
                snapshot_step=snapshot_step,
            )
        )
    return (
        usage_html
        + "<div style='margin:4px 0 8px;'>"
        + _metric_grid(chips)
        + "</div>"
    )


