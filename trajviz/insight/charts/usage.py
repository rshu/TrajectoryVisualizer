"""Step and agent usage charts: tokens, duration, and tools."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Callable

from ._layout import _add_legend_hint, _apply_chart_layout, _apply_dark, _empty_figure, _truncate_chart_label
import plotly.graph_objects as go

from ..parser import infer_non_cache_input
from ..palette import (
    AGENT_COLORS,
    CHART_ACCENT,
    DURATION_ERROR_COLORS,
    SESSION_COLORS,
    TOKEN_COLORS,
)
from trajviz.tool_vocab import parse_skill_name
from ._timeline import _legend_label, bind_timeline_agents
from ..metrics import (
    spawn_wait_seconds,
    step_duration_excluding_spawn,
    tool_call_stats_duration_ms,
)
from ..shell_cmd import tool_chart_name
from ..step_errors import step_error_kind


def _add_token_bar_traces(
    fig: go.Figure,
    x_values: list,
    fresh_input: list,
    cache_read: list,
    output: list,
    reasoning: list,
    *,
    cache_write: list | None = None,
    include_empty: bool = False,
    x_label: str = "Step",
) -> None:
    """Add token category bar traces to a figure.

    By default only adds traces that have non-zero data to avoid misleading legends
    (e.g., trajectories that only report total tokens, no breakdown).

    Pass ``include_empty=True`` to force every trace into the legend (useful when
    the viewer wants to see which fields are tracked even if they happen to be
    zero across the entire trajectory).

    Pass ``cache_write`` to add a 5th stacked trace. For OpenCode this sample
    always reports zero, but the field is present and worth showing.
    """
    traces = [
        ("Fresh Input", fresh_input, TOKEN_COLORS["fresh_input"]),
        ("Cache Read", cache_read, TOKEN_COLORS["cache_read"]),
        ("Output", output, TOKEN_COLORS["output"]),
        ("Reasoning", reasoning, TOKEN_COLORS["reasoning"]),
    ]
    if cache_write is not None:
        traces.append(("Cache Write", cache_write, TOKEN_COLORS["cache_write"]))
    for name, values, color in traces:
        if include_empty or any(v > 0 for v in values):
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=values,
                    name=name,
                    marker_color=color,
                    hovertemplate=f"{(x_label + ' ') if x_label else ''}%{{x}}<br>{name}: %{{y:,.0f}}<extra></extra>",
                )
            )


def _detect_outliers(values: list[float], threshold: float = 2.0) -> list[tuple[int, float, str]]:
    """Return (index, value, label) for values exceeding *threshold* σ from mean.

    Returns empty list if fewer than 10 values or no outliers found.
    """
    if len(values) < 10:
        return []
    clean = [v for v in values if v is not None and v > 0]
    if len(clean) < 5:
        return []
    mean = statistics.mean(clean)
    stdev = statistics.stdev(clean) if len(clean) > 1 else 0
    if stdev == 0:
        return []
    outliers = []
    for i, v in enumerate(values):
        if v is not None and v > 0 and (v - mean) > threshold * stdev:
            outliers.append((i, v, "spike"))
    return outliers


def build_token_chart(steps: list[dict], dark: bool = False, *, format: str | None = None) -> go.Figure:
    """Stacked bar of token breakdown over steps (non-overlapping segments).

    Default segments: fresh_input + cache_read
                      + net_output (output - reasoning) + reasoning = total

    Per-format overrides:
    - ``format in ("opencode", "codearts")``: render all five token fields
      stacked — Fresh Input, Cache Read, Output, Reasoning, Cache Write — with
      every trace forced into the legend even when a field is zero across the
      trajectory.  Both formats expose the same complete token schema.
    - No breakdown available: fall back to a single ``Total``
      bar.
    """
    if not steps:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig

    indices = list(range(len(steps)))
    cache_r = [s["tokens"]["cache_read"] for s in steps]
    fresh_input = [
        infer_non_cache_input(
            total_tokens=s["tokens"]["total"],
            input_tokens=s["tokens"]["input"],
            output_tokens=s["tokens"]["output"],
            reasoning_tokens=s["tokens"].get("reasoning", 0),
            cache_read_tokens=s["tokens"]["cache_read"],
        )
        for s in steps
    ]
    reasoning_t = [s["tokens"].get("reasoning", 0) for s in steps]
    output_t = [s["tokens"]["output"] for s in steps]
    net_output = [max(0, output - reasoning) for output, reasoning in zip(output_t, reasoning_t, strict=False)]
    cache_w = [s["tokens"].get("cache_write", 0) or 0 for s in steps]

    # Detect if token breakdown is available (any non-zero input/output/cache)
    has_breakdown = any(
        s["tokens"]["input"] > 0 or s["tokens"]["output"] > 0 or s["tokens"]["cache_read"] > 0 for s in steps
    )

    fig = go.Figure()
    if format in ("opencode", "codearts") and has_breakdown:
        # OpenCode / CodeArts view: show all five fields explicitly, forcing
        # zero traces into the legend so no existing legend item disappears.
        # These formats report output and reasoning as disjoint fields, so use
        # the raw output value instead of subtracting reasoning a second time.
        _add_token_bar_traces(
            fig, indices, fresh_input, cache_r, output_t, reasoning_t, cache_write=cache_w, include_empty=True
        )
    elif has_breakdown:
        _add_token_bar_traces(fig, indices, fresh_input, cache_r, net_output, reasoning_t)
    else:
        # No breakdown available — single "Total" trace.
        totals = [s["tokens"]["total"] for s in steps]
        fig.add_trace(
            go.Bar(
                x=indices,
                y=totals,
                name="Total",
                marker_color=CHART_ACCENT,
                hovertemplate="Step %{x}<br>Total: %{y:,.0f}<extra></extra>",
            )
        )

    _apply_chart_layout(
        fig,
        "Token Usage by Step",
        xaxis="Step",
        yaxis="Tokens",
        height=380,
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _add_legend_hint(fig)
    _apply_dark(fig, dark)
    return fig


def build_duration_chart(
    steps: list[dict],
    phases: list[dict] | None = None,
    dark: bool = False,
    compression_steps: list[int] | None = None,
) -> go.Figure:
    """Bar chart of step durations with average line.

    System errors (scaffold tools) and tool errors (agentic/custom) are
    separate legend series. Optional context compression markers.

    Spawn/delegation wait (``task``, ``Agent``, …) is subtracted from each
    bar so parent steps blocked on a child agent do not create false spikes.
    """
    if not steps:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig

    chart_durs = [step_duration_excluding_spawn(s) for s in steps]
    durations = [d if d is not None else 0 for d in chart_durs]
    spawn_adjusted = [
        d is not None and spawn_wait_seconds(s) > 0
        for s, d in zip(steps, chart_durs, strict=True)
    ]

    real_durations = [d for d in chart_durs if d is not None]
    avg_d = sum(real_durations) / len(real_durations) if real_durations else 0

    kinds = [step_error_kind(s) for s in steps]
    step_ids = [int(s.get("index", i)) for i, s in enumerate(steps)]
    normal_x = [i for i, k in enumerate(kinds) if k is None]
    normal_y = [durations[i] for i in normal_x]
    system_x = [i for i, k in enumerate(kinds) if k == "system"]
    system_y = [durations[i] for i in system_x]
    tool_x = [i for i, k in enumerate(kinds) if k == "tool"]
    tool_y = [durations[i] for i in tool_x]

    # Detect outliers — will be added as scatter labels per legend group
    outlier_set = {idx for idx, _, _ in _detect_outliers(durations)}

    bar_width = 0.8

    def _hover(idxs: list[int], *, suffix: str = "") -> tuple[list, str]:
        custom = [[step_ids[i], " (excl. task wait)" if spawn_adjusted[i] else ""] for i in idxs]
        template = (
            "Step %{customdata[0]}<br>%{y:.1f}s%{customdata[1]}"
            f"{suffix}<extra></extra>"
        )
        return custom, template

    fig = go.Figure()
    normal_cd, normal_hover = _hover(normal_x)
    fig.add_trace(
        go.Bar(
            x=normal_x,
            y=normal_y,
            customdata=normal_cd,
            name="Normal",
            legendgroup="Normal",
            marker_color="#3b82f6",
            width=bar_width,
            hovertemplate=normal_hover,
        )
    )
    if system_x:
        system_cd, system_hover = _hover(system_x, suffix=" (system error)")
        fig.add_trace(
            go.Bar(
                x=system_x,
                y=system_y,
                customdata=system_cd,
                name="System Error",
                legendgroup="System Error",
                marker_color=DURATION_ERROR_COLORS["system"],
                width=bar_width,
                hovertemplate=system_hover,
            )
        )
    if tool_x:
        tool_cd, tool_hover = _hover(tool_x, suffix=" (tool error)")
        fig.add_trace(
            go.Bar(
                x=tool_x,
                y=tool_y,
                customdata=tool_cd,
                name="Tool Error",
                legendgroup="Tool Error",
                marker_color=DURATION_ERROR_COLORS["tool"],
                width=bar_width,
                hovertemplate=tool_hover,
            )
        )

    # Spike labels as scatter traces — grouped with their bar trace so they
    # show/hide together when the legend is toggled.
    series = [
        ("Normal", normal_x, normal_y),
        ("System Error", system_x, system_y),
        ("Tool Error", tool_x, tool_y),
    ]
    for group, gx, gy in series:
        spike_x = [gx[j] for j in range(len(gx)) if gx[j] in outlier_set]
        spike_y = [gy[j] for j in range(len(gx)) if gx[j] in outlier_set]
        spike_text = [f"{v:.1f}s" for v in spike_y]
        if spike_x:
            fig.add_trace(
                go.Scatter(
                    x=spike_x,
                    y=spike_y,
                    mode="text",
                    text=spike_text,
                    textposition="top center",
                    textfont=dict(size=9, color="#dc2626"),
                    legendgroup=group,
                    showlegend=False,
                    hoverinfo="skip",
                    cliponaxis=False,  # allow text to extend past the right edge
                )
            )
    # Avg line — dashed, lighter
    fig.add_hline(
        y=avg_d,
        line_dash="dash",
        line_color="#94a3b8",
        line_width=1,
        annotation_text=f"Avg: {avg_d:.1f}s",
        annotation_position="top left",
        annotation_font=dict(color="#64748b", size=11),
    )
    _apply_chart_layout(
        fig,
        "Step Duration",
        xaxis="Step",
        yaxis="Duration (s)",
        height=400,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5, itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    fig.update_layout(margin=dict(t=70), clickmode="event")
    fig.update_xaxes(range=[-0.5, len(steps) - 0.5])
    _add_legend_hint(fig)

    # Context compression markers (red vertical lines)
    if compression_steps:
        for step_idx in compression_steps:
            fig.add_vline(
                x=step_idx,
                line_dash="dot",
                line_color="#dc2626",
                line_width=1.5,
                annotation_text="compressed",
                annotation_position="top",
                annotation_font=dict(size=8, color="#dc2626"),
            )

    _apply_dark(fig, dark)
    return fig


def build_tool_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bar chart of tool call frequency by name, stacked by agent."""
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    has_agents = len(color_map) > 1

    agent_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_tools: dict[str, int] = defaultdict(int)
    for s in steps:
        if not isinstance(s, dict):
            continue
        agent = agent_id_of(s)
        for tc in s.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = tool_chart_name(tc)
            agent_tool[agent][name] += 1
            all_tools[name] += 1

    if not all_tools:
        fig = _empty_figure(300, "No tool calls recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    sorted_tools = sorted(all_tools.keys(), key=lambda t: all_tools[t])
    display_names = [_truncate_chart_label(n) for n in sorted_tools]

    fig = go.Figure()
    if has_agents:
        # Stacked bars per agent, matching swimlane / workflow labeling
        for agent_id in sorted(color_map.keys(), key=lambda a: color_map[a]):
            label = _legend_label(agent_id, labels)
            counts = [agent_tool[agent_id].get(t, 0) for t in sorted_tools]
            if not any(counts):
                continue
            color = SESSION_COLORS[color_map[agent_id] % len(SESSION_COLORS)]
            fig.add_trace(
                go.Bar(
                    y=display_names,
                    x=counts,
                    orientation="h",
                    name=label,
                    marker_color=color,
                    hovertemplate=f"{label}: " + "%{x} call(s)<extra></extra>",
                )
            )
    else:
        counts = [all_tools[t] for t in sorted_tools]
        fig.add_trace(
            go.Bar(
                y=display_names,
                x=counts,
                orientation="h",
                marker_color=CHART_ACCENT,
                text=[str(c) for c in counts],
                textposition="outside",
                cliponaxis=False,
                customdata=sorted_tools,
                hovertemplate="%{customdata}: %{x} call(s)<extra></extra>",
            )
        )

    max_label = max(len(n) for n in display_names)
    _apply_tool_hbar_layout(
        fig,
        "Tool Call Frequency" + (" by Agent" if has_agents else ""),
        xaxis="Count",
        n_tools=len(sorted_tools),
        max_label=max_label,
        barmode="stack" if has_agents else "relative",
    )
    if has_agents:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),margin=dict(t=150),)
    _apply_dark(fig, dark)
    return fig


def build_tool_duration_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bars of tool duration, stacked by individual timed calls.

    Each segment is one tool invocation. Hover shows that segment's duration
    and the step index where it ran (plus the agent label when multiple agents
    are present).

    Spawn/delegation tools (``task``, ``Agent``, …) are omitted: their duration
    is wall-clock for the whole child agent, which swamps real tool timings and
    double-counts work already shown via the child's own calls.
    """
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    has_agents = len(color_map) > 1

    calls_by_tool: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    total_secs: dict[str, float] = defaultdict(float)
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        agent = agent_id_of(s)
        step_idx = int(s.get("index", i))
        for tc in s.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            ms = tool_call_stats_duration_ms(tc)
            if ms is None:
                continue
            name = tool_chart_name(tc)
            secs = ms / 1000.0
            calls_by_tool[name].append((step_idx, secs, agent))
            total_secs[name] += secs

    if not total_secs:
        fig = _empty_figure(300, "No tool-call timing recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    sorted_tools = sorted(total_secs.keys(), key=lambda t: total_secs[t])
    display_names = [_truncate_chart_label(n) for n in sorted_tools]

    fig = go.Figure()
    legend_seen: set[str] = set()
    for tool_name, y_label in zip(sorted_tools, display_names, strict=True):
        base = 0.0
        for step_idx, secs, agent_id in calls_by_tool[tool_name]:
            if has_agents:
                color = SESSION_COLORS[color_map.get(agent_id, 0) % len(SESSION_COLORS)]
                agent_label = _legend_label(agent_id, labels)
                show_legend = agent_id not in legend_seen
                if show_legend:
                    legend_seen.add(agent_id)
                hover_extra = f"<br>{agent_label}"
            else:
                color = CHART_ACCENT
                agent_label = ""
                show_legend = False
                hover_extra = ""

            fig.add_trace(
                go.Bar(
                    y=[y_label],
                    x=[secs],
                    base=[base],
                    orientation="h",
                    name=agent_label,
                    legendgroup=agent_id if has_agents else None,
                    showlegend=show_legend,
                    marker=dict(
                        color=color,
                        line=dict(width=1, color="rgba(255,255,255,0.9)"),
                    ),
                    customdata=[step_idx],
                    hovertemplate=(
                        f"{tool_name}<br>Step %{{customdata}}"
                        f"<br>{secs:.1f}s{hover_extra}<extra></extra>"
                    ),
                )
            )
            base += secs

    max_label = max(len(n) for n in display_names)
    _apply_tool_hbar_layout(
        fig,
        "Tool Call Duration" + (" by Agent" if has_agents else ""),
        xaxis="Duration (s)",
        n_tools=len(sorted_tools),
        max_label=max_label,
        barmode="overlay",
    )
    fig.update_layout(clickmode="event")
    fig.update_yaxes(categoryorder="array", categoryarray=display_names)
    if has_agents:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="right", x=1.0),margin=dict(t=200))
    _apply_dark(fig, dark)
    return fig


def _apply_tool_hbar_layout(
    fig: go.Figure,
    title: str,
    *,
    xaxis: str,
    n_tools: int,
    max_label: int,
    barmode: str,
) -> None:
    """Shared layout for horizontal tool frequency / duration bars."""
    _apply_chart_layout(
        fig,
        title,
        xaxis=xaxis,
        height=max(250, 50 * n_tools),
        barmode=barmode,
        margin=dict(l=max(140, max_label * 7 + 20), r=60, t=50, b=40),
    )


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba()`` string."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _spread_sankey_y(n: int) -> list[float]:
    """Even y positions in (0, 1), inset so first/last nodes are not clipped."""
    if n <= 1:
        return [0.5]
    return [0.08 + 0.84 * i / (n - 1) for i in range(n)]


def _trunc_skill_label(name: str, limit: int = 28) -> str:
    return name if len(name) <= limit else name[: limit - 3] + "..."


def collect_skill_calls(
    steps: list[dict],
    agent_id_of: Callable[[dict], str],
) -> Counter[tuple[str, str]]:
    """Count ``(agent_id, skill_id)`` pairs from Skill-tool invocations."""
    counts: Counter[tuple[str, str]] = Counter()
    for step in steps:
        if not isinstance(step, dict):
            continue
        agent = agent_id_of(step)
        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            skill = parse_skill_name(str(tc.get("tool_name") or ""), tc.get("input"))
            if skill:
                counts[(agent, skill)] += 1
    return counts


def build_skill_agent_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Sankey diagram of Skill-tool invocations: which agent called which skill."""
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    counts = collect_skill_calls(steps, agent_id_of)
    if not counts:
        fig = _empty_figure(300, "No Skill-tool invocations recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    skill_totals: Counter[str] = Counter()
    agent_totals: Counter[str] = Counter()
    for (agent, skill), n in counts.items():
        skill_totals[skill] += n
        agent_totals[agent] += n

    agent_ids = [aid for aid in sorted(color_map.keys(), key=lambda a: color_map[a]) if agent_totals[aid]]
    skills = [name for name, _ in skill_totals.most_common()]

    node_labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []

    agent_ys = _spread_sankey_y(len(agent_ids))
    skill_ys = _spread_sankey_y(len(skills))

    agent_index: dict[str, int] = {}
    for i, aid in enumerate(agent_ids):
        agent_index[aid] = len(node_labels)
        label = _legend_label(aid, labels)
        node_labels.append(f"{label} ({agent_totals[aid]})")
        node_colors.append(SESSION_COLORS[color_map[aid] % len(SESSION_COLORS)])
        node_x.append(0.02)
        node_y.append(agent_ys[i])

    skill_index: dict[str, int] = {}
    skill_color = "#64748b"
    for i, skill in enumerate(skills):
        skill_index[skill] = len(node_labels)
        node_labels.append(f"{_trunc_skill_label(skill)} ({skill_totals[skill]})")
        node_colors.append(skill_color)
        node_x.append(0.98)
        node_y.append(skill_ys[i])

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []
    hovers: list[str] = []
    for (agent, skill), n in counts.items():
        sources.append(agent_index[agent])
        targets.append(skill_index[skill])
        values.append(n)
        agent_color = SESSION_COLORS[color_map[agent] % len(SESSION_COLORS)]
        link_colors.append(_hex_rgba(agent_color, 0.38))
        agent_label = _legend_label(agent, labels)
        noun = "call" if n == 1 else "calls"
        hovers.append(f"{agent_label} → {skill}<br>{n} {noun}")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=14,
                    thickness=16,
                    line=dict(width=0.5, color="rgba(148,163,184,0.45)"),
                    label=node_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                    hovertemplate="%{label}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    customdata=hovers,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )
        ]
    )
    n_nodes = max(len(agent_ids), len(skills))
    _apply_chart_layout(
        fig,
        "Skill Calls by Agent",
        height=max(300, 48 * n_nodes + 80),
        margin=dict(l=16, r=16, t=90, b=24),
    )
    _apply_dark(fig, dark)
    return fig


def build_agent_token_chart(agent_summaries: list[dict], dark: bool = False) -> go.Figure:
    """Grouped bar chart showing token breakdown per agent.

    Uses the same vertical grouped-bar layout for single- and multi-agent
    sessions so the single-agent case still surfaces the four token
    categories (Fresh / Cache Read / Output / Reasoning) — just with one
    cluster instead of many.
    """
    if not agent_summaries:
        fig = _empty_figure(240, "No agent activity recorded.")
        _apply_dark(fig, dark)
        return fig

    labels = [a["label"] for a in agent_summaries]
    has_breakdown = any(a["input_tokens"] > 0 or a["output_tokens"] > 0 for a in agent_summaries)

    fig = go.Figure()
    if has_breakdown:
        from ..parser import infer_non_cache_input

        # Schema-tolerant fresh-input: handles both Claude Code (input includes
        # cache) and OpenCode (input is already cache-excluded).
        fresh_input = [
            infer_non_cache_input(
                a["total_tokens"],
                a["input_tokens"],
                a["output_tokens"],
                a["reasoning_tokens"],
                a["cache_read_tokens"],
            )
            for a in agent_summaries
        ]
        cache_read = [a["cache_read_tokens"] for a in agent_summaries]
        output = [max(0, a["output_tokens"] - a["reasoning_tokens"]) for a in agent_summaries]
        reasoning = [a["reasoning_tokens"] for a in agent_summaries]
        _add_token_bar_traces(fig, labels, fresh_input, cache_read, output, reasoning, x_label="")
    else:
        # No token breakdown available — show total tokens per agent
        session_palette = AGENT_COLORS  # shared palette so agent colors match across views
        totals = [a["total_tokens"] for a in agent_summaries]
        colors = [session_palette[i % len(session_palette)] for i in range(len(labels))]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=totals,
                name="Total Tokens",
                marker_color=colors,
                text=[f"{t:,}" for t in totals],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>%{y:,.0f} tokens<extra></extra>",
            )
        )

    _apply_chart_layout(
        fig,
        "Token Breakdown by Agent" if has_breakdown else "Total Tokens by Agent",
        xaxis="Agent",
        yaxis="Tokens (count)",
        height=320,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    fig.update_layout(margin=dict(t=70))
    _apply_dark(fig, dark)
    return fig
