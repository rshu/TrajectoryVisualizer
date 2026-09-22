"""Session activity charts: files, plan progress, errors, and context pressure."""

from __future__ import annotations

from ._layout import _add_dummy_marker_legend, _add_legend_hint, _apply_chart_layout, _apply_dark, _empty_figure
import plotly.graph_objects as go

from ..palette import SESSION_COLORS, TOKEN_COLORS
from ._timeline import _legend_label, bind_timeline_agents

_FILE_INTERACTION_COLORS = {
    "read": "#3b82f6",  # blue
    "write": "#10b981",  # green
    "search": "#f59e0b",  # orange
    "skill": "#eab308",  # gold
}

_FILE_INTERACTION_SYMBOLS = {
    "read": "circle",
    "write": "square",
    "search": "triangle-up",
    "skill": "star",
}

_FILE_INTERACTION_LEGEND = {
    "read": "read (circle)",
    "write": "write (square)",
    "search": "search (triangle)",
    "skill": "skill (star)",
}


def _add_shape_legend(fig: go.Figure, types_present: set[str]) -> None:
    """Legend entries that spell out marker shape (circle/square/triangle/star)."""
    _add_dummy_marker_legend(
        fig,
        [
            (_FILE_INTERACTION_LEGEND[itype], color, _FILE_INTERACTION_SYMBOLS[itype])
            for itype, color in _FILE_INTERACTION_COLORS.items()
            if itype in types_present
        ],
        legendgroup="shape",
        size=10,
    )


_COMPACTION_KIND_LABEL = {
    "compaction_part": "compacted",
    "compaction_message": "compacted",
    "summary": "compacted",
    "compress_step": "compacted",
    "tool_prune": "pruned",
    "occupancy_drop": "compacted",
}

# Occupancy must not share Fresh Input's blue — after compaction the stack is
# almost entirely fresh, and a blue occupancy line disappears into the fill.
_OCCUPANCY_LINE_COLOR = "#7c3aed"
_COMPACTION_MARKER_COLOR = "#e11d48"

# Row pitch for the Diagnostics file timeline. Keep in sync with the
# expander JS in insight.build_ui (Gradio forces Plotly autosize, so the
# pane height is restored from this same formula).
FILE_INTERACTION_ROW_PX = 28
FILE_INTERACTION_CHROME_PX = 120
FILE_INTERACTION_MIN_HEIGHT = 340
# Y-axis labels keep the path head and filename; the middle becomes "...".
FILE_PATH_LABEL_LIMIT = 40
_PATH_ELLIPSIS = "..."


def file_interaction_chart_height(unique_files: int) -> int:
    """Pixel height so every file row is visible without overlapping labels."""
    return max(
        FILE_INTERACTION_MIN_HEIGHT,
        FILE_INTERACTION_ROW_PX * unique_files + FILE_INTERACTION_CHROME_PX,
    )


def _ellipsize_middle_chars(text: str, limit: int, ellipsis: str = _PATH_ELLIPSIS) -> str:
    """Character-level middle ellipsis; extra characters go to the tail (filename)."""
    if len(text) <= limit:
        return text
    keep = limit - len(ellipsis)
    if keep <= 1:
        return (text[: max(0, keep)] + ellipsis)[:limit]
    left = max(1, keep // 3)
    right = keep - left
    return text[:left] + ellipsis + text[-right:]


def shorten_middle_path(path: str, limit: int = FILE_PATH_LABEL_LIMIT) -> str:
    """Keep the start and end of *path*; replace omitted components with ``...``."""
    if not path or len(path) <= limit:
        return path
    sep = "/" if "/" in path else "\\"
    parts = path.split(sep)
    if len(parts) <= 2:
        return _ellipsize_middle_chars(path, limit)

    def render(head: list[str], tail: list[str], omitted: bool) -> str:
        bits = list(head)
        if omitted:
            bits.append(_PATH_ELLIPSIS)
        bits.extend(tail)
        if parts[0] == "":
            return sep + sep.join(bits[1:])
        return sep.join(bits)

    head = parts[:2] if parts[0] == "" and len(parts) > 2 else parts[:1]
    tail = [parts[-1]]
    middle = parts[len(head) : -1]
    label = render(head, tail, bool(middle))
    if len(label) > limit:
        if parts[0] == "" and len(head) > 1:
            head = [""]
            label = render(head, tail, True)
        if len(label) > limit:
            return _ellipsize_middle_chars(path, limit)

    while middle:
        trial_tail = [middle[-1], *tail]
        trial = render(head, trial_tail, len(middle) > 1)
        if len(trial) <= limit:
            tail = trial_tail
            middle = middle[:-1]
            continue
        trial_head = [*head, middle[0]]
        trial = render(trial_head, tail, len(middle) > 1)
        if len(trial) <= limit:
            head = trial_head
            middle = middle[1:]
            continue
        break
    return render(head, tail, bool(middle))


def unique_short_paths(paths: list[str], limit: int = FILE_PATH_LABEL_LIMIT) -> dict[str, str]:
    """Map each full path to a unique middle-ellipsis label."""
    labels: dict[str, str] = {}
    used: set[str] = set()
    for path in paths:
        if path in labels:
            continue
        label = shorten_middle_path(path, limit)
        if label in used:
            found = None
            for extra in range(8, 96, 8):
                candidate = shorten_middle_path(path, limit + extra)
                if candidate not in used:
                    found = candidate
                    break
            if found is None:
                found = path
                n = 2
                while found in used:
                    found = f"{path} ({n})"
                    n += 1
            label = found
        labels[path] = label
        used.add(label)
    return labels


def build_file_interaction_chart(
    interactions: list[dict],
    target_files: set[str] | None = None,
    dark: bool = False,
    steps: list[dict] | None = None,
) -> go.Figure:
    """Build a Plotly scatter chart of file interactions across steps.

    x=step index, y=shortened file path (categorical). Color is interaction
    type for single-agent runs, and timeline agent (same palette as the
    swimlane) when more than one agent touched files. Marker shape is always
    read (circle) / write (square) / search (triangle) / skill (star); the
    legend spells those out. Target files are highlighted with a distinct
    marker border. Hover still shows the full path.
    """
    import os

    fig = go.Figure()
    if not interactions:
        _apply_chart_layout(fig, "File Interaction Timeline (no data)")
        _apply_dark(fig, dark)
        return fig

    target_files = target_files or set()

    def _norm(p: str) -> str:
        return os.path.normpath(p) if p else p

    norm_targets = {_norm(t) for t in target_files}

    color_map: dict[str, int] = {"": 0}
    labels: dict[str, str] = {"": "main"}
    agent_id_of = None
    step_by_idx: dict = {}
    if steps:
        color_map, labels, agent_id_of = bind_timeline_agents(steps)
        step_by_idx = {s["index"]: s for s in steps if isinstance(s, dict) and "index" in s}

    def _interaction_agent(item: dict) -> str:
        if agent_id_of is None:
            return ""
        step = step_by_idx.get(item["step"])
        return agent_id_of(step) if step else ""

    file_order: list[str] = []
    seen_files: set[str] = set()
    for item in interactions:
        path = item["path"]
        if path not in seen_files:
            seen_files.add(path)
            file_order.append(path)
    display = unique_short_paths(file_order)

    agents_present = []
    seen_agents: set[str] = set()
    for item in interactions:
        aid = _interaction_agent(item)
        if aid not in seen_agents:
            seen_agents.add(aid)
            agents_present.append(aid)
    has_agents = len(seen_agents) > 1

    def _add_group(group: list[dict], *, name: str, color: str, symbol=None) -> None:
        x = [i["step"] for i in group]
        y = [display[i["path"]] for i in group]
        is_target = [_norm(i["path"]) in norm_targets for i in group]
        hover = [f"{name}<br>Step {i['step']}: {i['tool']} ({i['type']})<br>{i['path']}" for i in group]
        border_colors = ["#dc2626" if t else color for t in is_target]
        border_widths = [2 if t else 0 for t in is_target]
        sizes = [12 if i["type"] == "skill" else 10 for i in group]
        if symbol is None:
            symbol = [_FILE_INTERACTION_SYMBOLS.get(i["type"], "circle") for i in group]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=sizes,
                    symbol=symbol,
                    line=dict(color=border_colors, width=border_widths),
                ),
                hovertext=hover,
                hoverinfo="text",
            )
        )

    if has_agents:
        for agent_id in sorted(color_map.keys(), key=lambda a: color_map[a]):
            group = [i for i in interactions if _interaction_agent(i) == agent_id]
            if not group:
                continue
            label = _legend_label(agent_id, labels)
            color = SESSION_COLORS[color_map[agent_id] % len(SESSION_COLORS)]
            _add_group(group, name=label, color=color)
        for aid in agents_present:
            if aid in color_map:
                continue
            group = [i for i in interactions if _interaction_agent(i) == aid]
            if group:
                _add_group(
                    group,
                    name=_legend_label(aid, labels),
                    color=SESSION_COLORS[len(color_map) % len(SESSION_COLORS)],
                )
    else:
        for itype, color in _FILE_INTERACTION_COLORS.items():
            group = [i for i in interactions if i["type"] == itype]
            if group:
                _add_group(
                    group,
                    name=_FILE_INTERACTION_LEGEND.get(itype, itype),
                    color=color,
                    symbol=_FILE_INTERACTION_SYMBOLS.get(itype, "circle"),
                )

    types_present = {i["type"] for i in interactions}
    if has_agents:
        _add_shape_legend(fig, types_present)

    unique_files = len(file_order)
    chart_height = file_interaction_chart_height(unique_files)

    max_label_len = max((len(display[p]) for p in file_order), default=20)
    # Cap the left gutter so long paths don't steal the plot; automargin can
    # still grow it when labels would otherwise clip.
    left_margin = max(140, min(320, max_label_len * 8))
    title = "File Interaction Timeline by Agent" if has_agents else "File Interaction Timeline"

    # Title + legend in container coords with fixed pixel offsets so they
    # stay stacked (title above legend) even when the chart is tall.
    top_margin = 90
    title_y = 1 - 8 / chart_height
    legend_y = 1 - 42 / chart_height
    _apply_chart_layout(
        fig,
        title,
        xaxis="Step",
        yaxis="File",
        height=chart_height,
        margin=dict(l=left_margin, r=24, t=top_margin, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yref="container",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
        ),
        meta={"tv_chart_height": chart_height},
    )
    fig.update_layout(
        title=dict(
            text=title,
            y=title_y,
            yref="container",
            yanchor="top",
            x=0.5,
            xanchor="center",
        )
    )
    # categoryarray is bottom→top; reverse so the first-touched files sit at the top.
    fig.update_yaxes(
        fixedrange=False,
        automargin=True,
        categoryorder="array",
        categoryarray=list(reversed([display[p] for p in file_order])),
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_layout(dragmode="pan", autosize=True)
    _apply_dark(fig, dark)
    return fig


def build_plan_timeline_chart(
    plan_history: list[dict],
    plan_metrics: dict,
    dark: bool = False,
) -> go.Figure:
    """Horizontal Gantt-style chart showing plan item progress over steps.

    Each bar represents a todo item, spanning from first in_progress to completed.
    Stalled items are highlighted in red.
    """
    items = plan_metrics.get("items", [])
    if not items:
        fig = _empty_figure(300, "No plan data available")
        _apply_dark(fig, dark)
        return fig

    stalled_contents = {s["content"] for s in plan_metrics.get("stalled", [])}

    # Sort by start_step so the timeline reads top→bottom in chronological order.
    # Plotly places the first-added bar at the *bottom* of a horizontal bar chart,
    # so we iterate in reverse (latest start first) to get earliest-at-top.
    # Items without a start_step (never started) fall to the bottom.
    def _sort_key(it: dict) -> tuple[int, int]:
        s = it.get("start_step")
        return (0, s) if s is not None else (1, 0)

    items = sorted(items, key=_sort_key, reverse=True)

    fig = go.Figure()
    y_labels = []
    for item in items:
        content = item["content"]
        short = content[:40] + "..." if len(content) > 40 else content
        y_labels.append(short)

        start = item.get("start_step")
        end = item.get("end_step")
        is_stalled = content in stalled_contents

        if start is not None and end is not None:
            color = "#dc2626" if is_stalled else "#059669"
            width = end - start
            fig.add_trace(
                go.Bar(
                    y=[short],
                    x=[max(width, 1)],
                    orientation="h",
                    base=start,
                    marker_color=color,
                    showlegend=False,
                    hovertext=f"{content}<br>Steps {start}→{end} ({width} steps)",
                    hoverinfo="text",
                    text=f"{width} steps",
                    textposition="inside",
                )
            )
        elif start is not None:
            fig.add_trace(
                go.Bar(
                    y=[short],
                    x=[1],
                    orientation="h",
                    base=start,
                    marker_color="#d97706",
                    showlegend=False,
                    hovertext=f"{content}<br>Started at step {start}, not completed",
                    hoverinfo="text",
                    text="stalled",
                    textposition="inside",
                )
            )
        elif end is not None:
            # Item went straight to "completed" without ever being marked
            # in_progress (common for OpenCode todo lists). Show a thin marker
            # at the completion step so the row isn't blank.
            fig.add_trace(
                go.Bar(
                    y=[short],
                    x=[1],
                    orientation="h",
                    base=max(0, end - 1),
                    marker_color="#3b82f6",
                    showlegend=False,
                    hovertext=f"{content}<br>Completed at step {end} (no in_progress recorded)",
                    hoverinfo="text",
                    text="completed",
                    textposition="inside",
                )
            )
        else:
            # Never started or completed — show a grey placeholder at x=0 so
            # the row appears in the y-axis instead of being silently dropped.
            fig.add_trace(
                go.Bar(
                    y=[short],
                    x=[0.5],
                    orientation="h",
                    base=0,
                    marker_color="#9ca3af",
                    showlegend=False,
                    hovertext=f"{content}<br>Never started",
                    hoverinfo="text",
                    text="not started",
                    textposition="outside",
                    cliponaxis=False,
                )
            )

    # Compact bar height (22px each + 90px padding) keeps the chart from
    # ballooning vertically when only a few items have explicit timing.
    chart_height = max(220, 22 * len(items) + 90)

    _apply_chart_layout(
        fig,
        "Plan Progress Timeline",
        xaxis="Step",
        height=chart_height,
        margin=dict(l=max(200, max((len(lbl) for lbl in y_labels), default=10) * 6 + 20), r=40, t=50, b=40),
    )
    # Tighter bar gap so individual items don't visually balloon when the
    # surrounding container is wide (e.g., right column of a 2-col layout).
    fig.update_layout(bargap=0.25)
    _apply_dark(fig, dark)
    return fig


def _pressure_series_colors(agents: list[dict]) -> dict[str, str]:
    """Distinct hue per pressure series, in first-appearance order."""
    return {agent.get("agent_id", ""): SESSION_COLORS[i % len(SESSION_COLORS)] for i, agent in enumerate(agents)}


def build_context_pressure_chart(
    steps: list[dict],
    *,
    agent_key: str | None = None,
    raw: dict | None = None,
    dark: bool = False,
    window_limit: int | float | str | None = None,
    series: dict | None = None,
    highlight_step: int | None = None,
) -> go.Figure:
    """Context-window occupancy over global step index, with compaction markers.

    Overlay mode (two or more agents) draws one occupancy line per agent in a
    distinct color. Compaction is drawn on that agent's series only — never as
    a full-height line across every session. Single-agent mode stacks fresh vs
    cache under an occupancy line and, when a window limit is known, adds
    70%/90% bands.
    """
    from ..context_usage import context_pressure_series

    if series is None:
        series = context_pressure_series(
            steps, agent_key=agent_key, raw=raw, window_limit=window_limit,
        )
    agents = series.get("agents") or []
    events = series.get("events") or []
    window_limit = series.get("window_limit")
    has_points = any(a.get("points") for a in agents)
    if not has_points:
        fig = _empty_figure(380, "No context occupancy data.")
        _apply_dark(fig, dark)
        return fig

    fig = go.Figure()
    single = len(agents) == 1
    color_map = _pressure_series_colors(agents)
    y_max = 0
    for agent in agents:
        for point in agent.get("points") or []:
            y_max = max(y_max, int(point.get("occupancy") or 0))
    if isinstance(window_limit, (int, float)) and window_limit > 0:
        y_max = max(y_max, int(window_limit))

    if single:
        agent = agents[0]
        color = _OCCUPANCY_LINE_COLOR
        points = agent.get("points") or []
        xs = [p["step"] for p in points]
        fresh = [p["fresh"] for p in points]
        cache = [p["cache_read"] for p in points]
        occ = [p["occupancy"] for p in points]
        turns = [p["local_turn"] for p in points]
        pct_suffix = []
        for occupancy in occ:
            if isinstance(window_limit, (int, float)) and window_limit > 0:
                pct_suffix.append(f"<br>Pressure: {100.0 * occupancy / window_limit:.1f}%")
            else:
                pct_suffix.append("")
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=fresh,
                name="Fresh input",
                mode="lines",
                line=dict(width=0.5, color=TOKEN_COLORS["fresh_input"]),
                stackgroup="occ",
                fillcolor="rgba(59,130,246,0.35)",
                hovertemplate=("Step %{x}<br>Turn %{customdata}<br>Fresh: %{y:,}<extra></extra>"),
                customdata=turns,
                legendgroup="tokens",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=cache,
                name="Cache read",
                mode="lines",
                line=dict(width=0.5, color=TOKEN_COLORS["cache_read"]),
                stackgroup="occ",
                fillcolor="rgba(52,211,153,0.35)",
                hovertemplate=("Step %{x}<br>Turn %{customdata}<br>Cache read: %{y:,}<extra></extra>"),
                customdata=turns,
                legendgroup="tokens",
            )
        )
        hover = [
            f"{agent['label']}<br>Step {x}<br>Turn {turn}<br>Occupancy: {o:,}{pct}"
            for x, turn, o, pct in zip(xs, turns, occ, pct_suffix, strict=False)
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=occ,
                name="Occupancy",
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                hovertext=hover,
                hoverinfo="text",
                legendgroup=agent.get("agent_id", ""),
            )
        )
    else:
        for agent in agents:
            points = agent.get("points") or []
            if not points:
                continue
            agent_id = agent.get("agent_id", "")
            color = color_map.get(agent_id, SESSION_COLORS[0])
            xs = [p["step"] for p in points]
            occ = [p["occupancy"] for p in points]
            turns = [p["local_turn"] for p in points]
            hover = []
            for x, turn, o, p in zip(
                xs,
                turns,
                occ,
                points,
                strict=False,
            ):
                pct = ""
                if isinstance(window_limit, (int, float)) and window_limit > 0:
                    pct = f"<br>Pressure: {100.0 * o / window_limit:.1f}%"
                hover.append(
                    f"{agent['label']}<br>Step {x}<br>Turn {turn}"
                    f"<br>Occupancy: {o:,}<br>Fresh: {p['fresh']:,}"
                    f"<br>Cache: {p['cache_read']:,}{pct}"
                )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=occ,
                    name=agent["label"],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=6, color=color),
                    hovertext=hover,
                    hoverinfo="text",
                    legendgroup=agent_id,
                )
            )

    _add_compaction_markers(
        fig,
        events,
        agents=agents,
        color_map=color_map,
        y_max=y_max,
        marker_color=_COMPACTION_MARKER_COLOR if single else None,
    )
    _add_snapshot_marker(fig, agents, highlight_step)

    if isinstance(window_limit, (int, float)) and window_limit > 0:
        fig.add_hline(
            y=window_limit,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Window {window_limit:,}",
            annotation_position="top right",
        )
        if single:
            fig.add_hrect(
                y0=window_limit * 0.7,
                y1=window_limit * 0.9,
                fillcolor="rgba(217,119,6,0.08)",
                line_width=0,
            )
            fig.add_hrect(
                y0=window_limit * 0.9,
                y1=window_limit,
                fillcolor="rgba(220,38,38,0.10)",
                line_width=0,
            )

    title = "Context Window Pressure"
    if single:
        title = f"Context Window Pressure — {agents[0]['label']}"
    _apply_chart_layout(
        fig,
        title,
        xaxis="Step",
        yaxis="Tokens (occupancy)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    if y_max > 0:
        fig.update_yaxes(range=[0, y_max * 1.08])
    _add_legend_hint(fig)
    _apply_dark(fig, dark)
    return fig


def _add_snapshot_marker(
    fig: go.Figure,
    agents: list[dict],
    highlight_step: int | None,
) -> None:
    """Open circle on the selected occupancy turn (pre-compaction snapshot)."""
    if highlight_step is None:
        return
    for agent in agents:
        for point in agent.get("points") or []:
            if int(point.get("step") or 0) != highlight_step:
                continue
            occ = int(point.get("occupancy") or 0)
            fig.add_trace(
                go.Scatter(
                    x=[highlight_step],
                    y=[occ],
                    mode="markers",
                    name="Selected window",
                    marker=dict(
                        size=16,
                        symbol="circle-open",
                        color=_OCCUPANCY_LINE_COLOR,
                        line=dict(width=3, color=_OCCUPANCY_LINE_COLOR),
                    ),
                    hovertext=(
                        f"Selected window<br>Step {highlight_step}"
                        f"<br>Occupancy: {occ:,}"
                    ),
                    hoverinfo="text",
                    showlegend=True,
                )
            )
            return


def _add_compaction_markers(
    fig: go.Figure,
    events: list[dict],
    *,
    agents: list[dict],
    color_map: dict[str, str],
    y_max: int,
    marker_color: str | None = None,
) -> None:
    """Per-agent compaction diamonds and drop stems — not full-height vlines."""
    if not events:
        return
    from ..context_usage import coalesce_compaction_events

    labels = {a.get("agent_id", ""): a.get("label") or "main" for a in agents}
    by_agent: dict[str, list[dict]] = {}
    for event in events:
        agent_id = event.get("agent", "")
        if agent_id not in labels:
            continue
        by_agent.setdefault(agent_id, []).append(event)

    for agent_id, agent_events in by_agent.items():
        color = marker_color or color_map.get(agent_id) or SESSION_COLORS[0]
        agent_label = labels.get(agent_id, "main")
        coalesced = coalesce_compaction_events(agent_events)
        xs: list[int] = []
        ys: list[float] = []
        hover: list[str] = []
        for event in coalesced:
            step = int(event.get("step") or 0)
            kind = event.get("kind") or ""
            kind_label = _COMPACTION_KIND_LABEL.get(kind, kind)
            before = event.get("occupancy_before")
            after = event.get("occupancy_after")
            dropped = event.get("dropped")
            y = before if isinstance(before, (int, float)) and before > 0 else after
            if not isinstance(y, (int, float)) or y <= 0:
                y = y_max * 0.5 if y_max else 0
            drop_txt = f"<br>−{int(dropped):,} tokens" if dropped else ""
            after_txt = f"<br>After: {int(after):,}" if isinstance(after, (int, float)) and after > 0 else ""
            hover.append(f"{agent_label}<br>Step {step}<br>{kind_label}{drop_txt}{after_txt}")
            xs.append(step)
            ys.append(float(y))
            if (
                isinstance(before, (int, float))
                and before > 0
                and isinstance(after, (int, float))
                and 0 < after < before
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[step, step],
                        y=[before, after],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                        legendgroup=agent_id,
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{agent_label} compact",
                    )
                )
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=f"{agent_label} compact",
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="diamond",
                    color=color,
                    line=dict(width=1.5, color="#fff"),
                ),
                legendgroup=agent_id,
                hovertext=hover,
                hoverinfo="text",
            )
        )
