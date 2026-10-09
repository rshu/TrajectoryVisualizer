"""Plotly chart builders for Converge trajectory comparison visualization."""

from __future__ import annotations

# Pre-import pandas before plotly to avoid circular import error
# in plotly's basevalidators when running inside Gradio async threads.
try:
    import pandas  # noqa: F401
except ImportError:
    pass

import plotly.graph_objects as go

from .milestones import MILESTONE_LABELS, MILESTONE_NAMES


# -- Layout helpers -----------------------------------------------------------

_TPL = "plotly_white"

# Milestone names/labels live in milestones.py, next to extract_milestones, so
# the chart and the HTML table cannot drift apart (C5). Aliased to the private
# names this module already used; the miss fallback stays `.get(name, name)`
# here and `.title()`-cased in rendering.py, so no rendered string changes.
_MILESTONE_NAMES = MILESTONE_NAMES
_MILESTONE_LABELS = MILESTONE_LABELS

_REF_COLOR = "#1d4ed8"   # blue
_CMP_COLOR = "#dc2626"   # red
_DELTA_COLOR = "#6b7280"  # gray

_PATTERN_COLORS = {
    "broad_exploration": "#3b82f6",
    "reverted_and_rewritten": "#ef4444",    # was write_retry
    "iterative_refinement": "#f87171",      # lighter red (less severe)
    "error_recovery_overhead": "#f59e0b",
    "premature_validation": "#8b5cf6",
    "redundant_search": "#06b6d4",
    "dead_end_branch": "#ec4899",
    "ordering_inefficiency": "#84cc16",
}


def _empty_figure(height: int = 380, message: str | None = None) -> go.Figure:
    """Return a blank Plotly figure, optionally with a centered message."""
    fig = go.Figure()
    if message:
        fig.add_annotation(
            text=message, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font_size=16,
        )
    fig.update_layout(template=_TPL, height=height)
    return fig


def _apply_chart_layout(
    fig: go.Figure, title: str,
    xaxis: str | None = None, yaxis: str | None = None,
    height: int = 380, **kwargs,
) -> None:
    """Apply standard chart layout."""
    layout = dict(
        title=title,
        template=_TPL,
        height=height,
        autosize=True,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    if xaxis:
        layout["xaxis_title"] = xaxis
    if yaxis:
        layout["yaxis_title"] = yaxis
    layout.update(kwargs)
    fig.update_layout(**layout)


# -- Milestone timeline -------------------------------------------------------

def build_milestone_timeline_chart(
    ref_milestones: dict[str, int | None],
    cmp_milestones: dict[str, int | None],
) -> go.Figure:
    """Dual-track horizontal chart showing milestone markers for reference and compared.

    Reference milestones on the top track (y=1), compared on the bottom track (y=0),
    with connecting lines showing deltas between matching milestones.
    """
    if not ref_milestones and not cmp_milestones:
        return _empty_figure(message="No milestone data available")

    fig = go.Figure()

    # Reference track (top, y=1)
    ref_x, ref_text = [], []
    for name in _MILESTONE_NAMES:
        val = ref_milestones.get(name)
        if val is not None:
            ref_x.append(val)
            ref_text.append(_MILESTONE_LABELS.get(name, name))

    if ref_x:
        fig.add_trace(go.Scatter(
            x=ref_x, y=[1] * len(ref_x),
            mode="markers+text",
            marker=dict(size=12, color=_REF_COLOR, symbol="diamond"),
            text=ref_text,
            textposition="top center",
            textfont=dict(size=9),
            name="Reference",
            hovertemplate="<b>%{text}</b><br>Step: %{x}<extra>Reference</extra>",
        ))

    # Compared track (bottom, y=0)
    cmp_x, cmp_text = [], []
    for name in _MILESTONE_NAMES:
        val = cmp_milestones.get(name)
        if val is not None:
            cmp_x.append(val)
            cmp_text.append(_MILESTONE_LABELS.get(name, name))

    if cmp_x:
        fig.add_trace(go.Scatter(
            x=cmp_x, y=[0] * len(cmp_x),
            mode="markers+text",
            marker=dict(size=12, color=_CMP_COLOR, symbol="diamond"),
            text=cmp_text,
            textposition="bottom center",
            textfont=dict(size=9),
            name="Compared",
            hovertemplate="<b>%{text}</b><br>Step: %{x}<extra>Compared</extra>",
        ))

    # Connecting lines for deltas
    for name in _MILESTONE_NAMES:
        ref_val = ref_milestones.get(name)
        cmp_val = cmp_milestones.get(name)
        if ref_val is not None and cmp_val is not None:
            delta = cmp_val - ref_val
            delta_label = f"+{delta}" if delta > 0 else str(delta)
            fig.add_trace(go.Scatter(
                x=[ref_val, cmp_val],
                y=[1, 0],
                mode="lines",
                line=dict(color=_DELTA_COLOR, width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_annotation(
                x=(ref_val + cmp_val) / 2, y=0.5,
                text=delta_label,
                showarrow=False,
                font=dict(size=9, color=_DELTA_COLOR),
            )

    _apply_chart_layout(
        fig, "Milestone Timeline",
        xaxis="Step Index", height=320,
        yaxis_visible=False,
        yaxis_range=[-0.5, 1.5],
        yaxis_tickvals=[0, 1],
        yaxis_ticktext=["Compared", "Reference"],
        yaxis_showticklabels=True,
    )
    return fig


# -- Segment cost chart -------------------------------------------------------

def build_segment_cost_chart(
    segment_data: dict,
    milestone_order_matches: bool,
) -> go.Figure:
    """Grouped bar chart for segment costs.

    When milestone_order_matches is True, uses segment_comparison (paired bars).
    Otherwise uses reference_segments + compared_segments (separate bars).
    """
    fig = go.Figure()

    if milestone_order_matches:
        comparisons = segment_data.get("segment_comparison", [])
        if not comparisons:
            return _empty_figure(message="No segment comparison data")

        labels = [c["segment"] for c in comparisons]
        overheads = [c.get("overhead", 0) for c in comparisons]

        fig.add_trace(go.Bar(
            x=labels, y=overheads,
            name="Overhead Ratio",
            marker_color=_CMP_COLOR,
            text=[f"{v:.2f}x" for v in overheads],
            textposition="outside",
        ))
        _apply_chart_layout(
            fig, "Segment Cost Overhead (Compared / Reference)",
            xaxis="Segment", yaxis="Overhead Ratio",
        )
    else:
        ref_segs = segment_data.get("reference_segments", [])
        cmp_segs = segment_data.get("compared_segments", [])

        if not ref_segs and not cmp_segs:
            return _empty_figure(message="No segment data")

        if ref_segs:
            fig.add_trace(go.Bar(
                x=[s["segment"] for s in ref_segs],
                y=[s.get("tokens", 0) for s in ref_segs],
                name="Reference",
                marker_color=_REF_COLOR,
            ))
        if cmp_segs:
            fig.add_trace(go.Bar(
                x=[s["segment"] for s in cmp_segs],
                y=[s.get("tokens", 0) for s in cmp_segs],
                name="Compared",
                marker_color=_CMP_COLOR,
            ))
        _apply_chart_layout(
            fig, "Segment Token Costs",
            xaxis="Segment", yaxis="Tokens",
            barmode="group",
        )

    return fig


# -- Divergence waterfall chart ------------------------------------------------

def build_divergence_waterfall_chart(patterns: list[dict]) -> go.Figure:
    """Waterfall chart showing cost breakdown by divergence pattern category.

    Patterns are grouped by type and ordered by total tokens descending.
    """
    if not patterns:
        return _empty_figure(message="No divergence patterns detected")

    # Aggregate by pattern type
    by_type: dict[str, int] = {}
    for p in patterns:
        ptype = p.get("type", "unknown")
        tokens = p.get("estimated_extra_cost", {}).get("tokens", 0)
        by_type[ptype] = by_type.get(ptype, 0) + tokens

    # Sort by tokens descending
    sorted_types = sorted(by_type.items(), key=lambda x: x[1], reverse=True)

    if not sorted_types or all(v == 0 for _, v in sorted_types):
        return _empty_figure(message="No measurable divergence cost")

    labels = [t for t, _ in sorted_types] + ["Total"]
    values = [v for _, v in sorted_types] + [sum(v for _, v in sorted_types)]
    measures = ["relative"] * len(sorted_types) + ["total"]

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=measures,
        text=[f"{v:,}" for v in values],
        textposition="outside",
        connector=dict(line=dict(color="#d1d5db", width=1)),
        increasing=dict(marker=dict(color="#ef4444")),
        decreasing=dict(marker=dict(color="#22c55e")),
        totals=dict(marker=dict(color="#6366f1")),
    ))

    _apply_chart_layout(
        fig, "Divergence Cost Breakdown by Pattern",
        xaxis="Pattern", yaxis="Extra Tokens",
    )
    return fig


# -- Anchor class recall chart ------------------------------------------------

def build_anchor_class_chart(anchor_analysis: dict | None) -> go.Figure:
    """Grouped bar chart showing per-class anchor-write recall for reference and compared.

    X-axis = file class, Y-axis = recall percentage.
    Reference bars are blue, compared bars are orange.
    """
    if not anchor_analysis:
        return _empty_figure(message="No anchor analysis available")

    # Per-class data is in reference/compared write_recall_by_class dicts
    ref_by_class = anchor_analysis.get("reference", {}).get("write_recall_by_class", {})
    cmp_by_class = anchor_analysis.get("compared", {}).get("write_recall_by_class", {})
    file_classes = anchor_analysis.get("file_classes", {})
    all_classes = sorted(set(ref_by_class.keys()) | set(cmp_by_class.keys()) | set(file_classes.keys()))

    if not all_classes:
        return _empty_figure(message="No per-class anchor data")

    classes = all_classes
    ref_recalls = [(ref_by_class.get(c) or 0) * 100 for c in classes]
    cmp_recalls = [(cmp_by_class.get(c) or 0) * 100 for c in classes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=classes,
        y=ref_recalls,
        name="Reference",
        marker_color=_REF_COLOR,
        text=[f"{v:.1f}%" for v in ref_recalls],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        x=classes,
        y=cmp_recalls,
        name="Compared",
        marker_color="#f97316",  # orange
        text=[f"{v:.1f}%" for v in cmp_recalls],
        textposition="outside",
    ))

    _apply_chart_layout(
        fig, "Anchor Write Recall by File Class",
        xaxis="File Class", yaxis="Recall (%)",
        barmode="group",
    )
    return fig
