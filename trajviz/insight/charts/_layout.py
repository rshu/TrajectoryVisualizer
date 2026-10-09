"""Shared Plotly chrome for insight charts."""

from collections import Counter
from collections.abc import Sequence

# Pre-import pandas before plotly to avoid circular import error
# in plotly's basevalidators when running inside Gradio async threads.
try:
    import pandas  # noqa: F401
except ImportError:
    pass

import plotly.graph_objects as go

from ..palette import PLOTLY_DARK_TEMPLATE

_TPL = "plotly_white"


def _apply_dark(fig: go.Figure, dark: bool) -> go.Figure:
    """Apply dark-mode layout overrides when *dark* is True."""
    if dark:
        fig.update_layout(**PLOTLY_DARK_TEMPLATE)
    return fig


def _empty_figure(height: int = 380, message: str | None = None) -> go.Figure:
    """Return a blank Plotly figure, optionally with a centered message."""
    fig = go.Figure()
    if message:
        fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
    fig.update_layout(template=_TPL, height=height)
    return fig


def _apply_chart_layout(
    fig: go.Figure, title: str, xaxis: str | None = None, yaxis: str | None = None, height: int = 380, **kwargs
) -> None:
    """Apply standard chart layout (template + margins + responsive sizing).

    The title is centered horizontally (``x=0.5``, ``xanchor="center"``) and
    pinned to the top of the container (``y=0.99``, ``yanchor="top"``). The
    top margin is 90px so a horizontal legend above the plot area
    (``y=1.06``) can wrap to a second row without overlapping the title.

    Centering — rather than left-aligning — avoids collisions with the
    Gradio ``gr.Plot`` chrome: Gradio draws a small tab label in the
    top-left corner of every chart widget, which would sit on top of a
    left-aligned title.
    """
    layout = dict(
        title=dict(text=title, y=0.99, x=0.5, xanchor="center", yanchor="top", font=dict(size=16)),
        template=_TPL,
        height=height,
        autosize=True,
        margin=dict(t=90, b=40, l=60, r=20),
    )
    if xaxis:
        layout["xaxis_title"] = xaxis
    if yaxis:
        layout["yaxis_title"] = yaxis
    layout.update(kwargs)
    fig.update_layout(**layout)


def _add_legend_hint(fig: go.Figure) -> None:
    """Add a subtle 'click legend to toggle' hint at the bottom-right."""
    fig.add_annotation(
        text="Click legend items to show/hide series",
        xref="paper",
        yref="paper",
        x=1.0,
        y=-0.12,
        showarrow=False,
        font=dict(size=9, color="#9ca3af"),
        xanchor="right",
    )


def _truncate_chart_label(name: str, limit: int = 30) -> str:
    """Shorten a chart axis/legend label with a trailing ellipsis."""
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


def _truncate_chart_labels(names: Sequence[str], limit: int = 30) -> list[str]:
    """Truncate a whole label set at once, keeping the labels distinct.

    Two names that agree on their first ``limit - 3`` characters truncate to
    the same string, and Plotly treats equal category labels as one category:
    the rows merge and their bars are drawn on top of each other. Long MCP tool
    ids (``mcp__<server>__<verb>``) are the realistic source. A colliding label
    therefore gets a ``~N`` counter, budgeted inside *limit* so the axis gutter
    still fits. With no collision the plain truncation is returned unchanged,
    so existing charts are byte-identical.
    """
    labels = [_truncate_chart_label(n, limit) for n in names]
    counts = Counter(labels)
    if all(count == 1 for count in counts.values()):
        return labels
    seen: Counter[str] = Counter()
    out: list[str] = []
    for name, label in zip(names, labels, strict=True):
        if counts[label] == 1:
            out.append(label)
            continue
        seen[label] += 1
        suffix = f"~{seen[label]}"
        out.append(_truncate_chart_label(name, limit - len(suffix)) + suffix)
    return out


def _add_dummy_marker_legend(
    fig: go.Figure,
    entries: list[tuple],
    *,
    legendgroup: str,
    legendrank: int = 2000,
    size: int = 10,
) -> None:
    """Add non-data legend rows for shape keys.

    Each entry is ``(name, color, symbol)`` or
    ``(name, color, symbol, line_dict)``.
    """
    for entry in entries:
        name, color, symbol = entry[0], entry[1], entry[2]
        marker: dict = {"color": color, "size": size, "symbol": symbol}
        if len(entry) > 3 and entry[3]:
            marker["line"] = entry[3]
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker=marker,
                hoverinfo="skip",
                legendgroup=legendgroup,
                legendrank=legendrank,
            )
        )
