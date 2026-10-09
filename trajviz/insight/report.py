"""Self-contained HTML snapshot of Overview, charts, and Patterns.

The live dashboard is Gradio. This module reuses the same presenters and writes
a single file that opens in a browser without the server — charts stay
interactive via Plotly's CDN bundle.
"""

from __future__ import annotations

import html
import os
import re
import tempfile
from datetime import UTC, datetime
from typing import Any

import plotly.graph_objects as go

from .loaders import FORMAT_LABELS
from .presenters.overview import (
    build_chart_outputs,
    build_diagnostics_outputs,
    build_overview_outputs,
    load_warnings_html,
)
from .presenters.patterns import (
    build_antipattern_html,
    render_failure_patterns_html,
    render_tool_sequences_html,
)
from .session import LoadedSession, LoadError, build_loaded_session, load_session
from .styles import APP_CSS

_PLOTLY_CDN = "cdn"

_MD_TABLE_SEP = re.compile(r"^\|[\s:|\-]+\|$")
_HEADING = re.compile(r"^(#{2,3})\s+(.*)$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_STYLE_TAG = re.compile(r"</?style[^>]*>", re.IGNORECASE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# The only pre-built markup the dashboard presenters put in their markdown is a
# chip grid: one `<div ...>` line per `_metric_grid` call. Anything else that
# starts a line with `<` came out of the trajectory.
_PASSTHROUGH_BLOCK = re.compile(r"^</?div\b", re.IGNORECASE)
_LIVE_MARKUP = re.compile(
    r"(?:^|[\s\"'/])on\w+\s*=|<\s*/?\s*(?:script|iframe|object|embed|svg|math|style|link|base)\b"
    r"|javascript\s*:",
    re.IGNORECASE,
)
# A markdown cell delimiter is an unescaped `|`; `formatting._md_cell_text`
# writes `\|` for a pipe that came out of the trajectory.
_MD_CELL_SEP = re.compile(r"(?<!\\)\|")


class ReportError(ValueError):
    """Raised when a trajectory cannot be turned into a report."""


def build_report_html(
    source: LoadedSession | dict,
    *,
    file_path: str | None = None,
    dark: bool = False,
    inline_plotly: bool = False,
) -> str:
    """Return a full HTML document for a loaded session or raw trajectory dict.

    The charts load plotly.js from its CDN by default, which keeps the file
    small but needs the network when the report is opened. ``inline_plotly``
    embeds the library (several MB) so the report works offline and makes no
    request at all.
    """
    session = _as_session(source, file_path=file_path)
    ov = build_overview_outputs(session)
    ch = build_chart_outputs(session, dark=dark)
    dg = build_diagnostics_outputs(session, dark=dark)
    pat_tool = render_tool_sequences_html(session.tool_sequences)
    pat_fail = render_failure_patterns_html(session.failure_patterns)
    antipattern = build_antipattern_html(session)
    issues_panel = ov["issues_html"]

    figures: list[tuple[str, go.Figure]] = [
        ("Token usage", ch["tok_fig"]),
        ("Step duration", ch["dur_fig"]),
        ("Context-window pressure", dg["diag_pressure_chart"]),
        ("Tool-call frequency", ch["tl_fig"]),
        ("Tool-call duration", ch["tool_dur_fig"]),
        ("Skill calls by agent", ch["skill_fig"]),
        ("Tool outcome timeline", ch["tool_outcome_fig"]),
        ("Tokens by agent", ch["agent_tok_fig"]),
        ("Agent swimlane", ch["swimlane_fig"]),
        ("File interactions", dg["diag_file_chart"]),
        ("Plan timeline", ch["plan_timeline_fig"]),
    ]

    basename = os.path.basename(session.path) or "trajectory"
    fmt_label = FORMAT_LABELS.get(session.format, session.format or "unknown")
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    title = f"TrajViz report — {basename}"

    include_js: str | bool = True if inline_plotly else _PLOTLY_CDN

    def charts(items: list[tuple[str, go.Figure]]) -> str:
        nonlocal include_js
        html_part, include_js = _charts_html(items, include_js)
        return html_part

    theme_class = "tv-theme-dark" if dark else "tv-theme-light"
    sections: list[str] = [
        _header_html(title, basename, fmt_label, generated, load_warnings_html(session)),
        _section(
            "Summary",
            issues_panel
            + ov["kpi_html"]
            + ov["session_detail"],
        ),
        charts(figures[:2]),
        _section("Deep dive", _mixed_md_to_html(ov["metrics_text"])),
        _section("Context utilization", dg["diag_pressure_html"]),
        charts(figures[2:3]),
        _section("Tools", _mixed_md_to_html(ov["behavior_text"])),
        charts(figures[3:6]),
        _section("Agents", ch["agent_cards_html"]),
        charts(figures[6:8]),
        _section(
            "Diagnostics",
            dg["diag_summary_html"] + dg["diag_rootcause_html"],
        ),
        charts(figures[8:11]),
        _section("Hotspots", _mixed_md_to_html(ov["hotspots_text"])),
        _section("Per-message metrics", _mixed_md_to_html(ov["per_message_text"])),
        _section(
            "Patterns",
            "<h3>Recurring tool sequences</h3>" + pat_tool + "<h3>Failure patterns</h3>" + pat_fail + antipattern,
        ),
    ]

    body = "\n".join(s for s in sections if s)
    return (
        f"<!DOCTYPE html>\n<html lang='en' class='{theme_class}'>\n<head>\n"
        f"<meta charset='utf-8'>\n<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<meta name='color-scheme' content='{'dark' if dark else 'light'}'>\n"
        f"<meta name='{_SOURCE_META}' content='{html.escape(_source_sha(session))}'>\n"
        f"<title>{html.escape(title)}</title>\n"
        f"<style>\n{_embedded_css()}\n</style>\n"
        "</head>\n<body>\n"
        f"<article class='tv-report'>\n{body}\n</article>\n"
        "</body>\n</html>\n"
    )


_SOURCE_META = "trajviz-source-sha256"
_SOURCE_META_RE = re.compile(r"<meta name='" + _SOURCE_META + r"' content='([^']*)'>")


def _source_sha(session: LoadedSession) -> str:
    raw = session.raw if isinstance(session.raw, dict) else {}
    sha = raw.get("_source_sha256")
    return sha if isinstance(sha, str) else ""


def _report_source(path: str) -> str | None:
    """The source sha a report on disk records, or None if it records none."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return None
    match = _SOURCE_META_RE.search(head)
    return match.group(1) if match else None


def _free_report_path(directory: str, name: str, sha: str) -> str:
    """A path for *name* in *directory* that will not clobber another trajectory's report.

    The name comes from the trajectory's stem, which two different runs can
    share (the same task from two harnesses). A file already holding a report
    of OTHER bytes is left alone and a numbered sibling is used; a report of the
    same bytes, or one that predates the stamp, is replaced as before.
    """
    stem, ext = os.path.splitext(name)
    candidate = os.path.join(directory, name)
    n = 2
    while os.path.exists(candidate):
        recorded = _report_source(candidate)
        if recorded is None or recorded == sha:
            return candidate
        candidate = os.path.join(directory, f"{stem}-{n}{ext}")
        n += 1
    return candidate


def write_report_file(
    source: LoadedSession | dict,
    dest: str | None = None,
    *,
    file_path: str | None = None,
    dark: bool = False,
    inline_plotly: bool = False,
) -> str:
    """Write the report to *dest* (file or directory) and return the path.

    An explicit file *dest* is written as asked. When the name is derived (a
    directory *dest*), another trajectory's report is never overwritten.
    """
    session = _as_session(source, file_path=file_path)
    name = _report_basename(session.path)
    if dest is None:
        dest = os.path.join(tempfile.mkdtemp(prefix="trajviz-report-"), name)
    elif os.path.isdir(dest):
        dest = _free_report_path(dest, name, _source_sha(session))
    html_out = build_report_html(session, dark=dark, inline_plotly=inline_plotly)
    parent = os.path.dirname(os.path.abspath(dest))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        handle.write(html_out)
    return dest


def _as_session(source: LoadedSession | dict, *, file_path: str | None = None) -> LoadedSession:
    if isinstance(source, LoadedSession):
        return source
    raw = source
    if not isinstance(raw, dict) or not raw:
        raise ReportError("Load a trajectory first.")
    if raw.get("_error"):
        raise ReportError(str(raw["_error"]))
    path = file_path or raw.get("_source_path") or "trajectory"
    return build_loaded_session(str(path), raw)


def report_from_path(
    path: str, dest: str | None = None, *, dark: bool = False, inline_plotly: bool = False,
) -> str:
    """Load *path* and write a report. Raises ``ReportError`` on ingest failure."""
    result = load_session(path)
    if isinstance(result, LoadError):
        raise ReportError(result.message)
    return write_report_file(result, dest, dark=dark, inline_plotly=inline_plotly)


def _report_basename(file_path: str) -> str:
    stem = os.path.splitext(os.path.basename(file_path or "trajectory"))[0] or "trajectory"
    safe = re.sub(r"[^\w.-]+", "-", stem).strip("-.") or "trajectory"
    return f"{safe}-trajviz-report.html"


def _embedded_css() -> str:
    app = _STYLE_TAG.sub("", APP_CSS)
    app = _HTML_COMMENT.sub("", app)
    layout = """
html.tv-theme-light { color-scheme: light; }
html.tv-theme-dark { color-scheme: dark; }
body { margin: 0; background: var(--ov-bg, #f6f8fc); color: var(--ov-text, #0f172a); }
.tv-report { max-width: 1100px; margin: 0 auto; padding: 24px 20px 72px; }
.tv-report-kicker { font-size: 12px; color: var(--ov-muted); letter-spacing: 0.04em; text-transform: uppercase; }
.tv-report h1 { font-size: 22px; margin: 4px 0 8px; letter-spacing: -0.02em; }
.tv-meta { font-size: 13px; color: var(--ov-muted); margin-bottom: 20px; }
.tv-note { margin: 8px 0 16px; padding: 8px 12px; background: var(--ov-anomaly-bg, #fef3c7);
  border-left: 3px solid var(--ov-warn, #b45309); color: var(--ov-anomaly-text, #92400e); font-size: 13px; }
.tv-section { margin: 28px 0; }
.tv-section h2 { font-size: 16px; margin: 0 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--ov-border, #dce3ef); }
.tv-section h3 { font-size: 14px; margin: 16px 0 8px; }
.tv-chart { margin: 8px 0 20px; }
.tv-md table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0 16px; }
.tv-md th, .tv-md td { border-bottom: 1px solid var(--ov-border, #dce3ef); padding: 6px 10px; text-align: left; }
.tv-md th { background: var(--ov-table-header-bg, #f1f5f9); font-size: 11px; text-transform: uppercase; }
.tv-md p { margin: 0 0 8px; }
.tv-md code { font-size: 12px; background: var(--ov-code-bg, #eef3ff); padding: 1px 4px; border-radius: 3px; }
@media print {
  .tv-chart { break-inside: avoid; }
}
"""
    return layout + "\n" + app


def _header_html(title: str, basename: str, fmt_label: str, generated: str, warnings_html: str) -> str:
    note_html = f"<div class='tv-note'>{warnings_html}</div>" if (warnings_html or "").strip() else ""
    return (
        "<header>"
        "<div class='tv-report-kicker'>TrajViz HTML report</div>"
        f"<h1>{html.escape(basename)}</h1>"
        f"<p class='tv-meta'>{html.escape(fmt_label)} &middot; generated {html.escape(generated)}"
        " &middot; Overview, charts, and Patterns (not a live Gradio session)</p>"
        f"{note_html}"
        "</header>"
    )


def _section(title: str, inner: str) -> str:
    if not (inner or "").strip():
        return ""
    return (
        f"<section class='tv-section' id='{html.escape(_slug(title))}'><h2>{html.escape(title)}</h2>{inner}</section>"
    )


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _figure_is_empty(fig: Any) -> bool:
    if fig is None or not isinstance(fig, go.Figure):
        return True
    # `_empty_figure(height, message)` carries its explanation as an annotation
    # on a figure with no traces. Testing traces alone dropped every "no
    # tool-call timing recorded in this trajectory" panel from the export — the
    # reader then could not tell a missing capability from a missing section.
    # Only a figure with neither traces nor annotations is a true placeholder.
    return len(fig.data) == 0 and not (fig.layout.annotations or ())


def _charts_html(items: list[tuple[str, go.Figure]], include_js: str | bool) -> tuple[str, str | bool]:
    parts: list[str] = []
    js = include_js
    for title, fig in items:
        if _figure_is_empty(fig):
            continue
        div = fig.to_html(
            full_html=False,
            include_plotlyjs=js,
            config={"displaylogo": False, "responsive": True},
        )
        js = False
        parts.append(
            f"<section class='tv-section tv-chart' id='{html.escape(_slug(title))}'>"
            f"<h2>{html.escape(title)}</h2>{div}</section>"
        )
    return "\n".join(parts), js


def _mixed_md_to_html(text: str) -> str:
    """Render dashboard markdown that already embeds HTML metric chips."""
    if not text or not str(text).strip():
        return ""
    lines = str(text).splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        raw_line = lines[i]
        stripped = raw_line.strip()
        heading = _HEADING.match(stripped)
        if heading:
            tag = "h2" if heading.group(1) == "##" else "h3"
            out.append(f"<{tag}>{_inline(heading.group(2))}</{tag}>")
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < n and _MD_TABLE_SEP.match(lines[i + 1].strip()):
            block = [stripped, lines[i + 1].strip()]
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            out.append(_md_table_to_html(block))
            continue
        if stripped.startswith("<"):
            # Pass through only TrajViz's own chip-grid markup. A trajectory
            # string that reaches here (a newline in an agent id used to end the
            # table block and hand the next line straight to the browser) is
            # escaped like any other paragraph.
            if _PASSTHROUGH_BLOCK.match(stripped) and not _LIVE_MARKUP.search(stripped):
                out.append(raw_line)
            else:
                out.append(f"<p>{_inline(stripped)}</p>")
            i += 1
            continue
        if not stripped:
            i += 1
            continue
        out.append(f"<p>{_inline(stripped)}</p>")
        i += 1
    return "<div class='tv-md'>" + "\n".join(out) + "</div>"


def _md_table_to_html(block: list[str]) -> str:
    def cells(line: str) -> list[str]:
        parts = _MD_CELL_SEP.split(line.strip())
        # Drop the empty parts produced by the row's leading/trailing delimiter.
        if parts and not parts[0].strip():
            parts = parts[1:]
        if parts and not parts[-1].strip():
            parts = parts[:-1]
        return [c.strip().replace("\\|", "|") for c in parts]

    header = cells(block[0])
    body_rows = [cells(row) for row in block[2:]]
    head = "".join(f"<th>{_inline(c)}</th>" for c in header)
    rows = []
    for row in body_rows:
        padded = row + [""] * (len(header) - len(row))
        rows.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in padded[: len(header)]) + "</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    escaped = _CODE.sub(r"<code>\1</code>", escaped)
    if escaped.startswith("*") and escaped.endswith("*") and len(escaped) > 2:
        escaped = f"<em>{escaped[1:-1]}</em>"
    return escaped
