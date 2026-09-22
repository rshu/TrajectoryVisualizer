"""HTML/code rendering, card styles, and workflow HTML generation."""

import html
import json
import re
from datetime import datetime, UTC

from pygments import highlight as _pygments_highlight
from pygments.formatters import HtmlFormatter as _HtmlFormatter
from pygments.lexers import get_lexer_by_name as _get_lexer, TextLexer as _TextLexer

from .charts import bind_timeline_agents
from .context_usage import PRESSURE_MAIN_AGENT, pressure_agent_key
from .metrics import tool_call_duration_ms
from .palette import AGENT_COLORS, AGENT_CSS_COLORS
from .parser import _optional_token_count
from .step_errors import step_error_kind
from .styles import WORKFLOW_CSS
from .workflow_role import workflow_role


_ROLE_COLORS = {
    "user": ("var(--wf-bg-user)", "var(--wf-border-user)", "User"),
    "assistant": ("var(--wf-bg-assistant)", "var(--wf-border-assistant)", "Assistant"),
    "task": ("var(--wf-bg-reasoning)", "var(--wf-border-reasoning)", "Task"),
    "system": ("var(--wf-bg-default)", "var(--wf-border-default)", "System"),
    "compaction": ("var(--wf-bg-default)", "var(--wf-border-default)", "Compaction"),
}

_ROLE_BADGE_STYLES = {
    "user": "background:var(--wf-border-user);color:white;",
    "assistant": "background:var(--wf-border-assistant);color:white;",
    "task": "background:var(--wf-border-reasoning);color:white;",
    "system": "background:var(--wf-border-default);color:white;",
    "compaction": "background:var(--wf-border-default);color:white;",
    "tool": "background:var(--wf-border-reasoning);color:white;",
}


def workflow_role_label(step: dict) -> str:
    """Human-readable Workflow role badge for *step*."""
    role = workflow_role(step)
    colors = _ROLE_COLORS.get(role)
    if colors:
        return colors[2]
    return role.title() if role else ""


def _card_style(step: dict) -> tuple[str, str, str, str | None]:
    """Return (bg_color, border_color, label, error_kind) for a step card.

    Colors are CSS variable references so they adapt to the active theme.
    ``error_kind`` is ``\"system\"``, ``\"tool\"``, or ``None``.
    """
    stored = step.get("role", "")
    stored_s = stored if isinstance(stored, str) else str(stored or "")
    err_kind = step_error_kind(step)
    if err_kind == "tool":
        return "var(--wf-bg-error)", "var(--wf-border-error)", "Tool Error", err_kind
    if err_kind == "system":
        return (
            "var(--wf-bg-system-error)",
            "var(--wf-border-system-error)",
            "System Error",
            err_kind,
        )
    if step.get("finish") == "stop" or step.get("finish") == "end_turn":
        return "var(--wf-bg-final)", "var(--wf-border-final)", "Final", None
    if step["tool_call_count"] > 0:
        return "var(--wf-bg-tool)", "var(--wf-border-tool)", "Tool Calls", None
    if step["has_reasoning"] and stored_s == "assistant":
        return "var(--wf-bg-reasoning)", "var(--wf-border-reasoning)", "Reasoning", None
    role = workflow_role(step)
    bg, border, label = _ROLE_COLORS.get(
        role, ("var(--wf-bg-default)", "var(--wf-border-default)", role.title()),
    )
    return bg, border, label, None


_CODE_FENCE_RE = re.compile(
    r"```(\w*)\n(.*?)```",
    re.DOTALL,
)

# Matches runs of 3+ backticks that were NOT consumed by _CODE_FENCE_RE.
# These are unbalanced/orphaned fences (e.g. from truncated model output)
# that would otherwise open a code block in markdown-it and swallow content.
_ORPHAN_FENCE_RE = re.compile(r"`{3,}")


_pygments_formatter = _HtmlFormatter(nowrap=True, style="github-dark")


def _highlight_code(code: str, lang: str) -> str:
    """Syntax-highlight a code string using Pygments."""
    try:
        lexer = _get_lexer(lang, stripall=True)
    except Exception:
        lexer = _TextLexer(stripall=True)
    return _pygments_highlight(code, lexer, _pygments_formatter)


def _neutralize_orphan_fences(text: str) -> str:
    """Replace runs of 3+ backticks with single backtick-escaped equivalents.

    Turns e.g. ````` into `` `​`​` `` (backticks separated by zero-width
    spaces) so they render visibly but never open a code fence in markdown-it.
    Only call this on segments already known to be *outside* balanced fences.
    """
    return _ORPHAN_FENCE_RE.sub(
        lambda m: "\u200b".join("`" for _ in range(len(m.group()))),
        text,
    )


def _md_to_html_preview(text: str) -> str:
    """Convert text with markdown fenced code blocks to HTML.

    Code fences (```lang ... ```) become syntax-highlighted <pre><code> blocks.
    Everything else is html-escaped.  Orphan backtick fences are neutralized.
    """
    parts: list[str] = []
    last_end = 0
    for m in _CODE_FENCE_RE.finditer(text):
        before = text[last_end:m.start()]
        if before:
            parts.append(_neutralize_orphan_fences(html.escape(before)))
        lang = m.group(1) or "text"
        code = m.group(2).rstrip("\n")
        highlighted = _highlight_code(code, lang)
        lang_escaped = html.escape(lang)
        parts.append(
            f'<div class="wf-code-block">'
            f'<span class="wf-code-lang">{lang_escaped}</span>'
            f'<pre class="wf-code-hl"><code>{highlighted}</code></pre>'
            f'</div>'
        )
        last_end = m.end()
    tail = text[last_end:]
    if tail:
        parts.append(_neutralize_orphan_fences(html.escape(tail)))
    return "".join(parts) if parts else html.escape(text)


_ROLE_FILTER_CHIPS = ["Assistant", "User"]
_FEATURE_FILTER_CHIPS = ["Tool Calls", "Errors", "Reasoning"]
_ALL_FEATURE_FILTER = "All"
AGENT_FILTER_PREFIX = "agent:"
AGENT_ALL_FILTER = "agent:All"
MAIN_AGENT_FILTER = f"{AGENT_FILTER_PREFIX}{PRESSURE_MAIN_AGENT}"


def agent_filter_token(agent_id: str) -> str:
    """CSV / chip token for a timeline agent id (empty id → main)."""
    return f"{AGENT_FILTER_PREFIX}{pressure_agent_key(agent_id)}"


def agent_id_from_filter_token(token: str) -> str | None:
    """Parse an ``agent:<id>`` filter token to a timeline id.

    Returns None for non-agent tokens and for ``agent:All``.
    """
    if not token.startswith(AGENT_FILTER_PREFIX):
        return None
    rest = token[len(AGENT_FILTER_PREFIX):]
    if rest == "All":
        return None
    if rest == PRESSURE_MAIN_AGENT:
        return ""
    return rest


def workflow_agent_chip_options(steps: list[dict]) -> list[tuple[str, str, int]]:
    """``(token, label, color_index)`` for multi-agent Workflow chips, else []."""
    color_map, labels, _agent_id_of = bind_timeline_agents(steps)
    if len(color_map) <= 1:
        return []
    return [
        (agent_filter_token(aid), labels[aid], idx)
        for aid, idx in color_map.items()
    ]


def _render_one_agent_card(a: dict, agent_hex: str) -> str:
    """Build one agent KPI card. Shared by single- and multi-agent paths."""
    label = html.escape(a["label"])
    full_id = html.escape(a.get("agent_id", ""))

    # Spawning link (only meaningful for sub-agents)
    spawn_html = ""
    if a.get("spawned_by_step") is not None:
        sidx = a["spawned_by_step"]
        spawn_html = (
            f"<div class='agent-card-spawn'>"
            f"<span class='insight-step-link' onclick=\"{_diag_jump_onclick(sidx)}\">"
            f"Spawned at step #{sidx}</span>"
            f"</div>"
        )

    _cache_display = (
        "N/A" if a["cache_read_tokens"] == 0 and a["total_tokens"] > 0
        else f"{a['cache_efficiency_pct']:.1f}%"
    )
    return (
        f"<div class='agent-card' title='{full_id}'"
        f" style='border-left:4px solid {agent_hex};'>"
        f"<div class='agent-card-header'>"
        f"<span class='agent-card-label' style='color:{agent_hex};'>{label}</span>"
        f"<span class='agent-card-steps'>{a['step_count']} steps</span>"
        f"</div>"
        f"{spawn_html}"
        f"<div class='agent-card-grid'>"
        f"<div><span class='agent-kpi-val'>{a['total_tokens']:,}</span><span class='agent-kpi-label'>Tokens</span></div>"
        f"<div><span class='agent-kpi-val'>{a['total_duration_s']:.1f}s</span><span class='agent-kpi-label'>Duration</span></div>"
        f"<div><span class='agent-kpi-val'>{a['tool_call_count']}</span><span class='agent-kpi-label'>Tool Calls</span></div>"
        f"<div><span class='agent-kpi-val'>{a['error_count']}</span><span class='agent-kpi-label'>Errors</span></div>"
        f"<div><span class='agent-kpi-val'>{_cache_display}</span><span class='agent-kpi-label'>Cache %</span></div>"
        f"<div><span class='agent-kpi-val'>{a['tokens_per_second']:,.0f}</span><span class='agent-kpi-label'>Tok/s</span></div>"
        f"</div>"
        f"</div>"
    )


def render_agent_summary_cards(agent_summaries: list[dict]) -> str:
    """Render per-agent summary cards as styled HTML.

    For single-agent sessions, render a single card so the user still sees the
    agent's stats in this section (rather than an empty placeholder). The card
    template is identical to the multi-agent case.
    """
    if not agent_summaries:
        return (
            "<div style='padding:2em;color:var(--ov-muted);text-align:center;font-size:14px;'>"
            "No agent activity recorded.</div>"
        )

    cards = [
        _render_one_agent_card(a, AGENT_COLORS[aidx % len(AGENT_COLORS)])
        for aidx, a in enumerate(agent_summaries)
    ]
    return "<div class='agent-cards-grid'>" + "".join(cards) + "</div>"


def render_filter_chips(
    active: list[str] | None = None,
    *,
    agent_options: list[tuple[str, str, int]] | None = None,
) -> str:
    """Render the Workflow filter chip panel.

    Roles are a required multi-select (OR within the group). Step features are
    also ORed, while ``All`` means that no feature predicate is applied. When
    *agent_options* is non-empty (multi-agent trajectories), a third Agent
    group mirrors feature semantics with ``agent:All`` / ``agent:…`` tokens.
    The delegated browser handler enforces these states; the backend ANDs the
    groups together.
    """
    agent_options = agent_options or []
    if active is None:
        active = [*_ROLE_FILTER_CHIPS, _ALL_FEATURE_FILTER]
        if agent_options:
            active = [*active, AGENT_ALL_FILTER]
    active_set = set(active)

    def _chip(
        *,
        data_filter: str,
        label: str,
        group: str,
        extra_class: str = "",
        style: str = "",
    ) -> str:
        classes = ["filter-chip"]
        if extra_class:
            classes.append(extra_class)
        is_active = data_filter in active_set
        if is_active:
            classes.append("chip-active")
        escaped_filter = html.escape(data_filter, quote=True)
        escaped_label = html.escape(label)
        style_attr = f" style='{html.escape(style, quote=True)}'" if style else ""
        pressed = "true" if is_active else "false"
        return (
            f"<button type='button' class='{' '.join(classes)}'"
            f" data-filter='{escaped_filter}' data-filter-group='{group}'"
            f" aria-pressed='{pressed}'{style_attr}>"
            f"{escaped_label}</button>"
        )

    def _group(key: str, title: str, hint: str, chips_html: str) -> str:
        return (
            f"<div class='filter-group' data-filter-group-container='{key}'>"
            f"<div class='filter-group-label'>{html.escape(title)}"
            f"<span>{html.escape(hint)}</span></div>"
            f"<div class='filter-options'>{chips_html}</div>"
            "</div>"
        )

    role_chips = "".join(
        _chip(data_filter=name, label=name, group="role")
        for name in _ROLE_FILTER_CHIPS
    )
    feature_chips = _chip(
        data_filter=_ALL_FEATURE_FILTER,
        label=_ALL_FEATURE_FILTER,
        group="feature",
        extra_class="filter-chip-all",
    ) + "".join(
        _chip(data_filter=name, label=name, group="feature")
        for name in _FEATURE_FILTER_CHIPS
    )

    groups = [
        _group("role", "Role", "select at least one", role_chips),
        _group("feature", "Step feature", "match any selected", feature_chips),
    ]

    summary = "Role: Assistant or User &middot; Step feature: All"
    reset_title = "Restore all roles and remove the step feature restriction"
    if agent_options:
        agent_chips = _chip(
            data_filter=AGENT_ALL_FILTER,
            label="All",
            group="agent",
            extra_class="filter-chip-all",
        )
        for token, label, color_idx in agent_options:
            hex_color = AGENT_COLORS[color_idx % len(AGENT_COLORS)]
            agent_chips += _chip(
                data_filter=token,
                label=label,
                group="agent",
                style=f"border-left:3px solid {hex_color};",
            )
        groups.append(_group("agent", "Agent", "match any selected", agent_chips))
        summary += " &middot; Agent: All"
        reset_title = (
            "Restore all roles and remove step feature / agent restrictions"
        )

    return (
        "<div class='filter-panel' id='wf-filter-bar'>"
        + "".join(groups)
        + "</div>"
        "<div class='filter-summary' id='wf-filter-summary'>"
        f"<span id='wf-filter-query'>{summary}</span>"
        "<button type='button' class='reset-filters' data-wf-action='reset-filters'"
        f" title='{html.escape(reset_title, quote=True)}'>"
        "Reset filters</button>"
        "</div>"
    )


def render_toc_sidebar(steps: list[dict], collapsed: bool = False) -> str:
    """Generate an HTML ``<nav>`` listing step numbers and role badges for a TOC sidebar.

    ``collapsed`` re-applies the user's ``toc-hidden`` state so re-renders
    (filter or search changes) don't pop a hidden sidebar back open.
    """
    if not steps:
        return ""
    items: list[str] = []
    for step in steps:
        idx = step.get("index", 0)
        role = workflow_role(step)
        role_style = _ROLE_BADGE_STYLES.get(role, "background:var(--wf-border-default);color:white;")
        onclick = (
            f"(function(){{"
            f"var c=document.getElementById('wf-card-{idx}');"
            f"if(!c)return;"
            f"if(typeof window.tvFocusWorkflowCard==='function'){{window.tvFocusWorkflowCard(c);}}"
            f"else{{c.scrollIntoView({{behavior:'smooth',block:'center'}});c.click();}}"
            f"}})()"
        )
        toc_indent = "padding-left:16px;" if step.get("is_sub_agent") else ""
        items.append(
            f"<div class='toc-entry' onclick=\"{onclick}\" data-step-idx='{idx}' style='{toc_indent}'>"
            f"<span class='toc-num'>#{idx}</span>"
            f"<span class='wf-badge' style='{role_style};font-size:11px;padding:2px 6px;'>"
            f"{html.escape(workflow_role_label(step))}</span>"
            f"</div>"
        )
    nav_class = "wf-toc-sidebar toc-hidden" if collapsed else "wf-toc-sidebar"
    return (
        f"<nav class='{nav_class}' id='wf-toc-sidebar'>"
        "<div class='toc-title'>Steps</div>"
        + "".join(items)
        + "</nav>"
    )


def render_workflow_html(steps: list[dict]) -> str:
    """Render vertical card flow as self-contained HTML with scroll container."""
    if not steps:
        return "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>No steps to display.</div>"

    css = WORKFLOW_CSS
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    has_agents = len(color_map) > 1

    cards_html = []
    for i, step in enumerate(steps):
        bg, border, label, err_kind = _card_style(step)
        dur = f"{step['duration']}s" if step["duration"] is not None else "\u2014"
        tok = f"{step['tokens']['total']:,}"
        preview = _md_to_html_preview(step["text_preview"]) if step["text_preview"] else "\u2014"

        part_icons = []
        for p in step["parts"]:
            t = p.get("type", "")
            if t == "reasoning":
                part_icons.append("thought")
            elif t == "tool_call":
                part_icons.append("tool")
            elif t == "text":
                part_icons.append("text")
        icon_str = " \u00b7 ".join(sorted(set(part_icons))) if part_icons else ""

        tc_info = f'<span>{step["tool_call_count"]} tool(s)</span>' if step["tool_call_count"] else ''
        if err_kind == "tool":
            err_info = (
                f'<span style="color:var(--wf-border-error)">'
                f'{step["error_count"]} tool err</span>'
            )
        elif err_kind == "system":
            err_info = (
                '<span style="color:var(--wf-border-system-error)">system err</span>'
            )
        else:
            err_info = ''

        # Agent badge with per-agent color (same identity as swimlane / tool chart)
        agent_badge = ''
        agent_left_border = ""
        if has_agents:
            aid = agent_id_of(step)
            aidx = color_map.get(aid, 0)
            agent_bg, agent_border = AGENT_CSS_COLORS[aidx % len(AGENT_CSS_COLORS)]
            agent_hex = AGENT_COLORS[aidx % len(AGENT_COLORS)]
            agent_left_border = f"border-left:4px solid {agent_hex};"
            agent_label = labels.get(aid) or (aid or "main")
            agent_badge = (
                f'<span class="wf-badge" style="background:{agent_bg};color:{agent_border};'
                f'border:1px solid {agent_border};font-size:9px;">{html.escape(agent_label)}</span>'
            )

        role = workflow_role(step)
        role_style = _ROLE_BADGE_STYLES.get(role, "background:var(--wf-border-default);color:white;")
        role_label = workflow_role_label(step)

        orig_idx = step.get("index", i)

        sub_agent_badge = ""
        sub_indent = ""
        if step.get("is_sub_agent"):
            sub_agent_badge = '<span class="wf-badge" style="background:var(--wf-border-reasoning);color:white;font-size:10px;">sub-agent</span>'
            sub_indent = "margin-left:24px;"

        card = f"""
        <div class="wf-card" id="wf-card-{orig_idx}" data-step-idx="{orig_idx}"
             style="background:{bg};border-color:{border};{agent_left_border}{sub_indent}">
            <div class="wf-header">
                <span class="wf-badge" style="background:{border};color:white;">#{orig_idx}</span>
                <span class="wf-badge" style="{role_style}">{html.escape(role_label)}</span>
                {"" if label == role_label else f'<span class="wf-badge" style="background:transparent;color:{border};border:1px solid {border};">{html.escape(label)}</span>'}
                {sub_agent_badge}
                {agent_badge}
                <span class="wf-icons">{icon_str}</span>
            </div>
            <div class="wf-meta">
                <span>{dur}</span>
                <span>{tok} tok</span>
                {tc_info}{err_info}
            </div>
            <div class="wf-preview">{preview}</div>
        </div>
        """
        cards_html.append(card)
        if i < len(steps) - 1:
            cards_html.append('<div class="wf-connector"></div>')

    return (
        css
        + '<div class="wf-scroll"><div class="wf-container">'
        + "\n".join(cards_html)
        + '</div></div>'
    )


def _fmt_timestamp(ms):
    """Convert epoch-milliseconds to readable ``YYYY-MM-DD HH:MM:SS`` (UTC)."""
    if not isinstance(ms, (int, float)):
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def _format_step_header(step: dict) -> str:
    """Build the styled HTML header banner and metadata table for a step detail panel."""
    bg, border, label, _err_kind = _card_style(step)
    role = workflow_role(step)
    role_style = _ROLE_BADGE_STYLES.get(role, "background:var(--wf-border-default);color:white;")

    rows: list[tuple[str, str]] = [("Role", workflow_role_label(step))]
    _optional = [
        ("agent", "Agent"), ("mode", "Mode"), ("model_id", "Model"),
        ("provider_id", "Provider"),
    ]
    for key, field in _optional:
        if step.get(key):
            rows.append((field, step[key]))
    if step.get("duration") is not None:
        rows.append(("Duration", f"{step['duration']}s"))

    created_str = _fmt_timestamp(step.get("time_created_ms"))
    if created_str:
        rows.append(("Created", created_str))
    completed_str = _fmt_timestamp(step.get("time_completed_ms"))
    if completed_str:
        rows.append(("Completed", completed_str))

    if step.get("finish"):
        rows.append(("Finish", step["finish"]))
    if step["tool_call_count"] > 0:
        rows.append(("Tool calls", str(step["tool_call_count"])))
    if step["error_count"] > 0:
        rows.append(("Errors", str(step["error_count"])))

    if step.get("is_sub_agent"):
        rows.append(("Sub-agent", "Yes"))
    if step.get("session_depth") is not None:
        rows.append(("Session depth", str(step["session_depth"])))
    if step.get("session_title"):
        rows.append(("Session title", step["session_title"]))
    if step.get("parent_session_id"):
        rows.append(("Parent session", step["parent_session_id"]))

    _id_fields = [
        ("id", "ID"), ("parent_id", "Parent ID"), ("session_id", "Session"),
        ("cwd", "CWD"), ("message_id", "Message ID"),
    ]
    for key, field in _id_fields:
        if step.get(key):
            rows.append((field, step[key]))
    if step.get("root") and step.get("root") != step.get("cwd"):
        rows.append(("Root", step["root"]))

    # Banner
    banner = (
        f"<div class='dp-header' style='background:{border};'>"
        f"<span class='dp-badge'>#{step['index']}</span>"
        f"<span class='dp-badge' style='{role_style}'>{html.escape(workflow_role_label(step))}</span>"
        f"Step {step['index']} &mdash; {html.escape(label)}"
        f"</div>"
    )

    # Metadata table
    tr_parts = []
    for field, value in rows:
        str_value = str(value)
        escaped_val = html.escape(str_value)
        # Wrap code-like values
        if isinstance(value, str) and any(c in value for c in ("/", ".", "-")) and len(value) > 8:
            escaped_val = f"<code>{escaped_val}</code>"
        tr_parts.append(f"<tr><td>{html.escape(field)}</td><td>{escaped_val}</td></tr>")

    table = f"<table class='dp-meta-table'>{''.join(tr_parts)}</table>"
    return banner + table


def _format_tool_call_detail(p: dict) -> str:
    """Render a single tool_call part as a styled HTML block."""
    inp = p.get("input", {})
    out = p.get("output", "")
    inp_str = json.dumps(inp, indent=2, ensure_ascii=False) if isinstance(inp, dict) else str(inp)
    if not isinstance(out, str):
        if isinstance(out, list) and all(isinstance(b, dict) and b.get("type") == "text" for b in out):
            # unwrap content-block lists to readable text
            out = "\n".join(str(b.get("text", "")) for b in out)
        else:
            try:
                out = json.dumps(out, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                out = str(out)
    if len(out) > 2000:
        out = out[:2000] + "\n... (truncated)"

    tc_dur = ""
    dur_ms = tool_call_duration_ms(p)
    if dur_ms is not None:
        tc_dur = f" &mdash; {round(dur_ms / 1000, 2)}s"

    meta_parts: list[str] = []
    tool_id = p.get("tool_id", "")
    if tool_id:
        meta_parts.append(f"<code>{html.escape(tool_id)}</code>")
    tc_meta = p.get("metadata") or {}
    handled = {"output", "input", "preview"}
    if isinstance(tc_meta, dict):
        if tc_meta.get("sessionId"):
            sid = str(tc_meta["sessionId"])
            display = f"{sid[:16]}\u2026" if len(sid) > 16 else sid
            meta_parts.append(f"Session: <code>{html.escape(display)}</code>")
            handled.add("sessionId")
        meta_model = tc_meta.get("model")
        if isinstance(meta_model, dict):
            if meta_model.get("modelID"):
                meta_parts.append(f"Model: <code>{html.escape(str(meta_model['modelID']))}</code>")
            if meta_model.get("providerID"):
                meta_parts.append(f"Provider: <code>{html.escape(str(meta_model['providerID']))}</code>")
            handled.add("model")
        elif meta_model:
            meta_parts.append(f"Model: <code>{html.escape(str(meta_model))}</code>")
            handled.add("model")
        if tc_meta.get("truncated"):
            meta_parts.append("truncated")
        handled.add("truncated")
        for mk, mv in tc_meta.items():
            if mk in handled or mv is None or mv == "" or mv == {} or mv == []:
                continue
            if isinstance(mv, (list, dict)):
                continue
            if isinstance(mv, str) and len(mv) > 60:
                mv = mv[:57] + "..."
            meta_parts.append(f"{html.escape(mk)}: <code>{html.escape(str(mv))}</code>")
    if isinstance(inp, dict) and inp.get("subagent_type"):
        meta_parts.append(f"Subagent: <code>{html.escape(str(inp['subagent_type']))}</code>")

    meta_line = f"<div class='dp-tool-meta'>{' &middot; '.join(meta_parts)}</div>" if meta_parts else ""

    has_error = p.get("error") or p.get("status") == "error"
    section_cls = "dp-section dp-section-tool-error" if has_error else "dp-section dp-section-tool"

    tool_name = html.escape(p.get("tool_name", "?"))
    status = html.escape(p.get("status", "?"))
    # Build a meaningful title: prefer explicit title, fall back to tool_name + key arg
    raw_title = p.get("title") or ""
    if not raw_title and isinstance(inp, dict):
        # Use first meaningful arg as title hint
        for k in ("command", "file_path", "pattern", "description", "prompt"):
            v = inp.get(k, "")
            if v:
                raw_title = str(v)[:80]
                break
    title = html.escape(raw_title) if raw_title else tool_name

    # Input/Output details
    inp_detail = (
        f"<details class='dp-details'><summary>Input</summary>"
        f"<div class='dp-details-body'><pre>{html.escape(inp_str)}</pre></div>"
        f"</details>"
    )
    out_detail = ""
    if out:
        out_detail = (
            f"<details class='dp-details'><summary>Output</summary>"
            f"<div class='dp-details-body'><pre>{html.escape(str(out))}</pre></div>"
            f"</details>"
        )

    error_detail = ""
    tc_error = p.get("error")
    if tc_error:
        err_str = tc_error if isinstance(tc_error, str) else json.dumps(tc_error, indent=2, ensure_ascii=False)
        error_detail = (
            f"<details class='dp-details' open><summary>Error</summary>"
            f"<div class='dp-details-body'><pre>{html.escape(err_str)}</pre></div>"
            f"</details>"
        )

    return (
        f"<div class='{section_cls}'>"
        f"<div class='dp-section-title'>Tool</div>"
        f"<div class='dp-tool-header'><code>{tool_name}</code> &mdash; "
        f"<code>{status}</code>{tc_dur}</div>"
        f"<div style='font-weight:600;margin-bottom:4px;color:var(--ov-text);'>{title}</div>"
        f"{meta_line}"
        f"{inp_detail}{out_detail}{error_detail}"
        f"</div>"
    )


def _format_text_section(p: dict, section_type: str) -> str:
    """Render a text or reasoning part as a styled HTML section card."""
    cls = "dp-section-text" if section_type == "text" else "dp-section-reasoning"
    label = "Text" if section_type == "text" else "Reasoning"
    if section_type == "text" and p.get("synthetic"):
        label = "Synthetic context"
    text = p.get("text", "")
    # Use the code-fence-aware HTML renderer for content with code blocks
    rendered = _md_to_html_preview(text) if text else ""
    return (
        f"<div class='dp-section {cls}'>"
        f"<div class='dp-section-title'>{label}</div>"
        f"<div class='dp-content'>{rendered}</div>"
        f"</div>"
    )


def _render_diff_lines(diff_text: str) -> str:
    """Parse unified-diff text and emit HTML with line-level coloring.

    Added lines get class ``diff-add``, removed lines ``diff-del``,
    and context/header lines get ``diff-ctx``.
    """
    lines: list[str] = []
    for raw_line in diff_text.splitlines():
        escaped = html.escape(raw_line)
        if raw_line.startswith("+"):
            lines.append(f"<span class='diff-add'>{escaped}</span>")
        elif raw_line.startswith("-"):
            lines.append(f"<span class='diff-del'>{escaped}</span>")
        elif raw_line.startswith("@@"):
            lines.append(f"<span class='diff-hunk'>{escaped}</span>")
        else:
            lines.append(f"<span class='diff-ctx'>{escaped}</span>")
    return "\n".join(lines)


def _split_diff_by_file(diff_text: str) -> list[tuple[str, str]]:
    """Split a multi-file unified diff into (filepath, diff_chunk) pairs."""
    file_re = re.compile(r"^diff --git a/.+ b/(.+)$", re.MULTILINE)
    matches = list(file_re.finditer(diff_text))
    if not matches:
        return [("patch", diff_text)]
    result: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(diff_text)
        result.append((m.group(1), diff_text[m.start():end]))
    return result


def _format_patch_section(p: dict) -> str:
    """Render a patch part as a styled HTML section card."""
    patch_hash = p.get("hash", "")
    patch_files = p.get("files", [])
    patch_id = p.get("id", "")
    diff_content = p.get("diff_content", "")
    meta_parts = []
    if patch_hash:
        meta_parts.append(f"<code>{html.escape(patch_hash[:12])}</code>")
    if patch_id:
        meta_parts.append(f"<code>{html.escape(patch_id)}</code>")
    meta_line = f"<div class='dp-tool-meta'>{' &middot; '.join(meta_parts)}</div>" if meta_parts else ""
    files_html = ""
    if patch_files:
        items = "".join(f"<li><code>{html.escape(f)}</code></li>" for f in patch_files)
        files_html = f"<div style='margin-top:4px;font-size:12px;'><strong>Files:</strong><ul style='margin:2px 0 0 16px;'>{items}</ul></div>"

    diff_html = ""
    if diff_content:
        file_chunks = _split_diff_by_file(diff_content)
        diff_sections: list[str] = []
        for filepath, chunk in file_chunks:
            rendered = _render_diff_lines(chunk)
            diff_sections.append(
                f"<details class='dp-diff-file'>"
                f"<summary><code>{html.escape(filepath)}</code></summary>"
                f"<pre class='dp-diff-pre'><code>{rendered}</code></pre>"
                f"</details>"
            )
        diff_html = (
            f"<details class='dp-diff-toggle'>"
            f"<summary>Show diff</summary>"
            f"<div class='dp-diff-content'>{''.join(diff_sections)}</div>"
            f"</details>"
        )

    return (
        f"<div class='dp-section dp-section-patch'>"
        f"<div class='dp-section-title'>Patch</div>"
        f"{meta_line}{files_html}{diff_html}"
        f"</div>"
    )


_METRIC_TOKEN_FIELDS = (
    ("total", "Total Tokens"),
    ("input", "Input Tokens"),
    ("output", "Output Tokens"),
    ("cache_read", "Cache Read"),
    ("cache_write", "Cache Write"),
)


def _unavailable_metric_fields(step: dict) -> list[str]:
    """Return required Metrics fields that cannot be shown reliably.

    Only the token counts are required. Duration is structurally absent on
    some steps with genuine token data (e.g. the final step of a Claude Code
    trajectory, whose duration is computed from the next step's timestamp),
    so it and the derived Throughput / Cache Ratio rows degrade to ``n/a``
    per row instead of making the whole table unavailable.
    """
    missing = list(step.get("_metrics_unavailable_fields") or [])
    tokens = step.get("tokens")
    if not isinstance(tokens, dict):
        tokens = {}

    for key, label in _METRIC_TOKEN_FIELDS:
        if _optional_token_count(tokens, key) is None and label not in missing:
            missing.append(label)
    return missing


def _format_metrics_unavailable(missing: list[str]) -> str:
    description = "The trajectory does not provide complete per-step metrics."
    missing_text = ", ".join(html.escape(field) for field in missing)
    return (
        "<div class='dp-metrics-unavailable' role='status'>"
        "<div class='dp-metrics-unavailable-icon' aria-hidden='true'>i</div>"
        "<div>"
        "<div class='dp-metrics-unavailable-title'>Metrics unavailable</div>"
        f"<div class='dp-metrics-unavailable-description'>{description}</div>"
        f"<div class='dp-metrics-unavailable-fields'>Missing or unavailable: "
        f"{missing_text}</div>"
        "</div></div>"
    )


def _format_metrics_tab(step: dict) -> str:
    """Render the Metrics table, or one explicit unavailable state.

    A real ``0`` in any token count renders as ``0``; the table is replaced
    by the unavailable notice only when a required token count is genuinely
    missing. Reasoning tokens are optional (formats that never report them
    show ``n/a`` on that row). Duration and the derived rows show ``n/a``
    individually when they cannot be computed, so complete token data is
    never hidden by a missing timing.
    """
    missing = _unavailable_metric_fields(step)
    if missing:
        return _format_metrics_unavailable(missing)

    tokens = step["tokens"]
    duration = step.get("duration")
    duration_available = (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and duration >= 0
    )
    duration_text = f"{duration}s" if duration_available else "n/a"
    if duration_available and duration > 0:
        throughput_text = f"{tokens['total'] / duration:,.0f} tok/s"
    else:
        throughput_text = "n/a"
    if tokens["total"] > 0:
        cache_ratio_text = f"{tokens['cache_read'] / tokens['total'] * 100:.1f}%"
    else:
        cache_ratio_text = "n/a"
    reasoning = _optional_token_count(tokens, "reasoning")
    reasoning_text = f"{reasoning:,}" if reasoning is not None else "n/a"
    rows = [
        ("Total Tokens", f"{tokens['total']:,}"),
        ("Input Tokens", f"{tokens['input']:,}"),
        ("Output Tokens", f"{tokens['output']:,}"),
        ("Reasoning Tokens", reasoning_text),
        ("Cache Read", f"{tokens['cache_read']:,}"),
        ("Cache Write", f"{tokens['cache_write']:,}"),
        ("Duration", duration_text),
        ("Throughput", throughput_text),
        ("Cache Ratio", cache_ratio_text),
    ]

    tr_parts = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>"
        for k, v in rows
    )
    return f"<table class='dp-meta-table'>{tr_parts}</table>"


def _format_raw_tab(step: dict) -> str:
    """Render the Raw tab content for a step detail panel."""
    # Build a clean dict with key fields
    raw_data = {
        "index": step.get("index"),
        "role": step["role"],
        "tokens": step["tokens"],
        "duration": step.get("duration"),
        "finish": step.get("finish"),
        "tool_call_count": step["tool_call_count"],
        "error_count": step["error_count"],
        "has_reasoning": step["has_reasoning"],
    }
    for k in ("agent", "model_id", "mode", "provider_id"):
        if step.get(k):
            raw_data[k] = step[k]
    raw_str = json.dumps(raw_data, indent=2, ensure_ascii=False, default=str)
    return (
        f"<div class='dp-details-body'>"
        f"<pre>{html.escape(raw_str)}</pre>"
        f"</div>"
    )


def format_step_detail(step: dict) -> str:
    """Format detail panel for a selected step as a tabbed HTML string.

    Three tabs: Content (default), Metrics, Raw.
    """
    idx = step.get("index", 0)

    # Breadcrumb
    breadcrumb = (
        f"<div class='dp-breadcrumb'>"
        f"<span onclick=\"{_JS_GOTO_WORKFLOW}\">Workflow</span>"
        f" › Step {idx}"
        f"</div>"
    )

    header = _format_step_header(step)

    # Content tab
    content_parts = []
    for p in step["parts"]:
        ptype = p.get("type", "unknown")
        if ptype == "text":
            content_parts.append(_format_text_section(p, "text"))
        elif ptype == "reasoning":
            content_parts.append(_format_text_section(p, "reasoning"))
        elif ptype == "tool_call":
            content_parts.append(_format_tool_call_detail(p))
        elif ptype in ("step_start", "step_finish"):
            pass
        elif ptype == "snapshot":
            content_parts.append(
                "<div class='dp-section dp-section-snapshot'>"
                "<div class='dp-section-title'>Snapshot</div>"
                "<em>data omitted</em></div>"
            )
        elif ptype == "patch":
            content_parts.append(_format_patch_section(p))
        else:
            content_parts.append(
                f"<div class='dp-section'>"
                f"<div class='dp-section-title'>{html.escape(ptype)}</div>"
                f"</div>"
            )

    content_html = "\n".join(content_parts) if content_parts else "<em>No content</em>"
    metrics_html = _format_metrics_tab(step)
    raw_html = _format_raw_tab(step)

    return (
        f"<div class='dp-panel'>"
        f"{breadcrumb}"
        f"{header}"
        f"<div class='dp-tabs'>"
        f"<div class='dp-tab dp-tab-active' data-tab='content'>Content</div>"
        f"<div class='dp-tab' data-tab='metrics'>Metrics</div>"
        f"<div class='dp-tab' data-tab='raw'>Raw</div>"
        f"</div>"
        f"<div class='dp-tab-content dp-tab-visible' data-tab-content='content'>{content_html}</div>"
        f"<div class='dp-tab-content' data-tab-content='metrics'>{metrics_html}</div>"
        f"<div class='dp-tab-content' data-tab-content='raw'>{raw_html}</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Diagnostic renderers
# ---------------------------------------------------------------------------

def _diag_jump_onclick(idx: int) -> str:
    """JS onclick to switch to Workflow tab and scroll to a step card.

    Prefers ``window.tvGotoWorkflowStep`` (bound on app load for duration-chart
    clicks) so overview badges and Plotly jumps share one path.
    """
    return (
        f"(function(){{"
        f"if(typeof window.tvGotoWorkflowStep==='function'){{"
        f"window.tvGotoWorkflowStep({idx});return;}}"
        f"{_JS_GOTO_WORKFLOW}"
        f"setTimeout(function(){{"
        f"var c=document.getElementById('wf-card-{idx}');"
        f"if(c){{"
        f"if(typeof window.tvFocusWorkflowCard==='function'){{window.tvFocusWorkflowCard(c);}}"
        f"else{{c.scrollIntoView({{behavior:'smooth',block:'center'}});c.click();}}"
        f"}}"
        f"}},200);"
        f"}})()"
    )




def build_root_cause_html(clusters: list[dict]) -> str:
    """Render root-cause candidate summary panel."""
    if not clusters:
        return ""

    from .diagnostics import format_root_cause_summary
    summaries = format_root_cause_summary(clusters)

    items = []
    for i, (cluster, summary) in enumerate(zip(clusters, summaries, strict=False)):
        badge_class = "diag-rc-primary" if i == 0 else "diag-rc-secondary"
        first_step = cluster["first_step"]
        onclick = _diag_jump_onclick(first_step)
        items.append(
            f"<div class='diag-rc-item {badge_class}' onclick=\"{onclick}\" style='cursor:pointer;'>"
            f"<span class='diag-rc-rank'>#{i + 1}</span> "
            f"<span class='diag-rc-text'>{html.escape(summary)}</span>"
            f"</div>"
        )
    return "<div class='diag-rc-panel'>" + "".join(items) + "</div>"


# ---------------------------------------------------------------------------
# Score visualization renderers
# ---------------------------------------------------------------------------

# Label-based Workflow-tab jump: robust to tab insertions/reordering
# (positional tabs[i] indexing broke when the Attribution tab shifted the order).
_JS_GOTO_WORKFLOW = (
    "if(typeof window.tvClickMainTab==='function'){window.tvClickMainTab('Workflow');}"
    "else{var tabs=document.querySelectorAll('button[role=tab]');"
    "for(var ti=0;ti<tabs.length;ti++){if(tabs[ti].textContent.trim()==='Workflow'){tabs[ti].click();break;}}}"
)








# ---------------------------------------------------------------------------
# DECAF failure attribution (Attribution tab)
# ---------------------------------------------------------------------------

_STRENGTH_STYLE = {
    "deductive":      ("#059669", "Deductive"),       # set arithmetic, no model
    "associational":  ("#d97706", "Associational"),   # objective trajectory fact
    "model_inferred": ("#6366f1", "Model-inferred"),  # LLM judge
}
_ATTR_CAP_LABEL = {
    "requirement_understanding": "Requirement Understanding",
    "task_planning": "Task Planning",
    "code_localization": "Code Localization",
    "code_editing": "Code Editing",
    "code_verification": "Code Verification",
    "self_repair_loop": "Self-Repair Loop",
    "tool_use": "Tool Use",
}
_LINK_ICON = {"observation": "•", "inference": "↳", "conclusion": "⇒"}


def _attr_notice(reason: str, warn: bool = True) -> str:
    color = "var(--ov-warn)" if warn else "var(--ov-muted)"
    return (f"<div style='padding:1.5em;color:{color};text-align:center;font-size:14px;"
            f"line-height:1.5;'>{html.escape(reason)}</div>")


def _attr_scorecard_html(scorecard: list[dict], primary: dict | None) -> str:
    """Seven capability cards; blamed capabilities colored by evidence tier."""
    cards = []
    prim_cap = (primary or {}).get("capability")
    for s in scorecard:
        cap = s["capability"]
        label = _ATTR_CAP_LABEL.get(cap, cap)
        if s.get("blamed"):
            color, tier_label = _STRENGTH_STYLE.get(s.get("tier"), ("#6b7280", "—"))
            badge = tier_label
            score = f"{s.get('weight', 0.0):.2f}"
            metric = (s.get("top_error") or "").replace("_", " ") or "blamed"
        elif s.get("assessed"):
            color, badge, score, metric = "#059669", "clean", "—", "no blame"
        else:
            # NOT assessed (e.g. judge capabilities with no cached verdict) — this
            # is NOT the same as "clean"; do not imply the capability was checked.
            color, badge, score, metric = "#9ca3af", "n/a", "—", "not assessed"
        star = " ★" if cap == prim_cap else ""
        cards.append(
            f"<div class='score-dim-card'>"
            f"<div class='score-dim-header'>"
            f"<span class='score-dim-name'>{html.escape(label)}{star}</span>"
            f"<span class='score-dim-badge' style='background:{color};'>{html.escape(badge)}</span>"
            f"</div>"
            f"<div class='score-dim-score' style='color:{color};'>{score}</div>"
            f"<div class='score-dim-metric'>{html.escape(metric)}</div>"
            f"</div>"
        )
    return "<div class='score-dim-grid'>" + "".join(cards) + "</div>"


def _attr_fault_html(fault: dict) -> str:
    """One collapsible fault panel: claim + strength badge + evidence chain."""
    cap, et = fault.get("capability", ""), fault.get("error_type", "")
    chain = fault.get("evidence_chain") or {}
    strength = chain.get("strength", "")
    weight = float(fault.get("blame_weight") or 0.0)
    if weight == 0:
        # arbiter-refuted candidate: neutral badge, never a tier color that
        # could read as an attributed fault
        color, tier_label = "#9ca3af", "refuted"
    else:
        color, tier_label = _STRENGTH_STYLE.get(strength, ("#6b7280", strength or "—"))
    primary_tag = " ★ primary" if fault.get("is_primary") else ""
    label = f"{_ATTR_CAP_LABEL.get(cap, cap)} → {et}"

    links_html = []
    for link in chain.get("links", []):
        icon = _LINK_ICON.get(link.get("kind", ""), "•")
        src = link.get("source", "")
        stmt = html.escape(link.get("statement", ""))
        quotes = ""
        for q in link.get("quotes", []) or []:
            quotes += (f"<div class='attr-quote'>step {html.escape(str(q.get('step')))}: "
                       f"{html.escape(str(q.get('text', '')))}</div>")
        links_html.append(
            f"<div class='attr-link'><span class='attr-link-icon'>{icon}</span>"
            f"<span class='attr-link-body'>{stmt}"
            f"<span class='attr-link-src'>{html.escape(src)}</span>{quotes}</span></div>")

    audit = (fault.get("audit") or {}).get("verdict", "")
    audit_html = (f"<div class='attr-audit'>audit: {html.escape(audit)}</div>"
                  if audit else "")
    return (
        f"<details class='judge-panel'{' open' if fault.get('is_primary') else ''}>"
        f"<summary>"
        f"<span class='judge-badge' style='background:{color};'>{html.escape(tier_label)}</span>"
        f" {html.escape(label)} "
        f"<span style='color:var(--ov-muted);font-size:12px;'>"
        f"(blame {weight:.2f}{primary_tag})</span>"
        f"</summary>"
        f"<div class='judge-reasoning'>{''.join(links_html)}{audit_html}</div>"
        f"</details>"
    )


def _attr_arbiter_html(arb: dict | None) -> str:
    """Render the arbiter's verdict prominently (a refutation is a first-class
    result, not something to bury under a zero-weight fault card)."""
    if not arb or arb.get("support") is None:
        return ""
    applied = arb.get("applied", "")
    refuted = applied in ("demoted_to_conjunctive", "refuted_reassigned",
                          "refuted_unattributed")
    color = "#dc2626" if refuted else ("#059669" if applied == "corroborated"
                                       else "#9ca3af")
    verb = {"demoted_to_conjunctive": "refuted (demoted to conjunctive)",
            "refuted_reassigned": "refuted (blame reassigned to a deductive fault)",
            "refuted_unattributed": "refuted (no attributed cause remains)",
            "corroborated": "corroborated", "noted": "noted"}.get(applied, applied)
    tgt = arb.get("target") or {}
    rationale = arb.get("rationale") or ""
    return (
        f"<div class='attr-banner'>"
        f"<div class='attr-banner-head'>"
        f"<span class='judge-badge' style='background:{color};'>Arbiter</span> "
        f"{html.escape(verb)} — <code>{html.escape(str(tgt.get('error_type', '?')))}</code>"
        f" ({html.escape(str(arb.get('confidence', '?')))})</div>"
        + (f"<div class='attr-banner-sub'>{html.escape(rationale)}</div>" if rationale else "")
        + "</div>")


def build_attribution_html(data: dict) -> str:
    """Render the DECAF attribution result: primary banner + arbiter verdict +
    capability scorecard + per-fault evidence chains (attributed and refuted
    candidates separated). ``data`` is a plain dict (an AttributionResult
    serialized) so rendering stays decoupled from ``awe``."""
    if not data.get("available"):
        return _attr_notice(data.get("reason") or "Attribution unavailable.")

    status = data.get("blame_status") or "?"
    primary = data.get("primary")
    if primary:
        head = (f"Primary cause: <b>{html.escape(_ATTR_CAP_LABEL.get(primary['capability'], primary['capability']))}</b> "
                f"→ <code>{html.escape(primary['error_type'])}</code>")
    else:
        head = f"No single dominant cause (<code>{html.escape(status)}</code>)"
    n_assessed = sum(1 for s in data.get("scorecard", []) if s.get("assessed"))
    tier_note = (f"{n_assessed}-capability assessment"
                 + (" (incl. judge verdict for this case)" if data.get("used_judge")
                    else " (deductive/associational slice)"))
    banner = (
        f"<div class='attr-banner'>"
        f"<div class='attr-banner-head'>{head}</div>"
        f"<div class='attr-banner-sub'>blame status: <b>{html.escape(status)}</b>"
        f" &middot; {html.escape(tier_note)}</div></div>")
    notes_html = "".join(
        f"<div class='attr-banner-sub'>&#9888; {html.escape(n)}</div>"
        for n in data.get("notes") or [])

    faults = data.get("faults", [])
    attributed = [f for f in faults if float(f.get("blame_weight") or 0) > 0]
    refuted = [f for f in faults if float(f.get("blame_weight") or 0) == 0]
    out = (banner + notes_html
           + _attr_arbiter_html(data.get("arbiter"))
           + _attr_scorecard_html(data.get("scorecard", []), primary))
    out += ("<div style='margin-top:12px;font-weight:600;font-size:13px;'>"
            "Diagnosed faults (evidence chains)</div>")
    out += "".join(_attr_fault_html(f) for f in attributed) or _attr_notice(
        "No attributed faults — no cause survived (see the arbiter verdict "
        "above / coverage gap).", warn=False)
    if refuted:
        out += ("<div style='margin-top:12px;font-weight:600;font-size:13px;"
                "color:var(--ov-muted);'>Refuted candidates (zero blame — kept "
                "for audit)</div>")
        out += "".join(_attr_fault_html(f) for f in refuted)
    return out



# ---------------------------------------------------------------------------
# Sub-Agent Delegation Summary
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Anti-Pattern Summary
# ---------------------------------------------------------------------------

_STEP_CHIP_STYLE = (
    "display:inline-block;padding:1px 6px;margin:0 4px 2px 0;"
    "border-radius:8px;background:var(--ov-table-header-bg);"
    "font-size:11px;font-variant-numeric:tabular-nums;cursor:pointer;"
)


def _step_link_chip(idx: int) -> str:
    """Clickable ``#N`` chip that jumps to Workflow step *idx*."""
    n = int(idx)
    return (
        f"<span class='insight-step-link' style='{_STEP_CHIP_STYLE}' "
        f"onclick=\"{_diag_jump_onclick(n)}\">#{n}</span>"
    )


def _step_link_chips(indices: list[int], *, limit: int = 8) -> str:
    """Deduped step chips; shows ``+N more`` when truncated."""
    seen: list[int] = []
    for raw in indices:
        if raw is None:
            continue
        n = int(raw)
        if n not in seen:
            seen.append(n)
    if not seen:
        return ""
    shown = seen[:limit]
    chips = "".join(_step_link_chip(n) for n in shown)
    extra = len(seen) - len(shown)
    if extra > 0:
        chips += (
            f"<span style='font-size:11px;color:var(--ov-muted);'>"
            f"+{extra} more</span>"
        )
    return f"<div style='margin-top:4px;'>{chips}</div>"


def _indices_for_step_range(start: int | None, end: int | None, *, max_span: int = 5) -> list[int]:
    """Expand a streak range to chips; long ranges keep only endpoints."""
    if start is None:
        return []
    if end is None or end == start:
        return [int(start)]
    lo, hi = int(start), int(end)
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo + 1 <= max_span:
        return list(range(lo, hi + 1))
    return [lo, hi]


def _antipattern_card(
    border_color: str,
    title: str,
    detail: str,
    why: str,
    *,
    steps_html: str = "",
) -> str:
    """Render a single anti-pattern card with a 'why this matters' line.

    *steps_html* is trusted markup built via :func:`_step_link_chips` (integers
    only). Title/detail/why are escaped — they may embed trajectory text.
    """
    title = html.escape(str(title))
    detail = html.escape(str(detail))
    why = html.escape(str(why))
    return (
        f"<div style='padding:8px 12px;background:var(--ov-card);"
        f"border-left:3px solid {border_color};border-radius:4px;margin-bottom:6px;'>"
        f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
        f"<span style='font-size:12px;font-weight:600;'>{title}</span>"
        f"<span style='font-size:12px;color:var(--ov-muted);'>{detail}</span>"
        f"</div>"
        f"{steps_html}"
        f"<div style='font-size:11px;color:var(--ov-muted);font-style:italic;margin-top:3px;'>"
        f"Why it matters: {why}</div>"
        f"</div>"
    )


def build_antipattern_summary_html(
    fruitless_streaks: list[dict],
    tool_selection: list[dict],
    plan_metrics: dict,
    error_count: int = 0,
    error_steps: list[int] | None = None,
) -> str:
    """Render an anti-pattern summary panel with Workflow jump chips."""
    cards = []

    # Platform/tool errors
    if error_count > 0:
        cards.append(_antipattern_card(
            "var(--ov-bad)",
            f"Tool errors ({error_count}×)",
            "detected from tool output (platform, permission, missing file)",
            "Failed tool calls cost tokens and turns to recover from, and often indicate "
            "environment problems (wrong path, missing dependency, sandbox limits) rather than agent mistakes — "
            "fix the environment and the agent may stop wandering.",
            steps_html=_step_link_chips(error_steps or []),
        ))

    # Fruitless streaks
    if fruitless_streaks:
        total_wasted = sum(s["length"] for s in fruitless_streaks)
        shown = fruitless_streaks[:3]
        streak_desc = ", ".join(
            f"steps {s['start_step']}-{s['end_step']} ({s['length']})" for s in shown
        )
        remaining = len(fruitless_streaks) - len(shown)
        if remaining > 0:
            remaining_len = sum(s["length"] for s in fruitless_streaks[len(shown):])
            streak_desc += f", +{remaining} more ({remaining_len})"
        streak_indices: list[int] = []
        for s in fruitless_streaks:
            streak_indices.extend(
                _indices_for_step_range(s.get("start_step"), s.get("end_step"))
            )
        cards.append(_antipattern_card(
            "var(--ov-warn)",
            f"Fruitless search streaks ({len(fruitless_streaks)}×)",
            f"{total_wasted} wasted steps — {streak_desc}",
            "Three or more consecutive searches that returned no matches. Each one still "
            "consumes tokens and latency; sustained streaks suggest the agent is looking "
            "in the wrong place rather than refining its query.",
            steps_html=_step_link_chips(streak_indices),
        ))

    # Tool selection
    if tool_selection:
        bash_steps = [f.get("step") for f in tool_selection if f.get("step") is not None]
        cards.append(_antipattern_card(
            "var(--ov-accent)",
            f"Bash-for-reading ({len(tool_selection)}×)",
            "steps used sed/cat/head instead of Read tool",
            "Reading files via shell pipes bypasses the Read tool's structure — "
            "no line numbers, no cross-turn cache, no output cap — which inflates "
            "context size and makes the trajectory harder to analyze.",
            steps_html=_step_link_chips(bash_steps),
        ))

    # Stalled plan items
    stalled = plan_metrics.get("stalled", [])
    if stalled:
        items_desc = ", ".join(f"'{s['content'][:30]}'" for s in stalled[:2])
        stall_steps: list[int] = []
        for s in stalled:
            stall_steps.extend(
                _indices_for_step_range(s.get("start_step"), s.get("end_step"))
            )
        cards.append(_antipattern_card(
            "var(--ov-warn)",
            f"Stalled plan items ({len(stalled)}×)",
            items_desc,
            "Items marked in_progress in TodoWrite but never marked completed, "
            "or completed more than 20 steps after they started. Often means the "
            "agent context-switched away and forgot to close the loop.",
            steps_html=_step_link_chips(stall_steps),
        ))

    if not cards:
        return "<div style='padding:12px;color:var(--ov-muted);text-align:center;font-size:13px;'>No anti-patterns detected</div>"

    return (
        "<div class='antipattern-summary'>"
        "<div style='font-size:13px;font-weight:600;margin:4px 0 8px;'>Anti-pattern summary</div>"
        + "".join(cards)
        + "</div>"
    )
