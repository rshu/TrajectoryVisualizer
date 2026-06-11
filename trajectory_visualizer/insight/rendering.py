"""HTML/code rendering, card styles, and workflow HTML generation."""

import html
import json
import re
from datetime import datetime, timezone

from pygments import highlight as _pygments_highlight
from pygments.formatters import HtmlFormatter as _HtmlFormatter
from pygments.lexers import get_lexer_by_name as _get_lexer, TextLexer as _TextLexer

from .charts import AGENT_CSS_COLORS, build_agent_color_map, AGENT_COLORS
from .styles import WORKFLOW_CSS


_ROLE_COLORS = {
    "user": ("var(--wf-bg-user)", "var(--wf-border-user)", "User"),
    "assistant": ("var(--wf-bg-assistant)", "var(--wf-border-assistant)", "Assistant"),
}

_ROLE_BADGE_STYLES = {
    "user": "background:var(--wf-border-user);color:white;",
    "assistant": "background:var(--wf-border-assistant);color:white;",
    "system": "background:var(--wf-border-default);color:white;",
    "tool": "background:var(--wf-border-reasoning);color:white;",
}


def _card_style(step: dict) -> tuple[str, str, str]:
    """Return (bg_color, border_color, label) for a step card.

    Colors are CSS variable references so they adapt to the active theme.
    """
    role = step["role"]
    if step["error_count"] > 0:
        return "var(--wf-bg-error)", "var(--wf-border-error)", "Error"
    if step.get("finish") == "stop" or step.get("finish") == "end_turn":
        return "var(--wf-bg-final)", "var(--wf-border-final)", "Final"
    if step["tool_call_count"] > 0:
        return "var(--wf-bg-tool)", "var(--wf-border-tool)", "Tool Calls"
    if step["has_reasoning"] and role == "assistant":
        return "var(--wf-bg-reasoning)", "var(--wf-border-reasoning)", "Reasoning"
    bg, border, label = _ROLE_COLORS.get(role, ("var(--wf-bg-default)", "var(--wf-border-default)", role.title()))
    return bg, border, label


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


def _escape_html_outside_fences(text: str) -> str:
    """Escape HTML tags in *text* but leave fenced-code blocks untouched.

    This prevents HTML-like fragments in assistant output (e.g. ``<thinking>``,
    ``<result>``) from being treated as real DOM when the markdown is later
    rendered with ``html=True`` (needed for ``<details>`` tags elsewhere).
    Content inside code fences is left as-is so markdown-it can handle it.

    Any unbalanced/orphaned backtick fences (3+) in non-fence segments are
    neutralized so they cannot open a spurious code block that swallows
    subsequent HTML (``<details>`` etc.).
    """
    parts: list[str] = []
    last_end = 0
    for m in _CODE_FENCE_RE.finditer(text):
        segment = html.escape(text[last_end:m.start()])
        parts.append(_neutralize_orphan_fences(segment))
        parts.append(m.group(0))          # fence untouched
        last_end = m.end()
    segment = html.escape(text[last_end:])
    parts.append(_neutralize_orphan_fences(segment))
    return "".join(parts)


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


_FILTER_CHIPS = ["Assistant", "User", "Tool Calls", "Errors", "Reasoning"]


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
            f"<span class='insight-step-link' onclick=\""
            f"(function(){{var tabs=document.querySelectorAll('.tab-nav button');"
            f"if(tabs.length>1)tabs[1].click();"
            f"setTimeout(function(){{var c=document.getElementById('wf-card-{sidx}');"
            f"if(c){{c.scrollIntoView({{behavior:'smooth',block:'center'}});c.click();}}"
            f"}},200);}})()\">Spawned at step #{sidx}</span>"
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


def render_filter_chips(active: list[str] | None = None, agent_labels: list[dict] | None = None) -> str:
    """Generate styled chip buttons for workflow filter categories.

    All chips are active by default. The HTML includes ``data-filter`` attributes
    and an ``onclick`` handler that toggles the ``chip-active`` class and writes
    the active values into a hidden Gradio Textbox (``#wf-filter-hidden``).
    """
    if active is None:
        active = list(_FILTER_CHIPS)
    active_set = set(active)
    chips: list[str] = []
    for name in _FILTER_CHIPS:
        cls = "filter-chip chip-active" if name in active_set else "filter-chip"
        chips.append(
            f"<span class='{cls}' data-filter='{html.escape(name)}'>"
            f"{html.escape(name)}</span>"
        )
    # Agent chips (after divider) when multi-agent
    if agent_labels and len(agent_labels) > 1:
        chips.append("<span class='filter-chip-divider'></span>")
        for i, al in enumerate(agent_labels):
            label = html.escape(al["label"])
            agent_hex = AGENT_COLORS[i % len(AGENT_COLORS)]
            filter_key = f"agent:{al['agent_id']}"
            chips.append(
                f"<span class='filter-chip filter-chip-agent chip-active'"
                f" data-filter='{html.escape(filter_key)}'"
                f" style='--agent-color:{agent_hex};'>"
                f"{label}</span>"
            )
    # Clear-all JS handler
    clear_js = (
        "(function(){"
        "var bar=document.getElementById('wf-filter-bar');"
        "if(!bar)return;"
        "bar.querySelectorAll('.filter-chip').forEach(function(c){c.classList.add('chip-active');});"
        "var active=Array.from(bar.querySelectorAll('.filter-chip.chip-active'))"
        ".map(function(c){return c.dataset.filter;});"
        "var h=document.querySelector('#wf-filter-hidden textarea,#wf-filter-hidden input');"
        "if(h){h.value=active.join(',');h.dispatchEvent(new Event('input',{bubbles:true}));}"
        "})()"
    )
    filter_summary = (
        f"<div class='filter-summary' id='wf-filter-summary'>"
        f"<span id='wf-filter-count'></span>"
        f"<span class='clear-all' onclick=\"{clear_js}\">Clear all</span>"
        f"</div>"
    )
    return (
        "<div class='filter-bar' id='wf-filter-bar'>" + "".join(chips) + "</div>"
        + filter_summary
    )


def render_toc_sidebar(steps: list[dict]) -> str:
    """Generate an HTML ``<nav>`` listing step numbers and role badges for a TOC sidebar."""
    if not steps:
        return ""
    items: list[str] = []
    for step in steps:
        idx = step.get("index", 0)
        role = step["role"]
        role_style = _ROLE_BADGE_STYLES.get(role, "background:var(--wf-border-default);color:white;")
        onclick = (
            f"(function(){{"
            f"var c=document.getElementById('wf-card-{idx}');"
            f"if(c){{c.scrollIntoView({{behavior:'smooth',block:'center'}});c.click();}}"
            f"}})()"
        )
        toc_indent = "padding-left:16px;" if step.get("is_sub_agent") else ""
        items.append(
            f"<div class='toc-entry' onclick=\"{onclick}\" data-step-idx='{idx}' style='{toc_indent}'>"
            f"<span class='toc-num'>#{idx}</span>"
            f"<span class='wf-badge' style='{role_style};font-size:11px;padding:2px 6px;'>"
            f"{html.escape(role.title())}</span>"
            f"</div>"
        )
    return (
        "<nav class='wf-toc-sidebar' id='wf-toc-sidebar'>"
        "<div class='toc-title'>Steps</div>"
        + "".join(items)
        + "</nav>"
    )


def render_workflow_html(steps: list[dict]) -> str:
    """Render vertical card flow as self-contained HTML with scroll container."""
    if not steps:
        return "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>No steps to display.</div>"

    css = WORKFLOW_CSS
    color_map = build_agent_color_map(steps)
    has_agents = len(color_map) > 1

    cards_html = []
    for i, step in enumerate(steps):
        bg, border, label = _card_style(step)
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
        err_info = f'<span style="color:var(--wf-border-error)">{step["error_count"]} err</span>' if step["error_count"] else ''

        # Agent badge with per-agent color
        agent_badge = ''
        agent_left_border = ""
        # Use effective agent (session_id for CodeArts sub-agents)
        agent_id = step.get("agent", "")
        if not agent_id and step.get("is_sub_agent") and step.get("session_id"):
            agent_id = step["session_id"]
        if has_agents:
            aidx = color_map.get(agent_id, 0)
            agent_bg, agent_border = AGENT_CSS_COLORS[aidx % len(AGENT_CSS_COLORS)]
            agent_hex = AGENT_COLORS[aidx % len(AGENT_COLORS)]
            agent_left_border = f"border-left:4px solid {agent_hex};"
            agent_label = "main" if not agent_id else (agent_id[:8] + "\u2026" if len(agent_id) > 8 else agent_id)
            agent_badge = (
                f'<span class="wf-badge" style="background:{agent_bg};color:{agent_border};'
                f'border:1px solid {agent_border};font-size:9px;">{html.escape(agent_label)}</span>'
            )

        role = step["role"]
        role_style = _ROLE_BADGE_STYLES.get(role, "background:var(--wf-border-default);color:white;")
        role_label = role.title()

        orig_idx = step.get("index", i)

        # CodeArts-specific badges (only render when fields are present)
        round_badge = ""
        round_num = step.get("round")
        if round_num is not None:
            round_badge = f'<span class="wf-badge" style="background:var(--ov-accent);color:white;font-size:10px;">R{round_num}</span>'

        sub_agent_badge = ""
        sub_indent = ""
        if step.get("is_sub_agent"):
            sub_agent_badge = '<span class="wf-badge" style="background:var(--wf-border-reasoning);color:white;font-size:10px;">sub-agent</span>'
            sub_indent = "margin-left:24px;"

        label_badges = _format_training_label_badges(step.get("training_label"))

        # Tool output preview (collapsed, for Bash commands)
        tool_output_html = ""
        tool_out = step.get("tool_output")
        if isinstance(tool_out, dict):
            bash_out = tool_out.get("bash", {}).get("content", "")
            if bash_out:
                escaped = html.escape(bash_out[:200])
                tool_output_html = (
                    f'<div class="wf-tool-output" style="font-size:11px;color:var(--ov-muted);'
                    f'background:var(--ov-code-bg);padding:4px 8px;border-radius:4px;margin-top:4px;'
                    f'max-height:60px;overflow:hidden;white-space:pre;font-family:monospace;">'
                    f'{escaped}</div>'
                )

        card = f"""
        <div class="wf-card" id="wf-card-{orig_idx}" data-step-idx="{orig_idx}"
             style="background:{bg};border-color:{border};{agent_left_border}{sub_indent}">
            <div class="wf-header">
                <span class="wf-badge" style="background:{border};color:white;">#{orig_idx}</span>
                {round_badge}
                <span class="wf-badge" style="{role_style}">{role_label}</span>
                {"" if label == role_label else f'<span class="wf-badge" style="background:transparent;color:{border};border:1px solid {border};">{label}</span>'}
                {sub_agent_badge}
                {label_badges}
                {agent_badge}
                <span class="wf-icons">{icon_str}</span>
            </div>
            <div class="wf-meta">
                <span>{dur}</span>
                <span>{tok} tok</span>
                {tc_info}{err_info}
            </div>
            <div class="wf-preview">{preview}</div>
            {tool_output_html}
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


def _format_training_label_badges(training_label: dict | None) -> str:
    if not isinstance(training_label, dict):
        return ""
    quality = training_label.get("quality", {}) if isinstance(training_label.get("quality"), dict) else {}
    value = training_label.get("value", {}) if isinstance(training_label.get("value"), dict) else {}
    decision = training_label.get("decision", {}) if isinstance(training_label.get("decision"), dict) else {}
    q = html.escape(str(quality.get("verdict", "?")))
    v = html.escape(str(value.get("tier", "?")))
    d = html.escape(str(decision.get("label", "?")))
    return (
        '<span class="wf-badge" style="background:#eef2ff;color:#3730a3;border:1px solid #818cf8;">'
        f'Q {q}</span>'
        '<span class="wf-badge" style="background:#ecfdf5;color:#047857;border:1px solid #34d399;">'
        f'V {v}</span>'
        '<span class="wf-badge" style="background:#fef3c7;color:#92400e;border:1px solid #f59e0b;">'
        f'{d}</span>'
    )


def _fmt_timestamp(ms):
    """Convert epoch-milliseconds to readable ``YYYY-MM-DD HH:MM:SS`` (UTC)."""
    if not isinstance(ms, (int, float)):
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _format_step_header(step: dict) -> str:
    """Build the styled HTML header banner and metadata table for a step detail panel."""
    bg, border, label = _card_style(step)
    role = step["role"]
    role_style = _ROLE_BADGE_STYLES.get(role, "background:var(--wf-border-default);color:white;")

    rows: list[tuple[str, str]] = [("Role", step['role'])]
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

    # CodeArts-specific metadata
    if step.get("round") is not None:
        rows.append(("Round", str(step["round"])))
    if step.get("is_sub_agent"):
        rows.append(("Sub-agent", "Yes"))
    if step.get("agent_id"):
        rows.append(("Agent ID", step["agent_id"]))

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
        f"<span class='dp-badge' style='{role_style}'>{html.escape(role.title())}</span>"
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


def _safe_fence(text: str, lang: str = "") -> str:
    """Wrap *text* in a fenced code block whose delimiter is longer than any backtick run inside.

    CommonMark allows opening fences of 3+ backticks; the closing fence must be
    at least as long.  By choosing a delimiter longer than any run in *text* we
    guarantee the fence is never prematurely closed.
    """
    longest = 2                              # minimum fence is 3
    for m in re.finditer(r"`+", text):
        longest = max(longest, len(m.group()))
    fence = "`" * (longest + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def _format_tool_call_detail(p: dict) -> str:
    """Render a single tool_call part as a styled HTML block."""
    inp = p.get("input", {})
    out = p.get("output", "")
    inp_str = json.dumps(inp, indent=2, ensure_ascii=False) if isinstance(inp, dict) else str(inp)
    if isinstance(out, str) and len(out) > 2000:
        out = out[:2000] + "\n... (truncated)"
    elif isinstance(out, dict):
        out = json.dumps(out, indent=2, ensure_ascii=False)
        if len(out) > 2000:
            out = out[:2000] + "\n... (truncated)"

    tc_dur = ""
    if p.get("time_start") and p.get("time_end"):
        tc_dur = f" &mdash; {round((p['time_end'] - p['time_start']) / 1000, 2)}s"

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
    title = html.escape(raw_title or tool_name)

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


def _format_metrics_tab(step: dict) -> str:
    """Render the Metrics tab content for a step detail panel."""
    tokens = step["tokens"]
    rows = [
        ("Total Tokens", f"{tokens['total']:,}"),
        ("Input Tokens", f"{tokens['input']:,}"),
        ("Output Tokens", f"{tokens['output']:,}"),
        ("Reasoning Tokens", f"{tokens['reasoning']:,}"),
        ("Cache Read", f"{tokens['cache_read']:,}"),
        ("Cache Write", f"{tokens['cache_write']:,}"),
    ]
    if step.get("duration") is not None:
        rows.append(("Duration", f"{step['duration']}s"))
        if tokens['total'] > 0 and step['duration'] > 0:
            tok_per_s = tokens['total'] / step['duration']
            rows.append(("Throughput", f"{tok_per_s:,.0f} tok/s"))
    if tokens['total'] > 0:
        cache_ratio = tokens['cache_read'] / tokens['total'] * 100
        rows.append(("Cache Ratio", f"{cache_ratio:.1f}%"))

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
        f"<span onclick=\"var tabs=document.querySelectorAll('.tab-nav button');"
        f"if(tabs.length>1)tabs[1].click();\">Workflow</span>"
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

    # CodeArts: question/prompt context
    question = step.get("question", [])
    if question:
        q_text = ""
        for q in question:
            if isinstance(q, dict):
                q_text = q.get("content", q.get("value", {}).get("input", ""))
                break
        if q_text:
            content_parts.insert(0,
                f"<div class='dp-section' style='border-left-color:var(--wf-border-reasoning);'>"
                f"<div class='dp-section-title'>Question / Prompt</div>"
                f"<div style='font-size:13px;white-space:pre-wrap;'>{html.escape(str(q_text)[:500])}</div>"
                f"</div>"
            )

    # CodeArts: full tool output
    tool_out = step.get("tool_output")
    if isinstance(tool_out, dict):
        for tool_name, tool_data in tool_out.items():
            out_content = tool_data.get("content", "") if isinstance(tool_data, dict) else str(tool_data)
            if out_content:
                escaped = html.escape(str(out_content)[:2000])
                content_parts.append(
                    f"<details class='dp-details'><summary>Tool Output ({html.escape(tool_name)})</summary>"
                    f"<div class='dp-details-body'><pre>{escaped}</pre></div>"
                    f"</details>"
                )

    training_label_html = _format_training_label_detail(step.get("training_label"))
    if training_label_html:
        content_parts.insert(0, training_label_html)

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


def _format_training_label_detail(training_label: dict | None) -> str:
    if not isinstance(training_label, dict):
        return ""
    quality = training_label.get("quality", {}) if isinstance(training_label.get("quality"), dict) else {}
    value = training_label.get("value", {}) if isinstance(training_label.get("value"), dict) else {}
    decision = training_label.get("decision", {}) if isinstance(training_label.get("decision"), dict) else {}
    tags = ", ".join(html.escape(str(tag)) for tag in value.get("tags", [])) or "none"
    defects = ", ".join(html.escape(str(flag)) for flag in quality.get("defect_flags", [])) or "none"
    reasons = ", ".join(html.escape(str(reason)) for reason in decision.get("reasons", [])) or "none"
    return (
        "<div class='dp-section' style='border-left-color:#2563eb;'>"
        "<div class='dp-section-title'>Training Labels</div>"
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;font-size:13px;'>"
        f"<div><strong>Behavior</strong><br>{html.escape(str(training_label.get('phase', '')))} / {html.escape(str(training_label.get('action', '')))}</div>"
        f"<div><strong>Quality</strong><br>{html.escape(str(quality.get('verdict', '')))} ({html.escape(str(quality.get('confidence', '')))} confidence)<br>Defects: {defects}</div>"
        f"<div><strong>Value</strong><br>{html.escape(str(value.get('tier', '')))} ({html.escape(str(value.get('confidence', '')))} confidence)<br>Tags: {tags}</div>"
        f"<div><strong>Decision</strong><br>{html.escape(str(decision.get('label', '')))}<br>Reasons: {reasons}</div>"
        "</div></div>"
    )


# ---------------------------------------------------------------------------
# Diagnostic renderers
# ---------------------------------------------------------------------------

def _diag_jump_onclick(idx: int) -> str:
    """JS onclick to switch to Workflow tab and scroll to a step card."""
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


def build_failure_chain_strip_html(chains: list[dict]) -> str:
    """Render a horizontal strip of clickable failure chain badges."""
    if not chains:
        return ""
    badges = []
    for c in chains:
        start, end = c["start"], c["end"]
        n = len(c["steps"])
        onclick = _diag_jump_onclick(start)
        if start == end:
            label = f"Chain: step {start} (1 step)"
        else:
            label = f"Chain: {start}\u2013{end} ({n} steps)"
        spawn_info = ""
        if c.get("spawning_step") is not None:
            spawn_info = f" <span style='font-size:10px;opacity:0.7;'>[from step {c['spawning_step']}]</span>"
        badges.append(
            f"<span class='diag-chain-badge' onclick=\"{onclick}\" style='cursor:pointer;'>"
            f"{html.escape(label)}{spawn_info}"
            f"</span>"
        )
    return "<div class='diag-chain-strip'>" + "".join(badges) + "</div>"


def build_bottleneck_cards_html(explanations: list[dict]) -> str:
    """Render bottleneck explanation cards with decomposition bars."""
    if not explanations:
        return ""
    cards = []
    for e in explanations:
        d = e["decomposition"]
        idx = e["step_idx"]
        onclick = _diag_jump_onclick(idx)

        # Build stacked bar segments
        bar_segments = []
        if d["tool_pct"] > 0:
            bar_segments.append(
                f"<div class='diag-bar-seg diag-bar-tool' style='width:{d['tool_pct']}%;'"
                f" title='Tool: {d['tool_s']}s ({d['tool_pct']}%)'></div>"
            )
        if d["inference_pct"] > 0:
            bar_segments.append(
                f"<div class='diag-bar-seg diag-bar-inference' style='width:{d['inference_pct']}%;'"
                f" title='Inference: {d['inference_s']}s ({d['inference_pct']}%)'></div>"
            )
        if d["idle_pct"] > 0:
            bar_segments.append(
                f"<div class='diag-bar-seg diag-bar-idle' style='width:{d['idle_pct']}%;'"
                f" title='Idle: {d['idle_s']}s ({d['idle_pct']}%)'></div>"
            )
        bar_html = "<div class='diag-bar'>" + "".join(bar_segments) + "</div>"

        cards.append(
            f"<div class='diag-bottleneck-card' onclick=\"{onclick}\" style='cursor:pointer;'>"
            f"<div class='diag-bottleneck-header'>"
            f"<span class='diag-bottleneck-step'>Step {idx}</span>"
            f"<span class='diag-bottleneck-dur'>{e['duration']:.1f}s</span>"
            f"</div>"
            f"{bar_html}"
            f"<div class='diag-bottleneck-text'>{html.escape(e['explanation'])}</div>"
            f"</div>"
        )
    return "<div class='diag-bottleneck-grid'>" + "".join(cards) + "</div>"


def build_root_cause_html(clusters: list[dict]) -> str:
    """Render root-cause candidate summary panel."""
    if not clusters:
        return ""

    from .diagnostics import format_root_cause_summary
    summaries = format_root_cause_summary(clusters)

    items = []
    for i, (cluster, summary) in enumerate(zip(clusters, summaries)):
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

_VERDICT_COLORS = {
    "good": "#059669",
    "warn": "#d97706",
    "bad": "#dc2626",
    "n/a": "#9ca3af",
}

_VERDICT_LABELS = {
    "good": "Good",
    "warn": "Warn",
    "bad": "Bad",
    "n/a": "N/A",
}

_DIMENSION_NAV_TARGETS = {
    "targeting": "file-interaction",
    "error_resilience": "failure-chain",
    "execution_efficiency": "bottleneck",
    "cost_efficiency": "cache",
}

_DIMENSION_DISPLAY_NAMES = {
    "targeting": "Targeting",
    "error_resilience": "Error Resilience",
    "execution_efficiency": "Execution Efficiency",
    "cost_efficiency": "Cost Efficiency",
}


def build_dimension_cards_html(dimensions: dict) -> str:
    """Render four dimension cards with sub-score, verdict badge, and driving metric."""
    if not dimensions:
        return ""

    cards = []
    for name, dim in dimensions.items():
        score = dim.get("score")
        verdict = dim.get("verdict", "n/a")
        color = _VERDICT_COLORS.get(verdict, "#9ca3af")
        label = _DIMENSION_DISPLAY_NAMES.get(name, name)
        score_str = f"{score:.0f}" if score is not None else "N/A"

        # Find the driving metric (lowest score)
        metrics = dim.get("metrics", {})
        driving_metric = ""
        if metrics:
            non_none = {k: v for k, v in metrics.items() if v is not None}
            if non_none:
                worst_key = min(non_none, key=non_none.get)
                driving_metric = f"{worst_key.replace('_', ' ')}: {non_none[worst_key]:.1f}" if isinstance(non_none[worst_key], float) else f"{worst_key.replace('_', ' ')}: {non_none[worst_key]}"

        # Navigation target
        nav = _DIMENSION_NAV_TARGETS.get(name, "")
        onclick = ""
        if nav:
            onclick = (
                f" onclick=\"(function(){{"
                f"var acc=document.querySelectorAll('.per-message-acc');"
                f"for(var i=0;i<acc.length;i++){{"
                f"var btn=acc[i].querySelector('button');"
                f"if(btn&&btn.textContent.indexOf('Diagnostics')>=0)"
                f"{{if(acc[i].classList.contains('open')===false)btn.click();break;}}"
                f"}}}})()\" style='cursor:pointer;'"
            )

        verdict_label = _VERDICT_LABELS.get(verdict, verdict)
        cards.append(
            f"<div class='score-dim-card'{onclick}>"
            f"<div class='score-dim-header'>"
            f"<span class='score-dim-name'>{html.escape(label)}</span>"
            f"<span class='score-dim-badge' style='background:{color};'>{html.escape(verdict_label)}</span>"
            f"</div>"
            f"<div class='score-dim-score' style='color:{color};'>{score_str}</div>"
            f"<div class='score-dim-metric'>{html.escape(driving_metric) if driving_metric else 'insufficient data'}</div>"
            f"</div>"
        )
    return "<div class='score-dim-grid'>" + "".join(cards) + "</div>"


def build_score_banner_badge_html(composite_score: float | None, verdict: str) -> str:
    """Render a compact score badge for the summary banner."""
    if composite_score is None:
        return ""
    color = _VERDICT_COLORS.get(verdict, "#9ca3af")
    label = _VERDICT_LABELS.get(verdict, verdict)
    return (
        f"<span class='score-banner-badge'>"
        f"<span class='score-banner-dot' style='background:{color};'></span>"
        f"Quality: {composite_score:.0f}/100 &middot; {html.escape(label)}"
        f"</span>"
    )


def build_judge_result_html(judge_result: dict | None) -> str:
    """Render collapsible LLM judge result panel."""
    if not judge_result:
        return ""

    verdict = judge_result.get("verdict", "uncertain")
    reasoning = judge_result.get("reasoning", "")
    flagged = judge_result.get("flagged_steps", [])
    color = _VERDICT_COLORS.get(
        "good" if verdict == "acceptable" else ("bad" if verdict == "poor" else "warn"),
        "#9ca3af",
    )
    verdict_display = verdict.title()

    flagged_html = ""
    if flagged:
        links = []
        for idx in flagged:
            onclick = _diag_jump_onclick(idx)
            links.append(f"<span class='insight-step-link' onclick=\"{onclick}\">step {idx}</span>")
        flagged_html = f"<div class='judge-flagged'>Flagged: {', '.join(links)}</div>"

    return (
        f"<details class='judge-panel'>"
        f"<summary>"
        f"<span class='judge-badge' style='background:{color};'>{html.escape(verdict_display)}</span>"
        f" LLM Judge Assessment"
        f"</summary>"
        f"<div class='judge-reasoning'>{html.escape(reasoning)}</div>"
        f"{flagged_html}"
        f"</details>"
    )


# ---------------------------------------------------------------------------
# Sub-Agent Delegation Summary
# ---------------------------------------------------------------------------

def build_subagent_summary_html(sessions: list[dict]) -> str:
    """Render a sub-agent delegation summary table."""
    if not sessions:
        return ""

    rows = ""
    for s in sessions:
        sid = s.get("session_id", "")[:12]
        spawn = s.get("spawn_step", "?")
        start = s.get("start_step", "?")
        end = s.get("end_step", "?")
        steps = s.get("step_count", 0)
        tokens = s.get("total_tokens", 0)
        tools = s.get("total_tools", 0)
        dur = s.get("total_duration", 0)
        rows += (
            f"<tr>"
            f"<td style='font-family:monospace;font-size:12px;'>{html.escape(sid)}</td>"
            f"<td style='text-align:center;'>{spawn}</td>"
            f"<td style='text-align:center;'>{start}–{end}</td>"
            f"<td style='text-align:right;'>{steps}</td>"
            f"<td style='text-align:right;'>{tokens:,}</td>"
            f"<td style='text-align:right;'>{tools}</td>"
            f"<td style='text-align:right;'>{dur:.0f}s</td>"
            f"</tr>"
        )

    total_steps = sum(s.get("step_count", 0) for s in sessions)
    total_tokens = sum(s.get("total_tokens", 0) for s in sessions)

    return (
        f"<div style='margin-bottom:12px;'>"
        f"<div style='font-size:13px;font-weight:600;margin-bottom:6px;'>"
        f"Sub-Agent Delegation — {len(sessions)} session(s), {total_steps} steps, {total_tokens:,} tokens</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        f"<thead><tr style='border-bottom:2px solid var(--ov-border);'>"
        f"<th style='text-align:left;padding:4px 8px;'>Session</th>"
        f"<th style='text-align:center;padding:4px 8px;'>Spawn</th>"
        f"<th style='text-align:center;padding:4px 8px;'>Steps</th>"
        f"<th style='text-align:right;padding:4px 8px;'>Count</th>"
        f"<th style='text-align:right;padding:4px 8px;'>Tokens</th>"
        f"<th style='text-align:right;padding:4px 8px;'>Tools</th>"
        f"<th style='text-align:right;padding:4px 8px;'>Duration</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )
