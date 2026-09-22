"""Workflow tab HTML: filter chips, cards, TOC, and detail store."""

from __future__ import annotations

import base64
import json

from ..charts import timeline_agent_id_of
from ..rendering import (
    AGENT_ALL_FILTER,
    AGENT_FILTER_PREFIX,
    agent_id_from_filter_token,
    format_step_detail,
    render_filter_chips,
    render_toc_sidebar,
    render_workflow_html,
    workflow_agent_chip_options,
    workflow_role,
)
from ..step_errors import step_error_kind

DETAIL_PLACEHOLDER = (
    "<div id='wf-detail-content'>"
    "<div data-wf-detail-placeholder='1' style='padding:2em 1em;text-align:center;color:var(--ov-muted);'>"
    "<p style='font-size:15px;margin-bottom:0.5em;'>Select a step to inspect</p>"
    "<p style='font-size:12px;'>Click any card on the left, or press <kbd>j</kbd>/<kbd>k</kbd> to navigate</p>"
    "</div></div>"
)
ROLE_FILTERS = ["Assistant", "User"]
FEATURE_FILTERS = ["Tool Calls", "Errors", "Reasoning"]
ALL_FEATURE_FILTER = "All"
FILTER_CHIPS_DEFAULT = [*ROLE_FILTERS, ALL_FEATURE_FILTER]


def _prerender_step_details(steps: list[dict]) -> str:
    """Pre-render all step details as HTML and return a base64-encoded JSON blob."""
    details = {}
    for step in steps:
        details[str(step["index"])] = format_step_detail(step)
    b64 = base64.b64encode(json.dumps(details).encode()).decode()
    return f'<div data-b64="{b64}" style="display:none"></div>'


def _workflow_step_labels(step: dict) -> set[str]:
    """Return every Workflow filter label that applies to *step*."""
    labels: set[str] = set()
    display = workflow_role(step)
    if display == "user":
        labels.add("User")
    elif step.get("role") == "assistant" or display in ("task", "system", "compaction"):
        # Task/system/compaction are stored as user in some exports but are
        # agent-protocol messages, not human turns.
        labels.add("Assistant")
    if step.get("tool_call_count", 0) > 0:
        labels.add("Tool Calls")
    if step_error_kind(step) is not None:
        labels.add("Errors")
    if step.get("has_reasoning"):
        labels.add("Reasoning")
    return labels


def filter_workflow_steps(
    steps: list[dict],
    active_filters: list[str],
    keyword: str = "",
) -> list[int]:
    """Return positions matching required roles, optional features/agents, and search.

    Roles are ORed with each other, selected features are ORed with each other,
    selected agents are ORed with each other, and the groups are ANDed.
    ``All`` / ``agent:All`` (or an omitted selection for that group) means that
    group imposes no predicate. Agent identity matches Workflow card coloring
    via :func:`timeline_agent_id_of`.
    """
    if not steps:
        return []

    keyword = (keyword or "").strip().lower()
    active = {str(value).strip() for value in active_filters if str(value).strip()}
    role_filters = active & set(ROLE_FILTERS)
    if not role_filters:
        return []
    feature_filters = active & set(FEATURE_FILTERS)
    restrict_features = ALL_FEATURE_FILTER not in active and bool(feature_filters)

    agent_tokens = {t for t in active if t.startswith(AGENT_FILTER_PREFIX)}
    restrict_agents = AGENT_ALL_FILTER not in agent_tokens and bool(agent_tokens)
    if restrict_agents:
        selected_agent_ids = {
            aid
            for token in agent_tokens
            if (aid := agent_id_from_filter_token(token)) is not None
        }
        if not selected_agent_ids:
            return []
        agent_id_of = timeline_agent_id_of(steps)
    else:
        selected_agent_ids = set()
        agent_id_of = None

    filtered: list[int] = []
    for position, step in enumerate(steps):
        labels = _workflow_step_labels(step)
        if not (labels & role_filters):
            continue
        if restrict_features and not (labels & feature_filters):
            continue
        if restrict_agents and agent_id_of(step) not in selected_agent_ids:
            continue

        if keyword:
            text = str(step.get("text_preview") or "").lower()
            tool_names = " ".join(
                str(tool_call.get("tool_name", ""))
                for tool_call in step.get("tool_calls", [])
                if isinstance(tool_call, dict)
            ).lower()
            tool_args = " ".join(
                str(tool_call.get("input", ""))
                for tool_call in step.get("tool_calls", [])
                if isinstance(tool_call, dict)
            ).lower()
            if keyword not in text and keyword not in tool_names and keyword not in tool_args:
                continue
        filtered.append(position)
    return filtered


def build_filtered_workflow_outputs(
    steps: list[dict],
    filter_csv: str,
    keyword: str,
    current_toc: str = "",
) -> tuple[str, str, str]:
    """Build filtered Workflow cards, count, and matching TOC HTML.

    ``current_toc`` carries the previous TOC HTML so a user-collapsed
    sidebar (``toc-hidden``) stays collapsed across re-renders.
    """
    if not steps:
        return (
            "<div style='padding:3em;color:var(--ov-muted);text-align:center;"
            "font-size:15px;'>Load a trajectory to see the step flow.</div>",
            "",
            "",
        )

    active_filters = [value.strip() for value in (filter_csv or "").split(",") if value.strip()]
    indices = filter_workflow_steps(steps, active_filters, keyword)
    filtered_steps = [steps[position] for position in indices]

    if not (set(active_filters) & set(ROLE_FILTERS)):
        workflow_html = (
            "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
            "Select at least one role to see steps.</div>"
        )
    else:
        workflow_html = render_workflow_html(filtered_steps)

    count_html = f"<div class='wf-count'>Showing {len(filtered_steps)} of {len(steps)} steps</div>"
    collapsed = "toc-hidden" in (current_toc or "")
    return workflow_html, count_html, render_toc_sidebar(filtered_steps, collapsed=collapsed)


def build_workflow_outputs(steps: list[dict]) -> dict:
    """Build workflow HTML, TOC, filter chips, and detail store."""
    wf_html = render_workflow_html(steps)
    wf_count = f"<div class='wf-count'>Showing {len(steps)} of {len(steps)} steps</div>"
    toc_html_val = render_toc_sidebar(steps)
    detail_store_val = _prerender_step_details(steps)

    agent_options = workflow_agent_chip_options(steps)
    active = list(FILTER_CHIPS_DEFAULT)
    if agent_options:
        active.append(AGENT_ALL_FILTER)
    wf_chips = render_filter_chips(active, agent_options=agent_options)
    wf_filter_val = ",".join(active)

    return {
        "wf_chips": wf_chips,
        "wf_filter_val": wf_filter_val,
        "wf_count": wf_count,
        "toc_html_val": toc_html_val,
        "wf_html": wf_html,
        "detail_store_val": detail_store_val,
    }
