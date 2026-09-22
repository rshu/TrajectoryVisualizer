"""Shared tool-call failure predicates (single source of truth).

Used by diagnostics, patterns, metrics, charts, and Issues so "failed tool
call" cannot drift across status sets or exit-code handling.
"""

from __future__ import annotations

from typing import Literal

from trajviz.tool_vocab import SYSTEM_TOOL_NAMES

# Union of former metrics/diagnostics/patterns status sets (lowercased).
FAILURE_STATUSES: frozenset[str] = frozenset({
    "error",
    "failed",
    "failure",
    "cancelled",
    "canceled",
    "timeout",
    "timed_out",
})

StepErrorKind = Literal["system", "tool"]


def tool_call_failed(tc: dict) -> bool:
    """True when a tool call clearly failed (status, exit code, or error fields)."""
    status = str(tc.get("status") or "").lower()
    if status in FAILURE_STATUSES:
        return True
    meta = tc.get("metadata")
    if isinstance(meta, dict) and meta.get("exit") not in (None, 0):
        return True
    return bool(tc.get("error") or tc.get("error_type"))


def tool_call_error_kind(tc: dict) -> StepErrorKind:
    """Classify one failed tool call as scaffold (system) or agentic (tool)."""
    name = tc.get("tool_name")
    if isinstance(name, str) and name in SYSTEM_TOOL_NAMES:
        return "system"
    return "tool"


def step_error_kind(step: dict) -> StepErrorKind | None:
    """Classify a step as system (scaffold) or tool (agentic) failure.

    System: Grep/Read/Write-style primitives, or ``finish == "error"``.
    Tool: Bash, Skill, Task, MCP, custom. Tool wins when both apply.
    See ``SYSTEM_TOOL_NAMES`` for the scaffold set.
    """
    saw_system = False
    for tc in step.get("tool_calls") or []:
        if not tool_call_failed(tc):
            continue
        name = tc.get("tool_name")
        if isinstance(name, str) and name in SYSTEM_TOOL_NAMES:
            saw_system = True
        else:
            return "tool"
    if saw_system:
        return "system"

    finish = step.get("finish") or ""
    if isinstance(finish, str) and finish.strip().lower() == "error":
        return "system"
    # Fallback when tool_calls were not attached but the step was flagged.
    if (step.get("error_count") or 0) > 0:
        return "tool"
    return None
