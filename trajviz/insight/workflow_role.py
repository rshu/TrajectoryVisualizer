"""Classify steps for Workflow badges and agent swimlane lanes.

Kept free of chart/HTML imports so swimlanes and rendering can both use it
without a circular dependency.
"""

from __future__ import annotations


def workflow_role(step: dict) -> str:
    """Role key used for Workflow badges, filters, and the user swimlane.

    OpenCode (and similar) record Task prompts, compaction checkpoints, and
    synthetic continues as ``role=user``. Those are agent-protocol messages,
    not human turns, so they must not count as User.
    """
    raw = step.get("role", "")
    role = raw if isinstance(raw, str) else str(raw or "")
    if role != "user":
        return role
    parts = step.get("parts") if isinstance(step.get("parts"), list) else []
    if step.get("is_compaction_checkpoint") or any(
        isinstance(part, dict) and part.get("type") == "compaction" for part in parts
    ):
        return "compaction"
    text_parts = [
        part for part in parts
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    if text_parts and all(part.get("synthetic") for part in text_parts):
        return "system"
    if step.get("is_sub_agent"):
        return "task"
    return "user"
