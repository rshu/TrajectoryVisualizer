"""Classify step failures as scaffold (system) vs agentic (tool) errors.

Predicates live in :mod:`trajviz.insight.tool_failure`; this module re-exports
them for existing imports.
"""

from __future__ import annotations

from .tool_failure import (  # noqa: F401
    FAILURE_STATUSES,
    StepErrorKind,
    step_error_kind,
    tool_call_error_kind,
    tool_call_failed,
)

__all__ = [
    "FAILURE_STATUSES",
    "StepErrorKind",
    "step_error_kind",
    "tool_call_error_kind",
    "tool_call_failed",
]
