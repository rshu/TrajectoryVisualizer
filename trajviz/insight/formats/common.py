"""Shared helpers for trajectory format converters."""

from datetime import datetime
from typing import Any


def safe_get(d: Any, *keys: Any, default: Any = None) -> Any:
    """Safe nested dict access."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def _iso_to_epoch_ms(iso_str: str | None) -> int | None:
    """Convert ISO 8601 string to epoch milliseconds.

    Non-string input (e.g. an already-numeric epoch timestamp in a
    hand-edited file) degrades to None (missing timestamp) instead of raising.
    """
    if not iso_str or not isinstance(iso_str, str):
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None

_TOOL_ERROR_PATTERNS = [
    # Platform / shell-level
    ("command not found", "platform_error"),
    ("not recognized as", "platform_error"),  # Windows "not recognized as internal or external command"
    ("command timed out", "platform_error"),
    # Permission / policy
    ("Permission denied", "permission_error"),
    ("permission denied", "permission_error"),
    ("rule which prevents", "permission_error"),  # OpenCode: "user has specified a rule which prevents..."
    ("must read file", "permission_error"),       # OpenCode: edit/write before read enforcement
    # Missing file / path
    ("No such file or directory", "missing_file"),
    ("cannot find the path", "missing_file"),
    ("ENOENT", "missing_file"),
    # Bad input / invalid args
    ("out of range", "bad_input"),
    # Generic tool failure (catch-all for tools that errored without a recognized cause)
    ("ripgrep failed", "tool_error"),
]


def _classify_tool_error(output: str) -> str | None:
    """Classify a Bash tool output as an error type, or None if no error detected."""
    if not output:
        return None
    for pattern, error_type in _TOOL_ERROR_PATTERNS:
        if pattern in output:
            return error_type
    return None
