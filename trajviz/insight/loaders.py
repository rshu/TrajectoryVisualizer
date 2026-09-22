"""Trajectory format detection and conversion (Claude Code, Cursor, OpenCode, CodeArts, ICode, Codex, Pi, DSH).

Converters live in ``trajviz.insight.formats``. This module is the public
facade: ``detect_format``, ``load_trajectory``, and the names callers already
import.
"""

from __future__ import annotations

from typing import Any

from .formats.claude_code import _convert_claude_code_to_internal
from .formats.codearts import _convert_codearts_metadata
from .formats.codex import _convert_codex_to_internal
from .formats.cursor import _convert_cursor_to_internal
from .formats.icode import _convert_icode_to_internal
from .formats.common import (  # noqa: F401  (re-exported public / test API)
    _classify_tool_error,
    _iso_to_epoch_ms,
    safe_get,
)
from .formats.dsh import (
    DshZipContents,
    _convert_dsh_to_internal,
    _dsh_drop_seed_prefix,  # noqa: F401
    _zip_dsh_members,  # noqa: F401
    parse_dsh_zip,
    resolve_dsh_session_path,
)
from .formats.opencode import _convert_opencode_metadata
from .formats.parse import _normalize_payload, _parse_trajectory_text, _path_ext
from .formats.pi import _convert_pi_to_internal
from .formats.sniff import (
    _EVENT_FORMATS,
    _FORMAT_STAMPS,
    _OBJECT_FORMATS,
    _detect_event_stream_format,
    _detect_object_format,
)

# Canonical display names for the detected trajectory formats.  Defined here,
# next to detect_format, as the single source of truth — UI/report modules
# import this mapping instead of maintaining their own drifting copies.
FORMAT_LABELS = {
    "ccsession": "Claude Code",
    "cursor": "Cursor",
    "codearts": "CodeArts",
    "icode": "ICode",
    "opencode": "OpenCode",
    "codex": "Codex CLI",
    "pi": "Pi",
    "dsh": "DeepSeek Harness",
}

# Dropdown choices for the Insight UI. Auto-detect is first so it is the
# default; an explicit format is still available as an override / mismatch gate.
FORMAT_DROPDOWN_CHOICES: list[tuple[str, str]] = [
    ("Auto-detect", ""),
    *((label, key) for key, label in FORMAT_LABELS.items()),
]


def _convert_codex_events(payload: list, *, source_path: str | None = None) -> dict:
    del source_path
    return _convert_codex_to_internal(payload)


def _convert_dsh_events(payload: list, *, source_path: str | None = None) -> dict:
    return _convert_dsh_to_internal(payload, source_path=source_path)


def _convert_pi_events(payload: list, *, source_path: str | None = None) -> dict:
    del source_path
    return _convert_pi_to_internal(payload)


_EVENT_CONVERTERS = {
    "codex": _convert_codex_events,
    "dsh": _convert_dsh_events,
    "pi": _convert_pi_events,
}
if set(_EVENT_CONVERTERS) != _EVENT_FORMATS:
    raise RuntimeError(
        "Event converters must match sniff._EVENT_FORMATS: "
        f"converters={sorted(_EVENT_CONVERTERS)} sniff={sorted(_EVENT_FORMATS)}"
    )
if set(FORMAT_LABELS) != _OBJECT_FORMATS | _EVENT_FORMATS:
    raise RuntimeError(
        "FORMAT_LABELS must match sniff object+event formats: "
        f"labels={sorted(FORMAT_LABELS)} sniff={sorted(_OBJECT_FORMATS | _EVENT_FORMATS)}"
    )


def detect_format(raw: Any) -> str:
    """Detect trajectory format from parsed content.

    Accepts a JSON object or an event array. Returns ``ccsession``,
    ``opencode``, ``codearts``, ``cursor``, ``icode``, ``codex``, ``pi``, ``dsh``,
    or ``unknown``.
    """
    if isinstance(raw, list):
        return _detect_event_stream_format(raw)
    if isinstance(raw, dict):
        return _detect_object_format(raw)
    return "unknown"


def _resolve_format(detected: str, hint: str | None) -> tuple[str, str | None]:
    """Apply ``format_hint`` to a detected format.

    Returns ``(fmt, reason)`` where ``reason`` is ``None`` (proceed),
    ``"unknown"`` (auto-detect found nothing), or ``"mismatch"``.

    Empty hint is auto-detect. A recognized hint on ``unknown`` *forces*
    that converter. A recognized hint on a different detected format is
    a mismatch (including Codex/Pi).
    """
    selected = hint if hint in FORMAT_LABELS else ""
    if not selected:
        if detected in ("", "unknown"):
            return "unknown", "unknown"
        return detected, None
    if detected in ("", "unknown"):
        return selected, None
    if detected != selected:
        return detected, "mismatch"
    return detected, None


def check_format_selection(detected: str, selected: str | None) -> str | None:
    """Return an error reason if the dropdown selection cannot load this file.

    ``None`` means proceed. Empty ``selected`` is auto-detect: any recognized
    format is accepted, and ``unknown`` is rejected so we do not render an
    empty dashboard. An explicit selection forces conversion when detection
    is ``unknown``, and rejects a different recognized format (including
    Codex/Pi).
    """
    _, reason = _resolve_format(detected, selected or None)
    return reason


_UNSUPPORTED_EVENT_STREAM = (
    "Unsupported event-array input; expected Codex JSONL "
    "(leading session_meta event), Pi JSONL "
    "(leading session event), or DeepSeek Harness JSONL "
    "(leading session event with createdAt / slash-typed body events)."
)


def _already_converted(raw: dict, fmt: str) -> bool:
    stamp = _FORMAT_STAMPS.get(fmt)
    return bool(stamp and raw.get(stamp) is True)


def _format_mismatch_error(selected: str, detected: str) -> dict:
    return {
        "_error": (
            f"Format mismatch: selected {FORMAT_LABELS.get(selected, selected)} "
            f"but file detected as {FORMAT_LABELS.get(detected, detected)}."
        ),
        "_error_code": "mismatch",
        "_selected": selected,
        "_detected": detected,
    }


def _kind_mismatch_error(fmt: str) -> dict:
    label = FORMAT_LABELS.get(fmt, fmt)
    if fmt in _EVENT_FORMATS:
        leading = "session_meta" if fmt == "codex" else "session"
        return {"_error": f"{label} expects a JSONL event array (leading {leading} event)."}
    return {"_error": f"{label} expects a JSON object, not an event array."}


def _apply_format(payload: Any, fmt: str, *, source_path: str | None = None) -> dict:
    """Convert parsed content to the internal step-model dict."""
    if fmt == "unknown":
        if isinstance(payload, dict):
            return payload
        return {"_error": _UNSUPPORTED_EVENT_STREAM}

    if isinstance(payload, dict) and _already_converted(payload, fmt):
        return payload

    if fmt in _EVENT_FORMATS:
        if not isinstance(payload, list):
            return _kind_mismatch_error(fmt)
        convert = _EVENT_CONVERTERS.get(fmt)
        if convert is None:
            return {"_error": f"Unknown event format: {fmt}"}
        return convert(payload, source_path=source_path)

    if not isinstance(payload, dict):
        return _kind_mismatch_error(fmt)
    if fmt == "ccsession":
        return _convert_claude_code_to_internal(payload)
    if fmt == "codearts":
        # Legacy-JSON exports are detected as CodeArts (so the UI labels them
        # correctly) but have no parser yet: their 'sender'/'content' messages
        # are not OpenCode-shaped and would silently produce empty steps.
        if safe_get(payload, "export_metadata", "source_format") == "codearts_legacy_json":
            return {
                "_error": "CodeArts legacy-JSON export detected "
                "(source_format=codearts_legacy_json): this legacy "
                "message schema is not yet supported. Re-export the "
                "session from the CodeArts SQLite database instead."
            }
        return _convert_codearts_metadata(payload)
    if fmt == "icode":
        return _convert_icode_to_internal(payload)
    if fmt == "cursor":
        return _convert_cursor_to_internal(payload)
    if fmt == "opencode":
        return _convert_opencode_metadata(payload)
    return {"_error": f"Unknown trajectory format: {fmt}"}


def _load_zip_trajectory(file_path: str, format_hint: str | None, source_sha: str) -> dict:
    """Load a DSH export zip (``session.jsonl`` + optional ``subagents/``)."""
    parsed = parse_dsh_zip(file_path)
    if not isinstance(parsed, DshZipContents):
        return parsed
    events = parsed.parent_events
    detected = detect_format(events)
    hint = format_hint if format_hint in FORMAT_LABELS else None
    fmt, reason = _resolve_format(detected, hint)
    if reason == "mismatch":
        return _format_mismatch_error(hint or "", detected)
    if fmt != "dsh":
        result = _apply_format(events, fmt, source_path=file_path)
    else:
        result = _convert_dsh_to_internal(
            events,
            source_path=file_path,
            child_event_lists=parsed.child_event_lists,
        )
    if "_error" in result:
        return result
    result["_source_path"] = file_path
    result["_source_sha256"] = source_sha
    return result


def load_trajectory(file_path: str, format_hint: str | None = None) -> dict:
    """Load a trajectory file via the content dispatcher.

    Reads the file once (sha256 of those exact bytes), sniffs JSON object vs
    event array vs JSONL, detects format, then converts. ``format_hint``
    forces a converter when detection is ``unknown`` (unmarked Claude dumps)
    and *requires* that format when detection already succeeded.

    DeepSeek Harness exports may be a ``session.jsonl``, a session directory
    containing that file plus ``subagents/``, or a zip of that layout.
    """
    if _path_ext(file_path) == ".log":
        return {"_error": "Unsupported file type: .log files are no longer supported."}

    file_path, path_error = resolve_dsh_session_path(file_path)
    if path_error:
        return {"_error": path_error}

    try:
        with open(file_path, "rb") as f:
            data = f.read()
        source_sha = _sha256_bytes(data)
    except OSError as exc:
        return {"_error": str(exc)}

    ext = _path_ext(file_path)
    if ext == ".zip" or (data.startswith(b"PK") and ext not in {".json", ".jsonl"}):
        return _load_zip_trajectory(file_path, format_hint, source_sha)

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return {"_error": str(exc)}

    payload, parse_error = _parse_trajectory_text(text, file_path)
    if parse_error:
        return {"_error": parse_error}

    payload = _normalize_payload(payload, file_path)
    detected = detect_format(payload)
    hint = format_hint if format_hint in FORMAT_LABELS else None
    fmt, reason = _resolve_format(detected, hint)
    if reason == "mismatch":
        return _format_mismatch_error(hint or "", detected)

    result = _apply_format(payload, fmt, source_path=file_path)
    if "_error" in result:
        return result
    result["_source_path"] = file_path
    # The displayed content's immutable identity — the sha256 of the EXACT
    # bytes parsed above (one read, one buffer): attribution requires the
    # canonical corpus file to still have these bytes at diagnosis time
    # (TOCTOU guard — never diagnose bytes the UI isn't showing).
    result["_source_sha256"] = source_sha
    return result


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
