"""JSON / JSONL payload parse and normalize (no import of loaders)."""

import json
import os
from typing import Any

from .sniff import _OBJECT_FORMATS, _detect_object_format


def _path_ext(file_path: str) -> str:
    return os.path.splitext(file_path)[1].lower()

def _parse_jsonl_events(text: str) -> tuple[list | None, str | None]:
    """Parse newline-delimited JSON, tolerating a truncated final line.

    Interior decode errors are fatal. A missing newline on the last
    non-empty line is treated as an in-progress append and dropped.
    """
    events: list = []
    pending: tuple[int, str] | None = None
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
        if not raw_line.strip():
            continue
        # Delay one non-empty line so only the true final line can
        # receive the append-in-progress tolerance below.
        if pending is None:
            pending = (line_number, raw_line)
            continue
        pending_number, pending_line = pending
        try:
            events.append(json.loads(pending_line))
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSONL at line {pending_number}: {exc.msg}"
        pending = (line_number, raw_line)

    if pending is not None:
        pending_number, pending_line = pending
        try:
            events.append(json.loads(pending_line))
        except json.JSONDecodeError as exc:
            # Rollouts are append-only. A missing newline on the final
            # object is the signal that the writer may still be appending.
            if pending_line.endswith(("\n", "\r")):
                return None, f"Invalid JSONL at line {pending_number}: {exc.msg}"
    return events, None


def _parse_trajectory_text(text: str, file_path: str) -> tuple[Any, str | None]:
    """Sniff content: JSON object, JSON array, or JSONL event stream.

    A complete JSON document (object or array) wins even when the path
    ends in ``.jsonl``, so a Claude/OpenCode dump saved with the wrong
    extension still loads. Multiple JSON values (``Extra data``) or a
    ``.jsonl`` path that is not one JSON document fall through to the
    JSONL parser (including last-line truncation tolerance).
    """
    ext = _path_ext(file_path)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg == "Extra data" or ext == ".jsonl":
            events, err = _parse_jsonl_events(text)
            if err:
                return None, err
            return events, None
        return None, str(exc)

    if isinstance(raw, (dict, list)):
        return raw, None
    return None, (
        f"Unsupported JSON input: top-level value is {type(raw).__name__}; "
        "expected a JSON object or event array."
    )


def _is_event_record(raw: dict) -> bool:
    """True when a JSON object is a Codex/Pi/DSH event, not a trajectory dump."""
    t = raw.get("type")
    if t == "session_meta":
        return True
    return t == "session" and isinstance(raw.get("id"), str) and bool(raw["id"])


def _unwrap_singleton_object_payload(payload: Any) -> Any:
    """If JSONL parsed as a single object-format dict, treat it as that object.

    Happens when a Claude/OpenCode/CodeArts/Cursor/ICode dump is saved as ``.jsonl`` and
    a trailing truncated line forced the JSONL parser (``json.loads`` of
    the whole file then fails).
    """
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        if _detect_object_format(payload[0]) in _OBJECT_FORMATS:
            return payload[0]
    return payload


def _looks_like_object_dump(raw: dict) -> bool:
    """True when an unknown dict still looks like a JSON trajectory object.

    Unmarked Claude dumps (no ``format`` field) must stay objects so
    ``format_hint`` can force ``ccsession``, even if the file was saved as
    ``.jsonl``.
    """
    if isinstance(raw.get("trajectory"), list):
        return True
    if isinstance(raw.get("messages"), list):
        return True
    if isinstance(raw.get("info"), dict):
        return True
    if isinstance(raw.get("meta"), dict) and isinstance(raw.get("state"), dict):
        return True
    session = raw.get("session")
    return isinstance(session, dict) and raw.get("type") is None


def _normalize_payload(payload: Any, file_path: str) -> Any:
    """Canonicalize parsed content to either a trajectory object or an event list.

    A one-line ``.jsonl`` file is valid JSON, so ``json.loads`` returns a
    dict. If that dict is not an object-format dump, wrap it as a
    single-event stream (Codex/Pi, or unknown JSONL). A ``session_meta`` /
    Pi ``session`` object is wrapped even without the ``.jsonl`` suffix.
    """
    payload = _unwrap_singleton_object_payload(payload)
    if isinstance(payload, dict):
        if _detect_object_format(payload) != "unknown":
            return payload
        if _looks_like_object_dump(payload):
            return payload
        if _path_ext(file_path) == ".jsonl" or _is_event_record(payload):
            return [payload]
    return payload
