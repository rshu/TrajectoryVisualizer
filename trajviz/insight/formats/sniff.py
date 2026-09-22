"""Object and event-stream format sniffing shared by parse and detect."""

from __future__ import annotations

_OBJECT_FORMATS = frozenset({"ccsession", "codearts", "cursor", "icode", "opencode"})
_EVENT_FORMATS = frozenset({"codex", "pi", "dsh"})
_FORMAT_STAMPS = {
    "ccsession": "_cc_format",
    "codearts": "_codearts_format",
    "cursor": "_cursor_format",
    "icode": "_icode_format",
    "codex": "_codex_format",
    "pi": "_pi_format",
    "dsh": "_dsh_format",
}
_CHRYS_EXPANDED_FORMAT = "chrys-expanded-session-v1"


def _looks_like_codex_jsonl(events: list) -> bool:
    """True when an event stream starts with Codex ``session_meta``."""
    return bool(events) and isinstance(events[0], dict) and events[0].get("type") == "session_meta"


def _looks_like_dsh_jsonl(events: list) -> bool:
    """True when an event stream is a DeepSeek Harness session log.

    DSH and Pi both lead with ``type: session`` plus a string ``id``. DSH
    headers carry ``createdAt`` (epoch ms) and/or ``delegationDepth`` /
    ``agentPreset`` / ``origin: subagent``; body events use slash-separated
    types (``user/message``, ``tool/call``).
    """
    if not events or not isinstance(events[0], dict):
        return False
    first = events[0]
    if first.get("type") != "session":
        return False
    ident = first.get("id")
    if not isinstance(ident, str) or not ident:
        return False
    created = first.get("createdAt")
    if isinstance(created, (int, float)) and not isinstance(created, bool):
        return True
    if "delegationDepth" in first or "agentPreset" in first:
        return True
    if first.get("origin") == "subagent":
        return True
    for event in events[1:12]:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if isinstance(etype, str) and "/" in etype:
            return True
    return False


def _looks_like_pi_jsonl(events: list) -> bool:
    """True when an event stream starts with a Pi ``session`` header.

    Distinct from Codex ``session_meta`` and DeepSeek Harness (checked first).
    Requires a non-empty string ``id`` so a generic ``{"type": "session"}``
    log line is not treated as Pi.
    """
    if not events or not isinstance(events[0], dict):
        return False
    first = events[0]
    if first.get("type") != "session":
        return False
    ident = first.get("id")
    return isinstance(ident, str) and bool(ident)


def _detect_event_stream_format(events: list) -> str:
    """Detect Codex / DSH / Pi from a parsed event array (JSONL or a JSON list)."""
    if not events or not isinstance(events[0], dict):
        return "unknown"
    if _looks_like_codex_jsonl(events):
        return "codex"
    # DSH before Pi: both lead with ``type: session`` + string ``id``.
    if _looks_like_dsh_jsonl(events):
        return "dsh"
    if _looks_like_pi_jsonl(events):
        return "pi"
    return "unknown"


def _looks_like_icode_object(raw: dict) -> bool:
    """True when a JSON object is an ICode / Chrys expanded session.

    Distinct from OpenCode ``info + messages``: ICode uses ``meta`` +
    ``state.messages`` with ``contents`` arrays, and stamps
    ``_chrys_export.format = chrys-expanded-session-v1``.
    """
    export = raw.get("_chrys_export")
    if isinstance(export, dict) and export.get("format") == _CHRYS_EXPANDED_FORMAT:
        return True
    meta = raw.get("meta")
    state = raw.get("state")
    if not isinstance(meta, dict) or not isinstance(state, dict):
        return False
    session_id = meta.get("session_id")
    messages = state.get("messages")
    if not (isinstance(session_id, str) and session_id and isinstance(messages, list)):
        return False
    for message in messages[:8]:
        if not isinstance(message, dict):
            continue
        contents = message.get("contents")
        if not isinstance(contents, list):
            continue
        if message.get("type") == "message":
            return True
        if message.get("role") in ("user", "assistant", "tool"):
            return True
    return False


def _detect_object_format(raw: dict) -> str:
    """Detect format of a parsed JSON object (raw export or already-converted)."""
    # Post-conversion markers: converters build/stamp a dict so a second pass
    # (UI gate, attribution, run-group) still reports the originating product.
    if raw.get("_cc_format") is True:
        return "ccsession"
    if raw.get("_codex_format") is True:
        return "codex"
    if raw.get("_pi_format") is True:
        return "pi"
    if raw.get("_dsh_format") is True:
        return "dsh"
    if raw.get("_icode_format") is True:
        return "icode"
    if raw.get("_cursor_format") is True:
        return "cursor"
    if raw.get("format") == "ccsession-trajectory":
        return "ccsession"
    # CodeArts exports use an OpenCode-compatible ``info + messages``
    # envelope.  Check their explicit export marker before the generic
    # OpenCode shape so the UI does not mislabel the originating product.
    export_metadata = raw.get("export_metadata")
    # Cursor consolidator exports also use info+messages; their marker must
    # win before the generic OpenCode shape (and before CodeArts, which has
    # its own source_format).
    if (
        isinstance(export_metadata, dict)
        and export_metadata.get("source_format") == "cursor_composer"
        and isinstance(raw.get("info"), dict)
        and isinstance(raw.get("messages"), list)
    ):
        return "cursor"
    if raw.get("_codearts_format") is True or (
        isinstance(export_metadata, dict)
        and export_metadata.get("schema_version") == 2
        and export_metadata.get("source_format") == "codearts_opencode_sqlite"
        and isinstance(raw.get("info"), dict)
        and isinstance(raw.get("messages"), list)
    ):
        return "codearts"
    # CodeArts legacy-JSON exports (codearts_consolidator.py emits
    # source_format "codearts_legacy_json") lack the OpenCode "info" dict but
    # are still CodeArts files — detect them so the UI's format-mismatch gate
    # and labeling work instead of falling through to "unknown".
    if (
        isinstance(export_metadata, dict)
        and export_metadata.get("schema_version") == 2
        and export_metadata.get("source_format") == "codearts_legacy_json"
        and isinstance(raw.get("messages"), list)
    ):
        return "codearts"
    # ICode before OpenCode: converted ICode files keep ``info + messages``
    # plus ``_chrys_export``, and must not be mislabeled as OpenCode.
    if _looks_like_icode_object(raw):
        return "icode"
    if isinstance(raw.get("info"), dict) and isinstance(raw.get("messages"), list):
        return "opencode"
    return "unknown"
