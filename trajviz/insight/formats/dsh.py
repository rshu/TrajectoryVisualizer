"""DeepSeek Harness JSONL / zip export → internal trajectory."""

import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any

from .parse import _normalize_payload, _parse_trajectory_text

# DeepSeek Harness tools are lowercase with JSON-string arguments. Map them
# onto the Claude Code / OpenCode vocabulary so file-interaction charts,
# TodoWrite plan tracking, and spawn annotation work without extra aliases.
_DSH_TOOL_NAMES = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "todo_write": "TodoWrite",
    "subagent": "Agent",
    "subagent_fork": "Agent",
}


def _dsh_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _dsh_event_ts_ms(event: dict) -> int | None:
    """Epoch-ms from a DSH event envelope (``time`` / packed ``time0`` / header)."""
    for key in ("time", "time0", "createdAt"):
        value = event.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return int(value)
    return None


def _dsh_event_seq(event: dict) -> int | None:
    for key in ("seq", "seq0"):
        value = event.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value) and value == int(value):
            return int(value)
    return None


def _dsh_coerce_seed_length(value: Any) -> int:
    """Positive int from JSON ``seedLength`` (int, whole float, or numeric string)."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value):
            return _dsh_coerce_seed_length(int(value))
        return 0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            if any(marker in text.lower() for marker in (".", "e")):
                return _dsh_coerce_seed_length(float(text))
            return _dsh_coerce_seed_length(int(text))
        except ValueError:
            return 0
    return 0


def _dsh_drop_seed_prefix(events: list[dict], seed_length: Any) -> list[dict]:
    """Drop a forked child's inherited parent prefix.

    Persisted DSH child logs copy the parent's events and keep their original
    ``seq`` values; ``seedLength`` is the first seq that belongs to the child.
    Compact fixtures without seq numbers treat ``seedLength`` as a slice index.
    """
    seed_length = _dsh_coerce_seed_length(seed_length)
    if seed_length <= 0:
        return events
    seqs = [_dsh_event_seq(event) for event in events]
    if any(seq is not None for seq in seqs):
        return [event for event, seq in zip(events, seqs, strict=False) if seq is not None and seq >= seed_length]
    return events[seed_length:]


def _dsh_iter_content(content: Any):
    if isinstance(content, str):
        if content:
            yield {"type": "text", "text": content}
        return
    if isinstance(content, dict):
        yield content
        return
    if not isinstance(content, list):
        return
    for item in content:
        if isinstance(item, str):
            if item:
                yield {"type": "text", "text": item}
        elif isinstance(item, dict):
            yield item


def _dsh_blocks_text(content: Any) -> str:
    chunks: list[str] = []
    for item in _dsh_iter_content(content):
        ctype = item.get("type")
        if ctype in ("text", "reasoning"):
            text = item.get("text") or ""
            if text:
                chunks.append(str(text))
        elif ctype in ("tool-result", "tool_result"):
            nested = _dsh_blocks_text(item.get("content"))
            if nested:
                chunks.append(nested)
    return "\n".join(chunks)


def _dsh_parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw} if raw else {}
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    if raw is None:
        return {}
    return {"raw": raw}


def _dsh_normalize_tool(name: str, arguments: Any) -> tuple[str, dict]:
    raw_name = name or "?"
    canonical = _DSH_TOOL_NAMES.get(str(raw_name).lower(), raw_name)
    args = _dsh_parse_args(arguments)
    if canonical in ("Read", "Write", "Edit") and "file_path" not in args:
        if "path" in args:
            args["file_path"] = args["path"]
        elif "filePath" in args:
            args["file_path"] = args["filePath"]
    if canonical == "Bash" and "command" not in args and "cmd" in args:
        args["command"] = args["cmd"]
    return canonical, args


def _dsh_tool_part(name: Any, arguments: Any, call_id: str) -> dict:
    tool_name, tool_input = _dsh_normalize_tool(
        name if isinstance(name, str) and name else "?",
        arguments,
    )
    return {
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_id": call_id,
        "status": "pending",
        "input": tool_input,
        "output": "",
        "state": {
            "status": "pending",
            "input": tool_input,
            "output": "",
            "metadata": {},
        },
        "metadata": {},
    }


def _dsh_stamp_tool_start(part: dict, ts: int | None) -> None:
    if not ts:
        return
    state = part.setdefault("state", {})
    if not isinstance(state, dict):
        return
    time_info = state.setdefault("time", {})
    if isinstance(time_info, dict) and "start" not in time_info:
        time_info["start"] = ts


def _dsh_apply_tool_result(
    part: dict,
    output: str,
    is_error: bool,
    ts: int | None,
    *,
    session_id: str,
) -> None:
    status = "error" if is_error else "success"
    part["output"] = output
    part["status"] = status
    if is_error:
        part["error"] = output
    state = part.setdefault("state", {})
    if isinstance(state, dict):
        state["status"] = status
        state["output"] = output
        if ts:
            time_info = state.setdefault("time", {})
            if isinstance(time_info, dict):
                time_info["end"] = ts
    child_id = _dsh_child_session_id_from_output(output)
    if not child_id:
        return
    for meta in (
        part.setdefault("metadata", {}),
        part.get("state", {}).setdefault("metadata", {}) if isinstance(part.get("state"), dict) else {},
    ):
        if isinstance(meta, dict):
            meta["sessionId"] = child_id
            if session_id:
                meta["parentSessionId"] = session_id


def _dsh_extract_usage(usage: Any) -> dict | None:
    """Map DSH usage {inputTokens, outputTokens, cacheReadTokens, reasoningTokens}."""
    if not isinstance(usage, dict):
        return None
    inp = _dsh_int(usage.get("inputTokens", usage.get("input", 0)))
    out = _dsh_int(usage.get("outputTokens", usage.get("output", 0)))
    reasoning = _dsh_int(usage.get("reasoningTokens", usage.get("reasoning", 0)))
    cache_read = _dsh_int(usage.get("cacheReadTokens", usage.get("cacheRead", 0)))
    cache_write = _dsh_int(usage.get("cacheWriteTokens", usage.get("cacheWrite", 0)))
    component_total = inp + out + reasoning + cache_read + cache_write
    vendor_total = _dsh_int(usage.get("totalTokens", usage.get("total", 0)))
    total = max(component_total, vendor_total)
    if not any((total, inp, out, reasoning, cache_read, cache_write)):
        return None
    return {
        "total": total,
        "input": inp,
        "output": out,
        "reasoning": reasoning,
        "cache": {"read": cache_read, "write": cache_write},
    }


def _dsh_child_session_id_from_output(output: str) -> str:
    if not isinstance(output, str) or not output:
        return ""
    match = re.search(r"started subagent\s+(\S+)", output, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).rstrip(".,;")


def _dsh_is_human_user_source(source: Any) -> bool:
    """True when a ``user/message`` is a human prompt, not a runtime notice.

    DSH tags sandbox snapshots as ``plugin`` and background-child notices as
    ``subagent-report`` / ``subagent-settled``. Compact fixtures omit ``kind``.
    """
    if not isinstance(source, dict):
        return True
    kind = source.get("kind")
    if not isinstance(kind, str) or not kind:
        return True
    return kind == "user"


def _dsh_tool_result_payload(data: dict) -> tuple[str, str, bool]:
    """Return ``(call_id, output_text, is_error)`` from a ``tool/result`` event."""
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    call_id = ""
    source = message.get("source")
    if isinstance(source, dict) and source.get("callId"):
        call_id = str(source.get("callId"))
    is_error = bool(data.get("error"))
    chunks: list[str] = []
    for item in _dsh_iter_content(message.get("content")):
        if item.get("type") in ("tool-result", "tool_result"):
            if not call_id and item.get("toolCallId"):
                call_id = str(item.get("toolCallId"))
            if item.get("isError"):
                is_error = True
            text = _dsh_blocks_text(item.get("content"))
            if text:
                chunks.append(text)
        elif item.get("type") in ("text", "reasoning") and item.get("text"):
            chunks.append(str(item.get("text")))
    output = "\n".join(chunks)
    if not output and isinstance(data.get("error"), dict):
        err = data["error"]
        name = err.get("name") or err.get("code") or "error"
        output = str(name)
        is_error = True
    return call_id, output, is_error


def _dsh_session_to_messages(events: list[dict]) -> tuple[dict, list[dict], str, str, str]:
    """Convert one DSH session's events into internal messages.

    Returns ``(session_header, messages, last_model, last_provider, title)``.
    """
    session: dict = {}
    body: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "session" and not session:
            session = event
            continue
        body.append(event)
    body = _dsh_drop_seed_prefix(body, session.get("seedLength"))

    session_id = session.get("id", "") if isinstance(session.get("id"), str) else ""
    parent_session_id = session.get("parentSession", "") if isinstance(session.get("parentSession"), str) else ""
    is_sub_agent = session.get("origin") == "subagent" or bool(parent_session_id)
    agent = session.get("agentPreset") or ""
    cwd = session.get("cwd") or ""

    messages: list[dict] = []
    pending_tools: dict[str, dict] = {}
    early_calls: dict[str, dict] = {}
    current_model = ""
    current_provider = ""
    title = ""

    def _append(role: str, ts: int | None, parts: list, tokens: dict | None = None, extra: dict | None = None) -> None:
        info: dict[str, Any] = {
            "role": role,
            "time": {"created": ts or 0},
            "sessionID": session_id,
            "isSubAgent": is_sub_agent,
            "agent": agent,
        }
        if parent_session_id:
            info["parentSessionID"] = parent_session_id
        if cwd:
            info["path"] = {"cwd": cwd}
        if tokens:
            info["tokens"] = tokens
        if extra:
            info.update(extra)
        messages.append({"info": info, "parts": parts})

    for event in body:
        etype = event.get("type")
        ts = _dsh_event_ts_ms(event)
        data = event.get("data") if isinstance(event.get("data"), dict) else {}

        if etype == "session/title":
            candidate = data.get("title")
            if isinstance(candidate, str) and candidate:
                title = candidate
            continue
        if etype == "request/context":
            if data.get("model"):
                current_model = str(data.get("model"))
            if data.get("provider"):
                current_provider = str(data.get("provider"))
            continue
        if etype == "request/header":
            header = data.get("header") if isinstance(data.get("header"), dict) else {}
            config = header.get("config") if isinstance(header.get("config"), dict) else {}
            if config.get("model"):
                current_model = str(config.get("model"))
            if config.get("provider"):
                current_provider = str(config.get("provider"))
            continue
        if etype == "tool/call":
            call_id = str(data.get("callId") or "")
            if not call_id:
                continue
            part = pending_tools.get(call_id)
            if part is not None:
                _dsh_stamp_tool_start(part, ts)
            else:
                early_calls[call_id] = {
                    "name": data.get("name") or "?",
                    "arguments": data.get("arguments"),
                    "start": ts,
                }
            continue
        if etype == "tool/result":
            call_id, output, is_error = _dsh_tool_result_payload(data)
            part = pending_tools.pop(call_id, None) if call_id else None
            if part is None and call_id:
                early = early_calls.pop(call_id, None)
                if early:
                    part = _dsh_tool_part(early.get("name"), early.get("arguments"), call_id)
                    _dsh_stamp_tool_start(part, early.get("start"))
                    _dsh_apply_tool_result(
                        part,
                        output,
                        is_error,
                        ts,
                        session_id=session_id,
                    )
                    _append("assistant", ts, [part])
                    continue
            if part is not None:
                _dsh_apply_tool_result(
                    part,
                    output,
                    is_error,
                    ts,
                    session_id=session_id,
                )
            else:
                _append(
                    "assistant",
                    ts,
                    [
                        {
                            "type": "tool",
                            "tool_name": "?",
                            "tool_id": call_id,
                            "status": "error" if is_error else "success",
                            "input": {},
                            "output": output,
                            **({"error": output} if is_error else {}),
                        }
                    ],
                )
            continue
        if etype == "user/message":
            source = data.get("source") if isinstance(data.get("source"), dict) else {}
            if not _dsh_is_human_user_source(source):
                continue
            parts = []
            for item in _dsh_iter_content(data.get("content")):
                ctype = item.get("type")
                if ctype in ("text", "reasoning") and item.get("text"):
                    parts.append({"type": "text", "text": item.get("text", "")})
            if parts:
                extra = {"id": data.get("id") or ""}
                _append("user", ts, parts, extra=extra)
            continue
        if etype != "assistant/message":
            continue

        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        msg_source = message.get("source") if isinstance(message.get("source"), dict) else {}
        if msg_source.get("model"):
            current_model = str(msg_source.get("model"))
        if msg_source.get("provider"):
            current_provider = str(msg_source.get("provider"))
        parts = []
        for item in _dsh_iter_content(message.get("content")):
            ctype = item.get("type")
            if ctype == "reasoning":
                parts.append({"type": "reasoning", "text": item.get("text") or ""})
            elif ctype == "text":
                parts.append({"type": "text", "text": item.get("text") or ""})
            elif ctype in ("tool-call", "tool_call", "toolCall"):
                call_id = str(item.get("id") or "")
                part = _dsh_tool_part(item.get("name", "?"), item.get("arguments"), call_id)
                early = early_calls.pop(call_id, None) if call_id else None
                if early:
                    _dsh_stamp_tool_start(part, early.get("start"))
                parts.append(part)
                if call_id:
                    pending_tools[call_id] = part
        tokens = _dsh_extract_usage(data.get("usage"))
        extra = {
            "id": message.get("id") or "",
            "modelID": current_model,
            "providerID": current_provider,
        }
        if parts or tokens:
            _append("assistant", ts, parts, tokens=tokens, extra=extra)

    for part in pending_tools.values():
        if part.get("status") == "pending":
            part["status"] = "error"
            state = part.get("state")
            if isinstance(state, dict):
                state["status"] = "error"

    return session, messages, current_model, current_provider, title


_DSH_CHILD_WALK_MAX_DEPTH = 8


def _dsh_child_paths_in_subagents_dir(
    sub_root: str,
    *,
    _depth: int = 0,
) -> list[str]:
    """Collect ``<id>/session.jsonl`` under a ``subagents/`` directory.

    Walks nested ``<id>/subagents/`` trees (grandchildren) up to
    ``_DSH_CHILD_WALK_MAX_DEPTH`` so a parent export that stores deeper
    delegations still merges.
    """
    if _depth > _DSH_CHILD_WALK_MAX_DEPTH or not os.path.isdir(sub_root):
        return []
    paths: list[str] = []
    try:
        names = sorted(os.listdir(sub_root))
    except OSError:
        return []
    seen: set[str] = set()
    for name in names:
        if any(sep in name for sep in ("/", "\\")) or name in (".", ".."):
            continue
        child_dir = os.path.join(sub_root, name)
        session = os.path.join(child_dir, "session.jsonl")
        if os.path.isfile(session):
            real = os.path.realpath(session)
            if real not in seen:
                seen.add(real)
                paths.append(session)
        nested = os.path.join(child_dir, "subagents")
        for nested_path in _dsh_child_paths_in_subagents_dir(
            nested,
            _depth=_depth + 1,
        ):
            real = os.path.realpath(nested_path)
            if real not in seen:
                seen.add(real)
                paths.append(nested_path)
    return paths


def _dsh_sibling_child_paths(source_path: str | None) -> list[str]:
    if not source_path:
        return []
    parent_dir = source_path if os.path.isdir(source_path) else os.path.dirname(os.path.abspath(source_path))
    return _dsh_child_paths_in_subagents_dir(os.path.join(parent_dir, "subagents"))


def _dsh_safe_session_id(session_id: str) -> str:
    """Reject session ids that could escape a search root as a folder name."""
    if not isinstance(session_id, str) or not session_id:
        return ""
    if any(sep in session_id for sep in ("/", "\\")) or ".." in session_id:
        return ""
    if len(session_id) > 200:
        return ""
    return session_id


def _dsh_export_dir_names(session_id: str) -> list[str]:
    """Folder names used by DSH GUI / zip exports for one session id."""
    session_id = _dsh_safe_session_id(session_id)
    if not session_id:
        return []
    names = [f"dsh-session-{session_id}"]
    if session_id.startswith("session-") and len(session_id) > 8:
        names.append(f"dsh-session-{session_id[len('session-') :]}")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _dsh_jsonl_session_id(path: str) -> str:
    """Session id from the first JSONL line, or empty if it is not a DSH header."""
    try:
        with open(path, encoding="utf-8-sig") as handle:
            line = handle.readline()
        payload = json.loads(line)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return ""
    if isinstance(payload, dict) and payload.get("type") == "session":
        ident = payload.get("id")
        if isinstance(ident, str) and ident:
            return ident
    return ""


def _dsh_export_root_child_paths(session_id: str) -> list[str]:
    """Child logs under ``TRAJVIZ_DSH_EXPORT_ROOT`` only (same-machine / tests).

    A hosted dashboard never sees the uploader's home directory; do not crawl
    cwd or ``~/Downloads``. A bare ``root/subagents/`` tree is used only when
    ``root/session.jsonl`` belongs to *this* session id.
    """
    extra = os.environ.get("TRAJVIZ_DSH_EXPORT_ROOT")
    if not extra:
        return []
    root = os.path.abspath(os.path.expanduser(extra))
    if not os.path.isdir(root):
        return []
    header_id = _dsh_jsonl_session_id(os.path.join(root, "session.jsonl"))
    if header_id and header_id == session_id:
        found = _dsh_child_paths_in_subagents_dir(os.path.join(root, "subagents"))
        if found:
            return found
    for name in _dsh_export_dir_names(session_id):
        found = _dsh_child_paths_in_subagents_dir(os.path.join(root, name, "subagents"))
        if found:
            return found
    return []


def _dsh_discover_child_log_paths(
    session_id: str,
    source_path: str | None,
    *,
    allow_export_root: bool = True,
) -> list[str]:
    """Find ``subagents/<id>/session.jsonl`` next to the loaded file.

    Hosted uploads are a single temp file (or a zip). Children come from a
    sibling ``subagents/`` tree, from zip members, or from an explicit
    ``TRAJVIZ_DSH_EXPORT_ROOT`` on this host (parent logs only).
    """
    paths = _dsh_sibling_child_paths(source_path)
    if paths:
        return paths
    if allow_export_root:
        return _dsh_export_root_child_paths(session_id)
    return []


def _dsh_read_event_list(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8-sig") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return []
    payload, err = _parse_trajectory_text(text, path)
    if err:
        return []
    payload = _normalize_payload(payload, path)
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    return []


def _fill_message_completion_times(messages: list[dict]) -> None:
    """Set ``time.completed`` from the next message in the *same* session.

    Parent and child logs are merged then sorted by timestamp. Filling from
    the globally next row attributes a parallel child's start to the parent
    (and sibling children to each other).
    """
    by_session: dict[str, list[dict]] = {}
    for msg in messages:
        info = msg.get("info") if isinstance(msg.get("info"), dict) else {}
        sid = info.get("sessionID")
        key = sid if isinstance(sid, str) else ""
        by_session.setdefault(key, []).append(msg)
    for group in by_session.values():
        for i in range(len(group) - 1):
            cur_info = group[i].get("info")
            nxt_info = group[i + 1].get("info")
            if not isinstance(cur_info, dict) or not isinstance(nxt_info, dict):
                continue
            cur_time = cur_info.get("time") if isinstance(cur_info.get("time"), dict) else None
            nxt_time = nxt_info.get("time") if isinstance(nxt_info.get("time"), dict) else None
            if not cur_time or not nxt_time:
                continue
            cur_t = cur_time.get("created")
            nxt_t = nxt_time.get("created")
            if (
                isinstance(cur_t, (int, float))
                and not isinstance(cur_t, bool)
                and isinstance(nxt_t, (int, float))
                and not isinstance(nxt_t, bool)
                and nxt_t >= cur_t
            ):
                cur_time["completed"] = nxt_t


def _dsh_ms_to_iso(ms: int | None) -> str:
    if not isinstance(ms, int) or ms <= 0:
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _dsh_session_header(events: list[dict]) -> dict:
    for event in events:
        if isinstance(event, dict) and event.get("type") == "session":
            return event
    return {}


def _dsh_keep_descendant_children(
    parent_id: str,
    child_event_lists: list[list[dict]],
) -> list[list[dict]]:
    """Drop logs whose ``parentSession`` is not this session or a discovered sibling."""
    if not parent_id:
        return [events for events in child_event_lists if events]
    parsed: list[tuple[str, str, list[dict]]] = []
    for events in child_event_lists:
        if not events:
            continue
        header = _dsh_session_header(events)
        child_id = header.get("id") if isinstance(header.get("id"), str) else ""
        parent_session = header.get("parentSession") if isinstance(header.get("parentSession"), str) else ""
        parsed.append((child_id, parent_session, events))
    child_ids = {child_id for child_id, _, _ in parsed if child_id}
    allowed_parents = {parent_id} | child_ids
    kept: list[list[dict]] = []
    for _child_id, parent_session, events in parsed:
        if parent_session and parent_session not in allowed_parents:
            continue
        kept.append(events)
    return kept


def _dsh_message_sort_key(msg: dict) -> tuple[float, int]:
    info = msg.get("info") if isinstance(msg.get("info"), dict) else {}
    time_info = info.get("time") if isinstance(info.get("time"), dict) else {}
    created = time_info.get("created") or 0
    try:
        created_n = float(created)
    except (TypeError, ValueError):
        created_n = 0.0
    return (created_n, 1 if info.get("isSubAgent") else 0)


def _convert_dsh_to_internal(
    events: list[dict],
    *,
    source_path: str | None = None,
    child_event_lists: list[list[dict]] | None = None,
) -> dict:
    """Convert DeepSeek Harness JSONL events into the trajviz internal format.

    DSH emits newline-delimited JSON with a ``session`` header and slash-typed
    body events (``user/message``, ``assistant/message``, ``tool/call``,
    ``tool/result``). Streaming chunk rows are ignored in favour of the
    completed ``assistant/message`` (which carries usage). Child sessions live
    under a sibling ``subagents/<id>/session.jsonl`` directory (or the same
    layout inside an export zip, including nested ``subagents/<id>/subagents/``)
    and are merged when present.
    """
    session, messages, model, provider, title = _dsh_session_to_messages(events)
    session_id = session.get("id", "") if isinstance(session.get("id"), str) else ""
    if child_event_lists is None:
        origin = session.get("origin")
        parent_session = session.get("parentSession")
        is_child_log = origin == "subagent" or (isinstance(parent_session, str) and bool(parent_session))
        child_event_lists = [
            _dsh_read_event_list(path)
            for path in _dsh_discover_child_log_paths(
                session_id,
                source_path,
                allow_export_root=not is_child_log,
            )
        ]
    child_event_lists = _dsh_keep_descendant_children(session_id, child_event_lists)
    sub_agent_ids: set[str] = set()
    for child_events in child_event_lists:
        if not child_events:
            continue
        child_session, child_messages, child_model, child_provider, child_title = _dsh_session_to_messages(child_events)
        child_id = child_session.get("id")
        if isinstance(child_id, str) and child_id:
            if child_id in sub_agent_ids:
                continue
            sub_agent_ids.add(child_id)
        if child_model and not model:
            model = child_model
        if child_provider and not provider:
            provider = child_provider
        if child_title:
            for msg in child_messages:
                info = msg.get("info") if isinstance(msg.get("info"), dict) else None
                if info is not None and not info.get("sessionTitle"):
                    info["sessionTitle"] = child_title
        messages.extend(child_messages)

    messages.sort(key=_dsh_message_sort_key)
    _fill_message_completion_times(messages)

    directory = session.get("cwd", "") or ""
    start_ms = _dsh_int(session.get("createdAt"))
    if not start_ms:
        for msg in messages:
            created = msg.get("info", {}).get("time", {}).get("created")
            if isinstance(created, (int, float)) and created:
                start_ms = int(created)
                break
    end_ms = 0
    for event in reversed(events):
        if isinstance(event, dict):
            ts = _dsh_event_ts_ms(event)
            if ts:
                end_ms = ts
                break
    if messages:
        last_info = messages[-1].get("info") if isinstance(messages[-1].get("info"), dict) else {}
        last_time = last_info.get("time") if isinstance(last_info.get("time"), dict) else {}
        last_created = last_time.get("created") or 0
        last_completed = last_time.get("completed") or 0
        end_ms = max(end_ms, int(last_created or 0), int(last_completed or 0))
    total_duration = round((end_ms - start_ms) / 1000.0, 3) if start_ms and end_ms and end_ms >= start_ms else 0
    total_tokens = sum(
        (m["info"].get("tokens") or {}).get("total", 0) for m in messages if isinstance(m.get("info"), dict)
    )
    started_at = _dsh_ms_to_iso(start_ms or None)
    finished_at = _dsh_ms_to_iso(end_ms or None)
    version = session.get("version")
    version_str = "" if version is None else str(version)

    info = {
        "id": session_id,
        "slug": "",
        "projectID": "",
        "directory": directory,
        "title": title,
        "version": version_str,
        "time": {"created": start_ms or 0, "updated": end_ms or 0},
    }

    return {
        "info": info,
        "messages": messages,
        "metadata": {
            "session_id": session_id,
            "directory": directory,
            "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1] if directory else "",
            "agent": "dsh",
            "model": model,
            "source": "dsh",
            "model_provider": provider,
            "originator": "DeepSeek Harness",
            "server_version": version_str,
            "timestamp_utc": started_at,
            "title": title,
            "sub_agent_count": len(sub_agent_ids),
            "agent_preset": session.get("agentPreset") or "",
        },
        "timing": {
            "total_duration": total_duration,
            "started_at": started_at,
            "finished_at": finished_at,
        },
        "output": {},
        "input": {},
        "token_usage": {"total_tokens": total_tokens},
        "stats": {},
        "_dsh_format": True,
    }


_DSH_ZIP_MAX_MEMBER_BYTES = 32 * 1024 * 1024
_DSH_ZIP_MAX_TOTAL_BYTES = 128 * 1024 * 1024
_DSH_ZIP_MAX_CHILD_MEMBERS = 64


def _zip_normalize_member(name: str) -> str:
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    return name


def _zip_is_session_jsonl(name: str) -> bool:
    name = _zip_normalize_member(name).rstrip("/")
    return name == "session.jsonl" or name.endswith("/session.jsonl")


def _zip_subagent_suffix(name: str) -> str | None:
    """Path after a ``subagents`` directory component, or None if not a child log."""
    parts = [part for part in _zip_normalize_member(name).split("/") if part]
    try:
        idx = parts.index("subagents")
    except ValueError:
        return None
    rest = parts[idx + 1 :]
    return "/".join(rest) if rest else None


def _zip_nested_child_id(rest: str) -> str:
    """Child id from the path after the first ``subagents/`` component.

    Accepts ``<id>/session.jsonl`` and nested
    ``<id>/subagents/<id>/session.jsonl``. Rejects extra path segments
    (``<id>/extra/session.jsonl``) and ``..``.
    """
    parts = [part for part in rest.split("/") if part]
    if len(parts) < 2 or parts[-1] != "session.jsonl":
        return ""
    body = parts[:-1]
    if len(body) % 2 != 1:
        return ""
    for i, part in enumerate(body):
        if i % 2 == 1:
            if part != "subagents":
                return ""
        elif part in (".", "..", "subagents"):
            return ""
    return body[-1]


def _zip_dsh_members(names: list[str]) -> tuple[str | None, list[tuple[str, str]]]:
    """Pick the parent ``session.jsonl`` and ``subagents/<id>/session.jsonl`` members.

    Parent is the shortest path whose basename is exactly ``session.jsonl`` and
    that is not under a ``subagents`` directory. Children are
    ``subagents/<id>/session.jsonl`` (DSH GUI zip root),
    ``.../subagents/<id>/session.jsonl`` (folder zip), or nested
    ``.../subagents/<id>/subagents/<id>/session.jsonl``. Duplicate child ids
    keep the shorter path.
    """
    jsonl_members = [
        _zip_normalize_member(name) for name in names if _zip_is_session_jsonl(name) and not name.endswith("/")
    ]
    parents = [name for name in jsonl_members if _zip_subagent_suffix(name) is None]
    parents.sort(key=lambda name: (name.count("/"), len(name)))
    parent = parents[0] if parents else None
    by_id: dict[str, str] = {}
    for name in jsonl_members:
        rest = _zip_subagent_suffix(name)
        if not rest:
            continue
        child_id = _zip_nested_child_id(rest)
        if not child_id:
            continue
        prev = by_id.get(child_id)
        if prev is None or (name.count("/"), len(name)) < (prev.count("/"), len(name)):
            by_id[child_id] = name
    children = sorted(by_id.items(), key=lambda item: item[0])
    return parent, children


def _zip_member_over_budget(archive: zipfile.ZipFile, member: str, used: list[int]) -> bool:
    try:
        info = archive.getinfo(member)
    except KeyError:
        return True
    size = info.file_size
    if size < 0 or size > _DSH_ZIP_MAX_MEMBER_BYTES:
        return True
    if used[0] + size > _DSH_ZIP_MAX_TOTAL_BYTES:
        return True
    used[0] += size
    return False


def resolve_dsh_session_path(file_path: str) -> tuple[str, str | None]:
    """If *file_path* is a DSH session directory, return its ``session.jsonl``.

    Returns ``(path, error)``. Directories without ``session.jsonl`` error.
    """
    if os.path.isdir(file_path):
        nested = os.path.join(file_path, "session.jsonl")
        if os.path.isfile(nested):
            return nested, None
        return file_path, f"Directory has no session.jsonl: {file_path}"
    return file_path, None


def _parse_zip_member_events(archive: zipfile.ZipFile, member: str) -> list[dict]:
    try:
        raw = archive.read(member)
    except KeyError:
        return []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return []
    payload, err = _parse_trajectory_text(text, member)
    if err:
        return []
    payload = _normalize_payload(payload, member)
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    return []


@dataclass(frozen=True)
class DshZipContents:
    """Parent JSONL events and optional child-session event lists from a zip."""

    parent_events: list[dict]
    child_event_lists: list[list[dict]] | None


def parse_dsh_zip(file_path: str) -> DshZipContents | dict:
    """Open a DSH export zip and parse ``session.jsonl`` plus child members.

    Returns ``DshZipContents`` or ``{"_error": ...}``. Format detection and
    conversion stay in the loaders facade so this module does not import it.
    """
    try:
        archive = zipfile.ZipFile(file_path)
    except (OSError, zipfile.BadZipFile) as exc:
        return {"_error": f"Invalid zip archive: {exc}"}
    with archive:
        parent_member, child_members = _zip_dsh_members(archive.namelist())
        if not parent_member:
            return {"_error": "Zip archive has no session.jsonl."}
        budget = [0]
        if _zip_member_over_budget(archive, parent_member, budget):
            return {"_error": f"Zip member too large: {parent_member}"}
        events = _parse_zip_member_events(archive, parent_member)
        if not events:
            return {"_error": f"Could not parse {parent_member} from zip."}
        child_lists: list[list[dict]] | None
        if child_members:
            child_lists = []
            for _child_id, member in child_members[:_DSH_ZIP_MAX_CHILD_MEMBERS]:
                if _zip_member_over_budget(archive, member, budget):
                    continue
                child_lists.append(_parse_zip_member_events(archive, member))
        else:
            child_lists = None
    return DshZipContents(parent_events=events, child_event_lists=child_lists)
