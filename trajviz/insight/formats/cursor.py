"""Cursor composer export (cursor_consolidator.py) → internal trajectory."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .common import safe_get

# JSONL / Composer names → TrajViz shared vocabulary.
_CURSOR_TOOL_NAMES = {
    "StrReplace": "Edit",
    "Shell": "Bash",
    "AwaitShell": "AwaitShell",
    "Task": "Task",
    "Read": "Read",
    "Write": "Write",
    "Grep": "Grep",
    "Glob": "Glob",
    "Delete": "Delete",
    "TodoWrite": "TodoWrite",
}


def _cursor_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _ms_to_iso(epoch_ms: Any) -> str:
    ms = _cursor_int(epoch_ms)
    if ms <= 0:
        return ""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()


def _normalize_tool_name(name: Any) -> str:
    raw = name if isinstance(name, str) and name else "?"
    return _CURSOR_TOOL_NAMES.get(raw, raw)


def _normalize_tool_input(name: str, payload: Any) -> dict:
    args = dict(payload) if isinstance(payload, dict) else {}
    if name in ("Read", "Write", "Edit") and "file_path" not in args:
        if "path" in args:
            args["file_path"] = args["path"]
        elif "filePath" in args:
            args["file_path"] = args["filePath"]
    if name == "Bash" and "command" not in args and "cmd" in args:
        args["command"] = args["cmd"]
    return args


def _normalize_parts(parts: list, *, session_id: str, message_id: str) -> list[dict]:
    normalized: list[dict] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                normalized.append({
                    "type": "text",
                    "text": text,
                    "id": part.get("id", ""),
                    "sessionID": part.get("sessionID", session_id),
                    "messageID": part.get("messageID", message_id),
                    "synthetic": bool(part.get("synthetic", False)),
                    "metadata": part.get("metadata") if isinstance(part.get("metadata"), dict) else {},
                })
            continue
        if ptype == "reasoning":
            normalized.append({
                "type": "reasoning",
                "text": part.get("text") or "",
                "time": part.get("time") if isinstance(part.get("time"), dict) else {},
                "id": part.get("id", ""),
                "sessionID": part.get("sessionID", session_id),
                "messageID": part.get("messageID", message_id),
            })
            continue
        if ptype not in ("tool", "tool_call"):
            normalized.append(part)
            continue
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        tool_name = _normalize_tool_name(
            part.get("tool") or part.get("tool_name") or part.get("name")
        )
        tool_input = _normalize_tool_input(
            tool_name, state.get("input", part.get("input", {})),
        )
        call_id = part.get("callID") or part.get("tool_id") or part.get("id") or ""
        status = state.get("status") or part.get("status") or "unknown"
        output = state.get("output", part.get("output", ""))
        metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
        new_state = dict(state)
        new_state["status"] = status
        new_state["input"] = tool_input
        new_state["output"] = output
        new_state["metadata"] = metadata
        normalized.append({
            "type": "tool",
            "tool": tool_name,
            "tool_name": tool_name,
            "callID": call_id,
            "tool_id": call_id,
            "state": new_state,
            "status": status,
            "input": tool_input,
            "output": output,
            "error": part.get("error") or state.get("error"),
            "id": part.get("id", call_id),
            "sessionID": part.get("sessionID", session_id),
            "messageID": part.get("messageID", message_id),
        })
    return normalized


def _convert_cursor_to_internal(raw: dict) -> dict:
    """Normalize a cursor_consolidator export into the shared step-model dict.

    Per-request billed tokens are **not** synthesized. Cursor stores a
    context-window occupancy snapshot (``info.promptTokenBreakdown`` and
    user-bubble ``tokens.context_window``). Per-step ``tokens.input/output/total``
    are consolidator estimates of logged text and tools, not API usage.
    """
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    messages = raw.get("messages") if isinstance(raw.get("messages"), list) else []
    export = raw.get("export_metadata") if isinstance(raw.get("export_metadata"), dict) else {}
    statistics = raw.get("statistics") if isinstance(raw.get("statistics"), dict) else {}
    sessions = raw.get("sessions") if isinstance(raw.get("sessions"), list) else []

    converted_messages: list[dict] = []
    user_count = 0
    asst_count = 0
    total_tool_calls = 0
    tool_breakdown: dict[str, int] = {}
    failed_tool_calls = 0
    sub_agent_ids: set[str] = set()
    token_totals = {"total": 0, "input": 0, "output": 0}
    has_step_timing = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        msg_info = message.get("info") if isinstance(message.get("info"), dict) else {}
        msg_info = dict(msg_info)
        role = msg_info.get("role") or message.get("role") or ""
        if role == "user":
            user_count += 1
        elif role == "assistant":
            asst_count += 1
        session_id = msg_info.get("sessionID") or info.get("id") or ""
        message_id = msg_info.get("id") or ""
        if msg_info.get("isSubAgent") and session_id:
            sub_agent_ids.add(str(session_id))
        tokens = msg_info.get("tokens")
        if isinstance(tokens, dict):
            token_totals["total"] += _cursor_int(tokens.get("total"))
            token_totals["input"] += _cursor_int(tokens.get("input"))
            token_totals["output"] += _cursor_int(tokens.get("output"))
        time_info = msg_info.get("time") if isinstance(msg_info.get("time"), dict) else {}
        if _cursor_int(time_info.get("created")) or _cursor_int(time_info.get("completed")):
            has_step_timing = True
        parts = _normalize_parts(
            message.get("parts") if isinstance(message.get("parts"), list) else [],
            session_id=str(session_id),
            message_id=str(message_id),
        )
        for part in parts:
            if part.get("type") != "tool":
                continue
            total_tool_calls += 1
            tname = str(part.get("tool") or "?")
            tool_breakdown[tname] = tool_breakdown.get(tname, 0) + 1
            status = str((part.get("state") or {}).get("status") or part.get("status") or "")
            if status.lower() in {"error", "failed", "failure"}:
                failed_tool_calls += 1
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            time_span = state.get("time") if isinstance(state.get("time"), dict) else {}
            if _cursor_int(time_span.get("start")) or _cursor_int(time_span.get("end")):
                has_step_timing = True
        converted_messages.append({"info": msg_info, "parts": parts})

    exported_sub = statistics.get("subagent_sessions", 0)
    if not isinstance(exported_sub, int):
        exported_sub = 0
    sub_agent_count = max(len(sub_agent_ids), exported_sub, max(0, len(sessions) - 1))

    created_ms = safe_get(info, "time", "created", default=0) or 0
    updated_ms = safe_get(info, "time", "updated", default=0) or created_ms
    started_at = _ms_to_iso(created_ms)
    finished_at = _ms_to_iso(updated_ms)
    duration_seconds = 0.0
    if isinstance(created_ms, (int, float)) and isinstance(updated_ms, (int, float)):
        if updated_ms > created_ms:
            duration_seconds = round((updated_ms - created_ms) / 1000.0, 3)

    breakdown = info.get("promptTokenBreakdown")
    if not isinstance(breakdown, dict):
        breakdown = None
    snapshot_total = _cursor_int((breakdown or {}).get("totalUsedTokens"))
    snapshot_limit = _cursor_int(info.get("contextTokenLimit")) or _cursor_int(
        (breakdown or {}).get("maxTokens")
    )
    summary = info.get("summary") if isinstance(info.get("summary"), dict) else {}

    raw["info"] = {
        **info,
        "id": info.get("id") or export.get("chat_id") or "",
        "title": info.get("title") or "",
        "directory": info.get("directory") or "",
        "time": {"created": created_ms or 0, "updated": updated_ms or 0},
    }
    raw["messages"] = converted_messages
    raw["metadata"] = {
        "session_id": raw["info"]["id"],
        "slug": "",
        "title": raw["info"]["title"],
        "directory": raw["info"]["directory"],
        "directory_name": str(raw["info"]["directory"]).replace("\\", "/").rsplit("/", 1)[-1],
        "agent": "cursor",
        "model": info.get("model") or "",
        "hostname": "",
        "platform": "",
        "python_version": "",
        "timestamp_utc": started_at,
        "branch": "",
        "generator_name": export.get("generator_name") or "cursor_consolidator",
        "generator_version": "",
        "format_version": str(export.get("schema_version") or 1),
        "sub_agent_count": sub_agent_count,
        "session_count": statistics.get("sessions") if isinstance(statistics.get("sessions"), int) else max(len(sessions), 1),
        "event_count": 0,
        "export_generated_at": export.get("generated_at", ""),
        "export_complete": export.get("complete"),
        "export_warnings": export.get("warnings", []),
        "token_semantics": export.get("token_semantics") or "context_window_snapshot+estimated_log_tokens",
        "context_snapshot": {
            "total_used_tokens": snapshot_total,
            "max_tokens": snapshot_limit,
            "usage_percent": info.get("contextUsagePercent"),
            "categories": (breakdown or {}).get("categories") if breakdown else [],
        },
    }
    raw["timing"] = {
        "total_duration": duration_seconds,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    raw["output"] = {
        "patch": "",
        "patch_length": 0,
        "patch_lines": 0,
        "has_patch": bool(summary.get("additions") or summary.get("deletions")),
        "error": None,
        "additions": summary.get("additions", 0) or 0,
        "deletions": summary.get("deletions", 0) or 0,
        "files_changed": summary.get("files", 0) or 0,
    }
    raw.setdefault("input", {"prompt": "", "prompt_length": 0})
    if token_totals["total"] or token_totals["input"] or token_totals["output"]:
        raw["token_usage"] = {
            "total_tokens": token_totals["total"],
            "prompt_tokens": token_totals["input"],
            "completion_tokens": token_totals["output"],
        }
    else:
        raw["token_usage"] = {}
    raw["stats"] = {
        "total_messages": user_count + asst_count,
        "user_messages": user_count,
        "assistant_messages": asst_count,
        "total_tool_calls": total_tool_calls,
        "tool_call_breakdown": tool_breakdown,
        "failed_tool_calls": failed_tool_calls,
        "reasoning_steps": 0,
        "sub_agent_count": sub_agent_count,
    }
    raw["_cursor_format"] = True
    raw["_source_format"] = "cursor"
    raw["_capabilities"] = {
        "has_timing": bool(duration_seconds or has_step_timing),
        "has_tool_calls": bool(total_tool_calls),
        "has_runtime_token_usage": False,
        "has_reasoning_content": False,
        "has_session_hierarchy": bool(sub_agent_count),
        "has_context_snapshot": bool(snapshot_total),
    }
    return raw
