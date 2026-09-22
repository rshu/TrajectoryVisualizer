"""Pi coding-agent JSONL → internal trajectory."""

import json
from typing import Any

from .common import _iso_to_epoch_ms

# Pi coding-agent tools use lowercase names and a `path` argument. Map them to
# the Claude Code / OpenCode vocabulary so file-interaction charts and write
# detection work without a second set of aliases.
_PI_TOOL_NAMES = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "grep": "Grep",
    "find": "Glob",
    "ls": "Glob",
}


def _pi_event_ts_ms(event: dict) -> int | None:
    """Prefer the event ISO timestamp; fall back to nested message epoch-ms."""
    ts = _iso_to_epoch_ms(event.get("timestamp") if isinstance(event.get("timestamp"), str) else None)
    if ts is not None:
        return ts
    msg = event.get("message")
    if isinstance(msg, dict):
        nested = msg.get("timestamp")
        if isinstance(nested, (int, float)) and not isinstance(nested, bool):
            value = int(nested)
            # Pi stores nested timestamps as epoch milliseconds.
            return value if value > 10**12 else value * 1000
        if isinstance(nested, str):
            return _iso_to_epoch_ms(nested)
    return None


def _pi_int(value: Any) -> int:
    """Coerce a Pi usage field to int; bools and non-numerics become 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _pi_extract_usage(usage: Any) -> dict | None:
    """Map Pi usage {input, output, cacheRead, cacheWrite, reasoning, totalTokens}.

    Pi's ``totalTokens`` is often input+output only (reasoning/cache omitted).
    Prefer the component sum so stacked token charts and the stored total agree.
    """
    if not isinstance(usage, dict):
        return None
    inp = _pi_int(usage.get("input", 0))
    out = _pi_int(usage.get("output", 0))
    reasoning = _pi_int(usage.get("reasoning", 0))
    cache_read = _pi_int(usage.get("cacheRead", 0))
    cache_write = _pi_int(usage.get("cacheWrite", 0))
    component_total = inp + out + reasoning + cache_read + cache_write
    vendor_total = _pi_int(usage.get("totalTokens"))
    total = max(component_total, vendor_total)
    if not any((total, inp, out, reasoning, cache_read, cache_write)):
        return None
    return {
        "total": total,
        "input": inp,
        "output": out,
        "reasoning": reasoning,
        "cache": {
            "read": cache_read,
            "write": cache_write,
        },
    }


def _pi_iter_content(content: Any):
    """Yield content dicts without treating a string as a character sequence."""
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


def _pi_content_text(content: Any) -> str:
    chunks: list[str] = []
    for item in _pi_iter_content(content):
        if item.get("type") == "text":
            chunks.append(str(item.get("text") or ""))
    return "\n".join(chunks)


def _pi_normalize_tool(name: str, arguments: Any) -> tuple[str, dict]:
    raw_name = name or "?"
    canonical = _PI_TOOL_NAMES.get(str(raw_name).lower(), raw_name)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    args = dict(arguments) if isinstance(arguments, dict) else ({} if arguments is None else {"raw": arguments})
    if canonical in ("Read", "Write", "Edit") and "file_path" not in args and "path" in args:
        args["file_path"] = args["path"]
    if canonical == "Bash" and "command" not in args and "cmd" in args:
        args["command"] = args["cmd"]
    return canonical, args


def _convert_pi_to_internal(events: list[dict]) -> dict:
    """Convert Pi coding-agent JSONL events into the trajviz internal format.

    Pi emits newline-delimited JSON with event types:
    - session: session ID, cwd, version
    - model_change / thinking_level_change: metadata (not steps)
    - message: nested {role: user|assistant|toolResult, content, usage, ...}

    Assistant ``toolCall`` parts are paired with later ``toolResult`` messages
    by ``toolCallId``.  The result matches the OpenCode internal message shape
    used by parse_steps().
    """
    session: dict = {}
    messages: list[dict] = []
    pending_tools: dict[str, dict] = {}
    current_model = ""
    current_provider = ""

    def _append(role: str, ts: int | None, parts: list, tokens: dict | None = None, extra: dict | None = None) -> None:
        info: dict[str, Any] = {
            "role": role,
            "time": {"created": ts or 0},
        }
        if tokens:
            info["tokens"] = tokens
        if extra:
            info.update(extra)
        messages.append({"info": info, "parts": parts})

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        ts = _pi_event_ts_ms(event)

        if etype == "session":
            session = event
            continue
        if etype == "model_change":
            if event.get("provider"):
                current_provider = str(event.get("provider"))
            if event.get("modelId"):
                current_model = str(event.get("modelId"))
            continue
        if etype in ("thinking_level_change",):
            continue
        if etype != "message":
            continue

        msg = event.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "user":
            parts = []
            for item in _pi_iter_content(msg.get("content")):
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
            if parts:
                _append("user", ts, parts, extra={"id": event.get("id") or ""})

        elif role == "assistant":
            if msg.get("model"):
                current_model = str(msg.get("model"))
            if msg.get("provider"):
                current_provider = str(msg.get("provider"))
            parts = []
            for item in _pi_iter_content(msg.get("content")):
                ctype = item.get("type")
                if ctype == "thinking":
                    text = item.get("thinking") or item.get("text") or ""
                    parts.append({"type": "reasoning", "text": text})
                elif ctype == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif ctype == "toolCall":
                    tool_name, tool_input = _pi_normalize_tool(
                        item.get("name", "?"),
                        item.get("arguments"),
                    )
                    call_id = item.get("id") or ""
                    part = {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "tool_id": call_id,
                        "status": "pending",
                        "input": tool_input,
                        "output": "",
                    }
                    parts.append(part)
                    if call_id:
                        pending_tools[call_id] = part
            error_message = msg.get("errorMessage")
            if error_message:
                parts.append({"type": "text", "text": f"[error] {error_message}"})
            tokens = _pi_extract_usage(msg.get("usage"))
            extra = {
                "id": event.get("id") or "",
                "modelID": current_model,
                "providerID": current_provider,
            }
            stop = msg.get("stopReason")
            if stop:
                extra["finish"] = stop
            if parts or tokens or stop == "error":
                if not parts and stop == "error":
                    parts = [{"type": "text", "text": "[error]"}]
                _append("assistant", ts, parts, tokens=tokens, extra=extra)

        elif role == "toolResult":
            call_id = msg.get("toolCallId") or ""
            output = _pi_content_text(msg.get("content"))
            is_error = bool(msg.get("isError"))
            status = "error" if is_error else "success"
            part = pending_tools.pop(call_id, None) if call_id else None
            if part is not None:
                part["output"] = output
                part["status"] = status
                if is_error:
                    part["error"] = output
            else:
                tool_name, _ = _pi_normalize_tool(msg.get("toolName") or "?", {})
                _append(
                    "assistant",
                    ts,
                    [
                        {
                            "type": "tool",
                            "tool_name": tool_name,
                            "tool_id": call_id,
                            "status": status,
                            "input": {},
                            "output": output,
                            **({"error": output} if is_error else {}),
                        }
                    ],
                )

    for part in pending_tools.values():
        if part.get("status") == "pending":
            part["status"] = "error"

    for i in range(len(messages) - 1):
        cur_t = messages[i]["info"]["time"].get("created")
        nxt_t = messages[i + 1]["info"]["time"].get("created")
        if isinstance(cur_t, (int, float)) and isinstance(nxt_t, (int, float)) and nxt_t >= cur_t:
            messages[i]["info"]["time"]["completed"] = nxt_t

    first_ts_iso = None
    last_ts_iso = None
    for event in events:
        if isinstance(event, dict) and isinstance(event.get("timestamp"), str):
            first_ts_iso = event["timestamp"]
            break
    for event in reversed(events):
        if isinstance(event, dict) and isinstance(event.get("timestamp"), str):
            last_ts_iso = event["timestamp"]
            break
    directory = session.get("cwd", "") or ""
    start_ms = _iso_to_epoch_ms(first_ts_iso)
    end_ms = _iso_to_epoch_ms(last_ts_iso)
    total_duration = (
        round((end_ms - start_ms) / 1000.0, 3) if isinstance(start_ms, int) and isinstance(end_ms, int) else 0
    )
    total_tokens = sum((m["info"].get("tokens") or {}).get("total", 0) for m in messages)
    last_model = current_model
    last_provider = current_provider

    info = {
        "id": session.get("id", ""),
        "slug": "",
        "projectID": "",
        "directory": directory,
        "title": "",
        "version": str(session.get("version", "") or ""),
        "time": {"created": start_ms or 0, "updated": end_ms or 0},
    }

    return {
        "info": info,
        "messages": messages,
        "metadata": {
            "session_id": session.get("id", ""),
            "directory": directory,
            "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1] if directory else "",
            "agent": "pi",
            "model": last_model,
            "source": "pi",
            "model_provider": last_provider,
            "originator": "Pi",
            "server_version": str(session.get("version", "") or ""),
            "timestamp_utc": first_ts_iso or "",
        },
        "timing": {
            "total_duration": total_duration,
            "started_at": first_ts_iso or "",
            "finished_at": last_ts_iso or "",
        },
        "output": {},
        "input": {},
        "token_usage": {"total_tokens": total_tokens},
        "stats": {},
        "_pi_format": True,
    }
