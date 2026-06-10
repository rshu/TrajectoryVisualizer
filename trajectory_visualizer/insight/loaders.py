"""Trajectory format detection and conversion (Claude Code, OpenCode, CodeArts, generic)."""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any


def safe_get(d: Any, *keys: Any, default: Any = None) -> Any:
    """Safe nested dict access."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def detect_format(raw: dict) -> str:
    """Detect trajectory format: 'ccsession', 'opencode', 'codearts', training, or 'unknown'."""
    # Post-conversion marker: the Claude Code converter builds a fresh dict and
    # sets this flag. Check it before the raw-format markers below so already-
    # converted trajectories still report as ccsession.
    if raw.get("_cc_format") is True:
        return "ccsession"
    if raw.get("format") == "ccsession-trajectory":
        return "ccsession"
    if raw.get("format") == "codearts":
        return "codearts"
    if isinstance(raw.get("info"), dict) and isinstance(raw.get("messages"), list):
        return "opencode"
    if _looks_like_training_conversation(raw):
        return "training_conversation"
    return "unknown"


def _looks_like_training_conversation(raw: dict) -> bool:
    """Return True when a payload looks like a simple training conversation JSON."""
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    seen_roles = set()
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        role = msg.get("role")
        if not isinstance(role, str) or not role:
            return False
        seen_roles.add(role)
    return bool(seen_roles & {"user", "assistant"})


def _estimate_message_tokens(content: str, reasoning_content: str = "") -> dict:
    """Return rough token estimates from content lengths without extra deps."""
    content_chars = len(content or "")
    reasoning_chars = len(reasoning_content or "")
    content_tokens = max(0, (content_chars + 3) // 4)
    reasoning_tokens = max(0, (reasoning_chars + 3) // 4)
    return {
        "estimated": True,
        "estimated_content_chars": content_chars,
        "estimated_reasoning_chars": reasoning_chars,
        "estimated_content_tokens": content_tokens,
        "estimated_reasoning_tokens": reasoning_tokens,
        "estimated_total": content_tokens + reasoning_tokens,
    }


def _convert_training_tool_calls(tool_calls: Any) -> list[dict]:
    """Convert OpenAI-style training tool calls into internal part records."""
    if not isinstance(tool_calls, list):
        return []
    out: list[dict] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            function = {}
        name = function.get("name") or call.get("name") or call.get("tool_name") or "tool"
        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments", call.get("input", {}))
        out.append({
            "type": "tool_call",
            "tool_name": str(name),
            "tool_id": str(call.get("id", call.get("tool_call_id", ""))),
            "status": "completed",
            "input": arguments,
            "output": "",
            "metadata": {},
        })
    return out


def _infer_training_scaffold(messages: list[dict]) -> dict:
    """Infer a likely training scaffold from system prompt and tool names."""
    system_prompt = ""
    tool_names: list[str] = []
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        system_prompt = str(messages[0].get("content", "") or "")
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                tool_names.append(str(fn["name"]))
    prompt_lower = system_prompt.lower()
    tool_set = set(tool_names)

    if "openhands" in prompt_lower:
        return {"id": "openhands", "label": "OpenHands", "confidence": "high"}
    if "claude agent sdk" in prompt_lower or "you are a claude agent" in prompt_lower:
        return {"id": "claude_agent_sdk", "label": "Claude Agent SDK", "confidence": "high"}
    if "<response>" in system_prompt and "<commands>" in system_prompt:
        return {"id": "xml_command_line", "label": "XML Command Scaffold", "confidence": "high"}
    if "claude code" in prompt_lower:
        return {"id": "claude_code_like", "label": "Claude Code-like", "confidence": "medium"}
    if "codex" in prompt_lower:
        return {"id": "codex_like", "label": "Codex-like", "confidence": "medium"}
    if tool_set & {"TodoWrite", "Read", "Grep", "Edit", "Glob", "BashOutput", "Task", "Bash"}:
        return {"id": "claude_code_like", "label": "Claude Code-like", "confidence": "medium"}
    if tool_names:
        return {"id": "generic_tool_calling", "label": "Generic Tool Calling", "confidence": "medium"}
    return {"id": "plain_conversation", "label": "Plain Conversation", "confidence": "high"}


def _convert_training_conversation_to_internal(raw: dict) -> dict:
    """Normalize training conversation JSON without trusting meta_info fields."""
    messages_out = []
    has_reasoning = False
    has_tool_calls = False
    source_messages = raw.get("messages", [])
    scaffold = _infer_training_scaffold(source_messages if isinstance(source_messages, list) else [])
    for idx, msg in enumerate(source_messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        if not role:
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        reasoning_content = msg.get("reasoning_content", "")
        if not isinstance(reasoning_content, str):
            reasoning_content = str(reasoning_content)
        if reasoning_content:
            has_reasoning = True
        tool_call_parts = _convert_training_tool_calls(msg.get("tool_calls"))
        if tool_call_parts or role == "tool":
            has_tool_calls = True
        messages_out.append({
            "index": idx,
            "role": role,
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_call_parts": tool_call_parts,
            "tool_call_id": str(msg.get("tool_call_id", "")),
            "tokens_estimate": _estimate_message_tokens(content, reasoning_content),
        })

    return {
        "format": "training-conversation",
        "messages": messages_out,
        "_analysis_profile": "training",
        "_source_format": "openai_conversations",
        "_training_scaffold": scaffold,
        "_capabilities": {
            "has_timing": False,
            "has_tool_calls": has_tool_calls,
            "has_runtime_token_usage": False,
            "has_reasoning_content": has_reasoning,
            "has_training_labels": False,
        },
    }


def _iso_to_epoch_ms(iso_str: str | None) -> int | None:
    """Convert ISO 8601 string to epoch milliseconds."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def _cc_extract_usage(usage: dict | None) -> dict:
    """Extract token usage from a Claude Code message's usage field."""
    if not isinstance(usage, dict):
        return {"total": 0, "input": 0, "output": 0, "reasoning": 0,
                "cache": {"read": 0, "write": 0}}
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
    # cache_write overlaps with input_tokens in Claude API — don't double-count
    total = inp + out + cache_read
    return {
        "total": total, "input": inp, "output": out, "reasoning": 0,
        "cache": {"read": cache_read, "write": cache_write},
    }


def _cc_content_to_parts(content: list, tool_result_map: dict | None = None) -> list:
    """Convert Claude Code content[] items to internal parts format."""
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        ctype = item.get("type", "")

        if ctype == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif ctype == "thinking":
            parts.append({"type": "reasoning", "text": item.get("text", "")})
        elif ctype == "tool_call" or ctype == "tool_use":
            tool_id = item.get("id", "")
            tool_name = item.get("name", "?")
            tool_input = item.get("input", {})
            # Check if we already have a result for this tool call
            result_info = (tool_result_map or {}).get(tool_id, {})
            status = result_info.get("status", "?")
            output = result_info.get("output", "")
            error = result_info.get("error")
            tool_exec = result_info.get("tool_execution")
            metadata = dict(tool_exec) if isinstance(tool_exec, dict) else {}
            parts.append({
                "type": "tool_call",
                "tool_name": tool_name,
                "tool_id": tool_id,
                "status": status,
                "title": item.get("caller", {}).get("type", "") if isinstance(item.get("caller"), dict) else "",
                "input": tool_input,
                "output": output,
                "error": error,
                "time_start": None,
                "time_end": None,
                "metadata": metadata,
            })
        elif ctype == "tool_result":
            # These are handled via tool_result_map; skip as standalone parts
            pass
        else:
            parts.append({"type": ctype, "raw": item})
    return parts


def _cc_build_tool_result_map(entries: list) -> dict:
    """Build a map of tool_use_id -> {output, status, error} from all user messages."""
    result_map: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "")
        content = entry.get("content", [])
        if not isinstance(content, list):
            continue

        # Direct user messages with tool_result
        if role == "user":
            tool_exec = entry.get("tool_execution")
            if not isinstance(tool_exec, dict):
                tool_exec = None
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    tool_id = item.get("tool_use_id", "") or item.get("tool_call_id", "")
                    is_error = item.get("is_error", False)
                    result_content = item.get("output", item.get("content", ""))
                    if tool_id:
                        result_map[tool_id] = {
                            "output": result_content,
                            "status": "error" if is_error else "success",
                            "error": result_content if is_error else None,
                            "tool_execution": tool_exec,
                        }

        # Events with nested user messages containing tool_result
        if role == "event":
            data = entry.get("data", {})
            if isinstance(data, dict):
                msg = data.get("message", {})
                if isinstance(msg, dict) and msg.get("type") == "user":
                    inner_msg = msg.get("message", {})
                    if isinstance(inner_msg, dict):
                        inner_tool_exec = inner_msg.get("tool_execution")
                        if not isinstance(inner_tool_exec, dict):
                            inner_tool_exec = None
                        inner_content = inner_msg.get("content", [])
                        if isinstance(inner_content, list):
                            for item in inner_content:
                                if isinstance(item, dict) and item.get("type") == "tool_result":
                                    tool_id = item.get("tool_use_id", "") or item.get("tool_call_id", "")
                                    is_error = item.get("is_error", False)
                                    result_content = item.get("output", item.get("content", ""))
                                    if tool_id:
                                        result_map[tool_id] = {
                                            "output": result_content,
                                            "status": "error" if is_error else "success",
                                            "error": result_content if is_error else None,
                                            "tool_execution": inner_tool_exec,
                                        }
    return result_map


def _cc_group_by_message_id(entries: list[dict]) -> list[dict]:
    """Group trajectory entries that share the same message_id into merged entries.

    Claude Code splits a single API response into multiple trajectory entries
    (one per content block: thinking, text, tool_call). These share the same
    message_id and request_id. We merge them to avoid double-counting tokens.

    The merged entry uses:
    - Content: concatenated from all entries (preserving order)
    - Usage: from the LAST entry (has final/cumulative output_tokens)
    - Timestamp: from the FIRST entry
    - stop_reason: from the LAST entry (only the final chunk has the real stop reason)
    - Other fields: from the first entry
    """
    if not entries:
        return []

    # Separate entries with and without message_id
    grouped: dict[str, list[dict]] = {}
    no_msgid: list[dict] = []
    order: list[str] = []  # Track insertion order of message_ids

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = entry.get("message_id", "")
        role = entry.get("role", "")
        if mid and role in ("user", "assistant"):
            if mid not in grouped:
                grouped[mid] = []
                order.append(mid)
            grouped[mid].append(entry)
        else:
            no_msgid.append(entry)

    # Merge groups
    merged: list[dict] = []
    for mid in order:
        group = grouped[mid]
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Sort by index to preserve order
        group.sort(key=lambda e: e.get("index", 0))
        first = group[0]
        last = group[-1]

        # Merge content arrays
        all_content = []
        for entry in group:
            content = entry.get("content", [])
            if isinstance(content, list):
                all_content.extend(content)

        merged_entry = {**first}
        merged_entry["content"] = all_content
        # Use last entry's usage (has cumulative/final output_tokens)
        merged_entry["usage"] = last.get("usage", first.get("usage"))
        # Use last entry's stop_reason (intermediate chunks have null)
        merged_entry["stop_reason"] = last.get("stop_reason") or first.get("stop_reason")
        merged.append(merged_entry)

    # Interleave back with non-message-id entries in original order
    # Use the index field to maintain ordering
    result = merged + no_msgid
    result.sort(key=lambda e: e.get("index", 0))
    return result


def _cc_convert_entry_to_step(entry: dict, tool_result_map: dict,
                              *, agent_id: str = "") -> dict | None:
    """Convert a single Claude Code trajectory entry to internal step format."""
    role = entry.get("role", "")
    if role not in ("user", "assistant"):
        return None

    content = entry.get("content", [])
    if not isinstance(content, list):
        content = []

    # Skip user messages that contain only tool_result items — their content
    # is already captured in the corresponding tool_call's output field.
    if role == "user" and content:
        has_non_result = any(
            isinstance(item, dict) and item.get("type") != "tool_result"
            for item in content
        )
        if not has_non_result:
            return None

    usage = _cc_extract_usage(entry.get("usage"))
    timestamp_ms = _iso_to_epoch_ms(entry.get("timestamp"))

    parts = _cc_content_to_parts(content, tool_result_map)

    # Compute derived fields
    tool_calls = [p for p in parts if p.get("type") == "tool_call"]
    errors = sum(1 for tc in tool_calls if tc.get("status") == "error")
    has_reasoning = any(p.get("type") == "reasoning" for p in parts)
    text_preview = ""
    for p in parts:
        if p.get("type") == "text" and p.get("text"):
            text_preview = p["text"]
            break
        if p.get("type") == "reasoning" and p.get("text") and not text_preview:
            text_preview = p["text"]
        if p.get("type") == "tool_call" and not text_preview:
            text_preview = f"[Tool: {p['tool_name']}]"

    model = entry.get("model", "")
    finish = entry.get("stop_reason", "")

    return {
        "role": role,
        "tokens": {
            "total": usage["total"],
            "input": usage["input"],
            "output": usage["output"],
            "reasoning": usage["reasoning"],
            "cache_read": usage["cache"]["read"],
            "cache_write": usage["cache"]["write"],
        },
        "duration": None,  # Will be computed from timestamps
        "parts": parts,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "error_count": errors,
        "has_reasoning": has_reasoning,
        "text_preview": text_preview,
        "finish": finish or "",
        "model_id": model,
        "provider_id": "",
        "time_created_ms": timestamp_ms,
        "time_completed_ms": None,
        "agent": agent_id or "",
        "mode": "",
        "message_id": entry.get("message_id", entry.get("uuid", "")),
        "id": entry.get("uuid", ""),
        "parent_id": entry.get("parent_uuid", ""),
        "session_id": "",
        "cwd": entry.get("cwd", ""),
        "root": "",
    }


def _cc_flatten_events(entries: list, tool_result_map: dict) -> list[dict]:
    """Flatten event entries with nested sub-agent messages into steps."""
    steps = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") != "event":
            continue
        event_type = entry.get("event_type", "")
        if event_type != "progress":
            continue

        data = entry.get("data", {})
        if not isinstance(data, dict):
            continue

        agent_id = data.get("agentId", "")
        msg = data.get("message", {})
        if not isinstance(msg, dict):
            continue

        msg_type = msg.get("type", "")
        inner_msg = msg.get("message", {})
        if not isinstance(inner_msg, dict):
            continue

        # Map event message type to role
        if msg_type == "assistant":
            inner_content = inner_msg.get("content", [])
            if not isinstance(inner_content, list):
                continue
            usage = _cc_extract_usage(inner_msg.get("usage"))
            timestamp_ms = _iso_to_epoch_ms(msg.get("timestamp") or entry.get("timestamp"))

            parts = _cc_content_to_parts(inner_content, tool_result_map)
            tool_calls = [p for p in parts if p.get("type") == "tool_call"]
            errors = sum(1 for tc in tool_calls if tc.get("status") == "error")
            has_reasoning = any(p.get("type") == "reasoning" for p in parts)
            text_preview = ""
            for p in parts:
                if p.get("type") in ("text", "reasoning") and p.get("text"):
                    text_preview = p["text"]
                    break
                if p.get("type") == "tool_call" and not text_preview:
                    text_preview = f"[Tool: {p['tool_name']}]"

            model = inner_msg.get("model", "")
            finish = inner_msg.get("stop_reason", "")

            steps.append({
                "role": "assistant",
                "tokens": {
                    "total": usage["total"],
                    "input": usage["input"],
                    "output": usage["output"],
                    "reasoning": usage["reasoning"],
                    "cache_read": usage["cache"]["read"],
                    "cache_write": usage["cache"]["write"],
                },
                "duration": None,
                "parts": parts,
                "tool_calls": tool_calls,
                "tool_call_count": len(tool_calls),
                "error_count": errors,
                "has_reasoning": has_reasoning,
                "text_preview": text_preview,
                "finish": finish or "",
                "model_id": model,
                "provider_id": "",
                "time_created_ms": timestamp_ms,
                "time_completed_ms": None,
                "agent": agent_id,
                "mode": "",
                "message_id": inner_msg.get("id", ""),
                "id": msg.get("uuid", entry.get("uuid", "")),
                "parent_id": entry.get("parent_uuid", ""),
                "session_id": "",
                "cwd": entry.get("cwd", ""),
                "root": "",
            })
        elif msg_type == "user":
            # User messages in events are typically tool results — already handled via
            # tool_result_map. But if they have text content, include them.
            inner_content = inner_msg.get("content", [])
            if not isinstance(inner_content, list):
                continue
            has_text = any(
                isinstance(item, dict) and item.get("type") == "text"
                for item in inner_content
            )
            if has_text:
                timestamp_ms = _iso_to_epoch_ms(msg.get("timestamp") or entry.get("timestamp"))
                parts = _cc_content_to_parts(inner_content, tool_result_map)
                steps.append({
                    "role": "user",
                    "tokens": {"total": 0, "input": 0, "output": 0, "reasoning": 0,
                               "cache_read": 0, "cache_write": 0},
                    "duration": None,
                    "parts": parts,
                    "tool_calls": [],
                    "tool_call_count": 0,
                    "error_count": 0,
                    "has_reasoning": False,
                    "text_preview": next(
                        (p["text"] for p in parts if p.get("type") == "text" and p.get("text")),
                        ""
                    ),
                    "finish": "",
                    "model_id": "",
                    "provider_id": "",
                    "time_created_ms": timestamp_ms,
                    "time_completed_ms": None,
                    "agent": agent_id,
                    "mode": "",
                    "message_id": "",
                    "id": msg.get("uuid", entry.get("uuid", "")),
                    "parent_id": entry.get("parent_uuid", ""),
                    "session_id": "",
                    "cwd": entry.get("cwd", ""),
                    "root": "",
                })

    return steps


def _cc_convert_sub_agent_trajectory(sub_agent: dict) -> list[dict]:
    """Convert a sub_agents[] entry's full trajectory into internal steps.

    The sub_agents[] top-level array contains complete sub-agent sessions with
    their own trajectory arrays — richer than the inline progress events.
    """
    agent_id = sub_agent.get("agent_id", "")
    sa_trajectory = sub_agent.get("trajectory", [])
    if not isinstance(sa_trajectory, list):
        return []

    # Merge entries that share the same message_id before processing
    sa_trajectory = _cc_group_by_message_id(sa_trajectory)

    # Build tool result map from sub-agent's own trajectory
    tool_result_map = _cc_build_tool_result_map(sa_trajectory)

    steps = []
    for entry in sa_trajectory:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "")
        if role in ("user", "assistant"):
            step = _cc_convert_entry_to_step(entry, tool_result_map, agent_id=agent_id)
            if step is not None:
                steps.append(step)
    return steps


def _convert_claude_code_to_internal(raw: dict) -> dict:
    """Convert Claude Code (ccsession-trajectory) format to internal format.

    Transforms the raw JSON into the structure expected by parse_steps() and
    compute_metrics(), so downstream code works unchanged.
    """
    import logging
    logger = logging.getLogger(__name__)

    # --- Format version check ---
    fmt_version = raw.get("format_version", "")
    if fmt_version and fmt_version != "1.0":
        logger.warning("Claude Code trajectory format_version %s (expected 1.0) — "
                        "some fields may not parse correctly", fmt_version)

    session = raw.get("session", {}) if isinstance(raw.get("session"), dict) else {}
    stats_raw = raw.get("statistics", {}) if isinstance(raw.get("statistics"), dict) else {}
    trajectory = raw.get("trajectory", []) if isinstance(raw.get("trajectory"), list) else []
    sub_agents_raw = raw.get("sub_agents", []) if isinstance(raw.get("sub_agents"), list) else []

    # --- Metadata ---
    started_at = session.get("started_at", "")
    ended_at = session.get("ended_at", "")
    models_used = session.get("models_used", [])
    model = models_used[0] if isinstance(models_used, list) and models_used else None
    generator = raw.get("generator", {}) if isinstance(raw.get("generator"), dict) else {}

    metadata = {
        "session_id": session.get("id", ""),
        "slug": session.get("slug", ""),
        "directory": session.get("working_directory", ""),
        "directory_name": os.path.basename(session.get("working_directory", "")),
        "agent": "claude-code",
        "model": model,
        "hostname": "",
        "platform": session.get("platform", ""),
        "python_version": "",
        "timestamp_utc": started_at,
        "branch": session.get("git_branch", ""),
        "ground_truth_patch": "",
        "baseline_commit": "",
        "sanitized": False,
        "server_url": "",
        "server_version": session.get("claude_code_version", ""),
        "generator_name": generator.get("name", ""),
        "generator_version": generator.get("version", ""),
        "format_version": fmt_version,
        "sub_agent_count": stats_raw.get("sub_agent_count", 0),
        "event_count": stats_raw.get("events", 0),
    }

    # --- Timing ---
    duration_seconds = session.get("duration_seconds")
    if duration_seconds is None and started_at and ended_at:
        start_ms = _iso_to_epoch_ms(started_at)
        end_ms = _iso_to_epoch_ms(ended_at)
        if start_ms is not None and end_ms is not None:
            duration_seconds = (end_ms - start_ms) / 1000.0
    timing = {
        "total_duration": round(duration_seconds, 3) if duration_seconds else 0,
        "started_at": started_at,
        "finished_at": ended_at,
    }

    # --- Stats & token usage ---
    tokens_raw = stats_raw.get("tokens", {}) if isinstance(stats_raw.get("tokens"), dict) else {}
    token_usage = {
        "total_tokens": (
            (tokens_raw.get("input", 0) or 0)
            + (tokens_raw.get("output", 0) or 0)
            + (tokens_raw.get("cache_read", 0) or 0)
            + (tokens_raw.get("cache_creation", 0) or 0)
        ),
        "prompt_tokens": tokens_raw.get("input", 0) or 0,
        "completion_tokens": tokens_raw.get("output", 0) or 0,
    }
    stats = {
        "total_messages": stats_raw.get("turns", 0),
        "user_messages": stats_raw.get("user_turns", 0),
        "assistant_messages": stats_raw.get("assistant_turns", 0),
        "total_tool_calls": stats_raw.get("tool_calls", 0),
        "tool_call_breakdown": stats_raw.get("tool_calls_by_name", {}),
        "failed_tool_calls": 0,
        "reasoning_steps": stats_raw.get("assistant_turns", 0),
    }

    # --- Convert main trajectory entries ---
    # Merge entries that share the same message_id to avoid double-counting tokens
    merged_trajectory = _cc_group_by_message_id(trajectory)

    # Build tool result map from merged trajectory
    tool_result_map = _cc_build_tool_result_map(merged_trajectory)

    converted_trajectory = []
    for entry in merged_trajectory:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "")
        if role in ("user", "assistant"):
            step = _cc_convert_entry_to_step(entry, tool_result_map)
            if step is not None:
                converted_trajectory.append(step)

    # --- Convert sub-agent trajectories from sub_agents[] ---
    # Prefer the full sub_agents[] trajectories over inline progress events,
    # as they contain complete message content, token usage, and timing.
    sub_agent_ids_processed = set()
    for sa in sub_agents_raw:
        if not isinstance(sa, dict):
            continue
        agent_id = sa.get("agent_id", "")
        sa_steps = _cc_convert_sub_agent_trajectory(sa)
        converted_trajectory.extend(sa_steps)
        if agent_id:
            sub_agent_ids_processed.add(agent_id)

    # Fallback: flatten inline progress events for any sub-agents NOT in sub_agents[].
    # This handles cases where progress events exist but sub_agents[] is missing.
    if not sub_agents_raw:
        event_steps = _cc_flatten_events(merged_trajectory, tool_result_map)
        converted_trajectory.extend(event_steps)
    else:
        # Only flatten events for agents not already processed via sub_agents[]
        event_steps = _cc_flatten_events(merged_trajectory, tool_result_map)
        for step in event_steps:
            if step.get("agent", "") not in sub_agent_ids_processed:
                converted_trajectory.append(step)

    # Sort by timestamp to maintain chronological order
    def _sort_key(step):
        ts = step.get("time_created_ms")
        return ts if isinstance(ts, (int, float)) else 0
    converted_trajectory.sort(key=_sort_key)

    # Assign indices and compute durations from consecutive timestamps
    for idx, step in enumerate(converted_trajectory):
        step["index"] = idx
    for i in range(len(converted_trajectory) - 1):
        t1 = converted_trajectory[i].get("time_created_ms")
        t2 = converted_trajectory[i + 1].get("time_created_ms")
        if isinstance(t1, (int, float)) and isinstance(t2, (int, float)):
            converted_trajectory[i]["duration"] = round((t2 - t1) / 1000.0, 2)
            converted_trajectory[i]["time_completed_ms"] = t2

    # --- Sub-agent summary for session display ---
    sub_agent_info = []
    for sa in sub_agents_raw:
        if not isinstance(sa, dict):
            continue
        sa_stats = sa.get("statistics", {}) if isinstance(sa.get("statistics"), dict) else {}
        sa_tokens = sa_stats.get("tokens", {}) if isinstance(sa_stats.get("tokens"), dict) else {}
        sub_agent_info.append({
            "agent_id": sa.get("agent_id", ""),
            "started_at": sa.get("started_at", ""),
            "ended_at": sa.get("ended_at", ""),
            "duration_seconds": sa.get("duration_seconds", 0),
            "spawned_by": sa.get("spawned_by_tool_call_id", ""),
            "turns": sa_stats.get("turns", 0),
            "tool_calls": sa_stats.get("tool_calls", 0),
            "tool_calls_by_name": sa_stats.get("tool_calls_by_name", {}),
            "tokens": {
                "input": sa_tokens.get("input", 0),
                "output": sa_tokens.get("output", 0),
                "cache_read": sa_tokens.get("cache_read", 0),
                "cache_creation": sa_tokens.get("cache_creation", 0),
            },
        })

    # Build the internal format that parse_steps()/compute_metrics() expects.
    return {
        "metadata": metadata,
        "input": {"prompt": "", "prompt_length": 0},
        "output": {"patch": "", "patch_length": 0, "patch_lines": 0,
                    "has_patch": False, "error": None},
        "timing": timing,
        "token_usage": token_usage,
        "stats": stats,
        "trajectory": [],  # Empty — we use _cc_parsed_steps instead
        "_cc_parsed_steps": converted_trajectory,
        "_cc_format": True,
        "_cc_sub_agents": sub_agent_info,
    }


def _convert_opencode_metadata(raw: dict) -> dict:
    """Populate metadata, timing, and output keys from OpenCode info structure.

    Mutates *raw* in place so that compute_metrics() and format_session_md()
    can read the standard fields.  Returns the same dict for convenience.
    """
    info = raw.get("info", {})
    if not isinstance(info, dict):
        return raw

    time_info = info.get("time", {}) if isinstance(info.get("time"), dict) else {}
    created_ms = time_info.get("created")
    updated_ms = time_info.get("updated")

    duration_seconds = 0.0
    started_at = ""
    finished_at = ""
    if isinstance(created_ms, (int, float)):
        started_at = datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc).isoformat()
    if isinstance(updated_ms, (int, float)):
        finished_at = datetime.fromtimestamp(updated_ms / 1000.0, tz=timezone.utc).isoformat()
    if isinstance(created_ms, (int, float)) and isinstance(updated_ms, (int, float)):
        duration_seconds = round((updated_ms - created_ms) / 1000.0, 3)

    summary = info.get("summary", {}) if isinstance(info.get("summary"), dict) else {}
    model_info = {}
    # Try to get model from first message's info
    messages = raw.get("messages", [])
    if isinstance(messages, list) and messages:
        first_msg_info = messages[0].get("info", {}) if isinstance(messages[0], dict) else {}
        if isinstance(first_msg_info, dict):
            model_info = first_msg_info.get("model", {}) if isinstance(first_msg_info.get("model"), dict) else {}

    raw["metadata"] = {
        "session_id": info.get("id", ""),
        "slug": info.get("slug", ""),
        "title": info.get("title", ""),
        "directory": info.get("directory", ""),
        "directory_name": info.get("directory", "").replace("\\", "/").rsplit("/", 1)[-1],
        "agent": "opencode",
        "model": model_info.get("modelID", ""),
        "hostname": "",
        "platform": "",
        "python_version": "",
        "timestamp_utc": started_at,
        "branch": "",
        "ground_truth_patch": "",
        "baseline_commit": "",
        "sanitized": False,
        "server_url": "",
        "server_version": info.get("version", ""),
        "generator_name": "opencode",
        "generator_version": info.get("version", ""),
        "format_version": "",
        "sub_agent_count": 0,
        "event_count": 0,
    }
    raw["timing"] = {
        "total_duration": duration_seconds,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    raw["output"] = {
        "patch": "", "patch_length": 0, "patch_lines": 0,
        "has_patch": bool(summary.get("additions") or summary.get("deletions")),
        "error": None,
        "additions": summary.get("additions", 0),
        "deletions": summary.get("deletions", 0),
        "files_changed": summary.get("files", 0),
    }
    raw.setdefault("input", {"prompt": "", "prompt_length": 0})

    # Compute raw stats from messages array
    messages = raw.get("messages", [])
    user_count = 0
    asst_count = 0
    total_tool_calls = 0
    tool_breakdown: dict[str, int] = {}
    total_input = 0
    total_output = 0
    for msg in (messages if isinstance(messages, list) else []):
        if not isinstance(msg, dict):
            continue
        msg_info = msg.get("info", {}) if isinstance(msg.get("info"), dict) else {}
        role = msg_info.get("role", "")
        if role == "user":
            user_count += 1
        elif role == "assistant":
            asst_count += 1
        # Count tokens
        tok = msg_info.get("tokens", {}) if isinstance(msg_info.get("tokens"), dict) else {}
        total_input += tok.get("input", 0) or 0
        total_output += tok.get("output", 0) or 0
        # Count tool calls from parts
        for part in (msg.get("parts", []) if isinstance(msg.get("parts"), list) else []):
            if isinstance(part, dict) and part.get("type") == "tool":
                total_tool_calls += 1
                tname = part.get("tool", "?")
                tool_breakdown[tname] = tool_breakdown.get(tname, 0) + 1

    raw.setdefault("token_usage", {
        "total_tokens": total_input + total_output,
        "prompt_tokens": total_input,
        "completion_tokens": total_output,
    })
    raw.setdefault("stats", {
        "total_messages": user_count + asst_count,
        "user_messages": user_count,
        "assistant_messages": asst_count,
        "total_tool_calls": total_tool_calls,
        "tool_call_breakdown": tool_breakdown,
        "failed_tool_calls": 0,
        "reasoning_steps": asst_count,
    })
    return raw


_CA_SENDER_TO_ROLE = {
    "User": "user", "user": "user",
    "AI": "assistant", "ai": "assistant",
    "DevMate": "assistant", "devmate": "assistant",
}


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


_ca_parse_timestamp = _iso_to_epoch_ms


def _ca_normalize_message(msg: dict, index: int, next_ts_ms: int | None) -> dict:
    """Convert a raw CodeArts message to OpenCode-like trajectory format.

    Preserves the original message under ``_codearts_raw`` for reference.
    """
    sender = msg.get("sender", "unknown")
    role = _CA_SENDER_TO_ROLE.get(sender, "assistant")
    ts_ms = _ca_parse_timestamp(msg.get("timestamp"))

    # Parse content blocks into parts
    parts: list[dict] = []
    tool_calls: list[dict] = []
    content = msg.get("content", [])
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            ctype = block.get("type", "")
            if ctype == "text":
                val = block.get("value", {})
                text = val.get("input", "") if isinstance(val, dict) else str(val)
                parts.append({"type": "text", "text": text})
            elif ctype == "output_text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif ctype == "function_call":
                tc = {
                    "type": "tool_call",
                    "tool_name": block.get("name", "unknown"),
                    "tool_id": block.get("id", ""),
                    "arguments": block.get("arguments", {}),
                    "text": block.get("text", ""),
                }
                parts.append(tc)
                tool_calls.append(tc)

    # Use more precise timing from collectionParams if available
    coll = msg.get("collectionParams", {})
    if isinstance(coll, dict):
        resp_start = coll.get("respStartTime")
        resp_end = coll.get("respEndTime")
    else:
        resp_start = resp_end = None

    # Build info block — extract ALL useful fields
    info: dict = {
        "role": role,
        "time": {},
        "id": msg.get("id", f"msg_{index}"),
        "sessionID": msg.get("chatId", ""),
        "responseMessageID": msg.get("responseMessageId", ""),
    }

    # Timing: prefer collectionParams precision, fall back to ISO timestamp
    created_ms = resp_start if isinstance(resp_start, (int, float)) else ts_ms
    completed_ms = resp_end if isinstance(resp_end, (int, float)) and resp_end != resp_start else next_ts_ms
    if created_ms is not None:
        info["time"]["created"] = created_ms
    if completed_ms is not None:
        info["time"]["completed"] = completed_ms

    # Tokens
    total_tokens = msg.get("total_tokens", 0)
    if total_tokens:
        info["tokens"] = {
            "total": total_tokens,
            "input": 0, "output": 0, "reasoning": 0,
            "cache": {"read": 0, "write": 0},
        }

    # Model
    model_id = msg.get("modelId", "")
    if model_id:
        info["modelID"] = model_id
    real_model = msg.get("realModelId", "")
    if real_model:
        info["realModelID"] = real_model

    # Finish reason
    is_completed = msg.get("isCompleted", False)
    if tool_calls:
        info["finish"] = "tool_use"
    elif role == "assistant" and is_completed:
        info["finish"] = "end_turn"
    elif role == "assistant":
        info["finish"] = "end_turn"

    # Round number
    if "round" in msg:
        info["round"] = msg["round"]

    # Agent info
    agent_info = msg.get("agentInfo", {})
    if isinstance(agent_info, dict) and agent_info:
        info["agent"] = agent_info.get("en_name", agent_info.get("agent_id", ""))
    # agentId: prefer top-level field, fall back to agentInfo.real_agent_id
    info["agentId"] = (
        msg.get("agentId", "")
        or (agent_info.get("real_agent_id", "") if isinstance(agent_info, dict) else "")
    )

    # Sub-agent tracking
    info["isSubAgent"] = (
        msg.get("isEmbedSubagent", False)
        or (isinstance(msg.get("extraParams"), dict)
            and msg["extraParams"].get("isSubAgent", False))
    )
    sub_agents = msg.get("subAgentMsgList", [])
    if sub_agents:
        info["subAgentMsgList"] = sub_agents

    # Question / prompt this step is responding to
    question = msg.get("question", [])
    if question:
        info["question"] = question

    # Tool execution output (operateCacheData)
    operate_cache = msg.get("operateCacheData")
    if isinstance(operate_cache, dict) and operate_cache:
        info["toolOutput"] = operate_cache
        # Classify tool errors from Bash output content
        bash_output = operate_cache.get("bash", {}).get("content", "") if isinstance(operate_cache.get("bash"), dict) else ""
        if bash_output and tool_calls:
            error_type = _classify_tool_error(bash_output)
            if error_type:
                for tc in tool_calls:
                    if tc.get("tool_name") == "Bash":
                        tc["status"] = "error"
                        tc["error"] = bash_output[:200]
                        tc["error_type"] = error_type

    # Output text (cleaner than parsing content blocks)
    output_text = msg.get("outputText", "")
    if output_text:
        info["outputText"] = output_text

    # Context compression — critical signal for context overflow diagnosis
    compress_msg = msg.get("compressMessage", "")
    if compress_msg:
        info["compressMessage"] = compress_msg

    # Task type classification (plan-execute, code-generate, code-fix, etc.)
    code_op = msg.get("codeOperateData", {})
    if isinstance(code_op, dict) and code_op.get("menuTask"):
        info["menuTask"] = code_op["menuTask"]

    # Request metadata from collectionParams (trigger, request type, card type)
    if isinstance(coll, dict):
        trigger = coll.get("trigger", "")
        if trigger:
            info["trigger"] = trigger
        req_type = coll.get("requestType", "")
        if req_type:
            info["requestType"] = req_type
        card_type = coll.get("cardType", "")
        if card_type:
            info["cardType"] = card_type

    return {"info": info, "parts": parts, "_codearts_raw": msg}


def _convert_codearts_metadata(raw: dict) -> dict:
    """Convert consolidated CodeArts JSON to internal Insight format.

    Parses raw messages into OpenCode-like trajectory entries and normalizes
    metadata/timing fields.  Original messages are preserved in each entry
    under ``_codearts_raw``.
    """
    md = raw.get("metadata", {})
    timing = raw.get("timing", {})
    stats = raw.get("statistics", {})
    messages = raw.get("messages", [])

    # Pre-compute timestamps for duration approximation
    ts_list = [_ca_parse_timestamp(m.get("timestamp")) for m in messages]

    # Convert messages to trajectory entries
    trajectory = []
    model_id = ""
    for i, msg in enumerate(messages):
        next_ts = ts_list[i + 1] if i + 1 < len(ts_list) else None
        entry = _ca_normalize_message(msg, i, next_ts_ms=next_ts)
        trajectory.append(entry)
        if not model_id:
            model_id = msg.get("modelId", "")

    # Fix cumulative token counts: CodeArts reports total_tokens as the
    # cumulative total for the conversation at each message, not the
    # per-message cost.  Convert to per-step deltas by subtracting the
    # previous assistant message's total.  Resets (where current < prev)
    # indicate a new context window / sub-task.
    prev_cumulative = 0
    for entry in trajectory:
        info = entry.get("info", {})
        tokens = info.get("tokens")
        if tokens and info.get("role") == "assistant":
            cumulative = tokens["total"]
            if cumulative >= prev_cumulative:
                tokens["total"] = cumulative - prev_cumulative
            # else: reset (new context window) — keep as-is
            prev_cumulative = cumulative
        elif info.get("role") == "user":
            # User messages don't carry tokens; don't reset prev
            pass

    # Set stop finish on last message
    if trajectory:
        last_ts = ts_list[-1] if ts_list else None
        if last_ts:
            trajectory[-1]["info"]["time"]["completed"] = last_ts
        trajectory[-1]["info"]["finish"] = "stop"

    # Compute total duration
    first_ts = ts_list[0] if ts_list else None
    last_ts = ts_list[-1] if ts_list else None
    total_duration = 0
    if first_ts and last_ts:
        total_duration = round((last_ts - first_ts) / 1000, 2)

    # Compute wall-clock duration from timing fields (more accurate than
    # message timestamps which may not cover idle time at start/end).
    timing_duration = total_duration
    t_start = _ca_parse_timestamp(timing.get("started_at"))
    t_end = _ca_parse_timestamp(timing.get("finished_at"))
    if t_start and t_end:
        timing_duration = round((t_end - t_start) / 1000, 2)

    # Directory: CodeArts doesn't have a working directory field.
    # metadata.title contains the full task prompt, not a directory name.

    # Session description from chat_base_info (agent-generated summary)
    cbi = raw.get("chat_base_info", {})
    session_desc = ""
    if isinstance(cbi, dict):
        session_desc = cbi.get("description", "")

    raw["trajectory"] = trajectory
    raw.setdefault("metadata", {}).update({
        "session_id": md.get("session_id", ""),
        "agent": md.get("agent", "CodeArts"),
        "agent_id": md.get("agent_id", ""),
        "model": model_id or md.get("model", ""),
        "directory_name": "",
        "description": session_desc,
        "generator_name": "codearts",
    })
    raw.setdefault("timing", {}).update({
        "started_at": timing.get("started_at", ""),
        "finished_at": timing.get("finished_at", ""),
        "total_duration": timing_duration,
    })
    raw.setdefault("statistics", {}).update({
        "total_messages": stats.get("total_messages", len(messages)),
        "user_messages": stats.get("user_messages", 0),
        "assistant_messages": stats.get("assistant_messages", 0),
    })

    return raw


# ---------------------------------------------------------------------------


def _convert_codex_to_internal(events: list[dict]) -> dict:
    """Convert Codex CLI JSONL events into the TrajectoryVisualizer internal format.

    Codex emits newline-delimited JSON with event types:
    - session_meta: session ID, cwd, model, version
    - turn_context: turn metadata
    - response_item: messages (user/assistant/developer), function_call, function_call_output, reasoning
    - event_msg: task_started, task_complete, token_count, agent_message

    We group these into "messages" matching the OpenCode internal format:
    each assistant turn = one message with parts (text, reasoning, tool calls).
    """
    # Extract session metadata
    session_meta = {}
    for e in events:
        if e.get("type") == "session_meta":
            session_meta = e.get("payload", {})
            break

    # Group events into assistant turns.
    # Pattern: user message → (reasoning → assistant text → function_calls → function_call_outputs)* → task_complete
    messages: list[dict] = []
    pending_tool_calls: dict[str, dict] = {}  # call_id -> function_call payload
    current_parts: list[dict] = []
    current_role = None
    current_timestamp = None

    def _flush_message():
        nonlocal current_parts, current_role, current_timestamp
        if current_parts and current_role:
            messages.append({
                "info": {
                    "role": current_role,
                    "time": {"created": current_timestamp or 0},
                },
                "parts": current_parts,
            })
        current_parts = []
        current_role = None
        current_timestamp = None

    for event in events:
        etype = event.get("type")
        ts = event.get("timestamp", "")

        if etype == "response_item":
            payload = event.get("payload", {})
            item_type = payload.get("type", "")
            role = payload.get("role", "")

            if item_type == "message":
                # Role change → flush previous message
                if role != current_role and current_parts:
                    _flush_message()

                current_role = role
                if not current_timestamp:
                    current_timestamp = ts

                for content in (payload.get("content") or []):
                    ctype = content.get("type", "")
                    text = content.get("text", "")
                    if ctype == "output_text":
                        current_parts.append({"type": "text", "text": text})
                    elif ctype == "input_text":
                        current_parts.append({"type": "text", "text": text})

            elif item_type == "reasoning":
                # If we already have tool call parts, this reasoning starts a new turn
                has_tool_parts = any(p.get("type") in ("tool_call", "tool") for p in current_parts)
                if has_tool_parts and current_role == "assistant":
                    _flush_message()
                current_role = current_role or "assistant"
                if not current_timestamp:
                    current_timestamp = ts
                summary = payload.get("summary", [])
                summary_text = " ".join(s.get("text", "") for s in summary) if summary else ""
                current_parts.append({"type": "reasoning", "text": summary_text})

            elif item_type == "function_call":
                if current_role != "assistant" and current_parts:
                    _flush_message()
                current_role = "assistant"
                if not current_timestamp:
                    current_timestamp = ts

                call_id = payload.get("call_id", "")
                name = payload.get("name", "exec_command")
                args_str = payload.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"raw": args_str}

                # Determine tool name and build normalized input
                cmd = args.get("cmd", "")
                tool_name = _classify_codex_command(name, cmd)
                normalized_input = _build_codex_tool_input(tool_name, cmd, args)

                pending_tool_calls[call_id] = {
                    "name": name,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "input": normalized_input,
                    "cmd": cmd,
                }

            elif item_type == "function_call_output":
                call_id = payload.get("call_id", "")
                output = payload.get("output", "")
                tc = pending_tool_calls.pop(call_id, {})

                # Determine status from output
                status = "success"
                if isinstance(output, str):
                    if "error" in output.lower()[:200] or "traceback" in output.lower()[:200]:
                        status = "error"
                    # Check exit code in output
                    if "exited with code" in output and "code 0" not in output:
                        status = "error"

                current_parts.append({
                    "type": "tool_call" if tc else "tool",
                    "tool_name": tc.get("tool_name", "Bash"),
                    "tool_id": call_id,
                    "status": status,
                    "input": tc.get("input", {}),
                    "output": output,
                })

        elif etype == "event_msg":
            msg_type = event.get("payload", {}).get("type", "")
            if msg_type == "task_complete":
                _flush_message()

    # Flush any remaining parts
    _flush_message()

    # Build metadata
    info = {
        "id": session_meta.get("id", ""),
        "slug": "",
        "projectID": "",
        "directory": session_meta.get("cwd", ""),
        "title": "",
        "version": session_meta.get("cli_version", ""),
        "time": {
            "created": events[0].get("timestamp", 0) if events else 0,
            "updated": events[-1].get("timestamp", 0) if events else 0,
        },
    }

    return {
        "info": info,
        "messages": messages,
        "metadata": {
            "source": "codex",
            "model_provider": session_meta.get("model_provider", "openai"),
            "originator": session_meta.get("originator", "Codex CLI"),
        },
        "timing": {},
        "output": {},
        "input": {},
        "token_usage": {},
        "stats": {},
    }


def _classify_codex_command(func_name: str, cmd: str) -> str:
    """Map a Codex exec_command to a TrajectoryVisualizer tool name.

    Codex uses exec_command for everything; we infer the intent from the command.
    """
    cmd_lower = cmd.lower().strip()

    # File reading
    if any(cmd_lower.startswith(p) for p in ["cat ", "head ", "tail ", "sed -n", "less "]):
        return "Read"
    if cmd_lower.startswith("rg ") or cmd_lower.startswith("grep "):
        return "Grep"
    if cmd_lower.startswith("find ") or cmd_lower.startswith("ls "):
        return "Glob"

    # File writing
    for pattern in ["cat >", "cat >>", "tee ", "echo >", "echo >>",
                     "sed -i", "patch ", "git apply"]:
        if pattern in cmd_lower:
            return "Write"

    # Testing
    if any(p in cmd_lower for p in ["pytest", "python -m pytest", "python -m unittest",
                                     "make test", "tox ", "npm test"]):
        return "Bash"  # test execution

    # Git
    if cmd_lower.startswith("git "):
        return "Bash"

    # Python execution
    if cmd_lower.startswith("python ") or cmd_lower.startswith("python3 "):
        return "Bash"

    return "Bash"


def _build_codex_tool_input(tool_name: str, cmd: str, raw_args: dict) -> dict:
    """Build a normalized input dict for Codex commands.

    Maps the raw Codex exec_command args into the format that
    canonical.py expects for each tool type:
    - Read: {"file_path": "..."}
    - Write: {"file_path": "..."}
    - Grep/Glob: {"pattern": "...", "path": "..."}
    - Bash: {"command": "..."}
    """
    import shlex

    if tool_name in ("Read",):
        # Extract file path from commands like:
        #   cat file.py, sed -n '1,20p' file.py, head -n 50 file.py
        #   nl -ba file.py | sed ...
        parts = cmd.split("|")[0].strip().split()  # take before first pipe
        # File path is usually the last arg that doesn't start with - or '
        file_path = ""
        for p in reversed(parts):
            if not p.startswith("-") and not p.startswith("'") and not p.startswith('"'):
                if "." in p or "/" in p:
                    file_path = p
                    break
        return {"file_path": file_path, "command": cmd}

    elif tool_name in ("Write",):
        # Extract file path from: sed -i, cat >, patch, git apply
        parts = cmd.split()
        file_path = ""
        for p in reversed(parts):
            if not p.startswith("-") and ("." in p or "/" in p):
                file_path = p
                break
        return {"file_path": file_path, "command": cmd}

    elif tool_name in ("Grep",):
        # Extract pattern and scope from: rg -n "pattern" dir/
        #   or: grep -rn "pattern" file
        parts = cmd.split()
        non_flag = [p for p in parts[1:] if not p.startswith("-")]
        pattern = non_flag[0].strip("'\"") if non_flag else ""
        path = non_flag[-1] if len(non_flag) > 1 else ""
        return {"pattern": pattern, "path": path, "command": cmd}

    elif tool_name in ("Glob",):
        # Extract path from: find dir, ls dir
        parts = cmd.split()
        path = parts[1] if len(parts) > 1 and not parts[1].startswith("-") else ""
        return {"path": path, "command": cmd}

    else:
        # Bash — pass cmd as "command" so canonical.py can parse it
        return {"command": cmd}


# Reject pathologically large uploads before reading them into memory. A
# trajectory is held several times over (raw dict + steps + gr.State copies +
# dumped string), so the working-set multiple of the file size is what matters.
_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


def load_trajectory(file_path: str, format_hint: str | None = None) -> dict:
    """Load trajectory file with error handling.

    Auto-detects format (ccsession / opencode / codearts / codex) and
    normalizes metadata.  Supports ``.json`` and ``.jsonl`` files.

    ``format_hint`` is used as a fallback when automatic detection returns
    ``"unknown"`` — useful for JSON files that lack format-identifying fields
    (e.g., Claude Code exports without the ``format`` marker).
    """
    if file_path.endswith(".log"):
        return {"_error": "Unsupported file type: .log files are no longer supported."}

    # Size guard (before any full read) to avoid self-inflicted OOM.
    try:
        size = os.path.getsize(file_path)
    except OSError as exc:
        return {"_error": str(exc)}
    if size > _MAX_FILE_BYTES:
        return {"_error": (
            f"File is too large ({size / 1e6:.0f} MB; limit "
            f"{_MAX_FILE_BYTES // (1024 * 1024)} MB). Trim the trajectory or raise "
            f"the limit if you understand the memory cost."
        )}

    # Handle Codex JSONL format
    if file_path.endswith(".jsonl"):
        try:
            events = []
            bad_lines = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Per-line recovery: a single truncated/corrupt line (common
                    # in crashed/killed sessions — exactly the runs users want to
                    # diagnose) must not discard every prior valid event.
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        bad_lines += 1
            if not events:
                return {"_error": "No parseable JSONL events found in file."}
            if events[0].get("type") == "session_meta":
                result = _convert_codex_to_internal(events)
                result["_source_path"] = file_path
                if bad_lines:
                    result["_load_warning"] = (
                        f"Skipped {bad_lines} unparseable line(s) of "
                        f"{bad_lines + len(events)}."
                    )
                return result
            return {"_error": "JSONL input is not supported for training conversations in v1; expected Codex JSONL format."}
        except OSError as exc:
            return {"_error": str(exc)}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return {"_error": str(exc)}

    # Top-level must be a JSON object; a list/scalar (a common heterogeneous
    # export shape) would otherwise crash detect_format() on raw.get(...).
    if not isinstance(raw, dict):
        return {"_error": (
            f"Expected a JSON object at the top level, got {type(raw).__name__}. "
            f"This does not look like a supported trajectory export."
        )}

    fmt = detect_format(raw)
    if fmt == "unknown" and format_hint in ("ccsession", "codearts", "opencode"):
        fmt = format_hint
    if fmt == "ccsession":
        result = _convert_claude_code_to_internal(raw)
    elif fmt == "codearts":
        result = _convert_codearts_metadata(raw)
    elif fmt == "opencode":
        result = _convert_opencode_metadata(raw)
    elif fmt == "training_conversation":
        result = _convert_training_conversation_to_internal(raw)
    else:
        result = raw
    result["_source_path"] = file_path
    return result
