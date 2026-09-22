"""Claude Code session dump → internal trajectory."""

import os

from .common import _iso_to_epoch_ms

def _cc_extract_usage(usage: dict | None) -> dict:
    """Extract token usage from a Claude Code message's usage field.

    Reasoning tokens are omitted: Claude Code / Anthropic usage does not report
    them, and a fabricated ``0`` was misread as a real measurement in Overview.
    """
    if not isinstance(usage, dict):
        return {"total": 0, "input": 0, "output": 0,
                "cache": {"read": 0, "write": 0}}
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
    # input_tokens, cache_creation_input_tokens and cache_read_input_tokens are
    # mutually exclusive in the Anthropic usage object, so the processed total is
    # their sum; cache_creation must be included to agree with the session total.
    total = inp + out + cache_read + cache_write
    return {
        "total": total, "input": inp, "output": out,
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


def _extract_tool_results(content: list, tool_exec: dict | None, result_map: dict) -> None:
    """Record tool_result items from a user message's content into *result_map*.

    Shared by the direct-user-message and event-nested ingestion paths of
    _cc_build_tool_result_map so the id-fallback and is_error semantics
    cannot diverge between them.
    """
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
            _extract_tool_results(content, tool_exec, result_map)

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
                            _extract_tool_results(inner_content, inner_tool_exec, result_map)
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


def _cc_build_step(parts: list, *, role: str, usage: dict | None = None,
                   timestamp_ms: int | None = None, model: str = "",
                   finish: str = "", agent: str = "", message_id: str = "",
                   step_id: str = "", parent_id: str = "", cwd: str = "",
                   tool_calls: list | None = None,
                   error_count: int | None = None,
                   has_reasoning: bool | None = None,
                   text_preview: str = "") -> dict:
    """Build one internal-format step dict — the single source of the step schema.

    Derived fields (tool_calls, error_count, has_reasoning) are computed from
    *parts* unless explicitly overridden.  ``usage=None`` yields zeroed token
    counts (no ``reasoning`` key — Claude Code does not report that field).
    Callers compute their own ``text_preview`` (the preview heuristics
    intentionally differ between the main-trajectory and event paths).
    """
    if tool_calls is None:
        tool_calls = [p for p in parts if p.get("type") == "tool_call"]
    if error_count is None:
        error_count = sum(1 for tc in tool_calls if tc.get("status") == "error")
    if has_reasoning is None:
        has_reasoning = any(p.get("type") == "reasoning" for p in parts)
    if usage is None:
        tokens = {"total": 0, "input": 0, "output": 0,
                  "cache_read": 0, "cache_write": 0}
    else:
        tokens = {
            "total": usage["total"],
            "input": usage["input"],
            "output": usage["output"],
            "cache_read": usage["cache"]["read"],
            "cache_write": usage["cache"]["write"],
        }
    return {
        "role": role,
        "tokens": tokens,
        "duration": None,  # Will be computed from timestamps
        "parts": parts,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "error_count": error_count,
        "has_reasoning": has_reasoning,
        "text_preview": text_preview,
        "finish": finish or "",
        "model_id": model,
        "provider_id": "",
        "time_created_ms": timestamp_ms,
        "time_completed_ms": None,
        "agent": agent,
        "mode": "",
        "message_id": message_id,
        "id": step_id,
        "parent_id": parent_id,
        "session_id": "",
        "cwd": cwd,
        "root": "",
    }


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

    # Preview: prefer text over reasoning (keep scanning after a reasoning hit)
    text_preview = ""
    for p in parts:
        if p.get("type") == "text" and p.get("text"):
            text_preview = p["text"]
            break
        if p.get("type") == "reasoning" and p.get("text") and not text_preview:
            text_preview = p["text"]
        if p.get("type") == "tool_call" and not text_preview:
            text_preview = f"[Tool: {p['tool_name']}]"

    return _cc_build_step(
        parts,
        role=role,
        usage=usage,
        timestamp_ms=timestamp_ms,
        model=entry.get("model", ""),
        finish=entry.get("stop_reason", ""),
        agent=agent_id or "",
        message_id=entry.get("message_id", entry.get("uuid", "")),
        step_id=entry.get("uuid", ""),
        parent_id=entry.get("parent_uuid", ""),
        cwd=entry.get("cwd", ""),
        text_preview=text_preview,
    )


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
            # Preview: first text-or-reasoning part wins (event-path heuristic)
            text_preview = ""
            for p in parts:
                if p.get("type") in ("text", "reasoning") and p.get("text"):
                    text_preview = p["text"]
                    break
                if p.get("type") == "tool_call" and not text_preview:
                    text_preview = f"[Tool: {p['tool_name']}]"

            steps.append(_cc_build_step(
                parts,
                role="assistant",
                usage=usage,
                timestamp_ms=timestamp_ms,
                model=inner_msg.get("model", ""),
                finish=inner_msg.get("stop_reason", ""),
                agent=agent_id,
                message_id=inner_msg.get("id", ""),
                step_id=msg.get("uuid", entry.get("uuid", "")),
                parent_id=entry.get("parent_uuid", ""),
                cwd=entry.get("cwd", ""),
                text_preview=text_preview,
            ))
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
                steps.append(_cc_build_step(
                    parts,
                    role="user",
                    timestamp_ms=timestamp_ms,
                    agent=agent_id,
                    step_id=msg.get("uuid", entry.get("uuid", "")),
                    parent_id=entry.get("parent_uuid", ""),
                    cwd=entry.get("cwd", ""),
                    # Event-path user steps never carry tool calls/usage
                    tool_calls=[],
                    error_count=0,
                    has_reasoning=False,
                    text_preview=next(
                        (p["text"] for p in parts if p.get("type") == "text" and p.get("text")),
                        ""
                    ),
                ))

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
        "directory": session.get("working_directory") or "",
        "directory_name": os.path.basename(session.get("working_directory") or ""),
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

    # Sort chronologically for display order.  A step whose timestamp is missing
    # keeps its original position (it carries the previous step's timestamp)
    # instead of collapsing to 0 and jumping ahead of the opening user prompt.
    _orig_order = {id(s): i for i, s in enumerate(converted_trajectory)}
    _carry: dict[int, float | None] = {}
    _last_ts: float | None = None
    for s in converted_trajectory:
        ts = s.get("time_created_ms")
        if isinstance(ts, (int, float)):
            _last_ts = ts
        _carry[id(s)] = _last_ts
    converted_trajectory.sort(key=lambda s: (
        _carry[id(s)] if _carry[id(s)] is not None else float("-inf"),
        _orig_order[id(s)],
    ))

    for idx, step in enumerate(converted_trajectory):
        step["index"] = idx

    # Compute per-step durations WITHIN each agent's own timeline.  A delegated
    # sub-agent runs concurrently while the main agent is blocked on the Task
    # tool, so the gap to the next step in the globally-interleaved list would
    # otherwise charge the main agent's idle wait to the sub-agent's step.
    by_agent: dict[str, list[dict]] = {}
    for step in converted_trajectory:
        by_agent.setdefault(step.get("agent", ""), []).append(step)
    for agent_steps in by_agent.values():
        # Pair consecutive *timestamped* steps within the agent, stepping over
        # any untimed step so its missing timestamp does not blank out the
        # duration of the timestamped step before it.
        timed = [s for s in agent_steps
                 if isinstance(s.get("time_created_ms"), (int, float))]
        for i in range(len(timed) - 1):
            # Skip assistant->user turn boundaries: any user step surviving
            # conversion is a real human prompt (tool_result-only user
            # messages were dropped), so the gap is human idle time between
            # turns, not model latency.  The turn-final assistant step keeps
            # duration=None, same as the last step of each agent timeline.
            if timed[i].get("role") == "assistant" and timed[i + 1].get("role") == "user":
                continue
            t1 = timed[i]["time_created_ms"]
            t2 = timed[i + 1]["time_created_ms"]
            if t2 >= t1:
                timed[i]["duration"] = round((t2 - t1) / 1000.0, 2)
                timed[i]["time_completed_ms"] = t2

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
