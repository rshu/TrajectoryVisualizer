"""Data loading, parsing, and aggregate metrics."""

from .loaders import safe_get

# Fields whose absence makes the whole Metrics table unavailable. Reasoning is
# optional: many formats never report it (show per-row N/A instead of fake 0).
_TOKEN_METRIC_FIELDS = {
    "total": "Total Tokens",
    "input": "Input Tokens",
    "output": "Output Tokens",
}


def _optional_token_count(tokens: dict, key: str) -> int | None:
    """Return an int token count when *key* was reported, else None."""
    value = tokens.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _missing_token_metric_fields(tokens_info: dict) -> list[str]:
    """Return display labels for token fields absent from the source payload."""
    missing = [
        label
        for key, label in _TOKEN_METRIC_FIELDS.items()
        if key not in tokens_info or tokens_info.get(key) is None
    ]
    cache = tokens_info.get("cache")
    if not isinstance(cache, dict) or "read" not in cache or cache.get("read") is None:
        missing.append("Cache Read")
    if not isinstance(cache, dict) or "write" not in cache or cache.get("write") is None:
        missing.append("Cache Write")
    return missing


def infer_non_cache_input(
    total_tokens: int,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    cache_read_tokens: int,
) -> int:
    """Infer fresh/non-cache input tokens across token-schema variants.

    Some traces report:
    - total = input + output + reasoning + cache_read  (input is already fresh)
    Others report:
    - total = input + output + reasoning               (input includes cache)
    """
    base = (input_tokens or 0) + (output_tokens or 0) + (reasoning_tokens or 0)
    total = total_tokens or 0
    cache_read = cache_read_tokens or 0

    # When no token breakdown is available (all components zero but total > 0),
    # we cannot infer fresh vs cached — treat entire total as fresh.
    if total > 0 and base == 0 and cache_read == 0:
        return total

    # Pick the interpretation whose implied total is closer to observed total.
    dist_fresh_input = abs(total - (base + cache_read))
    dist_cached_input = abs(total - base)
    if dist_fresh_input <= dist_cached_input:
        return max(0, input_tokens or 0)
    return max(0, (input_tokens or 0) - cache_read)


def _parse_parts(parts_raw: list) -> tuple[list, list, int, bool, str]:
    """Parse raw parts into structured parts, tool calls, error count, reasoning flag, and preview."""
    parts = []
    tool_calls = []
    errors = 0
    has_reasoning = False
    text_preview = ""
    synthetic_text_preview = ""

    for p in parts_raw:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type", "unknown")

        if ptype == "text":
            txt = p.get("text", "")
            parts.append({
                "type": "text", "text": txt,
                "synthetic": bool(p.get("synthetic", False)),
                "metadata": p.get("metadata", {}) if isinstance(p.get("metadata"), dict) else {},
                "time": p.get("time", {}) if isinstance(p.get("time"), dict) else {},
                "part_id": p.get("id", ""),
                "session_id": p.get("sessionID", ""),
                "message_id": p.get("messageID", ""),
            })
            if p.get("synthetic") and not synthetic_text_preview:
                synthetic_text_preview = txt
            elif not p.get("synthetic") and not text_preview:
                text_preview = txt
        elif ptype == "reasoning":
            parts.append({
                "type": "reasoning", "text": p.get("text", ""),
                "time": p.get("time", {}) if isinstance(p.get("time"), dict) else {},
                "part_id": p.get("id", ""),
                "session_id": p.get("sessionID", ""),
                "message_id": p.get("messageID", ""),
            })
            has_reasoning = True
            if not text_preview:
                text_preview = p.get("text", "")
        elif ptype in ("tool_call", "tool"):
            state = p.get("state", {})
            if not isinstance(state, dict):
                state = {"status": str(state)}
            tool_name = p.get("tool_name", p.get("name", p.get("tool", "?")))
            status = state.get("status", p.get("status", "?"))
            tool_input = state.get("input", p.get("input", p.get("arguments", {})))
            tool_output = state.get("output", p.get("output", ""))
            compacted_at = safe_get(state, "time", "compacted", default=None)
            tc = {
                "type": "tool_call", "tool_name": tool_name,
                "tool_id": p.get("tool_id", p.get("callID", p.get("id", ""))), "status": status,
                "title": state.get("title", ""),
                "input": tool_input,
                "output": tool_output,
                "error": p.get("error") or state.get("error") or None,
                "error_type": p.get("error_type"),
                "time_created": safe_get(p, "time", "created", default=None),
                "time_updated": safe_get(p, "time", "updated", default=None),
                "time_start": safe_get(state, "time", "start", default=None),
                "time_end": safe_get(state, "time", "end", default=None),
                "time_compacted": compacted_at,
                "duration_ms": safe_get(state, "metadata", "totalDurationMs", default=None),
                "metadata": state.get("metadata", {}),
                "part_id": p.get("id", ""),
                "session_id": p.get("sessionID", ""),
                "message_id": p.get("messageID", ""),
            }
            parts.append(tc)
            tool_calls.append(tc)
            if status == "error":
                errors += 1
            if not text_preview:
                text_preview = f"[Tool: {tool_name}] {tc['title']}"
        elif ptype in ("step_start", "step-start"):
            parts.append({
                "type": "step_start", "name": p.get("name", ""),
                "time": p.get("time", {}) if isinstance(p.get("time"), dict) else {},
                "part_id": p.get("id", ""),
            })
        elif ptype in ("step_finish", "step-finish"):
            parts.append({
                "type": "step_finish", "name": p.get("name", ""),
                "reason": p.get("reason", ""),
                "tokens": p.get("tokens", {}) if isinstance(p.get("tokens"), dict) else {},
                "cost": p.get("cost"),
                "time": p.get("time", {}) if isinstance(p.get("time"), dict) else {},
                "part_id": p.get("id", ""),
            })
        elif ptype == "compaction":
            summary_text = p.get("summary") or p.get("text") or ""
            if not isinstance(summary_text, str):
                summary_text = str(summary_text) if summary_text else ""
            parts.append({
                "type": "compaction",
                "summary": summary_text,
                "reason": p.get("reason", ""),
                "recent": p.get("recent", ""),
                "time": p.get("time", {}) if isinstance(p.get("time"), dict) else {},
                "part_id": p.get("id", ""),
                "session_id": p.get("sessionID", ""),
                "message_id": p.get("messageID", ""),
            })
            if not text_preview and summary_text:
                text_preview = summary_text
        elif ptype == "snapshot":
            parts.append({"type": "snapshot", "data": p.get("data", p.get("snapshot", {}))})
        elif ptype == "patch":
            patch_raw = p.get("raw", p)
            if not isinstance(patch_raw, dict):
                patch_raw = {}
            parts.append({
                "type": "patch", "hash": patch_raw.get("hash", ""),
                "files": patch_raw.get("files", []), "id": patch_raw.get("id", ""),
                "session_id": patch_raw.get("sessionID", ""),
                "message_id": patch_raw.get("messageID", ""),
                "diff_content": patch_raw.get("diff", patch_raw.get("diff_content", "")),
            })
        else:
            parts.append({"type": ptype, "raw": p})

    return parts, tool_calls, errors, has_reasoning, text_preview or synthetic_text_preview


def parse_steps(raw: dict) -> list[dict]:
    """Normalize each message in trajectory[] into a step dict."""
    # If already parsed by Claude Code converter, return directly (still backfill
    # the final step's duration, which the fast-path would otherwise skip).
    if raw.get("_cc_format") and "_cc_parsed_steps" in raw:
        steps = raw["_cc_parsed_steps"]
        _fill_missing_last_step_duration(steps, raw)
        return steps

    trajectory = raw.get("trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        trajectory = raw.get("messages", [])
    if not isinstance(trajectory, list):
        return []

    steps = []
    for idx, msg in enumerate(trajectory):
        if not isinstance(msg, dict):
            continue
        info = msg.get("info") if isinstance(msg.get("info"), dict) else {}
        message_type = msg.get("type") or info.get("type") or ""
        if not isinstance(message_type, str):
            message_type = ""
        is_compaction_checkpoint = message_type == "compaction"
        role = msg.get("role") or safe_get(info, "role", default="")
        if not role:
            role = "compaction" if is_compaction_checkpoint else "?"
        summary_flag = info.get("summary", False)

        tokens_info = safe_get(info, "tokens", default={})
        if not isinstance(tokens_info, dict):
            tokens_info = {}
        metrics_unavailable_fields = _missing_token_metric_fields(tokens_info)
        metrics_source_format = ""
        reasoning = _optional_token_count(tokens_info, "reasoning")
        tokens = {
            "total": tokens_info.get("total", 0) or 0,
            "input": tokens_info.get("input", 0) or 0,
            "output": tokens_info.get("output", 0) or 0,
            "cache_read": safe_get(tokens_info, "cache", "read", default=0) or 0,
            "cache_write": safe_get(tokens_info, "cache", "write", default=0) or 0,
        }
        if reasoning is not None:
            tokens["reasoning"] = reasoning
        # Formats that log per-message window contributions (ICode) resolve
        # the live context-window occupancy in their converter; per-step
        # totals are that step's own tokens, not the window size.
        window = _optional_token_count(tokens_info, "context_window")
        if window is not None and window >= 0:
            tokens["context_window"] = window

        t_created = safe_get(info, "time", "created", default=None)
        t_completed = safe_get(info, "time", "completed", default=None)
        duration = None
        if (
            isinstance(t_created, (int, float)) and t_created > 0
            and isinstance(t_completed, (int, float))
        ):
            duration = round((t_completed - t_created) / 1000.0, 2)

        raw_parts = msg.get("parts", [])
        if not isinstance(raw_parts, list):
            raw_parts = []
        parts, tool_calls, errors, has_reasoning, text_preview = _parse_parts(raw_parts)

        finish = safe_get(info, "finish", default="")
        path_info = safe_get(info, "path", default={})
        if not isinstance(path_info, dict):
            path_info = {}
        steps.append({
            "index": idx, "raw_index": idx, "role": role, "tokens": tokens, "duration": duration,
            "parts": parts, "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls), "error_count": errors,
            "has_reasoning": has_reasoning, "text_preview": text_preview,
            "finish": finish,
            "model_id": safe_get(info, "modelID", default=""),
            "provider_id": safe_get(info, "providerID", default=""),
            "time_created_ms": t_created, "time_completed_ms": t_completed,
            "agent": safe_get(info, "agent", default=""),
            "mode": safe_get(info, "mode", default=""),
            "message_id": (
                msg.get("message_id", "")
                or (info.get("id", "") if raw.get("_codearts_format") else "")
            ),
            "id": safe_get(info, "id", default=""),
            "parent_id": safe_get(info, "parentID", default=""),
            "session_id": safe_get(info, "sessionID", default=""),
            "cwd": path_info.get("cwd", ""), "root": path_info.get("root", ""),
            "is_sub_agent": info.get("isSubAgent", False),
            "parent_session_id": info.get("parentSessionID", ""),
            "session_depth": info.get("sessionDepth"),
            "session_title": info.get("sessionTitle", ""),
            "summary": summary_flag,
            "message_type": message_type,
            "is_compaction_checkpoint": is_compaction_checkpoint,
            "compaction_reason": info.get("reason", "") if is_compaction_checkpoint else "",
            "_metrics_unavailable_fields": metrics_unavailable_fields,
            "_metrics_source_format": metrics_source_format,
        })

    _fill_missing_last_step_duration(steps, raw)
    _annotate_spawned_subagents(steps, raw)
    return steps


def spawned_child_session_id(
    metadata: object,
    *,
    caller_session_id: str = "",
    root_session_id: str = "",
) -> str:
    """Child session id from Task-tool metadata, or '' if this is not a spawn.

    Ignores self-spawns and (when given) the trajectory root session, which
    nested tool parts sometimes echo incorrectly.
    """
    if not isinstance(metadata, dict):
        return ""
    child_id = (
        metadata.get("sessionId")
        or metadata.get("sessionID")
        or metadata.get("session_id")
    )
    if not isinstance(child_id, str) or not child_id:
        return ""
    if caller_session_id and child_id == caller_session_id:
        return ""
    if root_session_id and child_id == root_session_id:
        return ""
    return child_id


def _annotate_spawned_subagents(steps: list[dict], raw: dict | None = None) -> None:
    """Mark child sessions spawned via Task/agent tools as sub-agents.

    OpenCode exports often leave ``isSubAgent`` unset on child messages while
    recording ``metadata.sessionId`` / ``parentSessionId`` on the parent's
    task tool part. Infer those links so timelines, agent cards, and
    sub-agent session grouping see the real hierarchy.

    Never treats the trajectory root session as a spawned child (nested tool
    parts can incorrectly echo the parent session id).
    """
    root_sid = ""
    if isinstance(raw, dict):
        info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
        candidate = info.get("id") or ""
        if isinstance(candidate, str):
            root_sid = candidate
    if not root_sid:
        for step in steps:
            if isinstance(step, dict) and step.get("session_id"):
                root_sid = str(step["session_id"])
                break

    child_to_parent: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        parent_sid = step.get("session_id") or ""
        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            meta = tc.get("metadata") if isinstance(tc.get("metadata"), dict) else {}
            child_id = spawned_child_session_id(
                meta,
                caller_session_id=str(parent_sid or ""),
                root_session_id=root_sid,
            )
            if not child_id:
                continue
            parent_from_meta = (
                meta.get("parentSessionId")
                or meta.get("parentSessionID")
                or meta.get("parent_session_id")
                or parent_sid
            )
            if (
                isinstance(parent_from_meta, str)
                and parent_from_meta
                and parent_sid
                and parent_from_meta != parent_sid
            ):
                # Metadata disagrees with the calling session — skip.
                continue
            if child_id not in child_to_parent:
                child_to_parent[child_id] = (
                    parent_from_meta if isinstance(parent_from_meta, str) else ""
                )

    if not child_to_parent:
        return

    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = step.get("session_id") or ""
        if sid not in child_to_parent:
            continue
        step["is_sub_agent"] = True
        if not step.get("parent_session_id"):
            parent = child_to_parent[sid]
            if parent:
                step["parent_session_id"] = parent


def _fill_missing_last_step_duration(steps: list[dict], raw: dict) -> None:
    """Backfill duration on the final step when its completion timestamp is missing.

    Trajectory recorders sometimes close the session before writing the last
    message's ``time.completed``, leaving the step with ``duration=None`` even
    though it took real wall time. We substitute the trajectory's end timestamp
    (from ``raw.timing.finished_at`` or the latest completion seen across
    sibling steps) as a best-effort finish.
    """
    if not steps:
        return
    last = steps[-1]
    if last.get("duration") is not None:
        return
    start_ms = last.get("time_created_ms")
    if not isinstance(start_ms, (int, float)) or start_ms <= 0:
        return

    # Prefer the session's finished_at timestamp (authoritative when present).
    end_ms: float | None = None
    timing = raw.get("timing") if isinstance(raw.get("timing"), dict) else {}
    finished_at = timing.get("finished_at")
    if isinstance(finished_at, str) and finished_at:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            end_ms = dt.timestamp() * 1000
        except (ValueError, TypeError):
            end_ms = None

    # Fall back to the latest completion timestamp observed on earlier steps.
    if end_ms is None:
        completed = [s.get("time_completed_ms") for s in steps[:-1]
                     if isinstance(s.get("time_completed_ms"), (int, float))]
        if completed:
            end_ms = max(completed)

    if end_ms is None or end_ms <= start_ms:
        return

    last["duration"] = round((end_ms - start_ms) / 1000.0, 2)
    if not last.get("time_completed_ms"):
        last["time_completed_ms"] = end_ms
