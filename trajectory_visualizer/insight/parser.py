"""Data loading, parsing, and aggregate metrics."""

# Re-export loader functions for backward compatibility
from .loaders import (  # noqa: F401
    safe_get,
    detect_format,
    load_trajectory,
)

# Re-export metric functions for backward compatibility
from .metrics import (  # noqa: F401
    build_message_metrics,
    compute_metrics,
    compute_health_verdict,
    validate_token_integrity,
    extract_agent_info,
    compute_agent_summary,
    generate_agent_insights,
    effective_agent,
)

# Re-export formatting functions for backward compatibility
from .formatting import (  # noqa: F401
    format_session_md,
    format_performance_md,
    format_behavioral_md,
    format_output_md,
    format_banner_html,
    wall_clock_fmt,
    build_analytics_dataframe,
    _build_hotspots_md,
    _build_per_message_md,
    _fmt_dict_as_table,
    _friendly_finish,
    _friendly_parts,
)

# Re-export label functions for backward compatibility
from .labels import (  # noqa: F401
    LABEL_PHASE_COLORS,
    load_labeled_json,
    aggregate_labels,
)


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

    # No reliable session total to disambiguate the two schemas (e.g. OpenCode
    # reports per-step input/output/cache but no `total`, so total==0). Default
    # to "input is already fresh" — the common modern convention where fresh
    # input is reported separately from cache_read. Subtracting cache_read here
    # would double-discount input that never contained the cached tokens.
    if total <= 0:
        return max(0, input_tokens or 0)

    # When no token breakdown is available (all components zero but total > 0),
    # we cannot infer fresh vs cached — treat entire total as fresh.
    if base == 0 and cache_read == 0:
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

    for p in parts_raw:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type", "unknown")

        if ptype == "text":
            txt = p.get("text", "")
            parts.append({"type": "text", "text": txt})
            if not text_preview:
                text_preview = txt
        elif ptype == "reasoning":
            parts.append({"type": "reasoning", "text": p.get("text", "")})
            has_reasoning = True
            if not text_preview:
                text_preview = p.get("text", "")
        elif ptype in ("tool_call", "tool"):
            state = p.get("state", {})
            if not isinstance(state, dict):
                state = {"status": str(state)}
            tool_name = p.get("tool_name", p.get("name", p.get("tool", "?")))
            status = state.get("status", p.get("status", "?"))
            # Input: OpenCode uses state.input, CodeArts uses p.arguments
            tool_input = state.get("input", p.get("input", p.get("arguments", {})))
            # Output: OpenCode uses state.output, CodeArts stores output in step-level toolOutput
            tool_output = state.get("output", p.get("output", ""))
            tc = {
                "type": "tool_call", "tool_name": tool_name,
                "tool_id": p.get("tool_id", p.get("callID", p.get("id", ""))), "status": status,
                "title": state.get("title", ""),
                "input": tool_input,
                "output": tool_output,
                "error": p.get("error") or state.get("error") or None,
                "error_type": p.get("error_type"),
                "time_start": safe_get(state, "time", "start", default=None),
                "time_end": safe_get(state, "time", "end", default=None),
                "duration_ms": safe_get(state, "metadata", "totalDurationMs", default=None),
                "metadata": state.get("metadata", {}),
            }
            parts.append(tc)
            tool_calls.append(tc)
            if status == "error":
                errors += 1
            if not text_preview:
                text_preview = f"[Tool: {tool_name}] {tc['title']}"
        elif ptype in ("step_start", "step-start"):
            parts.append({"type": "step_start", "name": p.get("name", "")})
        elif ptype in ("step_finish", "step-finish"):
            parts.append({"type": "step_finish", "name": p.get("name", "")})
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

    return parts, tool_calls, errors, has_reasoning, text_preview


def parse_steps(raw: dict) -> list[dict]:
    """Normalize each message in trajectory[] into a step dict."""
    # If already parsed by Claude Code converter, return directly
    if raw.get("_cc_format") and "_cc_parsed_steps" in raw:
        return raw["_cc_parsed_steps"]

    trajectory = raw.get("trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        trajectory = raw.get("messages", [])
    if not isinstance(trajectory, list):
        return []

    steps = []
    for idx, msg in enumerate(trajectory):
        if not isinstance(msg, dict):
            continue
        if raw.get("_analysis_profile") == "training":
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            parts = []
            text_preview = ""
            if reasoning:
                parts.append({"type": "reasoning", "text": reasoning})
                text_preview = reasoning
            if content:
                parts.append({"type": "text", "text": content})
                if not text_preview:
                    text_preview = content
            raw_tool_parts = msg.get("tool_call_parts", [])
            if not isinstance(raw_tool_parts, list):
                raw_tool_parts = []
            parsed_tool_parts, tool_calls, errors, _, tool_preview = _parse_parts(raw_tool_parts)
            parts.extend(parsed_tool_parts)
            if not text_preview and tool_preview:
                text_preview = tool_preview
            token_est = msg.get("tokens_estimate", {})
            tokens = {
                "total": 0,
                "input": 0,
                "output": 0,
                "reasoning": 0,
                "cache_read": 0,
                "cache_write": 0,
                "estimated": bool(token_est.get("estimated")),
                "estimated_total": token_est.get("estimated_total", 0) or 0,
                "estimated_content_tokens": token_est.get("estimated_content_tokens", 0) or 0,
                "estimated_reasoning_tokens": token_est.get("estimated_reasoning_tokens", 0) or 0,
                "estimated_content_chars": token_est.get("estimated_content_chars", 0) or 0,
                "estimated_reasoning_chars": token_est.get("estimated_reasoning_chars", 0) or 0,
            }
            steps.append({
                "index": msg.get("index", idx),
                "role": msg.get("role", "?"),
                "tokens": tokens,
                "duration": None,
                "parts": parts,
                "tool_calls": tool_calls,
                "tool_call_count": len(tool_calls),
                "error_count": errors,
                "has_reasoning": bool(reasoning),
                "text_preview": text_preview,
                "finish": "",
                "model_id": "",
                "provider_id": "",
                "time_created_ms": None,
                "time_completed_ms": None,
                "agent": "",
                "mode": "",
                "message_id": "",
                "id": "",
                "parent_id": "",
                "session_id": msg.get("tool_call_id", ""),
                "cwd": "",
                "root": "",
                "round": None,
                "is_sub_agent": False,
                "sub_agent_msg_list": [],
                "tool_output": None,
                "output_text": content,
                "agent_id": "",
                "question": [],
            })
            continue
        info = msg.get("info") if isinstance(msg.get("info"), dict) else {}
        role = msg.get("role") or safe_get(info, "role", default="?")

        tokens_info = safe_get(info, "tokens", default={})
        if not isinstance(tokens_info, dict):
            tokens_info = {}
        tokens = {
            "total": tokens_info.get("total", 0) or 0,
            "input": tokens_info.get("input", 0) or 0,
            "output": tokens_info.get("output", 0) or 0,
            "reasoning": tokens_info.get("reasoning", 0) or 0,
            "cache_read": safe_get(tokens_info, "cache", "read", default=0) or 0,
            "cache_write": safe_get(tokens_info, "cache", "write", default=0) or 0,
        }

        t_created = safe_get(info, "time", "created", default=None)
        t_completed = safe_get(info, "time", "completed", default=None)
        duration = None
        if isinstance(t_created, (int, float)) and isinstance(t_completed, (int, float)):
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
            "index": idx, "role": role, "tokens": tokens, "duration": duration,
            "parts": parts, "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls), "error_count": errors,
            "has_reasoning": has_reasoning, "text_preview": text_preview,
            "finish": finish,
            "model_id": safe_get(info, "modelID", default=""),
            "provider_id": safe_get(info, "providerID", default=""),
            "time_created_ms": t_created, "time_completed_ms": t_completed,
            "agent": safe_get(info, "agent", default=""),
            "mode": safe_get(info, "mode", default=""),
            "message_id": msg.get("message_id", ""),
            "id": safe_get(info, "id", default=""),
            "parent_id": safe_get(info, "parentID", default=""),
            "session_id": safe_get(info, "sessionID", default=""),
            "cwd": path_info.get("cwd", ""), "root": path_info.get("root", ""),
            # CodeArts-specific fields (absent for CC/OpenCode — safe defaults)
            "round": info.get("round"),
            "is_sub_agent": info.get("isSubAgent", False),
            "sub_agent_msg_list": info.get("subAgentMsgList", []),
            "tool_output": info.get("toolOutput"),
            "output_text": info.get("outputText", ""),
            "agent_id": info.get("agentId", ""),
            "question": info.get("question", []),
        })

    _fill_missing_last_step_duration(steps, raw)
    return steps


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
    if last.get("duration") not in (None, 0, 0.0):
        return
    start_ms = last.get("time_created_ms")
    if not isinstance(start_ms, (int, float)):
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
