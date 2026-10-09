"""Data loading, parsing, and aggregate metrics."""

import json
import math

from .formats.common import as_text
from .loaders import safe_get

# Fields whose absence makes the whole Metrics table unavailable. Reasoning is
# optional: many formats never report it (show per-row N/A instead of fake 0).
_TOKEN_METRIC_FIELDS = {
    "total": "Total Tokens",
    "input": "Input Tokens",
    "output": "Output Tokens",
}


def _optional_token_count(tokens: dict, key: str) -> int | None:
    """Return an int token count when *key* was reported, else None.

    NaN/Infinity are rejected: JSON permits those literals and ``int(inf)``
    raises ``OverflowError``, which would escape ``parse_steps``.
    """
    value = tokens.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return int(value)


def usable_token_count(value) -> int | None:
    """A token count that can be summed, or ``None`` when it cannot.

    A token count is a physical quantity: it is a finite, non-negative
    integer. Some exports violate that — OpenCode subtracts the cache read
    from the prompt size, which double-subtracts against providers whose
    ``input_tokens`` is already cache-exclusive, and writes a NEGATIVE input.
    Such a record cannot be repaired without guessing, so it is reported as
    unknown rather than summed into a total or clamped to a false zero.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return int(value)


def _finite_token(value) -> int | float:
    """A reported token value with NaN/Infinity replaced by 0."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(value):
        return 0
    return value


def cache_read_share(cache_read_tokens, total_tokens) -> float | None:
    """Cache-read share of one step's tokens, or ``None`` when that is not a share.

    Every supported format keeps the cache read INSIDE the step total —
    OpenCode and Claude Code as an addend, Codex nested inside ``input`` — so
    a self-consistent record satisfies ``0 <= cache_read <= total``. When it
    does not, the quotient is not a fraction and must not be rendered as a
    percentage: a corrupt OpenCode record produced a share above 25,000%
    reported as "strong cache reuse". The condition is per-step, not per-format.
    """
    cache_read = usable_token_count(cache_read_tokens)
    total = usable_token_count(total_tokens)
    if cache_read is None or total is None or total <= 0:
        return None
    if cache_read > total:
        return None
    return cache_read / total


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


def token_segments(
    total_tokens: int,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int = 0,
) -> dict:
    """Disjoint token segments that add up to the step's reported total.

    Formats disagree about how their fields nest, and so do models inside one
    format: Claude Code reports input, cache read, cache write and output as
    disjoint fields; Codex puts the cache read inside input and reasoning
    inside output; OpenCode/Opus keeps reasoning disjoint while OpenCode/GLM
    nests it inside output. Hard-coding any one reading drew the others wrong.

    Four readings, each with the total it implies:

    ==============================  ==============================
    fresh input, reasoning disjoint  input+output+reasoning+cache_read+cache_write
    fresh input, reasoning nested    input+output+cache_read+cache_write
    input incl. cache, disjoint      input+output+reasoning
    input incl. cache, nested        input+output
    ==============================  ==============================

    The reading whose implied total is closest to the observed total wins;
    ties go to the earlier row, which keeps the previous preference for a
    cache-exclusive input. Returns ``fresh``, ``cache_read``, ``cache_write``,
    ``output`` (excluding reasoning), ``reasoning`` and the chosen ``reading``.
    Segments are clamped at zero, so a corrupt step cannot draw a negative bar.
    """
    total = total_tokens or 0
    inp = input_tokens or 0
    out = output_tokens or 0
    rea = reasoning_tokens or 0
    cr = cache_read_tokens or 0
    cw = cache_write_tokens or 0

    if total > 0 and inp == 0 and out == 0 and rea == 0 and cr == 0 and cw == 0:
        # No breakdown at all: the whole total is all we know.
        return {"fresh": total, "cache_read": 0, "cache_write": 0, "output": 0,
                "reasoning": 0, "reading": "total_only"}

    readings = (
        ("fresh_disjoint", inp + out + rea + cr + cw),
        ("fresh_nested", inp + out + cr + cw),
        ("cached_disjoint", inp + out + rea),
        ("cached_nested", inp + out),
    )
    reading = min(readings, key=lambda r: abs(total - r[1]))[0]
    cache_exclusive = reading.startswith("fresh")
    nested = reading.endswith("nested")
    return {
        "fresh": max(0, inp if cache_exclusive else inp - cr),
        "cache_read": max(0, cr),
        "cache_write": max(0, cw) if cache_exclusive else 0,
        "output": max(0, out - rea if nested else out),
        "reasoning": max(0, rea),
        "reading": reading,
    }


def infer_non_cache_input(
    total_tokens: int,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int = 0,
) -> int:
    """Fresh (non-cache-read, non-cache-write) input tokens; see :func:`token_segments`."""
    return token_segments(total_tokens, input_tokens, output_tokens, reasoning_tokens,
                          cache_read_tokens, cache_write_tokens)["fresh"]


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


# ---------------------------------------------------------------------------
# Step-model type contract
# ---------------------------------------------------------------------------
# Converters copy these fields from the export as found, so a value of the
# wrong type (an id recorded as a list, a finish reason as a number, a tool
# output as a boolean) used to travel into every consumer that assumes a string
# and break the whole load at whichever one touched it first. The contract is
# enforced once, here, so the step model has stable types regardless of format.
# Real exports already satisfy it: normalisation is a no-op on well-formed data.
_STEP_TEXT_FIELDS = (
    "finish", "model_id", "provider_id", "agent", "mode", "message_id", "id",
    "parent_id", "session_id", "cwd", "root", "parent_session_id", "session_title",
    "text_preview", "message_type", "compaction_reason",
)
_STEP_TIME_FIELDS = ("time_created_ms", "time_completed_ms")
_TOOL_TEXT_FIELDS = ("tool_id", "title", "part_id", "session_id", "message_id")
_TOOL_TIME_FIELDS = ("time_created", "time_updated", "time_start", "time_end", "time_compacted")
# 9999-12-31T23:59:59.999Z, the last instant ``datetime`` can represent. A larger
# or non-positive epoch is not a clock reading, and formatting it raises.
_MAX_EPOCH_MS = 253_402_300_799_999


def _as_epoch_ms(value):
    """A plausible epoch-milliseconds number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0 or value > _MAX_EPOCH_MS:
        return None
    return value


def _as_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _normalize_tool_call(tc: dict) -> None:
    tc["tool_name"] = as_text(tc.get("tool_name")) or "?"
    tc["status"] = as_text(tc.get("status")) or "?"
    for key in _TOOL_TEXT_FIELDS:
        if key in tc:
            tc[key] = as_text(tc[key])
    for key in _TOOL_TIME_FIELDS:
        if key in tc:
            tc[key] = _as_epoch_ms(tc[key])
    if "duration_ms" in tc:
        tc["duration_ms"] = _as_number(tc["duration_ms"])
    inp = tc.get("input")
    if not isinstance(inp, (dict, str, list)):
        tc["input"] = as_text(inp) or {}
    out = tc.get("output")
    if not isinstance(out, (str, dict, list)):
        tc["output"] = as_text(out)
    err = tc.get("error")
    if isinstance(err, (dict, list)):
        # Keep the failure signal (truthiness) and its detail, as text.
        tc["error"] = json.dumps(err, default=str, ensure_ascii=False)[:2000]
    elif isinstance(err, bool):
        tc["error"] = "true" if err else None
    elif err is not None and not isinstance(err, str):
        tc["error"] = as_text(err) or None
    if "error_type" in tc and tc["error_type"] is not None and not isinstance(tc["error_type"], str):
        tc["error_type"] = as_text(tc["error_type"]) or None
    if "metadata" in tc and not isinstance(tc["metadata"], dict):
        tc["metadata"] = {}


def _normalize_step(step: dict) -> None:
    """Enforce the step model's field types in place (see the block comment above)."""
    role = step.get("role")
    if not isinstance(role, str) or not role:
        step["role"] = as_text(role) or "?"
    for key in _STEP_TEXT_FIELDS:
        if key in step:
            step[key] = as_text(step[key])
    for key in _STEP_TIME_FIELDS:
        if key in step:
            step[key] = _as_epoch_ms(step[key])
    if "duration" in step:
        step["duration"] = _as_number(step["duration"])
    if "session_depth" in step and (isinstance(step["session_depth"], bool)
                                    or not isinstance(step["session_depth"], int)):
        step["session_depth"] = None
    # Only coerced when present: Claude Code steps omit it on purpose and the
    # metrics layer branches on its absence. `summary` is deliberately left
    # alone — OpenCode records a dict there and consumers test `is True`.
    if "is_sub_agent" in step and not isinstance(step["is_sub_agent"], bool):
        flag = step["is_sub_agent"]
        step["is_sub_agent"] = bool(flag) if isinstance(flag, int) else False
    parts = step.get("parts")
    if not isinstance(parts, list):
        step["parts"] = parts = []
    for part in parts:
        if isinstance(part, dict):
            if "type" in part:
                part["type"] = as_text(part["type"])
            if "text" in part and not isinstance(part["text"], str):
                part["text"] = as_text(part["text"])
    calls = step.get("tool_calls")
    if not isinstance(calls, list):
        step["tool_calls"] = calls = []
    calls[:] = [tc for tc in calls if isinstance(tc, dict)]
    for tc in calls:
        _normalize_tool_call(tc)
    step["tool_call_count"] = len(calls)


def parse_steps(raw: dict) -> list[dict]:
    """Normalize each message in trajectory[] into a step dict."""
    # If already parsed by Claude Code converter, return directly (still backfill
    # the final step's duration, which the fast-path would otherwise skip).
    if raw.get("_cc_format") and "_cc_parsed_steps" in raw:
        steps = raw["_cc_parsed_steps"]
        for step in steps:
            _normalize_step(step)
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
        reasoning = _optional_token_count(tokens_info, "reasoning")
        # NaN/Infinity are dropped here rather than downstream: JSON permits
        # the bare literals, and a non-finite token count propagates into
        # every aggregate before raising deep inside statistics.median().
        # A NEGATIVE value is preserved — it is a real signal that the export
        # is self-inconsistent, and the metrics layer counts those steps
        # instead of silently summing them.
        tokens = {
            "total": _finite_token(tokens_info.get("total", 0)),
            "input": _finite_token(tokens_info.get("input", 0)),
            "output": _finite_token(tokens_info.get("output", 0)),
            "cache_read": _finite_token(safe_get(tokens_info, "cache", "read", default=0)),
            "cache_write": _finite_token(safe_get(tokens_info, "cache", "write", default=0)),
        }
        if reasoning is not None:
            tokens["reasoning"] = reasoning
        # Formats that log per-message window contributions (ICode) resolve
        # the live context-window occupancy in their converter; per-step
        # totals are that step's own tokens, not the window size.
        window = _optional_token_count(tokens_info, "context_window")
        if window is not None and window >= 0:
            tokens["context_window"] = window
        # A step that merged several model responses (Codex) also records its
        # largest single request: totals sum the responses, occupancy must not.
        peak = _optional_token_count(tokens_info, "prompt_peak")
        if peak is not None and peak >= 0:
            tokens["prompt_peak"] = peak
            peak_cached = _optional_token_count(tokens_info, "prompt_peak_cache_read")
            tokens["prompt_peak_cache_read"] = peak_cached if peak_cached is not None and 0 <= peak_cached <= peak else 0

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
        })

    for step in steps:
        _normalize_step(step)
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
