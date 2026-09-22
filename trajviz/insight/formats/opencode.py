"""OpenCode info+messages export → internal trajectory."""

from datetime import datetime, UTC

def _convert_opencode_metadata(raw: dict) -> dict:
    """Populate metadata, timing, and output keys from OpenCode info structure.

    Mutates *raw* in place so that compute_metrics() and the session-detail
    renderer can read the standard fields.  Returns the same dict for convenience.
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
        started_at = datetime.fromtimestamp(created_ms / 1000.0, tz=UTC).isoformat()
    if isinstance(updated_ms, (int, float)):
        finished_at = datetime.fromtimestamp(updated_ms / 1000.0, tz=UTC).isoformat()
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

    # Consolidated OpenCode/CodeArts exports flatten child-session messages into
    # the main messages array.  Count distinct child session IDs instead of
    # hard-coding zero.  A session_manifest is also accepted so a child with
    # no persisted messages is still represented accurately.
    sub_agent_session_ids: set[str] = set()
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        message_info = message.get("info", {})
        if not isinstance(message_info, dict):
            continue
        agent_name = message_info.get("agent", "")
        is_sub_agent = bool(message_info.get("isSubAgent")) or (
            isinstance(agent_name, str) and agent_name.endswith("(subagent)")
        )
        if not is_sub_agent:
            continue
        child_session_id = message_info.get("sessionID")
        if isinstance(child_session_id, str) and child_session_id:
            sub_agent_session_ids.add(child_session_id)
    manifest = raw.get("session_manifest", [])
    for entry in manifest if isinstance(manifest, list) else []:
        if not isinstance(entry, dict) or not entry.get("depth"):
            continue
        manifest_info = entry.get("info", {})
        child_session_id = manifest_info.get("id") if isinstance(manifest_info, dict) else None
        if isinstance(child_session_id, str) and child_session_id:
            sub_agent_session_ids.add(child_session_id)
    exported_stats = raw.get("statistics", {})
    exported_sub_agent_count = (
        exported_stats.get("subagent_sessions", 0)
        if isinstance(exported_stats, dict)
        else 0
    )
    if not isinstance(exported_sub_agent_count, int):
        exported_sub_agent_count = 0
    sub_agent_count = max(len(sub_agent_session_ids), exported_sub_agent_count)
    exported_event_count = (
        exported_stats.get("event_rows", 0) if isinstance(exported_stats, dict) else 0
    )
    if not isinstance(exported_event_count, int):
        exported_event_count = 0
    manifest_entries = manifest if isinstance(manifest, list) else []
    manifest_event_count = sum(
        len(entry.get("events", []))
        for entry in manifest_entries if isinstance(entry, dict)
        if isinstance(entry.get("events", []), list)
    )
    event_count = max(exported_event_count, manifest_event_count)

    raw["metadata"] = {
        "session_id": info.get("id", ""),
        "slug": info.get("slug", ""),
        "title": info.get("title", ""),
        "directory": info.get("directory") or "",
        "directory_name": (info.get("directory") or "").replace("\\", "/").rsplit("/", 1)[-1],
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
        "sub_agent_count": sub_agent_count,
        "event_count": event_count,
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
