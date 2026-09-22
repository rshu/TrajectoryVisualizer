"""CodeArts SQLite export adapter on the OpenCode message schema."""

from .opencode import _convert_opencode_metadata

def _convert_codearts_metadata(raw: dict) -> dict:
    """Normalize a CodeArts SQLite export without changing message data.

    CodeArts deliberately shares the OpenCode message/part schema, so the
    existing OpenCode normalization is the correct base.  This adapter keeps
    that behavior while restoring CodeArts product identity, export metadata,
    and the parent/sub-agent counts already present in the consolidated file.
    Token values are per-message values, not cumulative totals.
    """
    _convert_opencode_metadata(raw)

    export = raw.get("export_metadata", {})
    if not isinstance(export, dict):
        export = {}
    exported_stats = raw.get("statistics", {})
    if not isinstance(exported_stats, dict):
        exported_stats = {}
    manifest = raw.get("session_manifest", [])
    if not isinstance(manifest, list):
        manifest = []

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        raw["metadata"] = metadata

    # Prefer the first explicit per-message model identifier.  User messages
    # commonly store it in ``info.model`` while assistant messages place it
    # directly in ``info.modelID``.
    model_id = metadata.get("model", "")
    for msg in raw.get("messages", []):
        if not isinstance(msg, dict):
            continue
        msg_info = msg.get("info", {})
        if not isinstance(msg_info, dict):
            continue
        nested_model = msg_info.get("model", {})
        candidate = msg_info.get("modelID")
        if not candidate and isinstance(nested_model, dict):
            candidate = nested_model.get("modelID")
        if candidate:
            model_id = candidate
            break

    # sub_agent_count/event_count were already derived by the shared
    # OpenCode normalization (isSubAgent messages, session_manifest, and
    # exporter statistics) — reuse them instead of recomputing.
    sub_agent_count = metadata.get("sub_agent_count", 0)
    event_count = metadata.get("event_count", 0)
    session_count = exported_stats.get("sessions")
    if not isinstance(session_count, int):
        session_count = len(manifest) or 1

    metadata.update({
        "agent": "codearts",
        "model": model_id,
        "generator_name": "codearts",
        "generator_version": metadata.get("server_version", ""),
        "format_version": str(export.get("schema_version", 2)),
        "sub_agent_count": sub_agent_count,
        "session_count": session_count,
        "event_count": event_count,
        "export_generated_at": export.get("generated_at", ""),
        "export_complete": export.get("complete"),
        "export_warnings": export.get("warnings", []),
    })

    # Use the authoritative exporter statistics where provided, while keeping
    # the tool-name breakdown calculated by the OpenCode-compatible adapter.
    stats = raw.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
        raw["stats"] = stats
    stats.update({
        "total_messages": exported_stats.get("total_messages", stats.get("total_messages", 0)),
        "user_messages": exported_stats.get("user_messages", stats.get("user_messages", 0)),
        "assistant_messages": exported_stats.get("assistant_messages", stats.get("assistant_messages", 0)),
        "total_tool_calls": exported_stats.get("tool_parts", stats.get("total_tool_calls", 0)),
        "reasoning_steps": exported_stats.get("reasoning_parts", stats.get("reasoning_steps", 0)),
        "sub_agent_count": sub_agent_count,
    })

    token_totals = {
        "total": 0, "input": 0, "output": 0, "reasoning": 0,
        "cache_read": 0, "cache_write": 0,
    }
    for msg in raw.get("messages", []):
        if not isinstance(msg, dict):
            continue
        msg_info = msg.get("info", {})
        tokens = msg_info.get("tokens", {}) if isinstance(msg_info, dict) else {}
        if not isinstance(tokens, dict):
            continue
        token_totals["total"] += tokens.get("total", 0) or 0
        token_totals["input"] += tokens.get("input", 0) or 0
        token_totals["output"] += tokens.get("output", 0) or 0
        token_totals["reasoning"] += tokens.get("reasoning", 0) or 0
        cache = tokens.get("cache", {})
        if isinstance(cache, dict):
            token_totals["cache_read"] += cache.get("read", 0) or 0
            token_totals["cache_write"] += cache.get("write", 0) or 0

    raw["token_usage"] = {
        "total_tokens": token_totals["total"],
        "prompt_tokens": token_totals["input"],
        "completion_tokens": token_totals["output"],
        "reasoning_tokens": token_totals["reasoning"],
        "cache_read_tokens": token_totals["cache_read"],
        "cache_write_tokens": token_totals["cache_write"],
    }
    raw["_codearts_format"] = True
    raw["_source_format"] = "codearts"
    raw["_capabilities"] = {
        "has_timing": True,
        "has_tool_calls": bool(stats.get("total_tool_calls")),
        "has_runtime_token_usage": True,
        "has_reasoning_content": bool(exported_stats.get("reasoning_parts")),
        "has_session_hierarchy": bool(sub_agent_count),
    }
    return raw
