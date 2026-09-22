"""Codex CLI JSONL → internal trajectory."""

import json
import re

from .common import _iso_to_epoch_ms

def _convert_codex_to_internal(events: list[dict]) -> dict:
    """Convert Codex CLI JSONL events into the trajviz internal format.

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
        if isinstance(e, dict) and e.get("type") == "session_meta":
            payload = e.get("payload")
            session_meta = payload if isinstance(payload, dict) else {}
            break

    # Group events into assistant turns.
    # Pattern: user message → (reasoning → assistant text → function_calls → function_call_outputs)* → task_complete
    messages: list[dict] = []
    pending_tool_calls: dict[str, dict] = {}  # call_id -> function_call payload
    current_parts: list[dict] = []
    current_role = None
    current_timestamp = None
    current_tokens: dict | None = None
    previous_cumulative_usage: dict | None = None
    previous_usage_snapshot: tuple | None = None

    usage_fields = (
        "total_tokens", "input_tokens", "output_tokens",
        "reasoning_output_tokens", "cached_input_tokens",
    )

    def _usage_value(usage: dict, field: str) -> int | float:
        value = usage.get(field, 0)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    def _flush_message():
        nonlocal current_parts, current_role, current_timestamp, current_tokens
        if current_parts and current_role:
            info_block = {
                "role": current_role,
                "time": {"created": current_timestamp or 0},
            }
            if current_tokens:
                info_block["tokens"] = current_tokens
            messages.append({"info": info_block, "parts": current_parts})
        current_parts = []
        current_role = None
        current_timestamp = None
        current_tokens = None

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        # Codex records timestamps as ISO-8601 strings; the internal contract is
        # epoch milliseconds, so convert here (parse_steps and every timing
        # consumer discard non-numeric timestamps).
        ts = _iso_to_epoch_ms(event.get("timestamp"))

        if etype == "response_item":
            payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
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
                    if not isinstance(content, dict):
                        continue
                    ctype = content.get("type", "")
                    text = content.get("text", "")
                    if ctype == "output_text" or ctype == "input_text":
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
                summary_text = " ".join(
                    s.get("text", "") for s in summary if isinstance(s, dict)
                ) if isinstance(summary, list) else ""
                current_parts.append({"type": "reasoning", "text": summary_text})

            elif item_type in ("function_call", "custom_tool_call"):
                if current_role != "assistant" and current_parts:
                    _flush_message()
                current_role = "assistant"
                if not current_timestamp:
                    current_timestamp = ts

                call_id = payload.get("call_id", "")
                name = payload.get("name", "exec_command")
                args_str = payload.get("arguments") or payload.get("input") or "{}"
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"raw": args_str}
                if not isinstance(args, dict):
                    args = {"raw": args_str}

                # Determine tool name and build normalized input
                cmd = args.get("cmd") or args.get("command") or ""
                if not cmd and isinstance(args_str, str):
                    if name == "exec":
                        commands = _extract_codex_exec_commands(args_str)
                        cmd = " ; ".join(commands) if commands else args_str
                    elif name == "apply_patch":
                        cmd = args_str
                tool_name = _classify_codex_command(name, cmd)
                normalized_input = _build_codex_tool_input(tool_name, cmd, args, name)

                pending_tool_calls[call_id] = {
                    "name": name,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "input": normalized_input,
                    "cmd": cmd,
                }

            elif item_type in ("function_call_output", "custom_tool_call_output"):
                # An output can arrive after task_complete flushed the turn
                # (role/timestamp reset to None); restore them so the final
                # flush's role guard does not silently drop this part.
                current_role = current_role or "assistant"
                current_timestamp = current_timestamp or ts
                call_id = payload.get("call_id", "")
                output = payload.get("output", "")
                tc = pending_tool_calls.pop(call_id, {})

                # Determine status from output.  Structured metadata
                # (metadata.exit_code) is authoritative; the substring
                # heuristic is only a fallback when no exit code exists.
                status = "success"
                if isinstance(output, str):
                    exit_code = None
                    try:
                        output_data = json.loads(output)
                    except json.JSONDecodeError:
                        output_data = None
                    if isinstance(output_data, dict):
                        metadata = output_data.get("metadata")
                        if isinstance(metadata, dict):
                            candidate = metadata.get("exit_code")
                            if isinstance(candidate, int) and not isinstance(candidate, bool):
                                exit_code = candidate
                    if exit_code is not None:
                        status = "error" if exit_code != 0 else "success"
                    else:
                        # Anchored fallback so benign text ("Found 0 errors",
                        # "error-free") does not flag a successful call.
                        if re.search(r"(?i)\b(?:error:|traceback \(most recent call last\))",
                                     output[:200]):
                            status = "error"
                        # Check exit code reported in the output text
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
            payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
            msg_type = payload.get("type", "")
            if msg_type == "token_count":
                # last_token_usage is the per-response delta. Codex can repeat an
                # identical notification, so use the cumulative + last snapshots
                # together to de-duplicate it. Cumulative counters alone are not a
                # safe key: parallel agents can report the same cumulative value
                # with different, legitimate last-usage deltas.
                usage = payload.get("info", {}) if isinstance(payload.get("info"), dict) else {}
                last_usage = usage.get("last_token_usage")
                cumulative = usage.get("total_token_usage")
                tu = None
                cumulative_snapshot = (
                    tuple(_usage_value(cumulative, field) for field in usage_fields)
                    if isinstance(cumulative, dict) else None
                )
                last_snapshot = (
                    tuple(_usage_value(last_usage, field) for field in usage_fields)
                    if isinstance(last_usage, dict) else None
                )
                snapshot = (cumulative_snapshot, last_snapshot)
                if snapshot != (None, None):
                    if snapshot == previous_usage_snapshot:
                        continue
                    previous_usage_snapshot = snapshot

                if isinstance(last_usage, dict):
                    tu = last_usage
                elif isinstance(cumulative, dict):
                    if previous_cumulative_usage is None or any(
                        _usage_value(cumulative, field)
                        < _usage_value(previous_cumulative_usage, field)
                        for field in usage_fields
                    ):
                        tu = cumulative
                    else:
                        tu = {
                            field: _usage_value(cumulative, field)
                            - _usage_value(previous_cumulative_usage, field)
                            for field in usage_fields
                        }

                if isinstance(cumulative, dict):
                    previous_cumulative_usage = {
                        field: _usage_value(cumulative, field) for field in usage_fields
                    }

                if isinstance(tu, dict):
                    token_delta = {
                        "total": _usage_value(tu, "total_tokens"),
                        "input": _usage_value(tu, "input_tokens"),
                        "output": _usage_value(tu, "output_tokens"),
                        "reasoning": _usage_value(tu, "reasoning_output_tokens"),
                        "cache": {"read": _usage_value(tu, "cached_input_tokens"), "write": 0},
                    }
                    if not any(token_delta[field] for field in ("total", "input", "output", "reasoning")) \
                            and not token_delta["cache"]["read"]:
                        continue
                    if current_tokens is None:
                        current_tokens = token_delta
                    else:
                        # A displayed step may contain several model responses
                        # (for example commentary followed by a tool call). Each
                        # response has its own last_token_usage delta, so retain
                        # all of them instead of replacing the earlier usage.
                        for field in ("total", "input", "output", "reasoning"):
                            current_tokens[field] += token_delta[field]
                        current_tokens["cache"]["read"] += token_delta["cache"]["read"]
            elif msg_type == "task_complete":
                _flush_message()

    # Drain tool calls that never received a function_call_output (session
    # interrupted/truncated mid-command) so the final — often most diagnostic —
    # invocation is not silently dropped from the timeline.
    if pending_tool_calls:
        current_role = current_role or "assistant"
        for call_id, tc in pending_tool_calls.items():
            current_parts.append({
                "type": "tool_call",
                "tool_name": tc.get("tool_name", "Bash"),
                "tool_id": call_id,
                "status": "error",  # interrupted: call never produced an output
                "input": tc.get("input", {}),
                "output": "",
            })
        pending_tool_calls.clear()

    # Flush any remaining parts
    _flush_message()

    # Approximate each turn's completion as the next turn's start so per-step
    # durations exist (Codex is single-session; parse_steps backfills the final
    # turn from the session end timestamp).
    for i in range(len(messages) - 1):
        cur_t = messages[i]["info"]["time"].get("created")
        nxt_t = messages[i + 1]["info"]["time"].get("created")
        if isinstance(cur_t, (int, float)) and isinstance(nxt_t, (int, float)) and nxt_t >= cur_t:
            messages[i]["info"]["time"]["completed"] = nxt_t

    first_ts_iso = events[0].get("timestamp") if events and isinstance(events[0], dict) else None
    last_ts_iso = events[-1].get("timestamp") if events and isinstance(events[-1], dict) else None
    directory = session_meta.get("cwd", "") or ""
    start_ms = _iso_to_epoch_ms(first_ts_iso)
    end_ms = _iso_to_epoch_ms(last_ts_iso)
    total_duration = (
        round((end_ms - start_ms) / 1000.0, 3)
        if isinstance(start_ms, int) and isinstance(end_ms, int) else 0
    )
    total_tokens = sum(
        (m["info"].get("tokens") or {}).get("total", 0) for m in messages
    )

    # Build metadata
    info = {
        "id": session_meta.get("id", ""),
        "slug": "",
        "projectID": "",
        "directory": directory,
        "title": "",
        "version": session_meta.get("cli_version", ""),
        "time": {"created": start_ms or 0, "updated": end_ms or 0},
    }

    return {
        "info": info,
        "messages": messages,
        "metadata": {
            "session_id": session_meta.get("id", ""),
            "directory": directory,
            "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1],
            "agent": "codex",
            "model": session_meta.get("model", "") or "",
            "source": "codex",
            "model_provider": session_meta.get("model_provider", "openai"),
            "originator": session_meta.get("originator", "Codex CLI"),
            "server_version": session_meta.get("cli_version", ""),
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
        "_codex_format": True,
    }

def _extract_codex_exec_commands(source: str) -> list[str]:
    """Extract JSON-quoted ``cmd`` values from a modern custom ``exec`` input."""
    if not isinstance(source, str):
        return []
    pattern = re.compile(r'(?:["\']?cmd["\']?)\s*:\s*("(?:\\.|[^"\\])*")')
    commands = []
    for match in pattern.finditer(source):
        try:
            command = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(command, str):
            commands.append(command)
    return commands


def _classify_codex_command(func_name: str, cmd: str) -> str:
    """Map a Codex exec_command to a trajviz tool name.

    Codex uses exec_command for everything; we infer the intent from the command.
    """
    func_lower = str(func_name).lower().strip()
    cmd_lower = str(cmd).lower().strip()

    if func_lower == "apply_patch" or "tools.apply_patch" in cmd_lower:
        return "Write"
    if not cmd_lower and func_lower not in ("exec", "exec_command"):
        # Preserve non-shell Codex tools (for example spawn_agent or wait)
        # instead of flattening every call into an empty Bash command.
        return str(func_name) or "Bash"

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

    # Test, git, python, and all other shell commands are Bash
    return "Bash"


def _build_codex_tool_input(
    tool_name: str,
    cmd: str,
    raw_args: dict,
    func_name: str = "",
) -> dict:
    """Build a normalized input dict for Codex commands.

    Maps the raw Codex exec_command args into the format that
    canonical.py expects for each tool type:
    - Read: {"file_path": "..."}
    - Write: {"file_path": "..."}
    - Grep/Glob: {"pattern": "...", "path": "..."}
    - Bash: {"command": "..."}
    """
    raw_text = raw_args.get("raw", "") if isinstance(raw_args, dict) else ""

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
        patch_match = re.search(
            r"^\*\*\* (?:Add|Update|Delete) File: (.+)$",
            cmd,
            flags=re.MULTILINE,
        )
        if patch_match:
            file_path = patch_match.group(1).strip()
        else:
            parts = cmd.split()
            file_path = ""
            for p in reversed(parts):
                if not p.startswith("-") and ("." in p or "/" in p):
                    file_path = p
                    break
        result = {"file_path": file_path, "command": cmd}
        if str(func_name).lower() == "apply_patch" or cmd.startswith("*** Begin Patch"):
            result["patch"] = raw_text or cmd
        return result

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
        if not cmd and isinstance(raw_args, dict) and raw_args:
            return dict(raw_args)
        result = {"command": cmd}
        if raw_text and raw_text != cmd:
            result["raw_input"] = raw_text
        return result
