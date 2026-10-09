#!/usr/bin/env python3
"""Export complete CodeArts trajectories without rewriting persisted data.

The exporter supports both CodeArts storage generations:

* Legacy session folders with ``chat_baseInfo.json`` and one or more
  ``messages_<n>.json`` files.
* Current AgentKernel/OpenCode SQLite databases (``opencode.db``).

SQLite databases are opened in read-only mode.  Child sessions are discovered
from both the ``session.parent_id`` relationship and tool-result metadata.
Message token dictionaries and part timing dictionaries are preserved exactly;
the exporter never converts token totals to deltas and never replaces
``time.start``/``time.end`` with database row timestamps.

Examples:

    # Current CodeArts database
    python scripts/codearts_consolidator.py \
        C:/Users/me/.codeartsdoer/codearts-data/opencode.db \
        --session-id ses_abc -o trajectory.json

    # Session ID shorthand (uses CODEARTS_DATABASE/OPENCODE_DATABASE or the
    # standard ~/.codeartsdoer/codearts-data/opencode.db location)
    python scripts/codearts_consolidator.py ses_abc -o trajectory.json

    # Legacy CodeArts session folder
    python scripts/codearts_consolidator.py path/to/session-folder
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Sequence

try:
    from scripts import _common
except ImportError:  # Direct execution from scripts/.
    import _common  # type: ignore[no-redef]


SCHEMA_VERSION = 2
LEGACY_BASE_INFO_NAMES = ("chat_baseInfo.json", "chat_baseinfo.json")
LEGACY_MESSAGE_RE = re.compile(r"^messages_(\d+)\.json$", re.IGNORECASE)
REQUIRED_DB_COLUMNS = {
    # Keep the hard requirements deliberately small.  Older OpenCode-derived
    # CodeArts schemas lack newer metadata columns such as workspace_id and
    # share_url, but their persisted trajectory is still exportable.
    "session": {"id"},
    "message": {"id", "session_id", "data"},
    "part": {"id", "message_id", "session_id", "data"},
}


class ConsolidationError(RuntimeError):
    """Raised when input cannot be exported safely and completely."""


def _read_json(path: Path, expected_type: type) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConsolidationError(f"Cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, expected_type):
        raise ConsolidationError(
            f"Expected {expected_type.__name__} in {path}, got {type(value).__name__}"
        )
    return value


def _decode_db_json(value: Any, *, context: str, expected_type: type = dict) -> Any:
    """Decode a JSON database column without silently dropping corrupt rows."""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConsolidationError(f"Invalid UTF-8 in {context}: {exc}") from exc
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConsolidationError(f"Invalid JSON in {context}: {exc}") from exc
    if not isinstance(value, expected_type):
        raise ConsolidationError(
            f"Expected {expected_type.__name__} in {context}, got {type(value).__name__}"
        )
    return value


def _decode_optional_db_json(value: Any, *, context: str) -> Any:
    """Decode optional session JSON while retaining non-JSON legacy values."""
    if value is None or value == "":
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Some older builds stored plain strings in these columns.  Keeping
            # that string is lossless; silently omitting it would not be.
            return value
    return value


def _find_legacy_base_info(session_dir: Path) -> Path | None:
    if not session_dir.is_dir():
        return None
    files = {path.name.casefold(): path for path in session_dir.iterdir() if path.is_file()}
    for candidate in LEGACY_BASE_INFO_NAMES:
        match = files.get(candidate.casefold())
        if match is not None:
            return match
    return None


def _legacy_message_files(session_dir: Path) -> list[tuple[int, Path]]:
    matches: list[tuple[int, Path]] = []
    for path in session_dir.iterdir():
        if not path.is_file():
            continue
        match = LEGACY_MESSAGE_RE.match(path.name)
        if match:
            matches.append((int(match.group(1)), path))
    matches.sort(key=lambda item: (item[0], item[1].name.casefold()))
    return matches


def _missing_shard_summary(
    indices: Sequence[int], *, preview_limit: int = 20
) -> tuple[int, list[int]]:
    """Count missing shard indexes without allocating ``range(max_index)``."""
    missing_count = 0
    preview: list[int] = []
    previous = -1
    for index in indices:
        gap_start = previous + 1
        if index > gap_start:
            gap_size = index - gap_start
            missing_count += gap_size
            remaining = preview_limit - len(preview)
            if remaining > 0:
                preview.extend(range(gap_start, min(index, gap_start + remaining)))
        previous = index
    return missing_count, preview


def is_legacy_session_dir(path: Path) -> bool:
    files = _legacy_message_files(path) if path.is_dir() else []
    return _find_legacy_base_info(path) is not None and any(index == 0 for index, _ in files)


def consolidate_legacy_session(session_dir: str | Path) -> dict[str, Any]:
    """Merge every legacy ``messages_<n>.json`` shard in numeric order."""
    session_path = Path(session_dir).expanduser().resolve()
    base_path = _find_legacy_base_info(session_path)
    if base_path is None:
        raise ConsolidationError(f"No chat_baseInfo.json found in {session_path}")

    message_files = _legacy_message_files(session_path)
    if not message_files or message_files[0][0] != 0:
        raise ConsolidationError(f"No messages_0.json found in {session_path}")
    duplicate_indices = sorted(
        index
        for index in {item[0] for item in message_files}
        if sum(1 for candidate, _ in message_files if candidate == index) > 1
    )
    if duplicate_indices:
        raise ConsolidationError(
            "Ambiguous legacy message shards: multiple files map to indexes "
            f"{duplicate_indices}"
        )

    base_info: dict[str, Any] = _read_json(base_path, dict)
    messages: list[dict[str, Any]] = []
    shard_counts: list[dict[str, Any]] = []
    for index, path in message_files:
        shard: list[Any] = _read_json(path, list)
        invalid = [i for i, message in enumerate(shard) if not isinstance(message, dict)]
        if invalid:
            raise ConsolidationError(
                f"Expected message objects in {path}; invalid indexes: {invalid[:5]}"
            )
        messages.extend(shard)
        shard_counts.append({"index": index, "file": path.name, "messages": len(shard)})

    indices = [index for index, _ in message_files]
    missing_count, missing_preview = _missing_shard_summary(indices)
    session_id = str(base_info.get("chatId") or session_path.name)
    selected_agent = base_info.get("selectedGpt")
    selected_agent = selected_agent if isinstance(selected_agent, dict) else {}
    first_timestamp = messages[0].get("timestamp", "") if messages else ""
    last_timestamp = messages[-1].get("timestamp", "") if messages else ""
    user_messages = sum(1 for message in messages if message.get("sender") == "User")

    warnings: list[str] = []
    if missing_count:
        preview_text = f"{missing_preview}"
        if missing_count > len(missing_preview):
            preview_text += f" (first {len(missing_preview)} of {missing_count})"
        warnings.append(
            "Non-contiguous legacy message shards were found and all existing shards "
            f"were included; missing indexes: {preview_text}"
        )

    return {
        "format": "codearts",
        "metadata": {
            "session_id": session_id,
            "title": base_info.get("title", ""),
            "agent": selected_agent.get(
                "en_name", selected_agent.get("agent_id", "CodeArts")
            ),
            "agent_id": selected_agent.get("real_agent_id", ""),
            "model": "",
            "timestamp_utc": base_info.get("timestamp", ""),
            "context_tokens": base_info.get("contextToken", 0),
            "generator_name": "codearts_consolidator",
        },
        "timing": {"started_at": first_timestamp, "finished_at": last_timestamp},
        "messages": messages,
        "chat_base_info": base_info,
        "statistics": {
            "total_messages": len(messages),
            "user_messages": user_messages,
            "assistant_messages": len(messages) - user_messages,
            "message_shards": len(message_files),
        },
        "export_metadata": {
            "schema_version": SCHEMA_VERSION,
            "source_format": "codearts_legacy_json",
            "source_directory": session_path.name,
            "source_files": [base_path.name, *[path.name for _, path in message_files]],
            "shards": shard_counts,
            "message_policy": "preserved",
            "complete": not warnings,
            "warnings": warnings,
        },
    }


def _database_uri(path: Path) -> str:
    # Path.as_uri correctly escapes spaces and produces a SQLite-compatible
    # file:///C:/... URI on Windows.
    return f"{path.expanduser().resolve().as_uri()}?mode=ro"


def open_database_read_only(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise ConsolidationError(f"Database not found: {db_path}")
    try:
        connection = sqlite3.connect(_database_uri(db_path), uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        # A read-only URI prevents writes; an explicit transaction additionally
        # pins all SELECTs to one snapshot while AgentKernel continues writing
        # to its WAL in another process.
        connection.execute("BEGIN")
    except sqlite3.Error as exc:
        raise ConsolidationError(f"Cannot open database read-only at {db_path}: {exc}") from exc
    return connection


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}


def _row_value(row: sqlite3.Row, column: str, default: Any = None) -> Any:
    # sqlite3.Row has no __contains__ and iterates VALUES, so `.keys()` is required.
    return row[column] if column in row.keys() else default  # noqa: SIM118


def _validate_database_schema(connection: sqlite3.Connection) -> None:
    for table, required in REQUIRED_DB_COLUMNS.items():
        try:
            columns = _table_columns(connection, table)
        except sqlite3.Error as exc:
            raise ConsolidationError(f"Cannot inspect table {table}: {exc}") from exc
        missing = sorted(required.difference(columns))
        if missing:
            raise ConsolidationError(
                f"Unsupported CodeArts database: table {table} is missing columns {missing}"
            )


def _get_session_row(connection: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM session WHERE id = ?",
        (session_id,),
    ).fetchone()


def _session_info(row: sqlite3.Row) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "additions": _row_value(row, "summary_additions", 0),
        "deletions": _row_value(row, "summary_deletions", 0),
        "files": _row_value(row, "summary_files", 0),
    }
    diffs = _decode_optional_db_json(
        _row_value(row, "summary_diffs"), context=f"session {row['id']} summary_diffs"
    )
    if diffs is not None:
        summary["diffs"] = diffs

    time_info: dict[str, Any] = {
        "created": _row_value(row, "time_created"),
        "updated": _row_value(row, "time_updated"),
    }
    if _row_value(row, "time_compacting") is not None:
        time_info["compacting"] = _row_value(row, "time_compacting")
    if _row_value(row, "time_archived") is not None:
        time_info["archived"] = _row_value(row, "time_archived")

    info: dict[str, Any] = {
        "id": row["id"],
        "projectID": _row_value(row, "project_id"),
        "workspaceID": _row_value(row, "workspace_id"),
        "parentID": _row_value(row, "parent_id"),
        "slug": _row_value(row, "slug", ""),
        "directory": _row_value(row, "directory", ""),
        "title": _row_value(row, "title", ""),
        "version": _row_value(row, "version", ""),
        "shareURL": _row_value(row, "share_url"),
        "summary": summary,
        "time": time_info,
    }
    revert = _decode_optional_db_json(
        _row_value(row, "revert"), context=f"session {row['id']} revert"
    )
    permission = _decode_optional_db_json(
        _row_value(row, "permission"), context=f"session {row['id']} permission"
    )
    if revert is not None:
        info["revert"] = revert
    if permission is not None:
        info["permission"] = permission
    return info


def _optional_rows(
    connection: sqlite3.Connection,
    table: str,
    filter_column: str,
    filter_value: str,
) -> list[dict[str, Any]]:
    """Return rows from a known optional trajectory table, if available."""
    columns = _table_columns(connection, table)
    if not columns or filter_column not in columns:
        return []
    # The function is only called with fixed table/column names from this
    # module; quoting also keeps unusual future identifiers harmless.
    rows = connection.execute(
        f'SELECT * FROM "{table}" WHERE "{filter_column}" = ?',
        (filter_value,),
    ).fetchall()
    # sqlite3.Row iteration yields values, not keys — `.keys()` is required.
    return [{key: row[key] for key in row.keys()} for row in rows]  # noqa: SIM118


def _message_extra_map(
    connection: sqlite3.Connection, session_id: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _optional_rows(
        connection, "cag_message_extra", "session_id", session_id
    ):
        message_id = row.get("message_id")
        if isinstance(message_id, str) and message_id:
            grouped.setdefault(message_id, []).append(row)
    return grouped


def _attach_storage_fields(
    record: dict[str, Any],
    *,
    row_id: str,
    session_id: str,
    time_created: int | None,
    time_updated: int | None,
    message_id: str | None = None,
) -> None:
    """Add relational columns while leaving persisted JSON fields untouched."""
    record.setdefault("id", row_id)
    record.setdefault("sessionID", session_id)
    if message_id is not None:
        record.setdefault("messageID", message_id)

    storage: dict[str, Any] = {
        "id": row_id,
        "sessionID": session_id,
        "timeCreated": time_created,
        "timeUpdated": time_updated,
    }
    if message_id is not None:
        storage["messageID"] = message_id

    storage_key = "_codeartsStorage"
    suffix = 2
    while storage_key in record:
        storage_key = f"_codeartsStorage{suffix}"
        suffix += 1
    record[storage_key] = storage


def _message_parts(
    connection: sqlite3.Connection, message_id: str
) -> list[dict[str, Any]]:
    part_columns = _table_columns(connection, "part")
    order_by = "time_created, id" if "time_created" in part_columns else "id"
    rows = connection.execute(
        f"SELECT * FROM part WHERE message_id = ? ORDER BY {order_by}",
        (message_id,),
    ).fetchall()
    parts: list[dict[str, Any]] = []
    for row in rows:
        part = _decode_db_json(row["data"], context=f"part {row['id']}")
        _attach_storage_fields(
            part,
            row_id=row["id"],
            message_id=row["message_id"],
            session_id=row["session_id"],
            time_created=_row_value(row, "time_created"),
            time_updated=_row_value(row, "time_updated"),
        )
        parts.append(part)
    return parts


def _session_messages(
    connection: sqlite3.Connection,
    session_info: dict[str, Any],
    *,
    depth: int,
    traversal_parent_id: str | None,
) -> tuple[list[tuple[tuple[int, int, str], dict[str, Any]]], set[str]]:
    session_id = str(session_info["id"])
    message_columns = _table_columns(connection, "message")
    order_by = "time_created, id" if "time_created" in message_columns else "id"
    rows = connection.execute(
        f"SELECT * FROM message WHERE session_id = ? ORDER BY {order_by}",
        (session_id,),
    ).fetchall()

    messages: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
    metadata_children: set[str] = set()
    message_extras = _message_extra_map(connection, session_id)
    for ordinal, row in enumerate(rows):
        info = _decode_db_json(row["data"], context=f"message {row['id']}")
        _attach_storage_fields(
            info,
            row_id=row["id"],
            session_id=row["session_id"],
            time_created=_row_value(row, "time_created"),
            time_updated=_row_value(row, "time_updated"),
        )

        # Only fill a missing creation timestamp.  Existing raw message timing,
        # including completed timestamps, is never replaced.
        raw_time = info.get("time")
        if raw_time is None:
            info["time"] = {"created": _row_value(row, "time_created")}
        elif isinstance(raw_time, dict):
            raw_time.setdefault("created", _row_value(row, "time_created"))

        if depth > 0:
            if "isSubAgent" in info and info["isSubAgent"] is not True:
                info["_codeartsOriginalIsSubAgent"] = info["isSubAgent"]
            info["isSubAgent"] = True
            info.setdefault(
                "parentSessionID", session_info.get("parentID") or traversal_parent_id
            )
            info.setdefault("sessionDepth", depth)
            info.setdefault("sessionTitle", session_info.get("title", ""))

        parts = _message_parts(connection, row["id"])
        for part in parts:
            tool_name = part.get("tool")
            if (
                part.get("type") != "tool"
                or not isinstance(tool_name, str)
                or tool_name.casefold() not in {"task", "agent", "delegate", "subagent"}
            ):
                continue
            state = part.get("state")
            if not isinstance(state, dict):
                continue
            metadata = state.get("metadata")
            if not isinstance(metadata, dict):
                continue
            for key in ("sessionId", "sessionID", "session_id"):
                child_id = metadata.get(key)
                if isinstance(child_id, str) and child_id and child_id != session_id:
                    metadata_children.add(child_id)

        created = _row_value(row, "time_created")
        if not isinstance(created, (int, float)) and isinstance(info.get("time"), dict):
            created = info["time"].get("created")
        sort_created = int(created) if isinstance(created, (int, float)) else 0
        message_record: dict[str, Any] = {"info": info, "parts": parts}
        extras = message_extras.get(str(row["id"]), [])
        if extras:
            message_record["codearts_extra"] = extras
        messages.append(
            ((sort_created, ordinal, str(row["id"])), message_record)
        )
    return messages, metadata_children


def _relational_children(connection: sqlite3.Connection, session_id: str) -> set[str]:
    session_columns = _table_columns(connection, "session")
    if "parent_id" not in session_columns:
        return set()
    order_by = "time_created, id" if "time_created" in session_columns else "id"
    return {
        str(row[0])
        for row in connection.execute(
            f"SELECT id FROM session WHERE parent_id = ? ORDER BY {order_by}",
            (session_id,),
        ).fetchall()
    }


def _trajectory_statistics(
    messages: Sequence[dict[str, Any]], session_count: int
) -> dict[str, int]:
    role_counts: dict[str, int] = {}
    part_count = 0
    tool_parts = 0
    reasoning_parts = 0
    for message in messages:
        info = message.get("info", {})
        role = str(info.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        parts = message.get("parts", [])
        part_count += len(parts)
        tool_parts += sum(1 for part in parts if part.get("type") == "tool")
        reasoning_parts += sum(1 for part in parts if part.get("type") == "reasoning")
    return {
        "sessions": session_count,
        "subagent_sessions": max(0, session_count - 1),
        "total_messages": len(messages),
        "user_messages": role_counts.get("user", 0),
        "assistant_messages": role_counts.get("assistant", 0),
        "total_parts": part_count,
        "tool_parts": tool_parts,
        "reasoning_parts": reasoning_parts,
    }


def consolidate_database_session(
    database: str | Path,
    session_id: str,
    *,
    include_children: bool = True,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Export one current CodeArts session and, by default, all descendants."""
    if not session_id:
        raise ConsolidationError("A non-empty session ID is required for database export")
    if max_depth is not None and max_depth < 0:
        raise ConsolidationError("max_depth must be zero or greater")

    db_path = Path(database).expanduser().resolve()
    connection = open_database_read_only(db_path)
    try:
        _validate_database_schema(connection)
        root_row = _get_session_row(connection, session_id)
        if root_row is None:
            raise ConsolidationError(f"Session not found: {session_id}")

        root_info = _session_info(root_row)
        visited: set[str] = set()
        queue: list[tuple[str, int, str | None, str]] = [
            (session_id, 0, None, "root")
        ]
        sorted_messages: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
        session_manifest: list[dict[str, Any]] = []
        warnings: list[str] = []

        while queue:
            current_id, depth, traversal_parent, discovered_by = queue.pop(0)
            if current_id in visited:
                continue
            if max_depth is not None and depth > max_depth:
                warnings.append(
                    f"Skipped session {current_id}: maximum depth {max_depth} exceeded"
                )
                continue

            row = _get_session_row(connection, current_id)
            if row is None:
                warnings.append(
                    f"Referenced child session {current_id} does not exist in the database"
                )
                continue
            visited.add(current_id)
            info = _session_info(row)
            current_messages, metadata_children = _session_messages(
                connection,
                info,
                depth=depth,
                traversal_parent_id=traversal_parent,
            )
            sorted_messages.extend(current_messages)
            manifest_entry: dict[str, Any] = {
                "depth": depth,
                "discovered_by": discovered_by,
                "traversal_parent_id": traversal_parent,
                "message_count": len(current_messages),
                "info": info,
            }
            session_extras = _optional_rows(
                connection, "cag_session_extra", "session_id", current_id
            )
            todos = _optional_rows(connection, "todo", "session_id", current_id)
            events = _optional_rows(connection, "event", "aggregate_id", current_id)
            if session_extras:
                manifest_entry["codearts_extra"] = session_extras
            if todos:
                manifest_entry["todos"] = todos
            if events:
                manifest_entry["events"] = events
            session_manifest.append(manifest_entry)

            if not include_children:
                continue
            relational_children = _relational_children(connection, current_id)
            all_children = relational_children | metadata_children
            for child_id in sorted(all_children):
                if child_id in visited:
                    continue
                if child_id in relational_children and child_id in metadata_children:
                    relation = "parent_id+tool_metadata"
                elif child_id in relational_children:
                    relation = "parent_id"
                else:
                    relation = "tool_metadata"
                queue.append((child_id, depth + 1, current_id, relation))

        sorted_messages.sort(key=lambda item: item[0])
        messages = [message for _, message in sorted_messages]
        statistics = _trajectory_statistics(messages, len(session_manifest))
        statistics["session_extra_rows"] = sum(
            len(entry.get("codearts_extra", [])) for entry in session_manifest
        )
        statistics["message_extra_rows"] = sum(
            len(message.get("codearts_extra", [])) for message in messages
        )
        statistics["todo_rows"] = sum(
            len(entry.get("todos", [])) for entry in session_manifest
        )
        statistics["event_rows"] = sum(
            len(entry.get("events", [])) for entry in session_manifest
        )
        optional_tables = {
            table: ("included" if _table_columns(connection, table) else "not_present")
            for table in ("cag_session_extra", "cag_message_extra", "todo", "event")
        }
        # session_share contains a reusable secret.  It is intentionally never
        # embedded in an export, even when the table exists.
        optional_tables["session_share"] = "excluded_sensitive"
        return {
            "info": root_info,
            "messages": messages,
            "session_manifest": session_manifest,
            "statistics": statistics,
            "export_metadata": {
                "schema_version": SCHEMA_VERSION,
                "source_format": "codearts_opencode_sqlite",
                "source_database": db_path.name,
                "root_session_id": session_id,
                "generated_at": datetime.now(UTC).isoformat(),
                "database_mode": "read_only",
                "include_children": include_children,
                "max_depth": max_depth,
                "child_discovery": ["session.parent_id", "tool.state.metadata.sessionId"],
                "token_policy": "preserved",
                "message_time_policy": "preserved; missing created values filled from row",
                "part_time_policy": "preserved",
                "optional_tables": optional_tables,
                "complete": not warnings,
                "warnings": warnings,
            },
        }
    except sqlite3.Error as exc:
        raise ConsolidationError(f"Database error while exporting {session_id}: {exc}") from exc
    finally:
        connection.close()


def write_output(data: dict[str, Any], output_path: str | Path) -> None:
    """Write JSON atomically so an interrupted export cannot leave a partial file."""
    if str(output_path) == "-":
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _common.write_json_atomic(path, data)


def ensure_output_is_not_source(
    output_path: str | Path, source_paths: Iterable[str | Path]
) -> None:
    """Refuse to replace a database or legacy shard with exported JSON."""
    _common.ensure_output_does_not_overwrite(
        output_path,
        source_paths,
        exc=ConsolidationError,
        allow_stdout_dash=True,
    )


def _safe_filename_component(value: Any, fallback: str = "session") -> str:
    """Convert a persisted/session ID into one non-traversing filename part."""
    component = str(value or fallback)
    component = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", component)
    while ".." in component:
        component = component.replace("..", "__")
    component = component.strip(" .")
    if not component:
        component = fallback
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update({f"COM{index}" for index in range(1, 10)})
    reserved.update({f"LPT{index}" for index in range(1, 10)})
    if component.split(".", 1)[0].upper() in reserved:
        component = f"_{component}"
    # Leave room for the trajectory suffix on filesystems with a 255-byte-ish
    # component limit.  Unicode byte limits vary, but this also prevents absurd
    # names from malformed input.
    return component[:120].rstrip(" .") or fallback


def _default_database_path() -> Path:
    configured = os.environ.get("CODEARTS_DATABASE") or os.environ.get(
        "OPENCODE_DATABASE"
    )
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codeartsdoer" / "codearts-data" / "opencode.db"


def _resolve_mode(
    raw_input: str, explicit_session_id: str | None
) -> tuple[str, Path, str | None]:
    input_path = Path(raw_input).expanduser()
    if input_path.exists():
        resolved = input_path.resolve()
        if resolved.is_file():
            return "database", resolved, explicit_session_id
        database = resolved / "opencode.db"
        if database.is_file() and explicit_session_id:
            return "database", database, explicit_session_id
        if is_legacy_session_dir(resolved):
            return "legacy", resolved, None
        return "directory", resolved, explicit_session_id

    # Keep the existing opencode_consolidator convenience: a bare session ID
    # means "use the configured/default database".
    if raw_input.startswith("ses_"):
        if explicit_session_id and explicit_session_id != raw_input:
            raise ConsolidationError(
                f"Conflicting session IDs: {raw_input} and {explicit_session_id}"
            )
        return "database", _default_database_path().resolve(), raw_input
    raise ConsolidationError(f"Input not found: {input_path}")


def _print_summary(data: dict[str, Any], output: str | Path) -> None:
    stats = data.get("statistics", {})
    session_id = (
        data.get("metadata", {}).get("session_id")
        or data.get("info", {}).get("id")
        or "unknown"
    )
    print(f"Consolidated: {session_id}", file=sys.stderr)
    print(
        f"  Sessions: {stats.get('sessions', 1)}; "
        f"messages: {stats.get('total_messages', 0)}; "
        f"parts: {stats.get('total_parts', 0)}",
        file=sys.stderr,
    )
    print(f"  Output: {output}", file=sys.stderr)
    export_metadata = data.get("export_metadata", {})
    warnings = export_metadata.get("warnings", []) if isinstance(export_metadata, dict) else []
    for warning in warnings if isinstance(warnings, list) else []:
        print(f"  Warning: {warning}", file=sys.stderr)
    if isinstance(export_metadata, dict) and export_metadata.get("complete") is False:
        print("  Completeness: partial (see warnings above)", file=sys.stderr)


def _batch_legacy(
    parent: Path, output_dir: Path | None
) -> tuple[int, int, int, int]:
    processed = 0
    ignored = 0
    failed = 0
    partial = 0
    used_destinations: set[Path] = set()
    for subdir in sorted((path for path in parent.iterdir() if path.is_dir()), key=lambda p: p.name):
        if not is_legacy_session_dir(subdir):
            ignored += 1
            continue
        try:
            data = consolidate_legacy_session(subdir)
            session_id = data["metadata"]["session_id"]
            safe_session_id = _safe_filename_component(session_id)
            destination = (
                output_dir / f"{safe_session_id}_trajectory_v2.json"
                if output_dir is not None
                else subdir / f"{safe_session_id}_trajectory_v2.json"
            )
            resolved_destination = destination.resolve()
            if resolved_destination in used_destinations:
                raise ConsolidationError(
                    f"Multiple sessions map to the same output: {resolved_destination}"
                )
            ensure_output_is_not_source(
                destination,
                [
                    path
                    for path in [
                        _find_legacy_base_info(subdir),
                        *[path for _, path in _legacy_message_files(subdir)],
                    ]
                    if path is not None
                ],
            )
            used_destinations.add(resolved_destination)
            write_output(data, destination)
            _print_summary(data, destination)
            processed += 1
            if data.get("export_metadata", {}).get("complete") is False:
                partial += 1
        except (ConsolidationError, OSError) as exc:
            print(f"Error in {subdir.name}: {exc}", file=sys.stderr)
            failed += 1
    return processed, ignored, failed, partial


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a complete CodeArts trajectory from legacy JSON shards or the "
            "current opencode.db database."
        )
    )
    parser.add_argument(
        "input",
        help="Legacy session directory, opencode.db, database directory, or session ID",
    )
    parser.add_argument(
        "--session-id",
        help="Root session ID (required when input is a database path)",
    )
    parser.add_argument("--output", "-o", help="Output JSON path; use '-' for stdout")
    parser.add_argument(
        "--no-children",
        action="store_true",
        help="Export only the selected database session",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum child-session depth (default: unlimited)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Export all legacy session subdirectories under input",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        mode, source, session_id = _resolve_mode(args.input, args.session_id)

        if args.batch:
            if mode != "directory":
                raise ConsolidationError("--batch requires a parent directory")
            output_dir = Path(args.output).expanduser().resolve() if args.output else None
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
            processed, ignored, failed, partial = _batch_legacy(source, output_dir)
            print(
                f"Done: {processed} sessions exported, {ignored} ignored, "
                f"{failed} failed, {partial} partial.",
                file=sys.stderr,
            )
            if failed:
                return 1
            if partial:
                return 2
            return 0 if processed or not ignored else 1

        if mode == "legacy":
            if args.session_id:
                raise ConsolidationError("--session-id is only valid for database input")
            data = consolidate_legacy_session(source)
            safe_session_id = _safe_filename_component(data["metadata"]["session_id"])
            default_output = source / f"{safe_session_id}_trajectory_v2.json"
            protected_sources: list[Path] = [
                path
                for path in [
                    _find_legacy_base_info(source),
                    *[message_path for _, message_path in _legacy_message_files(source)],
                ]
                if path is not None
            ]
        elif mode == "database":
            if not session_id:
                raise ConsolidationError("--session-id is required for database input")
            data = consolidate_database_session(
                source,
                session_id,
                include_children=not args.no_children,
                max_depth=args.max_depth,
            )
            safe_session_id = _safe_filename_component(session_id)
            default_output = Path.cwd() / f"trajectory_{safe_session_id}_v2.json"
            protected_sources = [
                source,
                Path(f"{source}-wal"),
                Path(f"{source}-shm"),
            ]
        else:
            raise ConsolidationError(
                f"{source} is neither a legacy session nor an opencode.db directory"
            )

        output: str | Path = args.output or default_output
        ensure_output_is_not_source(output, protected_sources)
        write_output(data, output)
        _print_summary(data, output)
        return 2 if data.get("export_metadata", {}).get("complete") is False else 0
    except (ConsolidationError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
