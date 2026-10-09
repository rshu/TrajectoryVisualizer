#!/usr/bin/env python3
"""
Script to recursively export opencode sessions with all subagent sessions.

This script exports a parent session and all its subagent sessions into a single JSON file
with a flat structure. It finds child sessions by looking for completed tool calls with
sessionId in their metadata.

Usage:
    python scripts/opencode_consolidator.py <session-id> [output-file]

    If output-file is "-", output is written to stdout.

    Environment variables:
        OPENCODE_DATABASE: Path to opencode database (highest priority).
        OPENCODE_DB: OpenCode's native database path override.
        OPENCODE_DATA_DIR: OpenCode data directory (looks for opencode.db inside).
        XDG_DATA_HOME: XDG data root (default file: $XDG_DATA_HOME/opencode/opencode.db).

        Default file, when none of the above are set:
            ~/.local/share/opencode/opencode.db
        On Windows, if that file is missing, also checks:
            %LOCALAPPDATA%\\opencode\\opencode.db
            %APPDATA%\\opencode\\opencode.db

Example:
    # Export to file
    python scripts/opencode_consolidator.py ses_123abc456 export.json

    # Export to stdout
    python scripts/opencode_consolidator.py ses_123abc456 -

    # Use custom database path (POSIX)
    OPENCODE_DATABASE=/custom/path/opencode.db python scripts/opencode_consolidator.py ses_123abc456 output.json

    # Use custom database path (Windows PowerShell)
    $env:OPENCODE_DATABASE = "$env:USERPROFILE\\.local\\share\\opencode\\opencode.db"
    python scripts/opencode_consolidator.py ses_123abc456 output.json

Output structure:
    {
        "sessions": [
            {
                "info": { ... session info ... },
                "messages": [ ... session messages ... ]
            },
            ...
        ]
    }
"""

import sys
import json
import sqlite3
import os
from pathlib import Path
from typing import Any
from collections.abc import Iterable

try:
    from scripts import _common
except ImportError:  # Direct execution from scripts/.
    import _common  # type: ignore[no-redef]


def _on_windows() -> bool:
    return os.name == "nt"


def _database_uri(path: Path) -> str:
    """Read-only SQLite URI. Path.as_uri() emits file:///C:/... on Windows."""
    return f"{Path(path).expanduser().resolve().as_uri()}?mode=ro"


def _default_data_dirs() -> list[Path]:
    """Directories OpenCode may use for ``opencode.db``.

    OpenCode's own default (xdg-basedir) is ``~/.local/share/opencode`` on every
    OS, including Windows. Explicit ``OPENCODE_DATA_DIR`` / ``XDG_DATA_HOME``
    overrides replace that default. When no override is set on Windows, also
    search ``%LOCALAPPDATA%\\opencode`` and ``%APPDATA%\\opencode`` so outdated
    docs and native Windows layouts still resolve.
    """
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            dirs.append(path)

    if data := os.environ.get("OPENCODE_DATA_DIR"):
        add(Path(data).expanduser())
        return dirs

    if xdg := os.environ.get("XDG_DATA_HOME"):
        add(Path(xdg).expanduser() / "opencode")
        return dirs

    add(Path.home() / ".local" / "share" / "opencode")
    if _on_windows():
        for env_key in ("LOCALAPPDATA", "APPDATA"):
            value = os.environ.get(env_key)
            if value:
                add(Path(value).expanduser() / "opencode")
    return dirs


def get_db_path() -> Path:
    """Resolve the opencode SQLite database path.

    Precedence:
        1. ``OPENCODE_DATABASE`` (this tool's override)
        2. ``OPENCODE_DB`` (OpenCode's native override; absolute as-is, else
           relative to the primary data directory)
        3. ``opencode.db`` in the first existing candidate data directory
        4. ``opencode.db`` in the primary data directory (for the not-found error)
    """
    if configured := os.environ.get("OPENCODE_DATABASE"):
        return Path(configured).expanduser()

    data_dirs = _default_data_dirs()
    primary = data_dirs[0]

    opencode_db = os.environ.get("OPENCODE_DB")
    if opencode_db and opencode_db != ":memory:":
        named = Path(opencode_db).expanduser()
        if named.is_absolute():
            return named
        return primary / opencode_db

    for data_dir in data_dirs:
        candidate = data_dir / "opencode.db"
        if candidate.is_file():
            return candidate
    return primary / "opencode.db"


def export_session_and_collect_children(
    conn: sqlite3.Connection,
    session_id: str,
    visited: set[str],
    max_depth: int = 10,
    current_depth: int = 0,
    main_session_info: dict[str, Any] | None = None,
    all_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Export a session and recursively export all child sessions.

    Returns:
        A dict with 'info' (from the main/parent session) and 'messages' (from all sessions).
        Messages from subagent sessions have their agent name prefixed with the subagent type.
    """
    if all_messages is None:
        all_messages = []

    if current_depth >= max_depth:
        print(f"Warning: Max depth {max_depth} reached at session {session_id}", file=sys.stderr)
        return {}

    if session_id in visited:
        # With a shared visited set this covers both cycles and diamonds
        # (a session reachable through two different parents).
        print(f"Warning: Session {session_id} already exported, skipping", file=sys.stderr)
        return {}

    visited.add(session_id)

    # Get session info
    cursor = conn.execute(
        """
        SELECT id, project_id, workspace_id, parent_id, slug, directory, title, version,
               summary_additions, summary_deletions, summary_files,
               time_created, time_updated, time_compacting, time_archived
        FROM session WHERE id = ?
        """,
        (session_id,),
    )
    session_row = cursor.fetchone()

    if not session_row:
        raise ValueError(f"Session not found: {session_id}")

    session_info = {
        "id": session_row[0],
        "project_id": session_row[1],
        "workspace_id": session_row[2],
        "parent_id": session_row[3],
        "slug": session_row[4],
        "directory": session_row[5],
        "title": session_row[6],
        "version": session_row[7],
        "summary": {
            "additions": session_row[8],
            "deletions": session_row[9],
            "files": session_row[10],
        },
        "time": {
            "created": session_row[11],
            "updated": session_row[12],
        },
    }

    if session_row[13] is not None:
        session_info["time"]["compacting"] = session_row[13]
    if session_row[14] is not None:
        session_info["time"]["archived"] = session_row[14]

    # Get messages for this session
    messages = []
    cursor.execute(
        """
        SELECT id, session_id, time_created, time_updated, data
        FROM message WHERE session_id = ?
        ORDER BY time_created, id
        """,
        (session_id,),
    )

    for msg_row in cursor.fetchall():
        msg_id, msg_session_id, msg_time_created, msg_time_updated, msg_data = msg_row

        # Parse message data
        try:
            msg_info = json.loads(msg_data) if isinstance(msg_data, str) else msg_data
            msg_info["id"] = msg_id
            msg_info["sessionID"] = msg_session_id
            # Fill-only merge: never replace timing persisted inside the
            # message payload (especially time.completed, the model's real
            # finish time) with DB row write timestamps.  Row times are used
            # only when the payload carries no timing of its own.
            raw_time = msg_info.get("time")
            if not isinstance(raw_time, dict):
                raw_time = {}
                msg_info["time"] = raw_time
            raw_time.setdefault("created", msg_time_created)
            if "completed" not in raw_time:
                raw_time.setdefault("updated", msg_time_updated)

            # Get parts for this message
            parts = []
            cursor.execute(
                """
                SELECT id, message_id, session_id, time_created, time_updated, data
                FROM part WHERE message_id = ?
                ORDER BY time_created, id
                """,
                (msg_id,),
            )

            for part_row in cursor.fetchall():
                part_id, part_message_id, part_session_id, part_time_created, part_time_updated, part_data = (
                    part_row
                )
                try:
                    part = json.loads(part_data) if isinstance(part_data, str) else part_data
                    part["id"] = part_id
                    part["messageID"] = part_message_id
                    part["sessionID"] = part_session_id
                    part["time"] = {
                        "created": part_time_created,
                        "updated": part_time_updated,
                    }
                    parts.append(part)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse part data for {part_id}: {e}", file=sys.stderr)

            msg_info["parts"] = parts
            messages.append(msg_info)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse message data for {msg_id}: {e}", file=sys.stderr)

    # Get summary_diffs and revert data (these are stored as JSON text in the session table)
    cursor.execute(
        """
        SELECT summary_diffs, revert, permission
        FROM session WHERE id = ?
        """,
        (session_id,),
    )
    diff_row = cursor.fetchone()
    if diff_row[0]:
        try:
            session_info["summary"]["diffs"] = json.loads(diff_row[0]) if isinstance(diff_row[0], str) else diff_row[0]
        except json.JSONDecodeError:
            pass
    if diff_row[1]:
        try:
            session_info["revert"] = json.loads(diff_row[1]) if isinstance(diff_row[1], str) else diff_row[1]
        except json.JSONDecodeError:
            pass
    if diff_row[2]:
        try:
            session_info["permission"] = json.loads(diff_row[2]) if isinstance(diff_row[2], str) else diff_row[2]
        except json.JSONDecodeError:
            pass

    # If this is the first session (main session), save its info
    if main_session_info is None:
        main_session_info = session_info

    # Modify agent name for subagent messages and structure as needed
    is_subagent = current_depth > 0 or session_info.get("parent_id") is not None
    subagent_type = None
    if is_subagent:
        # Extract subagent type from title (e.g., "@general subagent")
        title = session_info.get("title", "")
        if "@general subagent" in title:
            subagent_type = "general"
        else:
            subagent_type = session_info.get("agent", "")

    # Restructure messages to have "info" and "parts" fields for trajviz compatibility
    restructured_messages = []
    for msg in messages:
        # Move message-level fields into "info" if they're at top level
        if "info" not in msg:
            msg_info = {}
            keys_to_move = ["role", "time", "agent", "model", "tokens", "system", "finish", "mode", "path",
                           "cost", "id", "sessionID", "parentID", "summary",
                           "modelID", "providerID"]  # Add these for OpenCode format
            for key in keys_to_move:
                if key in msg:
                    msg_info[key] = msg.pop(key)
            msg["info"] = msg_info

        # Convert time.updated to time.completed for assistant messages (duration calculation)
        if msg["info"].get("role") == "assistant" and "time" in msg["info"]:
            time_info = msg["info"]["time"]
            if "updated" in time_info and "completed" not in time_info:
                time_info["completed"] = time_info.pop("updated")
        # Remove time.updated from user messages to match original format
        elif msg["info"].get("role") == "user" and "time" in msg["info"]:
            time_info = msg["info"]["time"]
            time_info.pop("updated", None)

        # Modify agent name for subagent messages
        if is_subagent and "agent" in msg["info"]:
            agent_name = msg["info"].get("agent", "")
            if subagent_type and not agent_name.endswith("(subagent)"):
                msg["info"]["agent"] = f"{agent_name} (subagent)"

        restructured_messages.append(msg)

    # Replace messages list
    messages[:] = restructured_messages

    # Add this session's messages to all_messages
    all_messages.extend(messages)

    # Find child session IDs from tool metadata
    child_session_ids = set()
    for msg in messages:
        for part in msg.get("parts", []):
            if (
                part.get("type") == "tool"
                and part.get("state", {}).get("status") == "completed"
                and "metadata" in part.get("state", {})
            ):
                child_id = part["state"]["metadata"].get("sessionId")
                if child_id and child_id != session_id:
                    child_session_ids.add(child_id)

    # Recursively export children
    for child_id in sorted(child_session_ids):
        try:
            export_session_and_collect_children(
                conn,
                child_id,
                visited,  # Shared set: each session is exported exactly once
                max_depth,
                current_depth + 1,
                main_session_info,
                all_messages,
            )
        except Exception as e:
            print(f"Warning: Failed to export child session {child_id}: {e}", file=sys.stderr)

    return {
        "info": main_session_info,
        "messages": all_messages,
    }


def print_usage():
    """Print usage information."""
    print(f"Usage: {sys.argv[0]} <session-id> [output-file]", file=sys.stderr)
    print("\nArguments:", file=sys.stderr)
    print("  session-id    The session ID to export (e.g., ses_123abc456)", file=sys.stderr)
    print("  output-file   Optional output path. Use '-' for stdout.", file=sys.stderr)
    print("                Defaults to 'export-<session-id>.json'", file=sys.stderr)
    print("\nEnvironment variables:", file=sys.stderr)
    print("  OPENCODE_DATABASE  Path to opencode database (highest priority)", file=sys.stderr)
    print("  OPENCODE_DB        OpenCode's native database path override", file=sys.stderr)
    print("  OPENCODE_DATA_DIR  OpenCode data directory (opencode.db inside)", file=sys.stderr)
    print("  XDG_DATA_HOME      XDG data root ($XDG_DATA_HOME/opencode/opencode.db)", file=sys.stderr)
    print("                     Default: ~/.local/share/opencode/opencode.db", file=sys.stderr)
    print("                     Windows also checks %LOCALAPPDATA%\\opencode and", file=sys.stderr)
    print("                     %APPDATA%\\opencode when the default file is missing.", file=sys.stderr)
    print("\nExamples:", file=sys.stderr)
    print(f"  {sys.argv[0]} ses_123abc456", file=sys.stderr)
    print(f"  {sys.argv[0]} ses_123abc456 export.json", file=sys.stderr)
    print(f"  {sys.argv[0]} ses_123abc456 - > output.json", file=sys.stderr)
    print(f"  OPENCODE_DATABASE=/custom/opencode.db {sys.argv[0]} ses_123abc456", file=sys.stderr)
    print("  # Windows PowerShell:", file=sys.stderr)
    print(
        '  $env:OPENCODE_DATABASE = "$env:USERPROFILE\\.local\\share\\opencode\\opencode.db"',
        file=sys.stderr,
    )
    print(f"  python {sys.argv[0]} ses_123abc456 export.json", file=sys.stderr)


def ensure_output_is_not_source(
    output_path: str | Path, source_paths: Iterable[str | Path]
) -> None:
    """Refuse to replace the live database or one of its SQLite sidecars."""
    _common.ensure_output_does_not_overwrite(
        output_path,
        source_paths,
        exc=ValueError,
        allow_stdout_dash=True,
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_usage()
        sys.exit(0 if len(sys.argv) > 1 else 1)

    session_id = sys.argv[1]

    # Handle output file argument
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = f"export-{session_id}.json"

    db_path = get_db_path().expanduser()

    if not db_path.is_file():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        print(
            "Set OPENCODE_DATABASE to the full path of opencode.db "
            "(on Windows, typically %USERPROFILE%\\.local\\share\\opencode\\opencode.db).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        ensure_output_is_not_source(
            output_file,
            [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")],
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading from database: {db_path}", file=sys.stderr)

    conn = None
    try:
        # Read-only: never mutate the user's live store (a journal_mode=WAL
        # PRAGMA rewrites the DB header and creates -wal/-shm sidecar files).
        conn = sqlite3.connect(_database_uri(db_path), uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")

        # Export session with all children
        result = export_session_and_collect_children(conn, session_id, set())

# Sort messages by creation time
        if result.get("messages"):
            result["messages"].sort(key=lambda m: m.get("info", {}).get("time", {}).get("created", 0))

        output = json.dumps(result, indent=2, ensure_ascii=False)

        if output_file == "-":
            print(output)
        else:
            output_path = Path(output_file).expanduser()
            output_path.write_text(output, encoding="utf-8")
            print(f"Exported to {output_path}", file=sys.stderr)

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
