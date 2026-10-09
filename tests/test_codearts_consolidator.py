from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


TEST_FILE = Path(__file__).resolve()
PROJECT_ROOT = TEST_FILE.parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "codearts_consolidator.py"
if not MODULE_PATH.is_file():
    # Allows this test to run from an isolated staging directory as well as
    # from the repository's tests/ directory.
    MODULE_PATH = TEST_FILE.with_name("codearts_consolidator.py")
SPEC = importlib.util.spec_from_file_location("codearts_consolidator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
consolidator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = consolidator
SPEC.loader.exec_module(consolidator)


SESSION_SCHEMA = """
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    parent_id TEXT,
    slug TEXT,
    directory TEXT,
    title TEXT,
    version TEXT,
    share_url TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    summary_diffs TEXT,
    revert TEXT,
    permission TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    time_compacting INTEGER,
    time_archived INTEGER,
    workspace_id TEXT
);
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE cag_session_extra (
    id TEXT,
    session_id TEXT,
    has_origin_session INTEGER,
    origin_session_version TEXT,
    origin_session TEXT,
    extra_info TEXT,
    type TEXT
);
CREATE TABLE cag_message_extra (
    id TEXT,
    session_id TEXT,
    message_id TEXT,
    has_origin_message INTEGER,
    origin_message_version TEXT,
    origin_message TEXT,
    extra_info TEXT,
    source TEXT
);
CREATE TABLE todo (
    session_id TEXT,
    content TEXT,
    status TEXT,
    priority TEXT,
    position INTEGER,
    time_created INTEGER,
    time_updated INTEGER
);
CREATE TABLE event (
    id TEXT,
    aggregate_id TEXT,
    seq INTEGER,
    type TEXT,
    data TEXT
);
CREATE TABLE session_share (
    session_id TEXT,
    id TEXT,
    secret TEXT,
    url TEXT,
    time_created INTEGER,
    time_updated INTEGER
);
"""


def insert_session(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    parent_id: str | None,
    created: int,
) -> None:
    connection.execute(
        """
        INSERT INTO session (
            id, project_id, parent_id, slug, directory, title, version,
            share_url, summary_additions, summary_deletions, summary_files,
            summary_diffs, revert, permission, time_created, time_updated,
            time_compacting, time_archived, workspace_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            "project",
            parent_id,
            session_id,
            "C:/workspace",
            f"Session {session_id}",
            "test",
            None,
            1,
            2,
            3,
            json.dumps([{"file": "a.py"}]),
            None,
            json.dumps({"edit": "allow"}),
            created,
            created + 900,
            None,
            None,
            "workspace",
        ),
    )


def insert_message(
    connection: sqlite3.Connection,
    message_id: str,
    session_id: str,
    created: int,
    data: dict,
) -> None:
    connection.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (message_id, session_id, created, created + 100, json.dumps(data)),
    )


def insert_part(
    connection: sqlite3.Connection,
    part_id: str,
    message_id: str,
    session_id: str,
    created: int,
    data: dict,
) -> None:
    connection.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (part_id, message_id, session_id, created, created + 20, json.dumps(data)),
    )


class CodeArtsConsolidatorV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_database(self) -> tuple[Path, dict, dict, dict]:
        database = self.root / "opencode.db"
        connection = sqlite3.connect(database)
        connection.executescript(SESSION_SCHEMA)

        insert_session(connection, "root", parent_id=None, created=1_000)
        insert_session(connection, "child_parent", parent_id="root", created=2_000)
        insert_session(
            connection, "grandchild", parent_id="child_parent", created=2_500
        )
        insert_session(connection, "child_tool", parent_id=None, created=3_000)

        insert_message(
            connection,
            "msg_user",
            "root",
            1_010,
            {"role": "user", "time": {"created": 1_010}},
        )
        exact_tokens = {
            "input": 100,
            "output": 20,
            "reasoning": 7,
            "cache": {"read": 4, "write": 2},
            "total": 131,
        }
        insert_message(
            connection,
            "msg_assistant",
            "root",
            1_020,
            {
                "role": "assistant",
                "time": {"created": 1_020, "completed": 1_099},
                "tokens": exact_tokens,
                "agent": "build",
            },
        )
        exact_part_time = {"start": 1_021, "end": 1_030}
        insert_part(
            connection,
            "part_reasoning",
            "msg_assistant",
            "root",
            1_021,
            {"type": "reasoning", "text": "think", "time": exact_part_time},
        )
        insert_part(
            connection,
            "part_task",
            "msg_assistant",
            "root",
            1_022,
            {
                "type": "tool",
                "tool": "task",
                "state": {
                    "status": "completed",
                    "metadata": {"sessionId": "child_tool"},
                },
            },
        )
        exact_tokens_2 = {
            "input": 40,
            "output": 80,
            "reasoning": 3,
            "cache": {"read": 1, "write": 0},
            "total": 124,
        }
        insert_message(
            connection,
            "msg_assistant_2",
            "root",
            1_030,
            {
                "role": "assistant",
                "time": {"created": 1_030, "completed": 1_090},
                "tokens": exact_tokens_2,
                "agent": "build",
            },
        )
        # The same child is discoverable via parent_id and tool metadata.  It
        # must still be exported only once.
        insert_part(
            connection,
            "part_duplicate_child",
            "msg_assistant_2",
            "root",
            1_031,
            {
                "type": "tool",
                "tool": "task",
                "state": {"metadata": {"sessionId": "child_parent"}},
            },
        )
        insert_message(
            connection,
            "msg_child_parent",
            "child_parent",
            2_010,
            {"role": "assistant", "time": {"created": 2_010}},
        )
        insert_message(
            connection,
            "msg_grandchild",
            "grandchild",
            2_510,
            {"role": "assistant", "time": {"created": 2_510}},
        )
        # Cycle back to the root.  visited must prevent duplicate traversal.
        insert_part(
            connection,
            "part_cycle",
            "msg_grandchild",
            "grandchild",
            2_511,
            {
                "type": "tool",
                "tool": "task",
                "state": {"metadata": {"sessionId": "root"}},
            },
        )
        insert_message(
            connection,
            "msg_child_tool",
            "child_tool",
            3_010,
            {"role": "assistant", "time": {"created": 3_010}},
        )
        connection.execute(
            "INSERT INTO cag_session_extra VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("root", "root", None, None, None, None, "kernel"),
        )
        connection.execute(
            "INSERT INTO cag_message_extra VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "extra-message",
                "root",
                "msg_assistant",
                0,
                None,
                None,
                '{"trace":"kept"}',
                "kernel",
            ),
        )
        connection.execute(
            "INSERT INTO todo VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("root", "verify", "completed", "high", 0, 1_001, 1_002),
        )
        connection.execute(
            "INSERT INTO event VALUES (?, ?, ?, ?, ?)",
            ("event-1", "root", 1, "test", '{"value":1}'),
        )
        # This value must never appear anywhere in the exported JSON.
        connection.execute(
            "INSERT INTO session_share VALUES (?, ?, ?, ?, ?, ?)",
            ("root", "share-1", "TOP-SECRET", "https://example.invalid", 1, 2),
        )
        connection.commit()
        connection.close()
        return database, exact_tokens, exact_tokens_2, exact_part_time

    def test_legacy_reads_all_numeric_shards_without_rewriting_messages(self) -> None:
        session = self.root / "legacy"
        session.mkdir()
        (session / "chat_baseInfo.json").write_text(
            json.dumps({"chatId": "legacy-id", "title": "Legacy"}), encoding="utf-8"
        )
        shard_0 = [{"sender": "User", "content": "zero", "nested": {"x": 1}}]
        shard_2 = [{"sender": "Assistant", "content": "two"}]
        shard_10 = [{"sender": "Assistant", "content": "ten"}]
        for index, payload in ((0, shard_0), (10, shard_10), (2, shard_2)):
            (session / f"messages_{index}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

        result = consolidator.consolidate_legacy_session(session)

        self.assertEqual(result["messages"], shard_0 + shard_2 + shard_10)
        self.assertEqual(result["statistics"]["message_shards"], 3)
        self.assertEqual(
            [entry["index"] for entry in result["export_metadata"]["shards"]],
            [0, 2, 10],
        )
        self.assertTrue(result["export_metadata"]["warnings"])
        self.assertFalse(result["export_metadata"]["complete"])

    def test_legacy_rejects_duplicate_numeric_shard_indexes(self) -> None:
        session = self.root / "legacy-duplicates"
        session.mkdir()
        (session / "chat_baseInfo.json").write_text("{}", encoding="utf-8")
        (session / "messages_0.json").write_text("[]", encoding="utf-8")
        (session / "messages_1.json").write_text("[]", encoding="utf-8")
        (session / "messages_01.json").write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(
            consolidator.ConsolidationError, "multiple files map to indexes"
        ):
            consolidator.consolidate_legacy_session(session)

    def test_filename_sanitizing_and_large_shard_gaps_are_bounded(self) -> None:
        component = consolidator._safe_filename_component(r"..\..\CON/escape")
        self.assertNotIn("..", component)
        self.assertNotIn("\\", component)
        self.assertNotIn("/", component)
        self.assertNotEqual(component.upper(), "CON")

        count, preview = consolidator._missing_shard_summary([0, 1_000_000_000])
        self.assertEqual(count, 999_999_999)
        self.assertEqual(len(preview), 20)

    def test_batch_rejects_duplicate_output_names_instead_of_overwriting(self) -> None:
        parent = self.root / "batch"
        output_dir = self.root / "batch-output"
        parent.mkdir()
        output_dir.mkdir()
        for name, content in (("a", "first"), ("b", "second")):
            session = parent / name
            session.mkdir()
            (session / "chat_baseInfo.json").write_text(
                '{"chatId":"duplicate"}', encoding="utf-8"
            )
            (session / "messages_0.json").write_text(
                json.dumps([{"sender": "User", "content": content}]),
                encoding="utf-8",
            )

        with contextlib.redirect_stderr(io.StringIO()):
            processed, ignored, failed, partial = consolidator._batch_legacy(
                parent, output_dir
            )

        self.assertEqual((processed, ignored, failed, partial), (1, 0, 1, 0))
        output = json.loads(
            (output_dir / "duplicate_trajectory_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(output["messages"][0]["content"], "first")
        with contextlib.redirect_stderr(io.StringIO()):
            return_code = consolidator.main(
                [str(parent), "--batch", "--output", str(output_dir)]
            )
        self.assertEqual(return_code, 1)

    def test_database_export_preserves_tokens_and_part_times(self) -> None:
        database, exact_tokens, exact_tokens_2, exact_part_time = self.make_database()

        result = consolidator.consolidate_database_session(database, "root")

        messages = {message["info"]["id"]: message for message in result["messages"]}
        self.assertEqual(messages["msg_assistant"]["info"]["tokens"], exact_tokens)
        self.assertEqual(
            messages["msg_assistant_2"]["info"]["tokens"], exact_tokens_2
        )
        reasoning = next(
            part
            for part in messages["msg_assistant"]["parts"]
            if part["id"] == "part_reasoning"
        )
        self.assertEqual(reasoning["time"], exact_part_time)
        self.assertEqual(reasoning["_codeartsStorage"]["timeCreated"], 1_021)
        self.assertEqual(
            messages["msg_assistant"]["info"]["time"],
            {"created": 1_020, "completed": 1_099},
        )

    def test_database_export_discovers_both_child_relationships(self) -> None:
        database, _, _, _ = self.make_database()

        result = consolidator.consolidate_database_session(database, "root")

        manifest = {
            entry["info"]["id"]: entry for entry in result["session_manifest"]
        }
        self.assertEqual(
            set(manifest), {"root", "child_parent", "grandchild", "child_tool"}
        )
        self.assertEqual(
            manifest["child_parent"]["discovered_by"], "parent_id+tool_metadata"
        )
        self.assertEqual(manifest["grandchild"]["discovered_by"], "parent_id")
        self.assertEqual(manifest["grandchild"]["depth"], 2)
        self.assertEqual(manifest["child_tool"]["discovered_by"], "tool_metadata")
        child_infos = [
            message["info"]
            for message in result["messages"]
            if message["info"]["sessionID"] != "root"
        ]
        self.assertEqual(len(child_infos), 3)
        self.assertTrue(all(info["isSubAgent"] is True for info in child_infos))
        self.assertEqual(result["statistics"]["sessions"], 4)
        self.assertEqual(result["statistics"]["total_messages"], 6)

    def test_no_children_is_root_only(self) -> None:
        database, _, _, _ = self.make_database()

        result = consolidator.consolidate_database_session(
            database, "root", include_children=False
        )

        self.assertEqual(result["statistics"]["sessions"], 1)
        self.assertEqual(
            {message["info"]["sessionID"] for message in result["messages"]},
            {"root"},
        )

    def test_open_database_is_enforced_read_only(self) -> None:
        database, _, _, _ = self.make_database()

        connection = consolidator.open_database_read_only(database)
        try:
            self.assertTrue(connection.in_transaction)
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM message")
        finally:
            connection.close()

        check = sqlite3.connect(database)
        try:
            self.assertEqual(check.execute("SELECT count(*) FROM message").fetchone()[0], 6)
        finally:
            check.close()

    def test_optional_trajectory_extras_are_included_but_share_secret_is_not(self) -> None:
        database, _, _, _ = self.make_database()

        result = consolidator.consolidate_database_session(database, "root")
        serialized = json.dumps(result)
        root_manifest = next(
            entry for entry in result["session_manifest"] if entry["info"]["id"] == "root"
        )
        assistant = next(
            message
            for message in result["messages"]
            if message["info"]["id"] == "msg_assistant"
        )

        self.assertEqual(root_manifest["codearts_extra"][0]["type"], "kernel")
        self.assertEqual(root_manifest["todos"][0]["content"], "verify")
        self.assertEqual(root_manifest["events"][0]["id"], "event-1")
        self.assertEqual(assistant["codearts_extra"][0]["source"], "kernel")
        self.assertNotIn("TOP-SECRET", serialized)
        self.assertEqual(
            result["export_metadata"]["optional_tables"]["session_share"],
            "excluded_sensitive",
        )

    def test_older_minimal_schema_is_still_exportable(self) -> None:
        database = self.root / "old.db"
        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE session (id TEXT PRIMARY KEY);
            CREATE TABLE message (id TEXT, session_id TEXT, data TEXT);
            CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, data TEXT);
            INSERT INTO session VALUES ('old-root');
            INSERT INTO message VALUES (
                'old-message', 'old-root',
                '{"role":"assistant","time":{"created":123}}'
            );
            """
        )
        connection.commit()
        connection.close()

        result = consolidator.consolidate_database_session(database, "old-root")

        self.assertEqual(result["info"]["id"], "old-root")
        self.assertEqual(result["messages"][0]["info"]["id"], "old-message")
        self.assertEqual(result["messages"][0]["info"]["time"]["created"], 123)

    def test_depth_limit_is_reported_as_partial(self) -> None:
        database, _, _, _ = self.make_database()

        result = consolidator.consolidate_database_session(
            database, "root", max_depth=1
        )

        self.assertFalse(result["export_metadata"]["complete"])
        self.assertTrue(
            any("maximum depth" in warning for warning in result["export_metadata"]["warnings"])
        )

    def test_cli_refuses_to_overwrite_database_or_legacy_source(self) -> None:
        database, _, _, _ = self.make_database()
        with contextlib.redirect_stderr(io.StringIO()):
            return_code = consolidator.main(
                [
                    str(database),
                    "--session-id",
                    "root",
                    "--output",
                    str(database),
                ]
            )
        self.assertEqual(return_code, 1)
        self.assertEqual(database.read_bytes()[:16], b"SQLite format 3\x00")

        wal_path = Path(f"{database}-wal")
        with contextlib.redirect_stderr(io.StringIO()):
            return_code = consolidator.main(
                [
                    str(database),
                    "--session-id",
                    "root",
                    "--output",
                    str(wal_path),
                ]
            )
        self.assertEqual(return_code, 1)
        self.assertFalse(wal_path.exists())

        session = self.root / "protected-legacy"
        session.mkdir()
        base_info = session / "chat_baseInfo.json"
        messages = session / "messages_0.json"
        base_info.write_text('{"chatId":"protected"}', encoding="utf-8")
        original_messages = '[{"sender":"User","content":"safe"}]'
        messages.write_text(original_messages, encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            return_code = consolidator.main(
                [str(session), "--output", str(messages)]
            )
        self.assertEqual(return_code, 1)
        self.assertEqual(messages.read_text(encoding="utf-8"), original_messages)

    def test_cli_uses_safe_default_name_and_nonzero_for_partial_export(self) -> None:
        session = self.root / "unsafe-name"
        session.mkdir()
        (session / "chat_baseInfo.json").write_text(
            '{"chatId":"../../escape"}', encoding="utf-8"
        )
        (session / "messages_0.json").write_text("[]", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            return_code = consolidator.main([str(session)])
        self.assertEqual(return_code, 0)
        outputs = list(session.glob("*_trajectory_v2.json"))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].parent, session)

        database, _, _, _ = self.make_database()
        partial_output = self.root / "partial.json"
        with contextlib.redirect_stderr(io.StringIO()):
            return_code = consolidator.main(
                [
                    str(database),
                    "--session-id",
                    "root",
                    "--max-depth",
                    "1",
                    "--output",
                    str(partial_output),
                ]
            )
        self.assertEqual(return_code, 2)
        partial = json.loads(partial_output.read_text(encoding="utf-8"))
        self.assertFalse(partial["export_metadata"]["complete"])

    def test_output_is_detected_and_parsed_as_opencode(self) -> None:
        database, _, _, _ = self.make_database()
        result = consolidator.consolidate_database_session(database, "root")
        output = self.root / "trajectory.json"
        consolidator.write_output(result, output)

        if not (PROJECT_ROOT / "trajviz").is_dir():
            self.skipTest("trajviz package is not beside this staged test")
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from trajviz.insight.loaders import detect_format, load_trajectory
            from trajviz.insight.parser import parse_steps
            from trajviz.insight.patterns import extract_subagent_sessions

            self.assertEqual(detect_format(result), "codearts")
            loaded = load_trajectory(str(output))
            self.assertNotIn("_error", loaded)
            self.assertEqual(loaded["metadata"]["sub_agent_count"], 3)
            self.assertEqual(loaded["metadata"]["event_count"], 1)
            steps = parse_steps(loaded)
            self.assertEqual(len(steps), 6)
            self.assertEqual([step["raw_index"] for step in steps], list(range(6)))
            self.assertEqual(sum(1 for step in steps if step["is_sub_agent"]), 3)
            subagent_sessions = extract_subagent_sessions(steps, loaded["messages"])
            self.assertEqual(
                {session["session_id"] for session in subagent_sessions},
                {"child_parent", "grandchild", "child_tool"},
            )
            sessions_by_id = {
                session["session_id"]: session for session in subagent_sessions
            }
            self.assertEqual(sessions_by_id["child_tool"]["spawn_step"], 1)
            self.assertEqual(sessions_by_id["child_parent"]["spawn_step"], 2)
            self.assertIsNone(sessions_by_id["grandchild"]["spawn_step"])
        finally:
            sys.path.remove(str(PROJECT_ROOT))


if __name__ == "__main__":
    unittest.main()
