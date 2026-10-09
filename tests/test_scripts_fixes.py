"""Regression tests for the 2026-08 scripts/ review fixes.

Covers:
* B29 — opencode_consolidator exports diamond-reachable sessions exactly once.
* B30 — opencode_consolidator preserves persisted message timing (fill-only).
* B32 — the taxonomy version regex actually matches TAXONOMY_REFERENCE.md.
* C15 — the reserved-label documentation note does not leak into the parsed
  taxonomy (load_taxonomy's line format is load-bearing).
* R27/R28 — scripts/_common.py no-overwrite guard and atomic JSON write.
* R29 — the v1 step_labeler CLI routes through v2 and emits assistant-only
  records of a single uniform shape (and inherits the overwrite guard).
"""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import _common
from scripts import opencode_consolidator
from scripts import step_labeler
from scripts import step_labeler_v2


SCHEMA = """
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
    slug TEXT,
    directory TEXT,
    title TEXT,
    version TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    time_created INTEGER,
    time_updated INTEGER,
    time_compacting INTEGER,
    time_archived INTEGER,
    summary_diffs TEXT,
    revert TEXT,
    permission TEXT
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
"""


def _insert_session(conn: sqlite3.Connection, session_id: str, parent_id=None) -> None:
    conn.execute(
        "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id, "project", "workspace", parent_id, session_id, "/workspace",
            f"Session {session_id}", "test", 0, 0, 0, 1_000, 2_000, None, None,
            None, None, None,
        ),
    )


def _insert_message(
    conn: sqlite3.Connection,
    msg_id: str,
    session_id: str,
    payload: dict,
    time_created: int,
    time_updated: int,
) -> None:
    conn.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (msg_id, session_id, time_created, time_updated, json.dumps(payload)),
    )


def _insert_spawn_part(
    conn: sqlite3.Connection, part_id: str, msg_id: str, session_id: str, child_id: str
) -> None:
    data = {
        "type": "tool",
        "tool": "task",
        "state": {"status": "completed", "metadata": {"sessionId": child_id}},
    }
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (part_id, msg_id, session_id, 1_100, 1_200, json.dumps(data)),
    )


class OpencodeDiamondTraversalTests(unittest.TestCase):
    """B29: a session reachable via two parents must be exported exactly once."""

    def test_diamond_child_session_exported_once(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        # root spawns B and C; both B and C spawn D (diamond).
        _insert_session(conn, "ses_root")
        _insert_session(conn, "ses_b", "ses_root")
        _insert_session(conn, "ses_c", "ses_root")
        _insert_session(conn, "ses_d", "ses_b")

        _insert_message(conn, "m_root_b", "ses_root", {"role": "assistant"}, 1_000, 1_050)
        _insert_spawn_part(conn, "p_root_b", "m_root_b", "ses_root", "ses_b")
        _insert_message(conn, "m_root_c", "ses_root", {"role": "assistant"}, 1_100, 1_150)
        _insert_spawn_part(conn, "p_root_c", "m_root_c", "ses_root", "ses_c")

        _insert_message(conn, "m_b", "ses_b", {"role": "assistant"}, 1_200, 1_250)
        _insert_spawn_part(conn, "p_b_d", "m_b", "ses_b", "ses_d")
        _insert_message(conn, "m_c", "ses_c", {"role": "assistant"}, 1_300, 1_350)
        _insert_spawn_part(conn, "p_c_d", "m_c", "ses_c", "ses_d")

        _insert_message(conn, "m_d", "ses_d", {"role": "assistant"}, 1_400, 1_450)
        conn.commit()

        with contextlib.redirect_stderr(io.StringIO()):
            result = opencode_consolidator.export_session_and_collect_children(
                conn, "ses_root", set()
            )
        conn.close()

        exported_ids = [m["info"]["id"] for m in result["messages"]]
        self.assertEqual(exported_ids.count("m_d"), 1)
        self.assertEqual(len(exported_ids), len(set(exported_ids)))
        self.assertEqual(
            sorted(exported_ids), ["m_b", "m_c", "m_d", "m_root_b", "m_root_c"]
        )


class OpencodePersistedTimingTests(unittest.TestCase):
    """B30: DB row timestamps must never clobber persisted message timing."""

    def _export_single(self, payload: dict, created: int, updated: int) -> dict:
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        _insert_session(conn, "ses_root")
        _insert_message(conn, "m_0", "ses_root", payload, created, updated)
        conn.commit()
        with contextlib.redirect_stderr(io.StringIO()):
            result = opencode_consolidator.export_session_and_collect_children(
                conn, "ses_root", set()
            )
        conn.close()
        return result["messages"][0]["info"]

    def test_persisted_completed_time_survives_db_row_touch(self) -> None:
        info = self._export_single(
            {"role": "assistant", "time": {"created": 5_000, "completed": 10_000}},
            created=5_000,
            updated=99_999,  # row touched long after the model finished
        )
        self.assertEqual(info["time"], {"created": 5_000, "completed": 10_000})

    def test_row_times_fill_in_when_payload_has_no_timing(self) -> None:
        info = self._export_single({"role": "assistant"}, created=1_000, updated=1_500)
        # Fallback path: row created/updated used, updated renamed to completed.
        self.assertEqual(info["time"], {"created": 1_000, "completed": 1_500})


class TaxonomyVersionTests(unittest.TestCase):
    """B32/C15: version extraction works and reserved labels stay out."""

    TAXONOMY = Path(step_labeler.__file__).resolve().parent / "TAXONOMY_REFERENCE.md"

    def test_version_regex_matches_shipped_taxonomy_file(self) -> None:
        mapping, version = step_labeler.load_taxonomy(str(self.TAXONOMY))
        self.assertNotEqual(version, "unknown")
        self.assertEqual(version, "v1")
        self.assertEqual(
            set(mapping),
            {"understand", "plan", "implement", "debug", "validate", "report"},
        )

    def test_reserved_label_note_does_not_leak_into_taxonomy(self) -> None:
        mapping, _ = step_labeler.load_taxonomy(str(self.TAXONOMY))
        all_actions = {a for actions in mapping.values() for a in actions}
        self.assertNotIn("user", mapping)
        self.assertNotIn("unknown", mapping)
        self.assertNotIn("user_prompt", all_actions)
        self.assertNotIn("unknown", all_actions)


class CommonAtomicWriteTests(unittest.TestCase):
    """R28: shared atomic JSON write."""

    def test_writes_json_and_leaves_no_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            _common.write_json_atomic(out, {"key": "valué", "n": 3})
            text = out.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(json.loads(text), {"key": "valué", "n": 3})
            self.assertEqual(list(Path(tmp).glob(".out.json.*.tmp")), [])

    def test_failed_write_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            original = '{"existing": true}\n'
            out.write_text(original, encoding="utf-8")
            with self.assertRaises(TypeError):
                _common.write_json_atomic(out, {"bad": {1, 2}})  # not serializable
            self.assertEqual(out.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(tmp).glob(".out.json.*.tmp")), [])

    def test_replaces_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            out.write_text('{"old": 1}\n', encoding="utf-8")
            _common.write_json_atomic(out, {"new": 2})
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), {"new": 2})


class CommonOverwriteGuardTests(unittest.TestCase):
    """R27: shared no-overwrite guard raises the caller's exception type."""

    class GuardError(RuntimeError):
        pass

    def test_raises_given_exception_when_output_is_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.json"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaises(self.GuardError) as raised:
                _common.ensure_output_does_not_overwrite(
                    source,
                    [source],
                    exc=self.GuardError,
                    message="would overwrite: {source}",
                )
            self.assertIn(str(source.resolve()), str(raised.exception))

    def test_distinct_paths_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.json"
            source.write_text("{}", encoding="utf-8")
            _common.ensure_output_does_not_overwrite(
                Path(tmp) / "output.json", [source], exc=ValueError
            )

    def test_stdout_dash_bypass_is_opt_in(self) -> None:
        dash_source = Path("-")
        _common.ensure_output_does_not_overwrite(
            "-", [dash_source], exc=ValueError, allow_stdout_dash=True
        )
        with self.assertRaises(ValueError):
            _common.ensure_output_does_not_overwrite(
                "-", [dash_source], exc=ValueError, allow_stdout_dash=False
            )


class StepLabelerV1CliTests(unittest.TestCase):
    """R29/B31: v1 CLI delegates to v2 — assistant-only, uniform, guarded."""

    STEPS = [
        {
            "index": 0,
            "raw_index": 0,
            "role": "user",
            "text_preview": "Please fix the bug",
            "tokens": {},
            "parts": [],
            "tool_calls": [],
        },
        {
            "index": 1,
            "raw_index": 1,
            "role": "assistant",
            "text_preview": "Reading the code",
            "tokens": {"total": 10},
            "parts": [{"type": "reasoning", "text": "inspect files"}],
            "tool_calls": [],
        },
        {
            "index": 2,
            "raw_index": 2,
            "role": "assistant",
            "text_preview": "",  # empty step -> fallback label
            "tokens": {},
            "parts": [],
            "tool_calls": [],
        },
        {
            "index": 3,
            "raw_index": 3,
            "role": "system",
            "text_preview": "system context",
            "tokens": {},
            "parts": [],
            "tool_calls": [],
        },
    ]

    def _run_cli(self, argv: list[str]) -> None:
        with (
            patch.object(step_labeler, "_load_dotenv", lambda *a, **k: None),
            patch.object(step_labeler_v2, "load_all_steps", return_value=self.STEPS),
            patch.object(
                step_labeler,
                "call_llm",
                return_value='{"phase":"understand","action":"code_reading"}',
            ),
            patch.object(sys, "argv", argv),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            step_labeler.main()

    def test_v1_cli_emits_assistant_only_records_of_uniform_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = Path(tmp) / "trajectory.json"
            trajectory.write_text('{"messages": []}\n', encoding="utf-8")
            output = Path(tmp) / "trajectory_labeled.json"
            self._run_cli(
                [
                    "step_labeler.py", str(trajectory), "-o", str(output),
                    "--base-url", "https://example.invalid/v1",
                    "--api-key", "test", "--model", "test-model",
                ]
            )
            data = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(data["schema_version"], "trajectory_labels.v2")
        self.assertEqual(data["taxonomy_version"], "v1")
        steps = data["steps"]
        self.assertEqual([s["role"] for s in steps], ["assistant", "assistant"])
        self.assertEqual([s["index"] for s in steps], [1, 2])
        # Single uniform record shape across LLM-labeled and fallback records.
        key_sets = {tuple(sorted(s)) for s in steps}
        self.assertEqual(len(key_sets), 1)
        self.assertEqual(
            (steps[0]["phase"], steps[0]["action"], steps[0]["label_source"]),
            ("understand", "code_reading", "llm"),
        )
        self.assertEqual(
            (steps[1]["phase"], steps[1]["action"], steps[1]["label_source"]),
            ("unknown", "unknown", "fallback"),
        )

    def test_v1_cli_refuses_to_overwrite_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = Path(tmp) / "trajectory.json"
            original = '{"messages": []}\n'
            trajectory.write_text(original, encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                self._run_cli(
                    [
                        "step_labeler.py", str(trajectory), "-o", str(trajectory),
                        "--base-url", "https://example.invalid/v1",
                        "--api-key", "test", "--model", "test-model",
                    ]
                )
            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(trajectory.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
