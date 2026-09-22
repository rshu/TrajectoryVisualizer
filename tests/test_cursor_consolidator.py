from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import cursor_consolidator as consolidator


PARENT_ID = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"
CHILD_ID = "cccccccc-4444-5555-6666-dddddddddddd"


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


class CursorConsolidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.projects = self.root / "projects"
        self.session_dir = (
            self.projects / "demo-workspace" / "agent-transcripts" / PARENT_ID
        )
        self.parent_jsonl = self.session_dir / f"{PARENT_ID}.jsonl"
        self.child_jsonl = self.session_dir / "subagents" / f"{CHILD_ID}.jsonl"
        _write_jsonl(
            self.parent_jsonl,
            [
                {
                    "role": "user",
                    "message": {"content": [{"type": "text", "text": "Inspect the project"}]},
                },
                {
                    "role": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "Reading"},
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {"path": "/workspace/demo/app.py"},
                            },
                            {
                                "type": "tool_use",
                                "name": "Task",
                                "input": {"prompt": "explore", "subagent_type": "explore"},
                            },
                        ]
                    },
                },
                {"type": "turn_ended", "status": "success"},
            ],
        )
        _write_jsonl(
            self.child_jsonl,
            [
                {
                    "role": "user",
                    "message": {"content": [{"type": "text", "text": "explore"}]},
                },
                {
                    "role": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Grep",
                                "input": {"path": "/workspace/demo", "pattern": "TODO"},
                            }
                        ]
                    },
                },
                {"type": "turn_ended", "status": "success"},
            ],
        )
        self.database = self.root / "state.vscdb"
        composer = {
            "composerId": PARENT_ID,
            "name": "Cursor fixture",
            "createdAt": 1000,
            "lastUpdatedAt": 5000,
            "status": "completed",
            "unifiedMode": "agent",
            "isAgentic": True,
            "modelConfig": {"modelName": "grok-4.6", "maxMode": False},
            "promptTokenBreakdown": {
                "totalUsedTokens": 1200,
                "maxTokens": 256000,
                "categories": [
                    {"id": "conversation", "label": "Conversation", "estimatedTokens": 900}
                ],
            },
            "contextUsagePercent": 0.5,
            "filesChangedCount": 1,
            "totalLinesAdded": 3,
            "totalLinesRemoved": 1,
            "subagentComposerIds": [CHILD_ID],
            "newlyCreatedFiles": [],
            "originalFileStates": {
                "vscode-remote://wsl/workspace/demo/app.py": {"isNewlyCreated": False}
            },
            "workspaceIdentifier": {"uri": {"path": "/workspace/demo"}},
            "fullConversationHeadersOnly": [
                {"bubbleId": "bubble-user", "type": 1},
                {"bubbleId": "bubble-think", "type": 2},
                {"bubbleId": "bubble-read", "type": 2},
            ],
            "usageData": {},
        }
        bubbles = {
            "bubble-user": {
                "_v": 3,
                "type": 1,
                "bubbleId": "bubble-user",
                "createdAt": "1970-01-01T00:00:01.000Z",
                "contextWindowStatusAtCreation": {"tokensUsed": 800, "maxTokens": 256000},
            },
            "bubble-think": {
                "_v": 3,
                "type": 2,
                "bubbleId": "bubble-think",
                "startedAtMs": 1000,
                "thinkingDurationMs": 400,
                "turnDurationMs": 1500,
                "tokenCount": {"inputTokens": 0, "outputTokens": 0},
            },
            "bubble-read": {
                "_v": 3,
                "type": 2,
                "bubbleId": "bubble-read",
                "startedAtMs": 2100,
                "completedAtMs": 2300,
                "toolFormerData": {
                    "toolCallId": "call-read",
                    "status": "completed",
                    "name": "read_file_v2",
                    "rawArgs": json.dumps({"path": "/workspace/demo/app.py"}),
                    "result": {"totalLinesInFile": 40},
                    "additionalData": {"startedAtMs": 2100},
                },
            },
        }
        connection = sqlite3.connect(self.database)
        connection.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            "INSERT INTO cursorDiskKV VALUES (?, ?)",
            (f"composerData:{PARENT_ID}", json.dumps(composer)),
        )
        connection.execute(
            "INSERT INTO cursorDiskKV VALUES (?, ?)",
            (
                f"composerData:{CHILD_ID}",
                json.dumps({
                    "composerId": CHILD_ID,
                    "name": "explore child",
                    "promptTokenBreakdown": {
                        "totalUsedTokens": 500,
                        "maxTokens": 256000,
                    },
                }),
            ),
        )
        for bubble_id, payload in bubbles.items():
            connection.execute(
                "INSERT INTO cursorDiskKV VALUES (?, ?)",
                (f"bubbleId:{PARENT_ID}:{bubble_id}", json.dumps(payload)),
            )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _export(self, output: Path | None = None) -> dict:
        dest = output or (self.root / "export.json")
        code = consolidator.main([
            PARENT_ID,
            str(dest),
            "--projects-dir", str(self.projects),
            "--db", str(self.database),
        ])
        self.assertEqual(code, 0)
        return json.loads(dest.read_text(encoding="utf-8"))

    def test_joins_jsonl_subagents_and_composer_metadata(self) -> None:
        result = self._export()
        meta = result["export_metadata"]
        self.assertEqual(meta["source_format"], "cursor_composer")
        self.assertEqual(meta["schema_version"], 1)
        self.assertEqual(meta["token_semantics"], "context_window_snapshot+estimated_log_tokens")
        self.assertEqual(meta["chat_id"], PARENT_ID)
        self.assertEqual(result["info"]["title"], "Cursor fixture")
        self.assertEqual(result["info"]["model"], "grok-4.6")
        self.assertEqual(result["info"]["promptTokenBreakdown"]["totalUsedTokens"], 1200)
        self.assertEqual(result["statistics"]["sessions"], 2)
        self.assertEqual(result["statistics"]["subagent_sessions"], 1)
        self.assertEqual(len(result["messages"]), 4)

        parent_asst = result["messages"][1]
        self.assertEqual(parent_asst["info"]["role"], "assistant")
        self.assertEqual(parent_asst["info"]["finish"], "success")
        self.assertEqual(parent_asst["info"]["time"]["created"], 1000)
        self.assertEqual(parent_asst["info"]["time"]["completed"], 2500)
        self.assertEqual(parent_asst["info"]["time"]["thinkingMs"], 400)
        self.assertGreater(parent_asst["info"]["tokens"]["total"], 0)
        self.assertGreater(parent_asst["info"]["tokens"]["output"], 0)
        self.assertEqual(parent_asst["info"]["tokens"]["context_window"], 800)
        parent_user = result["messages"][0]
        self.assertEqual(parent_user["info"]["tokens"]["context_window"], 800)
        self.assertEqual(parent_user["info"]["time"]["created"], 1000)
        tools = [part for part in parent_asst["parts"] if part["type"] == "tool"]
        self.assertEqual(tools[0]["tool"], "Read")
        self.assertEqual(tools[0]["state"]["status"], "completed")
        self.assertEqual(tools[0]["state"]["time"]["start"], 2100)
        self.assertEqual(tools[0]["state"]["time"]["end"], 2300)
        self.assertEqual(tools[0]["state"]["metadata"]["totalDurationMs"], 200)
        self.assertIn("totalLinesInFile", tools[0]["state"]["output"])
        self.assertEqual(tools[1]["tool"], "Task")
        self.assertEqual(tools[1]["state"]["metadata"]["sessionId"], CHILD_ID)

        child_asst = result["messages"][3]
        self.assertTrue(child_asst["info"]["isSubAgent"])
        self.assertEqual(child_asst["info"]["parentSessionID"], PARENT_ID)
        self.assertEqual(child_asst["parts"][0]["tool"], "Grep")
        self.assertEqual(child_asst["parts"][0]["state"]["status"], "unknown")
        self.assertEqual(child_asst["info"]["tokens"]["context_window"], 500)

    def test_loader_detects_export_as_cursor_not_opencode(self) -> None:
        from trajviz.insight.loaders import detect_format, load_trajectory
        from trajviz.insight.parser import parse_steps

        dest = self.root / "export.json"
        self._export(dest)
        raw = json.loads(dest.read_text(encoding="utf-8"))
        self.assertEqual(detect_format(raw), "cursor")
        loaded = load_trajectory(str(dest))
        self.assertEqual(detect_format(loaded), "cursor")
        self.assertEqual(loaded["metadata"]["agent"], "cursor")
        self.assertFalse(loaded["_capabilities"]["has_runtime_token_usage"])
        self.assertTrue(loaded["_capabilities"]["has_timing"])
        self.assertEqual(loaded["metadata"]["context_snapshot"]["total_used_tokens"], 1200)
        self.assertGreater(loaded["token_usage"]["total_tokens"], 0)
        steps = parse_steps(loaded)
        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[1]["duration"], 1.5)
        self.assertEqual(steps[1]["tokens"]["context_window"], 800)
        self.assertGreater(steps[1]["tokens"]["total"], 0)
        self.assertEqual(steps[1]["tool_calls"][0]["time_start"], 2100)
        self.assertEqual(steps[1]["tool_calls"][0]["time_end"], 2300)
        self.assertEqual(steps[1]["tool_calls"][0]["duration_ms"], 200)
        self.assertEqual(steps[1]["tool_calls"][0]["tool_name"], "Read")
        self.assertIn("40", steps[1]["tool_calls"][0]["output"])
        self.assertTrue(steps[3]["is_sub_agent"])
        self.assertEqual(steps[1]["tool_calls"][1]["metadata"]["sessionId"], CHILD_ID)
        self.assertEqual(steps[3]["tokens"]["context_window"], 500)

    def test_jsonl_only_when_db_missing(self) -> None:
        with patch.object(consolidator, "resolve_state_vscdb", return_value=None):
            result = consolidator.consolidate_chat(
                PARENT_ID,
                projects_dir=self.projects,
                database_path=None,
            )
        self.assertTrue(result["export_metadata"]["warnings"])
        self.assertEqual(result["info"]["title"], "")
        self.assertEqual(result["messages"][1]["parts"][1]["state"]["status"], "unknown")

    def test_overwrite_returns_error_code(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            code = consolidator.main([
                PARENT_ID,
                str(self.parent_jsonl),
                "--projects-dir", str(self.projects),
                "--db", str(self.database),
            ])
        self.assertEqual(code, 1)
        self.assertTrue(self.parent_jsonl.read_text(encoding="utf-8").startswith("{"))

    def test_list_includes_chat_id_and_title(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = consolidator.main([
                "--list",
                "--projects-dir", str(self.projects),
                "--db", str(self.database),
            ])
        self.assertEqual(code, 0)
        text = buf.getvalue()
        self.assertIn(PARENT_ID, text)
        self.assertIn("Cursor fixture", text)
        listed_ids = [line.split()[0] for line in text.splitlines() if line.strip()]
        self.assertEqual(listed_ids, [PARENT_ID])


class CursorOccupancyMappingTests(unittest.TestCase):
    def test_percentage_remaining_uses_last_token_limit(self) -> None:
        used, cap = consolidator._occupancy_from_bubble(
            {
                "contextWindowStatusAtCreation": {
                    "tokensUsed": 268224,
                    "tokenLimit": 300000,
                    "percentageRemainingFloat": 10.592,
                }
            },
            256_000,
        )
        self.assertEqual(used, 268224)
        self.assertEqual(cap, 300_000)
        later, cap = consolidator._occupancy_from_bubble(
            {
                "contextWindowStatusAtCreation": {
                    "percentageRemainingFloat": 37.13203125,
                    "percentageRemaining": 37,
                }
            },
            cap,
        )
        self.assertEqual(later, 188604)
        self.assertEqual(cap, 300_000)

    def test_duplicate_jsonl_user_does_not_steal_later_occupancy_bubble(self) -> None:
        events = [
            {
                "role": "user",
                "message": {"content": [{
                    "type": "text",
                    "text": (
                        "<timestamp>Wednesday, Sep 16, 2026, 11:26 AM (UTC+8)</timestamp> "
                        "<user_query> First prompt </user_query>"
                    ),
                }]},
            },
            {
                "role": "user",
                "message": {"content": [{
                    "type": "text",
                    "text": (
                        "<timestamp>Wednesday, Sep 16, 2026, 11:26 AM (UTC+8)</timestamp> "
                        "<user_query> First prompt </user_query>"
                    ),
                }]},
            },
            {
                "role": "user",
                "message": {"content": [{
                    "type": "text",
                    "text": (
                        "<timestamp>Wednesday, Sep 16, 2026, 3:08 PM (UTC+8)</timestamp> "
                        "<user_query> Second prompt </user_query>"
                    ),
                }]},
            },
        ]
        bubbles = [
            {
                "type": 1,
                "bubbleId": "b0",
                "createdAt": "2026-09-16T03:25:59.000Z",
                "text": "First prompt",
            },
            {
                "type": 1,
                "bubbleId": "b1",
                "createdAt": "2026-09-16T07:08:11.000Z",
                "text": "Second prompt",
                "contextWindowStatusAtCreation": {
                    "tokensUsed": 268224,
                    "tokenLimit": 300000,
                    "percentageRemainingFloat": 10.592,
                },
            },
        ]
        users, assistants, _tools = consolidator._partition_composer_bubbles(bubbles)
        aligned = consolidator._align_role_bubbles(events, users, assistants)
        self.assertEqual(aligned[0]["bubbleId"], "b0")
        self.assertIsNone(aligned[1])
        self.assertEqual(aligned[2]["bubbleId"], "b1")
        messages, _tool_count, _cap = consolidator.events_to_messages(
            events, session_id="s1", bubbles=bubbles, occupancy_limit=256000,
        )
        self.assertNotIn("context_window", messages[0]["info"].get("tokens", {}))
        self.assertNotIn("context_window", messages[1]["info"].get("tokens", {}))
        self.assertEqual(messages[2]["info"]["tokens"]["context_window"], 268224)

    def test_session_snapshot_ramps_from_zero(self) -> None:
        messages = [
            {"info": {"role": "user", "tokens": {"total": 10}}},
            {"info": {"role": "assistant", "tokens": {"total": 10}}},
            {"info": {"role": "assistant", "tokens": {"total": 30}}},
            {"info": {"role": "assistant", "tokens": {"total": 60}}},
        ]
        consolidator._apply_session_occupancy_snapshot(
            messages, {"promptTokenBreakdown": {"totalUsedTokens": 1000}},
        )
        windows = [m["info"]["tokens"]["context_window"] for m in messages[1:]]
        self.assertEqual(windows[0], 0)
        self.assertEqual(windows[-1], 1000)
        self.assertLess(windows[1], windows[2])
        self.assertGreater(windows[1], 0)
