"""ICode / Chrys expanded-session loader."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from trajviz.insight.charts import build_token_chart
from trajviz.insight.context_usage import (
    context_usage_breakdown,
    detect_compaction_events,
    step_context_occupancy,
)
from trajviz.insight.loaders import detect_format, load_trajectory
from trajviz.insight.metrics import extract_agent_info
from trajviz.insight.parser import parse_steps
from trajviz.insight.patterns import extract_subagent_sessions
from trajviz.insight.presenters import trajectory_format_label
from trajviz.insight.tool_failure import tool_call_failed


FIXTURE = Path(__file__).parent / "fixtures" / "icode_minimal.json"


class ICodeLoaderTests(unittest.TestCase):
    def test_format_is_detected_before_generic_opencode(self):
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(detect_format(raw), "icode")

        generic = {"info": {"id": "ses_open"}, "messages": []}
        self.assertEqual(detect_format(generic), "opencode")

        codearts = json.loads(
            (Path(__file__).parent / "fixtures" / "codearts_minimal.json").read_text(encoding="utf-8")
        )
        self.assertEqual(detect_format(codearts), "codearts")

    def test_detects_meta_state_without_export_stamp(self):
        raw = {
            "meta": {"session_id": "session-x", "schema_version": 1},
            "state": {
                "messages": [
                    {"type": "message", "role": "user", "contents": [{"type": "text", "text": "hi"}]},
                ]
            },
        }
        self.assertEqual(detect_format(raw), "icode")

    def test_loader_preserves_product_identity_and_session_totals(self):
        loaded = load_trajectory(str(FIXTURE))

        self.assertNotIn("_error", loaded)
        self.assertEqual(detect_format(loaded), "icode")
        self.assertTrue(loaded.get("_icode_format"))
        self.assertEqual(loaded["metadata"]["agent"], "icode")
        self.assertEqual(loaded["metadata"]["generator_name"], "chrys-manager")
        self.assertEqual(loaded["metadata"]["session_id"], "session-main-0001")
        self.assertEqual(loaded["metadata"]["directory"], "/home/user/proj")
        self.assertEqual(loaded["metadata"]["directory_name"], "proj")
        self.assertEqual(loaded["metadata"]["model"], "test-model")
        self.assertEqual(loaded["metadata"]["session_count"], 2)
        self.assertEqual(loaded["metadata"]["sub_agent_count"], 1)
        self.assertEqual(loaded["token_usage"]["total_tokens"], 1200)
        self.assertEqual(loaded["token_usage"]["prompt_tokens"], 1000)
        self.assertEqual(loaded["token_usage"]["completion_tokens"], 200)

    def test_user_and_assistant_steps_skip_turn_markers(self):
        steps = parse_steps(load_trajectory(str(FIXTURE)))
        roles = [s["role"] for s in steps]
        self.assertEqual(roles.count("user"), 2)
        self.assertGreaterEqual(roles.count("assistant"), 5)
        self.assertNotIn("tool", roles)
        user = next(s for s in steps if s["role"] == "user")
        self.assertIn("lanczos caller", user["text_preview"])
        self.assertFalse(any(s.get("message_id") == "chatcmpl-0005" for s in steps))

    def test_glob_tool_paired_with_result(self):
        steps = parse_steps(load_trajectory(str(FIXTURE)))
        glob_call = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-glob-1"
        )
        self.assertEqual(glob_call["tool_name"], "Glob")
        self.assertEqual(glob_call["status"], "completed")
        self.assertIn("lanczos_caller.cpp", glob_call["output"])
        self.assertEqual(glob_call["input"]["pattern"], "**/lanczos_caller.cpp")

    def test_split_shell_calls_keep_separate_failures(self):
        steps = parse_steps(load_trajectory(str(FIXTURE)))
        tools = [tc for s in steps for tc in s["tool_calls"]]
        sh = next(tc for tc in tools if tc["tool_id"] == "call-sh-1")
        paired = next(tc for tc in tools if tc["tool_id"] == "")

        self.assertEqual(sh["tool_name"], "Bash")
        self.assertTrue(tool_call_failed(sh))
        self.assertEqual(sh["error_type"], "argument_parsing")
        self.assertIn("Argument parsing failed", sh["output"])
        self.assertEqual(sh["title"], "ls")

        self.assertEqual(paired["tool_name"], "Bash")
        self.assertEqual(paired["input"]["command"], "ls")
        self.assertTrue(tool_call_failed(paired))
        self.assertEqual(paired["error_type"], "tool_not_found")

    def test_subagent_is_flattened_and_annotated(self):
        loaded = load_trajectory(str(FIXTURE))
        steps = parse_steps(loaded)

        spawn = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-spawn-1"
        )
        self.assertEqual(spawn["tool_name"], "Agent")
        self.assertEqual(spawn["metadata"]["sessionId"], "inv-0001")
        self.assertEqual(spawn["metadata"]["parentSessionId"], "session-main-0001")
        self.assertFalse(tool_call_failed(spawn))

        child_steps = [s for s in steps if s.get("session_id") == "inv-0001"]
        self.assertTrue(child_steps)
        self.assertTrue(all(s.get("is_sub_agent") for s in child_steps))
        self.assertTrue(all(s.get("parent_session_id") == "session-main-0001" for s in child_steps))
        self.assertEqual(child_steps[0]["session_depth"], 1)
        self.assertEqual(child_steps[0]["session_title"], "Explore Agent")
        self.assertEqual(child_steps[0]["agent"], "Explore")

        grep = next(
            tc for s in child_steps for tc in s["tool_calls"] if tc["tool_id"] == "call-sub-1"
        )
        self.assertEqual(grep["tool_name"], "Grep")
        self.assertIn("lanczos", grep["output"])

        sessions = extract_subagent_sessions(steps, loaded.get("messages") or [])
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "inv-0001")
        self.assertIsNotNone(sessions[0]["spawn_step"])

        spawn_idx = next(s["index"] for s in steps if any(tc["tool_id"] == "call-spawn-1" for tc in s["tool_calls"]))
        first_child_idx = child_steps[0]["index"]
        summary = next(s for s in steps if s.get("message_id") == "chatcmpl-0004")
        self.assertLess(spawn_idx, first_child_idx)
        self.assertLess(first_child_idx, summary["index"])

    def test_session_header_uses_model_and_provider(self):
        steps = parse_steps(load_trajectory(str(FIXTURE)))
        model_id, provider_id, _ = extract_agent_info(steps)
        self.assertEqual(model_id, "test-model")
        self.assertEqual(provider_id, "openai")

    def test_timestamps_are_epoch_ms(self):
        steps = parse_steps(load_trajectory(str(FIXTURE)))
        stamped = [s for s in steps if s.get("time_created_ms")]
        self.assertTrue(stamped)
        for step in stamped:
            self.assertIsInstance(step["time_created_ms"], int)
            self.assertGreater(step["time_created_ms"], 10**12)

    def test_token_chart_falls_back_to_total_without_breakdown(self):
        steps = parse_steps(load_trajectory(str(FIXTURE)))
        figure = build_token_chart(steps, format="icode")
        self.assertEqual([trace.name for trace in figure.data], ["Total"])

    def test_context_utilization_per_step_tokens_and_compaction(self):
        # Chrys ``_group.token_count`` is each message's own window
        # contribution, split across the assistant call and its tool result.
        # Token Usage by Step shows the group's own tokens; Context
        # Utilization occupancy is the live window: running contributions
        # minus compacted ranges plus their summaries.
        loaded = load_trajectory(str(FIXTURE))
        steps = parse_steps(loaded)
        main = [s for s in steps if s.get("session_id") == "session-main-0001"]

        # Per-step total is the group's contribution (40 + 20), not a
        # monotonically increasing running window.
        first_tool = next(s for s in main if s.get("message_id") == "chatcmpl-0001")
        self.assertEqual(first_tool["tokens"]["total"], 60)
        totals = [s["tokens"]["total"] for s in main if s["role"] == "assistant"]
        self.assertEqual(totals[:3], [60, 55, 110])

        # The compressed context covering messages [0, 2) emits a compaction
        # checkpoint with its summary, and the window restarts from it.
        compaction = next(s for s in main if s["role"] == "compaction")
        self.assertTrue(compaction["is_compaction_checkpoint"])
        self.assertIn("lanczos caller", compaction["parts"][0]["summary"])
        self.assertTrue(any(e["kind"] == "compaction_message" for e in detect_compaction_events(steps)))

        # Live window after the last step: 12 (summary) + 20 + 30 + 25 + 50
        # + 60 + 70.
        last_main = next(s for s in reversed(main) if s["role"] == "assistant")
        self.assertEqual(step_context_occupancy(last_main)["occupancy"], 267)

        breakdown = context_usage_breakdown(steps, raw=loaded)
        self.assertEqual(breakdown["occupancy"], 267)
        self.assertEqual(sum(breakdown["buckets"].values()), 267)
        self.assertGreater(breakdown["buckets"]["summarized"], 0)
        self.assertFalse(breakdown["scaled"])

    def test_human_readable_format_label(self):
        self.assertEqual(trajectory_format_label("icode"), "ICode")

    def test_format_hint_mismatch_with_opencode(self):
        loaded = load_trajectory(str(FIXTURE), format_hint="opencode")
        self.assertEqual(loaded.get("_error_code"), "mismatch")
        self.assertEqual(loaded.get("_detected"), "icode")
        self.assertEqual(loaded.get("_selected"), "opencode")

    def test_already_converted_stamp_is_stable(self):
        loaded = load_trajectory(str(FIXTURE))
        self.assertEqual(detect_format(loaded), "icode")
        self.assertEqual(detect_format(loaded), "icode")

    def test_jsonl_singleton_object_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(FIXTURE.read_text(encoding="utf-8"))
            loaded = load_trajectory(path)
        self.assertNotIn("_error", loaded)
        self.assertEqual(detect_format(loaded), "icode")


if __name__ == "__main__":
    unittest.main()
