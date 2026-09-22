"""Pi coding-agent JSONL loader."""

import json
import os
import tempfile
import unittest

from trajviz.insight.loaders import detect_format, load_trajectory
from trajviz.insight.metrics import extract_agent_info
from trajviz.insight.parser import parse_steps


def _write(tmp: str, name: str, events: list) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return path


def _session(**extra):
    event = {
        "type": "session",
        "version": 3,
        "id": "sess-1",
        "timestamp": "2026-08-24T01:29:43.221Z",
        "cwd": "/home/user/proj",
    }
    event.update(extra)
    return event


def _msg(role, ts_iso, content, **msg_extra):
    message = {"role": role, "content": content, "timestamp": 1787535033022}
    message.update(msg_extra)
    return {
        "type": "message",
        "id": msg_extra.get("id", f"id-{role}"),
        "timestamp": ts_iso,
        "message": message,
    }


class PiLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.events = [
            _session(),
            {"type": "model_change", "id": "m1", "timestamp": "2026-08-24T01:30:18.190Z",
             "provider": "zenmux", "modelId": "z-ai/glm-5.3-free"},
            {"type": "thinking_level_change", "id": "t1",
             "timestamp": "2026-08-24T01:30:18.191Z", "thinkingLevel": "medium"},
            _msg("user", "2026-08-24T01:30:33.026Z",
                 [{"type": "text", "text": "explore the repo"}]),
            _msg("assistant", "2026-08-24T01:30:33.065Z", [],
                 stopReason="error",
                 errorMessage="No API key for provider: zenmux",
                 provider="zenmux", model="z-ai/glm-5.3-free",
                 usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                        "totalTokens": 0}),
            {"type": "model_change", "id": "m2", "timestamp": "2026-08-24T01:32:56.590Z",
             "provider": "cline", "modelId": "nvidia/nemotron-3.5-lightning:free"},
            _msg("assistant", "2026-08-24T01:33:02.137Z", [
                    {"type": "thinking", "thinking": "I should list the directory."},
                    {"type": "toolCall", "id": "call-1", "name": "bash",
                     "arguments": {"command": "ls /home/user/proj"}},
                 ],
                 stopReason="toolUse",
                 provider="cline", model="nvidia/nemotron-3.5-lightning:free",
                 usage={"input": 1728, "output": 87, "cacheRead": 0, "cacheWrite": 0,
                        "reasoning": 49, "totalTokens": 1815}),
            _msg("toolResult", "2026-08-24T01:33:03.369Z",
                 [{"type": "text", "text": "trajviz\nREADME.md"}],
                 toolCallId="call-1", toolName="bash", isError=False),
            _msg("assistant", "2026-08-24T01:33:10.013Z", [
                    {"type": "toolCall", "id": "call-2", "name": "read",
                     "arguments": {"path": "/home/user/proj/README.md"}},
                 ],
                 stopReason="toolUse",
                 usage={"input": 2000, "output": 40, "totalTokens": 2040}),
            _msg("toolResult", "2026-08-24T01:33:10.029Z",
                 [{"type": "text", "text": "# TrajViz"}],
                 toolCallId="call-2", toolName="read", isError=False),
            _msg("assistant", "2026-08-24T01:33:12.379Z", [
                    {"type": "toolCall", "id": "call-3", "name": "write",
                     "arguments": {"path": "/home/user/proj/DESIGN.md",
                                   "content": "# Design"}},
                 ],
                 stopReason="toolUse",
                 usage={"input": 2100, "output": 55, "totalTokens": 2155}),
            _msg("toolResult", "2026-08-24T01:33:12.390Z",
                 [{"type": "text", "text": "written"}],
                 toolCallId="call-3", toolName="write", isError=False),
            _msg("assistant", "2026-08-24T01:33:36.288Z", [
                    {"type": "thinking", "thinking": "Summarize for the user."},
                    {"type": "text", "text": "Here is an overview of the repo."},
                 ],
                 stopReason="stop",
                 usage={"input": 8962, "output": 782, "reasoning": 30,
                        "totalTokens": 9744}),
        ]

    def _load(self, events=None, name="session.jsonl"):
        return load_trajectory(_write(self.tmp, name, events or self.events))

    def test_detects_as_pi(self):
        raw = self._load()
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "pi")
        self.assertTrue(raw.get("_pi_format"))

    def test_does_not_steal_codex_jsonl(self):
        events = [
            {"type": "session_meta", "timestamp": "2026-01-05T12:00:00.000Z",
             "payload": {"id": "s1", "cwd": "/p", "model": "gpt-5"}},
        ]
        raw = self._load(events, name="codex.jsonl")
        self.assertEqual(detect_format(raw), "codex")

    def test_session_metadata(self):
        raw = self._load()
        self.assertEqual(raw["metadata"]["session_id"], "sess-1")
        self.assertEqual(raw["metadata"]["directory"], "/home/user/proj")
        self.assertEqual(raw["metadata"]["directory_name"], "proj")
        self.assertEqual(raw["metadata"]["agent"], "pi")
        self.assertEqual(raw["metadata"]["model"], "nvidia/nemotron-3.5-lightning:free")

    def test_user_and_assistant_steps(self):
        steps = parse_steps(self._load())
        roles = [s["role"] for s in steps]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        user = next(s for s in steps if s["role"] == "user")
        self.assertIn("explore the repo", user["text_preview"])

    def test_thinking_becomes_reasoning(self):
        steps = parse_steps(self._load())
        reasoned = [s for s in steps if s.get("has_reasoning")]
        self.assertTrue(reasoned)
        self.assertTrue(any(
            any(p.get("type") == "reasoning" for p in s["parts"])
            for s in reasoned
        ))

    def test_tool_call_paired_with_result(self):
        steps = parse_steps(self._load())
        bash = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-1"
        )
        self.assertEqual(bash["tool_name"], "Bash")
        self.assertEqual(bash["status"], "success")
        self.assertIn("command", bash["input"])
        self.assertIn("trajviz", bash["output"])

    def test_read_write_normalized_to_file_path(self):
        steps = parse_steps(self._load())
        read = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-2"
        )
        write = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-3"
        )
        self.assertEqual(read["tool_name"], "Read")
        self.assertEqual(read["input"]["file_path"], "/home/user/proj/README.md")
        self.assertEqual(write["tool_name"], "Write")
        self.assertEqual(write["input"]["file_path"], "/home/user/proj/DESIGN.md")

    def test_error_assistant_surfaces_message(self):
        steps = parse_steps(self._load())
        error_steps = [
            s for s in steps
            if s["role"] == "assistant" and "No API key" in (s.get("text_preview") or "")
        ]
        self.assertTrue(error_steps)
        self.assertEqual(error_steps[0].get("finish"), "error")

    def test_token_usage_ingested(self):
        steps = parse_steps(self._load())
        # Vendor totalTokens is 1815 (input+output); reasoning 49 is extra.
        assistant = [
            s for s in steps
            if s["role"] == "assistant" and s["tokens"]["input"] == 1728
        ]
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0]["tokens"]["output"], 87)
        self.assertEqual(assistant[0]["tokens"]["reasoning"], 49)
        self.assertEqual(assistant[0]["tokens"]["total"], 1728 + 87 + 49)

    def test_token_total_includes_cache(self):
        events = [
            _session(),
            _msg("assistant", "2026-08-24T01:33:02.137Z",
                 [{"type": "text", "text": "ok"}],
                 stopReason="stop",
                 usage={"input": 100, "output": 20, "cacheRead": 50,
                        "cacheWrite": 10, "reasoning": 5, "totalTokens": 120}),
        ]
        steps = parse_steps(self._load(events, name="cache.jsonl"))
        asst = next(s for s in steps if s["role"] == "assistant")
        self.assertEqual(asst["tokens"]["cache_read"], 50)
        self.assertEqual(asst["tokens"]["cache_write"], 10)
        self.assertEqual(asst["tokens"]["total"], 100 + 20 + 5 + 50 + 10)

    def test_string_content_is_one_text_part(self):
        events = [
            _session(),
            _msg("user", "2026-08-24T01:30:33.026Z", "explore the repo"),
            _msg("assistant", "2026-08-24T01:33:36.288Z",
                 "Here is an overview of the repo.",
                 stopReason="stop"),
        ]
        steps = parse_steps(self._load(events, name="string-content.jsonl"))
        user = next(s for s in steps if s["role"] == "user")
        asst = next(s for s in steps if s["role"] == "assistant")
        self.assertEqual(user["text_preview"], "explore the repo")
        self.assertEqual(len(user["parts"]), 1)
        self.assertEqual(asst["text_preview"], "Here is an overview of the repo.")
        self.assertEqual(len(asst["parts"]), 1)

    def test_session_header_uses_last_model(self):
        steps = parse_steps(self._load())
        model_id, provider_id, _ = extract_agent_info(steps)
        self.assertEqual(model_id, "nvidia/nemotron-3.5-lightning:free")
        self.assertEqual(provider_id, "cline")

    def test_timestamps_are_epoch_ms(self):
        steps = parse_steps(self._load())
        stamped = [s for s in steps if s.get("time_created_ms")]
        self.assertTrue(stamped)
        for s in stamped:
            self.assertIsInstance(s["time_created_ms"], int)
            self.assertGreater(s["time_created_ms"], 10**12)

    def test_dangling_tool_call_marked_error(self):
        events = [
            _session(),
            _msg("assistant", "2026-08-24T01:33:02.137Z", [
                {"type": "toolCall", "id": "call-x", "name": "bash",
                 "arguments": {"command": "pwd"}},
            ]),
        ]
        steps = parse_steps(self._load(events, name="dangling.jsonl"))
        tool = next(tc for s in steps for tc in s["tool_calls"])
        self.assertEqual(tool["status"], "error")
        self.assertEqual(tool["tool_id"], "call-x")

    def test_failed_tool_result_status(self):
        events = [
            _session(),
            _msg("assistant", "2026-08-24T01:33:02.137Z", [
                {"type": "toolCall", "id": "call-f", "name": "bash",
                 "arguments": {"command": "false"}},
            ]),
            _msg("toolResult", "2026-08-24T01:33:03.000Z",
                 [{"type": "text", "text": "exit 1"}],
                 toolCallId="call-f", toolName="bash", isError=True),
        ]
        steps = parse_steps(self._load(events, name="fail.jsonl"))
        tool = next(tc for s in steps for tc in s["tool_calls"])
        self.assertEqual(tool["status"], "error")
        self.assertEqual(tool["output"], "exit 1")

    def test_truncated_final_line_does_not_reject_file(self):
        path = os.path.join(self.tmp, "trunc.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for event in self.events[:4]:
                f.write(json.dumps(event) + "\n")
            f.write('{"type":"message","timestamp":"2026-')
        raw = load_trajectory(path)
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "pi")

    def test_unknown_jsonl_still_rejected(self):
        path = _write(self.tmp, "other.jsonl", [{"type": "not-a-session", "id": "x"}])
        raw = load_trajectory(path)
        self.assertIn("_error", raw)
        self.assertIn("Pi", raw["_error"])


if __name__ == "__main__":
    unittest.main()
