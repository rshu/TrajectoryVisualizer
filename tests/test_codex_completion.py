"""Codex runs that complete normally must say so.

The Codex converter never set a finish reason, so every assistant step carried
``finish=""``, even in rollouts that end in ``task_complete``.
Everything that asks "did this run stop normally?" (the converge outcome, the
batch report's per-task outcome, the multi-run "finished" column) reads the
last assistant step's finish, so every Codex run read as not finished,
including the ones that resolved their task. ``task_complete`` is Codex's own
statement that a turn ended normally; it now marks that turn's final assistant
message ``stop``, the value the other formats use. A rollout cut off before
``task_complete`` keeps an empty finish and still reads as not finished.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from trajviz.converge.alignment import _detect_success
from trajviz.insight.session import LoadedSession, load_session


def _ev(ts: int, etype: str, payload: dict) -> dict:
    return {"timestamp": f"2026-01-01T00:00:{ts:02d}Z", "type": etype, "payload": payload}


def _user(ts: int, text: str) -> dict:
    return _ev(ts, "response_item", {"type": "message", "role": "user",
                                     "content": [{"type": "input_text", "text": text}]})


def _assistant(ts: int, text: str) -> dict:
    return _ev(ts, "response_item", {"type": "message", "role": "assistant",
                                     "content": [{"type": "output_text", "text": text}]})


def _call(ts: int, call_id: str) -> list[dict]:
    return [
        _ev(ts, "response_item", {"type": "function_call", "name": "shell", "call_id": call_id,
                                  "arguments": json.dumps({"command": ["bash", "-lc", "ls"]})}),
        _ev(ts, "response_item", {"type": "function_call_output", "call_id": call_id,
                                  "output": json.dumps({"output": "a.py", "metadata": {"exit_code": 0}})}),
    ]


def _session(*events: dict) -> list[dict]:
    meta = _ev(0, "session_meta", {"id": "s", "cwd": "/w", "originator": "codex_cli_rs", "cli_version": "0.1"})
    return [meta, *events]


def _load(events: list[dict]) -> LoadedSession:
    path = os.path.join(tempfile.mkdtemp(), "rollout.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(e) for e in events))
    result = load_session(path)
    assert isinstance(result, LoadedSession), result
    return result


def _assistant_steps(session: LoadedSession) -> list[dict]:
    return [s for s in session.steps if s.get("role") == "assistant"]


class CodexCompletionTests(unittest.TestCase):
    def test_task_complete_marks_the_turns_last_assistant_step_stop(self):
        s = _load(_session(
            _user(1, "fix it"),
            _ev(2, "event_msg", {"type": "task_started"}),
            *_call(3, "c1"),
            _assistant(4, "done"),
            _ev(5, "event_msg", {"type": "task_complete"}),
        ))
        steps = _assistant_steps(s)
        self.assertEqual(steps[-1]["finish"], "stop")
        self.assertTrue(_detect_success(s.steps))

    def test_a_rollout_cut_off_before_task_complete_is_not_finished(self):
        s = _load(_session(_user(1, "fix it"), *_call(2, "c1"), _assistant(3, "working")))
        self.assertEqual(_assistant_steps(s)[-1]["finish"], "")
        self.assertFalse(_detect_success(s.steps))

    def test_each_completed_turn_ends_in_stop_and_mid_turn_steps_are_untouched(self):
        s = _load(_session(
            _user(1, "first"), *_call(2, "c1"), _assistant(3, "first done"),
            _ev(4, "event_msg", {"type": "task_complete"}),
            _user(5, "second"), *_call(6, "c2"), _assistant(7, "second done"),
            _ev(8, "event_msg", {"type": "task_complete"}),
        ))
        finishes = [step["finish"] for step in _assistant_steps(s)]
        self.assertEqual(finishes.count("stop"), 2, finishes)
        self.assertEqual(finishes[-1], "stop")

    def test_classic_shell_argv_becomes_a_command_string(self):
        # Older rollouts record the `shell` tool's command as argv; it used to
        # reach file-path extraction as a list and fail the whole analysis.
        s = _load(_session(_user(1, "fix it"), *_call(2, "c1"), _assistant(3, "done")))
        calls = [tc for step in s.steps for tc in step.get("tool_calls") or []]
        self.assertTrue(calls)
        self.assertEqual(calls[0]["input"].get("command"), "bash -lc ls")

    def test_task_complete_without_an_assistant_message_is_harmless(self):
        s = _load(_session(_user(1, "hello"), _ev(2, "event_msg", {"type": "task_complete"}),
                           _assistant(3, "late reply")))
        self.assertEqual(_assistant_steps(s)[-1]["finish"], "")


if __name__ == "__main__":
    unittest.main()
