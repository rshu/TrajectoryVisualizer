"""Realistic synthesized exports for the four formats added by PR #13.

The repo ships only two hand-written minimal fixtures (``cursor_minimal.json``,
``icode_minimal.json``) for ~2,900 lines of Cursor / ICode / DSH / Pi loader
code, and the 2,500-file corpus contains none of those formats.  The exports
built here are modelled on the structures the real producers emit — verified
against live DeepSeek Harness ``session.jsonl`` logs, live Chrys/ICode
``session.json`` state, and a live ``cursor_consolidator.py`` export — so they
exercise multi-turn conversations, parallel tool calls, sub-agents, failures,
unicode, JSON-string tool arguments and missing optional fields.

Every assertion here states a property that must stay true for the numbers the
dashboard publishes to mean anything: the right converter is chosen, no turn or
tool call is silently swallowed, failures stay failures, sub-agent work is
attributed to the child session, and token totals never contradict their own
components.  None of these tests assert a currently-observed value merely
because it is what the code prints today.
"""

import json
import os
import tempfile
import unittest
import zipfile

from trajviz.insight.formats.sniff import _EVENT_FORMATS, _OBJECT_FORMATS
from trajviz.insight.loaders import (
    FORMAT_LABELS,
    check_format_selection,
    detect_format,
    load_trajectory,
)
from trajviz.insight.metrics import compute_metrics
from trajviz.insight.parser import parse_steps

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

UNICODE_PROMPT = "分析这个代码仓 🚀 — review the parser, s'il vous plaît"


def _write_jsonl(directory: str, name: str, events: list) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def _write_json(directory: str, name: str, payload: dict) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


def _tool_calls(steps: list[dict]) -> list[dict]:
    return [tc for step in steps for tc in (step.get("tool_calls") or [])]


def _roles(steps: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.get("role")] = counts.get(step.get("role"), 0) + 1
    return counts


def _assert_token_accounting(case: unittest.TestCase, steps: list[dict], metrics: dict) -> None:
    """Per-step and aggregate token counts must be finite, non-negative, coherent.

    ``total`` is the number every downstream rate divides by, so it must at
    minimum cover the disjoint components (fresh input, output, cache read) and
    must never exceed them by more than the reasoning count, whichever
    convention a format uses for reasoning-inside-output.  Formats that report
    only a per-message total (ICode) are exempt from the upper bound.
    """
    for step in steps:
        tokens = step["tokens"]
        for key, value in tokens.items():
            case.assertIsInstance(value, int, f"token field {key} is not an int: {value!r}")
            case.assertGreaterEqual(value, 0, f"negative token field {key}")
        disjoint = tokens["input"] + tokens["output"] + tokens["cache_read"] + tokens["cache_write"]
        reasoning = tokens.get("reasoning", 0)
        case.assertGreaterEqual(
            tokens["total"], disjoint,
            f"step total {tokens['total']} is below its own components {disjoint}",
        )
        if disjoint:
            case.assertLessEqual(
                tokens["total"], disjoint + reasoning,
                f"step total {tokens['total']} exceeds components+reasoning {disjoint + reasoning}",
            )
    aggregate = metrics.get("tokens") or {}
    for key, value in aggregate.items():
        case.assertGreaterEqual(value, 0, f"negative aggregate token field {key}")
    if aggregate:
        case.assertGreaterEqual(
            aggregate["total"],
            aggregate["input"] + aggregate["output"] + aggregate["cache_read"] + aggregate["cache_write"],
        )


# --------------------------------------------------------------------------
# DeepSeek Harness
# --------------------------------------------------------------------------

DSH_PARENT_ID = "session-4e6805f9-4f75-47fe-a565-cfa65c50af8e"
DSH_CHILD_ID = "session-9a11b0c2-2d44-4c11-a0aa-77d5e2c31f90"
DSH_T0 = 1787041975311


def _dsh_assistant(seq, time_ms, message_id, content, usage=None, turn=1, step=1):
    data = {
        "turn": turn,
        "step": step,
        "message": {
            "id": message_id,
            "role": "assistant",
            "source": {"kind": "model", "provider": "deepseek-official", "model": "deepseek-v4-pro"},
            "content": content,
        },
    }
    if usage is not None:
        data["usage"] = usage
    return {"type": "assistant/message", "seq": seq, "time": time_ms, "data": data}


def _dsh_result(seq, time_ms, call_id, text, *, is_error=False, turn=1, step=1):
    return {
        "type": "tool/result",
        "seq": seq,
        "time": time_ms,
        "data": {
            "turn": turn,
            "step": step,
            "message": {
                "id": f"result-{call_id}",
                "role": "tool",
                "source": {"kind": "tool", "callId": call_id},
                "content": [{
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "isError": is_error,
                    "content": [{"type": "text", "text": text}],
                }],
            },
        },
    }


def dsh_parent_events() -> list[dict]:
    """A realistic multi-turn DSH parent log (two human turns, one delegation)."""
    return [
        {"type": "session", "version": 3, "id": DSH_PARENT_ID, "createdAt": DSH_T0,
         "cwd": "/w/proj", "isSeeded": False, "delegationDepth": 0, "agentPreset": "standard"},
        {"type": "permission/preset", "seq": 0, "time": DSH_T0 + 1, "data": {"preset": "workspace-write"}},
        {"type": "sandbox/mode", "seq": 1, "time": DSH_T0 + 2, "data": {"mode": "workspace-write"}},
        {"type": "request/context", "seq": 2, "time": DSH_T0 + 3,
         "data": {"model": "deepseek-v4-pro", "provider": "deepseek-official"}},
        {"type": "session/title", "seq": 3, "time": DSH_T0 + 4, "data": {"title": "Parser review"}},
        {"type": "turn/start", "seq": 4, "time": DSH_T0 + 10, "data": {"turn": 1}},
        # Human prompt (kind "user") — becomes a step.
        {"type": "user/message", "seq": 5, "time": DSH_T0 + 11,
         "data": {"role": "user", "id": "u1", "content": [{"type": "text", "text": UNICODE_PROMPT}],
                  "source": {"kind": "user", "rpcId": "rpc-1", "clientTimeZone": "Asia/Shanghai"}}},
        # Two parallel tool calls in one assistant turn.
        _dsh_assistant(
            6, DSH_T0 + 2000, "m1",
            [
                {"type": "reasoning", "text": "Look at the parser and run the suite."},
                {"type": "tool-call", "id": "call_00_read", "name": "read",
                 "arguments": '{"path": "/w/proj/parser.py"}'},
                {"type": "tool-call", "id": "call_01_bash", "name": "bash",
                 "arguments": '{"command": "pytest -q"}'},
            ],
            usage={"inputTokens": 7745, "outputTokens": 129, "cacheReadTokens": 0, "reasoningTokens": 52},
        ),
        {"type": "tool/call", "seq": 7, "time": DSH_T0 + 2100,
         "data": {"callId": "call_00_read", "name": "read", "arguments": '{"path": "/w/proj/parser.py"}'}},
        {"type": "tool/call", "seq": 8, "time": DSH_T0 + 2110,
         "data": {"callId": "call_01_bash", "name": "bash", "arguments": '{"command": "pytest -q"}'}},
        _dsh_result(9, DSH_T0 + 2500, "call_00_read", "File: /w/proj/parser.py (440 lines)\n1|import json"),
        _dsh_result(10, DSH_T0 + 9000, "call_01_bash",
                    "Error: cannot run pytest: file changed since it was read", is_error=True),
        # Runtime notice, NOT a human prompt — must not become a user step.
        {"type": "user/message", "seq": 11, "time": DSH_T0 + 9100,
         "data": {"role": "user", "id": "u-plugin",
                  "content": [{"type": "text", "text": "[sandbox snapshot taken]"}],
                  "source": {"kind": "plugin", "rpcId": "rpc-p"}}},
        {"type": "turn/end", "seq": 12, "time": DSH_T0 + 9200, "data": {"turn": 1}},
        {"type": "turn/start", "seq": 13, "time": DSH_T0 + 20000, "data": {"turn": 2}},
        {"type": "user/message", "seq": 14, "time": DSH_T0 + 20001,
         "data": {"role": "user", "id": "u2", "content": [{"type": "text", "text": "delegate the rest"}],
                  "source": {"kind": "user", "rpcId": "rpc-2"}}},
        # Delegation + a large output body.
        _dsh_assistant(
            15, DSH_T0 + 21000, "m2",
            [
                {"type": "text", "text": "Delegating."},
                {"type": "tool-call", "id": "call_02_sub", "name": "subagent",
                 "arguments": '{"prompt": "audit the loaders"}'},
            ],
            usage={"inputTokens": 594, "outputTokens": 706, "cacheReadTokens": 16128, "reasoningTokens": 451},
            turn=2, step=1,
        ),
        _dsh_result(16, DSH_T0 + 45000, "call_02_sub",
                    f"started subagent {DSH_CHILD_ID}\n" + ("x" * 20000), turn=2, step=1),
        # Final assistant turn whose tool call never gets a result.
        _dsh_assistant(
            17, DSH_T0 + 46000, "m3",
            [
                {"type": "tool-call", "id": "call_03_orphan", "name": "edit",
                 "arguments": '{"path": "/w/proj/a.py", "old": "a", "new": "b"}'},
            ],
            usage={"inputTokens": 100, "outputTokens": 40, "cacheReadTokens": 0, "reasoningTokens": 0},
            turn=2, step=2,
        ),
    ]


def dsh_child_events() -> list[dict]:
    """A realistic forked-child DSH log (``origin: subagent`` + ``parentSession``)."""
    return [
        {"type": "session", "version": 3, "id": DSH_CHILD_ID, "createdAt": DSH_T0 + 25000,
         "cwd": "/w/proj", "origin": "subagent", "parentSession": DSH_PARENT_ID,
         "delegationDepth": 1, "agentPreset": "explore"},
        {"type": "user/message", "seq": 0, "time": DSH_T0 + 25001,
         "data": {"role": "user", "id": "cu1", "content": [{"type": "text", "text": "audit the loaders"}],
                  "source": {"kind": "user"}}},
        _dsh_assistant(
            1, DSH_T0 + 26000, "cm1",
            [{"type": "tool-call", "id": "call_10_grep", "name": "grep",
              "arguments": '{"pattern": "detect_format"}'}],
            usage={"inputTokens": 300, "outputTokens": 20, "cacheReadTokens": 0, "reasoningTokens": 5},
        ),
        _dsh_result(2, DSH_T0 + 26500, "call_10_grep", "loaders.py:94: def detect_format"),
    ]


class DshSyntheticExportTests(unittest.TestCase):
    """DeepSeek Harness JSONL: turn, tool and sub-agent bookkeeping."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = _write_jsonl(self.tmp.name, "session.jsonl", dsh_parent_events())
        self.raw = load_trajectory(self.path)
        self.assertNotIn("_error", self.raw, self.raw.get("_error"))
        self.steps = parse_steps(self.raw)
        self.metrics = compute_metrics(self.steps, self.raw)

    def test_detects_as_deepseek_harness_not_codex_or_pi(self):
        """A DSH header carries ``type: session`` like Pi; sniff order must not confuse them.

        Mis-detection here is silent: another converter would produce a
        near-empty trajectory with no error banner.
        """
        self.assertEqual(detect_format(dsh_parent_events()), "dsh")
        self.assertEqual(self.raw.get("_dsh_format"), True)
        # A second detection pass runs in the UI gate and in attribution.
        self.assertEqual(detect_format(self.raw), "dsh")

    def test_every_human_turn_and_model_turn_becomes_exactly_one_step(self):
        """Step counts are the denominator of most published rates.

        Two human prompts and three assistant messages were encoded; the
        ``plugin``-sourced runtime notice is not a human turn.
        """
        self.assertEqual(_roles(self.steps), {"user": 2, "assistant": 3})

    def test_runtime_notices_are_not_counted_as_human_prompts(self):
        """``source.kind`` other than ``user`` marks harness-generated messages."""
        user_texts = [
            part.get("text")
            for step in self.steps if step["role"] == "user"
            for part in step["parts"]
        ]
        self.assertIn(UNICODE_PROMPT, user_texts)
        self.assertNotIn("[sandbox snapshot taken]", user_texts)

    def test_every_tool_call_block_survives_conversion_with_its_arguments(self):
        """Parallel calls in one assistant turn must each stay a distinct call.

        DSH ships arguments as a JSON *string*; losing the parse would blank
        every file path and command in the workflow view.
        """
        calls = _tool_calls(self.steps)
        self.assertEqual(len(calls), 4)
        by_id = {tc["tool_id"]: tc for tc in calls}
        self.assertEqual(sorted(by_id), ["call_00_read", "call_01_bash", "call_02_sub", "call_03_orphan"])
        self.assertEqual(by_id["call_00_read"]["tool_name"], "Read")
        self.assertEqual(by_id["call_00_read"]["input"].get("file_path"), "/w/proj/parser.py")
        self.assertEqual(by_id["call_01_bash"]["tool_name"], "Bash")
        self.assertEqual(by_id["call_01_bash"]["input"].get("command"), "pytest -q")
        self.assertEqual(by_id["call_02_sub"]["tool_name"], "Agent")

    def test_failed_and_unanswered_tool_calls_are_not_reported_as_successes(self):
        """A result flagged ``isError`` and a call that never returned are failures.

        Counting either as a success inflates the tool-success rate, a headline
        number in the dashboard and in any paper that quotes it.
        """
        by_id = {tc["tool_id"]: tc for tc in _tool_calls(self.steps)}
        self.assertEqual(by_id["call_01_bash"]["status"], "error")
        self.assertEqual(by_id["call_03_orphan"]["status"], "error")
        self.assertEqual(by_id["call_00_read"]["status"], "success")
        self.assertEqual(self.metrics["tool_call_count"], 4)
        self.assertEqual(self.metrics["tool_fail"], 2)

    def test_large_tool_output_is_preserved_not_truncated_away(self):
        """A 20 KB result body must reach the step model intact."""
        by_id = {tc["tool_id"]: tc for tc in _tool_calls(self.steps)}
        self.assertGreater(len(by_id["call_02_sub"]["output"]), 19000)

    def test_epoch_millisecond_timestamps_are_read_as_milliseconds(self):
        """DSH stamps epoch **ms**; reading them as seconds moves sessions by decades."""
        started = self.raw["timing"]["started_at"]
        self.assertTrue(started.startswith("2026-"), started)
        span_seconds = (dsh_parent_events()[-1]["time"] - DSH_T0) / 1000.0
        self.assertAlmostEqual(self.raw["timing"]["total_duration"], span_seconds, delta=1.0)

    def test_token_accounting_is_coherent(self):
        _assert_token_accounting(self, self.steps, self.metrics)

    def test_zip_export_attributes_child_work_to_the_child_session(self):
        """A DSH export zip merges ``subagents/<id>/session.jsonl``.

        If the child's steps were credited to the parent session, per-agent
        pressure, swimlanes and delegation accounting would all be wrong.
        """
        zip_path = os.path.join(self.tmp.name, "export.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "session.jsonl",
                "\n".join(json.dumps(e, ensure_ascii=False) for e in dsh_parent_events()),
            )
            archive.writestr(
                f"subagents/{DSH_CHILD_ID}/session.jsonl",
                "\n".join(json.dumps(e, ensure_ascii=False) for e in dsh_child_events()),
            )
        raw = load_trajectory(zip_path)
        self.assertNotIn("_error", raw, raw.get("_error"))
        self.assertEqual(detect_format(raw), "dsh")
        steps = parse_steps(raw)
        child = [s for s in steps if s["session_id"] == DSH_CHILD_ID]
        parent = [s for s in steps if s["session_id"] == DSH_PARENT_ID]
        self.assertTrue(child, "child session steps were dropped from the merged export")
        self.assertTrue(all(s["is_sub_agent"] for s in child))
        self.assertTrue(all(s["parent_session_id"] == DSH_PARENT_ID for s in child))
        self.assertFalse(any(s["is_sub_agent"] for s in parent))
        self.assertEqual(raw["metadata"]["sub_agent_count"], 1)
        # The child's own grep call must survive the merge.
        self.assertIn("call_10_grep", {tc["tool_id"] for tc in _tool_calls(steps)})


# --------------------------------------------------------------------------
# Pi
# --------------------------------------------------------------------------

PI_T0 = "2026-08-24T01:29:43.221Z"


def _pi_message(event_id, ts_iso, message):
    return {"type": "message", "id": event_id, "timestamp": ts_iso, "message": message}


def pi_events() -> list[dict]:
    """A realistic multi-turn Pi log with paired and unpaired tool calls."""
    return [
        {"type": "session", "version": 4, "id": "pi-sess-7", "timestamp": PI_T0, "cwd": "/w/proj"},
        {"type": "model_change", "id": "mc1", "timestamp": "2026-08-24T01:29:44.000Z",
         "provider": "zenmux", "modelId": "z-ai/glm-5.3"},
        {"type": "thinking_level_change", "id": "tl1", "timestamp": "2026-08-24T01:29:44.500Z",
         "thinkingLevel": "high"},
        _pi_message("e1", "2026-08-24T01:30:00.000Z",
                    {"role": "user", "content": [{"type": "text", "text": UNICODE_PROMPT}]}),
        _pi_message("e2", "2026-08-24T01:30:05.000Z", {
            "role": "assistant",
            "model": "z-ai/glm-5.3",
            "provider": "zenmux",
            "content": [
                {"type": "thinking", "thinking": "Read the file, then grep."},
                {"type": "text", "text": "Starting."},
                {"type": "toolCall", "id": "tc-1", "name": "read",
                 "arguments": '{"path": "/w/proj/a.py"}'},
                {"type": "toolCall", "id": "tc-2", "name": "grep",
                 "arguments": {"pattern": "detect_format"}},
            ],
            "usage": {"input": 1200, "output": 90, "cacheRead": 800, "cacheWrite": 0,
                      "reasoning": 40, "totalTokens": 1290},
            "stopReason": "tool-calls",
        }),
        _pi_message("e3", "2026-08-24T01:30:06.000Z", {
            "role": "toolResult", "toolCallId": "tc-1", "toolName": "read",
            "content": [{"type": "text", "text": "print('héllo')"}], "isError": False,
        }),
        _pi_message("e4", "2026-08-24T01:30:07.000Z", {
            "role": "toolResult", "toolCallId": "tc-2", "toolName": "grep",
            "content": "grep: /w/proj: Is a directory", "isError": True,
        }),
        _pi_message("e5", "2026-08-24T01:30:20.000Z",
                    {"role": "user", "content": "now write it"}),
        _pi_message("e6", "2026-08-24T01:30:25.000Z", {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "id": "tc-3", "name": "write",
                 "arguments": '{"path": "/w/proj/b.py", "content": "x"}'},
            ],
            "usage": {"input": 300, "output": 15, "cacheRead": 0, "cacheWrite": 64,
                      "reasoning": 0},
            "stopReason": "tool-calls",
        }),
    ]


class PiSyntheticExportTests(unittest.TestCase):
    """Pi coding-agent JSONL: pairing, error propagation, sniff isolation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = _write_jsonl(self.tmp.name, "pi.jsonl", pi_events())
        self.raw = load_trajectory(self.path)
        self.assertNotIn("_error", self.raw, self.raw.get("_error"))
        self.steps = parse_steps(self.raw)
        self.metrics = compute_metrics(self.steps, self.raw)

    def test_detects_as_pi_and_stays_pi_on_a_second_pass(self):
        """A Pi header has no Codex ``session_meta`` and no DSH body markers."""
        self.assertEqual(detect_format(pi_events()), "pi")
        self.assertEqual(detect_format(self.raw), "pi")

    def test_each_turn_becomes_one_step_and_metadata_events_do_not(self):
        """``model_change`` / ``thinking_level_change`` are settings, not turns."""
        self.assertEqual(_roles(self.steps), {"user": 2, "assistant": 2})

    def test_tool_results_are_paired_back_onto_their_call_by_id(self):
        """Pi delivers results as separate ``toolResult`` messages.

        A failed pairing would both lose the output and leave the call
        permanently ``pending`` -> reported as an error that never happened.
        """
        by_id = {tc["tool_id"]: tc for tc in _tool_calls(self.steps)}
        self.assertEqual(sorted(by_id), ["tc-1", "tc-2", "tc-3"])
        self.assertEqual(by_id["tc-1"]["status"], "success")
        self.assertEqual(by_id["tc-1"]["output"], "print('héllo')")
        self.assertEqual(by_id["tc-1"]["input"].get("file_path"), "/w/proj/a.py")

    def test_an_error_result_marks_its_call_failed(self):
        by_id = {tc["tool_id"]: tc for tc in _tool_calls(self.steps)}
        self.assertEqual(by_id["tc-2"]["status"], "error")
        self.assertEqual(by_id["tc-2"]["error"], "grep: /w/proj: Is a directory")

    def test_a_call_that_never_returned_is_not_a_success(self):
        by_id = {tc["tool_id"]: tc for tc in _tool_calls(self.steps)}
        self.assertEqual(by_id["tc-3"]["status"], "error")
        self.assertNotEqual(self.metrics["tool_success_rate"], 100.0)

    def test_string_content_is_not_iterated_character_by_character(self):
        """``content`` may be a bare string; treating it as a sequence explodes a
        prompt into hundreds of one-character text parts."""
        user_steps = [s for s in self.steps if s["role"] == "user"]
        second = user_steps[1]
        self.assertEqual(len(second["parts"]), 1)
        self.assertEqual(second["parts"][0]["text"], "now write it")

    def test_token_accounting_is_coherent(self):
        _assert_token_accounting(self, self.steps, self.metrics)


# --------------------------------------------------------------------------
# ICode / Chrys
# --------------------------------------------------------------------------

ICODE_SESSION_ID = "b6486c5d6d22"
ICODE_CHILD_ID = "inv-explore-1"


def _icode_msg(role, contents, props=None, message_id=""):
    message = {"type": "message", "role": role, "contents": contents,
               "additional_properties": props or {}}
    if message_id:
        message["message_id"] = message_id
    return message


def _icode_group(index, kind, tokens, created):
    return {
        "_group": {"id": f"group_msg_{index}", "kind": kind, "index": index,
                   "token_count": tokens, "token_estimator_v": 4},
        "_chrys_created_at": created,
    }


def icode_export() -> dict:
    """A realistic Chrys expanded-session export with a nested sub-agent."""
    child = {
        "meta": {
            "session_id": "child-sess-1",
            "invocation_id": ICODE_CHILD_ID,
            "parent_session_id": ICODE_SESSION_ID,
            "parent_provider_call_id": "call_02_explore",
            "agent_display_name": "explorer",
            "agent_profile": "explore",
            "model_id": "deepseek-v4-pro",
            "model_provider": "openrouter",
            "created_at": "2026-08-31T08:10:00+00:00",
            "ended_at": "2026-08-31T08:12:30+00:00",
        },
        "state": {
            "messages": [
                _icode_msg("user", [{"type": "text", "text": "map the loaders"}],
                           _icode_group(0, "user", 120, "2026-08-31T08:10:01+00:00")),
                _icode_msg("assistant", [
                    {"type": "function_call", "name": "grep", "call_id": "c-child-1",
                     "arguments": '{"pattern": "detect_format"}',
                     "additional_properties": {"_chrys_tool_kind": "search"}},
                ], _icode_group(1, "tool_call", 60, "2026-08-31T08:10:02+00:00")),
                _icode_msg("tool", [
                    {"type": "function_result", "call_id": "c-child-1",
                     "result": "loaders.py:94", "additional_properties": {}},
                ], _icode_group(1, "tool_call", 30, None)),
            ],
            "last_usage": {"system_overhead_tokens": 500},
        },
    }
    return {
        "meta": {
            "session_id": ICODE_SESSION_ID,
            "generated_title": "Loader audit",
            "primary_cwd": "/w/proj",
            "agent_profile": "standard",
            "model_id": "deepseek-v4-pro",
            "model_provider": "openrouter",
            "os_name": "darwin",
            "app_version": "0.9.1",
            "schema_version": 7,
            "created_at": "2026-08-31T08:02:51.187943+00:00",
            "updated_at": "2026-08-31T09:31:17.665203+00:00",
        },
        "state": {
            "messages": [
                _icode_msg("user", [{"type": "text", "text": UNICODE_PROMPT}],
                           _icode_group(0, "user", 361, "2026-08-31T08:02:51.187943+00:00")),
                _icode_msg("assistant", [
                    {"type": "text", "text": "Reading first."},
                    {"type": "text_reasoning", "protected_data": "opaque",
                     "additional_properties": {}},
                    {"type": "function_call", "name": "read_file", "call_id": "call_00_read",
                     "arguments": '{"path": "/w/proj/parser.py"}',
                     "additional_properties": {
                         "_chrys_tool_kind": "filesystem.read",
                         "_chrys_timing": {"started_at": "2026-08-31T08:03:00+00:00",
                                           "finished_at": "2026-08-31T08:03:00.500000+00:00"},
                     }},
                    {"type": "function_call", "name": "zsh", "call_id": "call_01_sh",
                     "arguments": '{"command": "pytest -q"}',
                     "additional_properties": {"_chrys_tool_kind": "shell"}},
                ], _icode_group(1, "tool_call", 534, "2026-08-31T08:03:00+00:00")),
                # Chrys inserts turn markers that are not conversation messages.
                _icode_msg("assistant", [], {"_chrys_kind": "turn", "_turn": 1}),
                _icode_msg("tool", [
                    {"type": "function_result", "call_id": "call_00_read",
                     "result": "File: /w/proj/parser.py (440 lines)",
                     "additional_properties": {}},
                    {"type": "function_result", "call_id": "call_01_sh",
                     "result": "", "exception": "CalledProcessError: exit 1",
                     "additional_properties": {"failed": True, "tool_error_kind": "process"}},
                ], _icode_group(1, "tool_call", 968, None)),
                _icode_msg("assistant", [
                    {"type": "function_call", "name": "explore_agent", "call_id": "call_02_explore",
                     "arguments": '{"prompt": "map the loaders"}',
                     "additional_properties": {"_chrys_tool_kind": "sub_agent"}},
                ], _icode_group(2, "tool_call", 210, "2026-08-31T08:09:59+00:00")),
                _icode_msg("tool", [
                    {"type": "function_result", "call_id": "call_02_explore",
                     "result": "explorer finished",
                     "additional_properties": {
                         "_chrys_tool_result_metadata": {"sub_agent_invocation_id": ICODE_CHILD_ID},
                     }},
                ], _icode_group(2, "tool_call", 90, None)),
            ],
            "last_usage": {"system_overhead_tokens": 21567},
            "total_session_tokens": 296664,
            "total_session_input_tokens": 293775,
            "total_session_output_tokens": 2889,
        },
        "_chrys_export": {
            "format": "chrys-expanded-session-v1",
            "producer": {"name": "chrys", "version": "0.9.1"},
            "complete_relative_to_persisted_data": True,
            "compressed_contexts": [],
        },
        "_chrys_sub_agent_sessions": [child],
    }


class IcodeSyntheticExportTests(unittest.TestCase):
    """ICode / Chrys expanded session: pairing, failures, sub-agent flattening."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = _write_json(self.tmp.name, "icode.json", icode_export())
        self.raw = load_trajectory(self.path)
        self.assertNotIn("_error", self.raw, self.raw.get("_error"))
        self.steps = parse_steps(self.raw)
        self.metrics = compute_metrics(self.steps, self.raw)

    def test_detects_as_icode_not_opencode(self):
        """ICode converts into an OpenCode-shaped ``info`` + ``messages`` dict.

        If the second detection pass fell through to OpenCode the UI would
        mislabel the product and format-specific rendering would change.
        """
        self.assertEqual(detect_format(icode_export()), "icode")
        self.assertEqual(detect_format(self.raw), "icode")

    def test_a_raw_session_without_the_export_marker_is_still_icode(self):
        """Users load ``~/.chrys/sessions/<id>/session.json`` directly; that file
        has ``meta`` + ``state.messages`` but no ``_chrys_export`` stamp."""
        payload = icode_export()
        payload.pop("_chrys_export")
        payload.pop("_chrys_sub_agent_sessions")
        self.assertEqual(detect_format(payload), "icode")

    def test_turn_marker_messages_do_not_become_steps(self):
        """``_chrys_kind: turn`` rows are bookkeeping, not assistant turns."""
        self.assertEqual(_roles(self.steps)["user"], 2)  # parent + child prompt
        self.assertEqual(
            len([s for s in self.steps if s["role"] == "assistant" and not s["parts"]]), 0
        )

    def test_results_are_paired_onto_their_call_and_failures_stay_failures(self):
        """Two results arrive in one ``tool`` message; both must find their call."""
        by_id = {tc["tool_id"]: tc for tc in _tool_calls(self.steps)}
        self.assertIn("call_00_read", by_id)
        self.assertIn("call_01_sh", by_id)
        self.assertEqual(by_id["call_00_read"]["status"], "completed")
        self.assertEqual(by_id["call_01_sh"]["status"], "error")
        self.assertGreaterEqual(self.metrics["tool_fail"], 1)

    def test_json_string_arguments_are_parsed_into_tool_input(self):
        """Chrys stores ``arguments`` as a JSON string."""
        by_id = {tc["tool_id"]: tc for tc in _tool_calls(self.steps)}
        self.assertEqual(by_id["call_00_read"]["input"].get("path"), "/w/proj/parser.py")
        self.assertEqual(by_id["call_01_sh"]["input"].get("command"), "pytest -q")

    def test_sub_agent_messages_are_flattened_and_attributed_to_the_child(self):
        """A nested ``_chrys_sub_agent_sessions`` entry must appear as child steps.

        The child's own id may be its invocation id or its session id, but it
        must differ from the parent's and carry the parent link, or delegated
        work gets billed to the parent's own agent lane.
        """
        child = [s for s in self.steps if s["is_sub_agent"]]
        self.assertTrue(child, "nested sub-agent session was dropped")
        self.assertTrue(all(s["parent_session_id"] == ICODE_SESSION_ID for s in child))
        self.assertTrue(all(s["session_id"] and s["session_id"] != ICODE_SESSION_ID for s in child))
        self.assertEqual(len({s["session_id"] for s in child}), 1)
        self.assertEqual(self.raw["metadata"]["sub_agent_count"], 1)
        self.assertIn("c-child-1", {tc["tool_id"] for tc in _tool_calls(self.steps)})

    def test_session_token_totals_come_from_the_session_state(self):
        self.assertEqual(self.raw["token_usage"]["total_tokens"], 296664)
        _assert_token_accounting(self, self.steps, self.metrics)


# --------------------------------------------------------------------------
# Cursor
# --------------------------------------------------------------------------

CURSOR_CHAT_ID = "2ea54dc1-0bf9-44ce-af68-c285319d1343"
CURSOR_CHILD_ID = "3434c363-df66-46bb-9721-ccb20cc71cc7"
CURSOR_T0 = 1786876398974


def _cursor_msg(role, message_id, session_id, parts, *, created, completed=None,
                is_sub=False, depth=0, parent="", tokens=None):
    info = {
        "role": role,
        "id": message_id,
        "sessionID": session_id,
        "agent": "cursor" if depth == 0 else "cursor (subagent)",
        "isSubAgent": is_sub,
        "sessionDepth": depth,
        "time": {"created": created, "completed": completed if completed is not None else created},
    }
    if parent:
        info["parentSessionID"] = parent
    if tokens:
        info["tokens"] = tokens
    return {"info": info, "parts": parts}


def _cursor_tool(name, call_id, session_id, message_id, *, status, input_, output="", error=None,
                 metadata=None):
    state = {"status": status, "input": input_, "output": output, "metadata": metadata or {}}
    if error is not None:
        state["error"] = error
    return {"type": "tool", "tool": name, "callID": call_id, "state": state,
            "id": call_id, "sessionID": session_id, "messageID": message_id}


def cursor_export() -> dict:
    """A realistic ``cursor_consolidator.py`` export with one sub-agent chat."""
    return {
        "info": {
            "id": CURSOR_CHAT_ID,
            "title": "Project overview",
            "directory": "/w/proj",
            "status": "completed",
            "model": "grok-4.6",
            "modelConfig": {"modelName": "grok-4.6", "maxMode": False},
            "time": {"created": CURSOR_T0, "updated": CURSOR_T0 + 17_586_554},
            "summary": {"additions": 31, "deletions": 9, "files": 4},
            "promptTokenBreakdown": {
                "totalUsedTokens": 42000, "maxTokens": 256000,
                "categories": [{"id": "conversation", "estimatedTokens": 40000}],
            },
            "contextUsagePercent": 0.164,
            "contextTokenLimit": 256000,
            "subagentComposerIds": [CURSOR_CHILD_ID],
        },
        "messages": [
            _cursor_msg("user", "bubble-u1", CURSOR_CHAT_ID, [
                {"type": "text", "text": UNICODE_PROMPT, "id": "bubble-u1:text:0",
                 "sessionID": CURSOR_CHAT_ID, "messageID": "bubble-u1"},
            ], created=CURSOR_T0, tokens={"input": 14, "output": 0, "total": 14,
                                          "context_window": 800}),
            _cursor_msg("assistant", "bubble-a1", CURSOR_CHAT_ID, [
                {"type": "text", "text": "Reading, then shelling out.",
                 "id": "bubble-a1:text:0", "sessionID": CURSOR_CHAT_ID, "messageID": "bubble-a1"},
                _cursor_tool("Read", "call-read", CURSOR_CHAT_ID, "bubble-a1",
                             status="completed", input_={"path": "/w/proj/app.py"},
                             output='{"totalLinesInFile": 40}'),
                _cursor_tool("Shell", "call-shell", CURSOR_CHAT_ID, "bubble-a1",
                             status="error", input_={"command": "pytest -q"},
                             output="exit 1", error="command failed"),
                _cursor_tool("StrReplace", "call-edit", CURSOR_CHAT_ID, "bubble-a1",
                             status="unknown", input_={"filePath": "/w/proj/app.py"}),
                _cursor_tool("Task", "call-task", CURSOR_CHAT_ID, "bubble-a1",
                             status="completed", input_={"prompt": "review tests"},
                             output="done",
                             metadata={"sessionId": CURSOR_CHILD_ID,
                                       "parentSessionId": CURSOR_CHAT_ID}),
            ], created=CURSOR_T0 + 2000, completed=CURSOR_T0 + 9000,
                tokens={"input": 900, "output": 220, "total": 1120, "context_window": 21000}),
            _cursor_msg("user", "child-u1", CURSOR_CHILD_ID, [
                {"type": "text", "text": "review tests", "id": "child-u1:text:0",
                 "sessionID": CURSOR_CHILD_ID, "messageID": "child-u1"},
            ], created=CURSOR_T0 + 3000, is_sub=True, depth=1, parent=CURSOR_CHAT_ID,
                tokens={"input": 3, "output": 0, "total": 3}),
            _cursor_msg("assistant", "child-a1", CURSOR_CHILD_ID, [
                _cursor_tool("Grep", "call-grep", CURSOR_CHILD_ID, "child-a1",
                             status="completed", input_={"pattern": "def test_"},
                             output="tests/test_a.py:1"),
            ], created=CURSOR_T0 + 4000, is_sub=True, depth=1, parent=CURSOR_CHAT_ID,
                tokens={"input": 40, "output": 12, "total": 52}),
        ],
        "sessions": [
            {"id": CURSOR_CHAT_ID, "parent_id": None, "depth": 0, "title": "Project overview",
             "transcript_path": "/w/.cursor/a.jsonl", "message_count": 2},
            {"id": CURSOR_CHILD_ID, "parent_id": CURSOR_CHAT_ID, "depth": 1, "title": "review tests",
             "transcript_path": "/w/.cursor/b.jsonl", "message_count": 2},
        ],
        "statistics": {
            "sessions": 2, "subagent_sessions": 1, "total_messages": 4,
            "user_messages": 2, "assistant_messages": 2,
            "tool_parts": 5, "tool_parts_with_output": 4,
        },
        "export_metadata": {
            "schema_version": 1,
            "source_format": "cursor_composer",
            "generator_name": "cursor_consolidator",
            "generated_at": "2026-09-22T12:00:00+00:00",
            "chat_id": CURSOR_CHAT_ID,
            "token_semantics": "context_window_snapshot+estimated_log_tokens",
            "complete": False,
            "warnings": ["Chat x already exported, skipping cycle/diamond"],
        },
    }


class CursorSyntheticExportTests(unittest.TestCase):
    """Cursor composer export: detection precedence, tool vocabulary, token semantics."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = _write_json(self.tmp.name, "cursor.json", cursor_export())
        self.raw = load_trajectory(self.path)
        self.assertNotIn("_error", self.raw, self.raw.get("_error"))
        self.steps = parse_steps(self.raw)
        self.metrics = compute_metrics(self.steps, self.raw)

    def test_detects_as_cursor_not_opencode_or_codearts(self):
        """The consolidator emits an OpenCode-compatible ``info`` + ``messages``
        envelope; only ``export_metadata.source_format`` distinguishes it."""
        self.assertEqual(detect_format(cursor_export()), "cursor")
        self.assertEqual(detect_format(self.raw), "cursor")

    def test_every_message_becomes_a_step_including_the_sub_agent_chat(self):
        self.assertEqual(_roles(self.steps), {"user": 2, "assistant": 2})

    def test_cursor_tool_names_map_onto_the_shared_vocabulary(self):
        """``Shell``/``StrReplace`` must become ``Bash``/``Edit`` or shell parsing,
        write detection and file-interaction charts silently do nothing."""
        names = {tc["tool_id"]: tc["tool_name"] for tc in _tool_calls(self.steps)}
        self.assertEqual(names["call-shell"], "Bash")
        self.assertEqual(names["call-edit"], "Edit")
        self.assertEqual(names["call-read"], "Read")
        by_id = {tc["tool_id"]: tc for tc in _tool_calls(self.steps)}
        self.assertEqual(by_id["call-edit"]["input"].get("file_path"), "/w/proj/app.py")

    def test_an_errored_tool_call_is_counted_as_a_failure(self):
        self.assertEqual(self.metrics["tool_call_count"], 5)
        self.assertGreaterEqual(self.metrics["tool_fail"], 1)
        by_id = {tc["tool_id"]: tc for tc in _tool_calls(self.steps)}
        self.assertEqual(by_id["call-shell"]["status"], "error")

    def test_sub_agent_chats_are_attributed_to_their_own_session(self):
        child = [s for s in self.steps if s["session_id"] == CURSOR_CHILD_ID]
        self.assertEqual(len(child), 2)
        self.assertTrue(all(s["is_sub_agent"] for s in child))
        self.assertTrue(all(s["parent_session_id"] == CURSOR_CHAT_ID for s in child))
        self.assertEqual(self.raw["metadata"]["sub_agent_count"], 1)

    def test_estimated_tokens_are_never_advertised_as_measured_api_usage(self):
        """Cursor does not persist billed tokens; every per-step count is a
        chars/4 estimate of logged text.  The capability flag is what tells a
        consumer not to publish these as API usage."""
        self.assertIs(self.raw["_capabilities"]["has_runtime_token_usage"], False)
        self.assertEqual(
            self.raw["metadata"]["token_semantics"],
            "context_window_snapshot+estimated_log_tokens",
        )

    def test_epoch_millisecond_timestamps_are_read_as_milliseconds(self):
        self.assertTrue(self.raw["timing"]["started_at"].startswith("2026-"))
        self.assertAlmostEqual(self.raw["timing"]["total_duration"], 17586.554, places=3)

    def test_token_accounting_is_coherent(self):
        _assert_token_accounting(self, self.steps, self.metrics)


# --------------------------------------------------------------------------
# Cross-format detection precedence
# --------------------------------------------------------------------------

def opencode_export() -> dict:
    """A minimal but realistic OpenCode export (``info`` + ``messages``)."""
    return {
        "info": {"id": "ses_abc", "slug": "s", "directory": "/w/proj",
                 "time": {"created": 1786876398974, "updated": 1786876498974}},
        "messages": [
            {"info": {"role": "user", "id": "m1", "sessionID": "ses_abc",
                      "time": {"created": 1786876398974}},
             "parts": [{"type": "text", "text": "hi"}]},
            {"info": {"role": "assistant", "id": "m2", "sessionID": "ses_abc",
                      "modelID": "glm-5", "providerID": "z",
                      "time": {"created": 1786876399974, "completed": 1786876400974},
                      "tokens": {"input": 10, "output": 5, "reasoning": 0,
                                 "cache": {"read": 0, "write": 0}, "total": 15}},
             "parts": [{"type": "text", "text": "ok"}]},
        ],
    }


def claude_code_export() -> dict:
    return {
        "format": "ccsession-trajectory",
        "metadata": {"session_id": "cc-1", "cwd": "/w/proj"},
        "trajectory": [
            {"role": "user", "content": "hi", "timestamp": "2026-08-01T00:00:00Z"},
        ],
    }


def codearts_export() -> dict:
    payload = opencode_export()
    payload["export_metadata"] = {
        "schema_version": 2,
        "source_format": "codearts_opencode_sqlite",
        "generator_name": "codearts_consolidator",
    }
    return payload


def codex_events() -> list[dict]:
    return [
        {"type": "session_meta", "timestamp": "2026-08-01T00:00:00.000Z",
         "payload": {"id": "cx-1", "cwd": "/w/proj", "originator": "Codex CLI"}},
        {"type": "response_item", "timestamp": "2026-08-01T00:00:01.000Z",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]}},
    ]


class FormatPrecedenceTests(unittest.TestCase):
    """The four new sniffers must never claim a file belonging to an older format."""

    def test_older_formats_are_still_detected_as_themselves(self):
        """A silent mis-claim has no error path: every downstream number would
        be computed by the wrong converter."""
        self.assertEqual(detect_format(opencode_export()), "opencode")
        self.assertEqual(detect_format(claude_code_export()), "ccsession")
        self.assertEqual(detect_format(codearts_export()), "codearts")
        self.assertEqual(detect_format(codex_events()), "codex")

    def test_each_synthesized_export_detects_as_its_own_format(self):
        self.assertEqual(detect_format(cursor_export()), "cursor")
        self.assertEqual(detect_format(icode_export()), "icode")
        self.assertEqual(detect_format(dsh_parent_events()), "dsh")
        self.assertEqual(detect_format(pi_events()), "pi")

    def test_a_codex_log_is_not_claimed_by_the_dsh_or_pi_sniffers(self):
        """Codex leads with ``session_meta``; DSH/Pi lead with ``session``."""
        self.assertEqual(detect_format(codex_events()), "codex")

    def test_selecting_the_wrong_format_is_reported_as_a_mismatch(self):
        """The dropdown gate is the only guard against a forced mis-parse."""
        cases = [
            (cursor_export(), "cursor"),
            (icode_export(), "icode"),
            (dsh_parent_events(), "dsh"),
            (pi_events(), "pi"),
            (opencode_export(), "opencode"),
        ]
        for payload, own in cases:
            detected = detect_format(payload)
            self.assertEqual(detected, own)
            self.assertIsNone(check_format_selection(detected, own))
            for other in FORMAT_LABELS:
                if other == own:
                    continue
                self.assertEqual(
                    check_format_selection(detected, other), "mismatch",
                    f"selecting {other} for a {own} file was not rejected",
                )

    def test_format_labels_cover_every_detectable_format(self):
        """``FORMAT_LABELS`` is the single source of display names; a format
        missing from it renders as a bare key and breaks the mismatch gate."""
        self.assertEqual(set(FORMAT_LABELS), _OBJECT_FORMATS | _EVENT_FORMATS)
        for payload in (cursor_export(), icode_export(), opencode_export(),
                        claude_code_export(), codearts_export()):
            self.assertIn(detect_format(payload), FORMAT_LABELS)
        for events in (dsh_parent_events(), pi_events(), codex_events()):
            self.assertIn(detect_format(events), FORMAT_LABELS)
        for label in FORMAT_LABELS.values():
            self.assertTrue(label and label.strip(), "empty format label")


if __name__ == "__main__":
    unittest.main()
