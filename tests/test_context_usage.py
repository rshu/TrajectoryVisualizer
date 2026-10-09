"""Context-window composition: bucket attribution and Diagnostics HTML."""

import json
import unittest

from trajviz.insight.context_usage import (
    PRESSURE_ALL_AGENTS,
    PRESSURE_MAIN_AGENT,
    SNAPSHOT_CURRENT,
    context_usage_breakdown,
    estimate_tokens,
    format_context_usage_html,
    format_token_count,
    parse_usage_snapshot,
    usage_snapshot_choices,
)
from trajviz.insight.formatting import format_context_pressure_html
from trajviz.tool_vocab import SPAWN_TOOL_NAMES


def _tokens(*, inp, out=0, reasoning=0, cache_read=0):
    total = inp + out + reasoning + cache_read
    return {
        "total": total,
        "input": inp,
        "output": out,
        "reasoning": reasoning,
        "cache_read": cache_read,
    }


def _chars(n_tokens: int) -> str:
    """String whose ~4-chars/token estimate equals ``n_tokens``."""
    return "x" * (n_tokens * 4)


def _step(
    idx,
    *,
    role="assistant",
    agent="",
    session_id="",
    tokens=None,
    parts=None,
    tool_calls=None,
    model_id="",
    is_sub_agent=False,
    summary=False,
):
    return {
        "index": idx,
        "role": role,
        "agent": agent,
        "is_sub_agent": is_sub_agent,
        "session_id": session_id,
        "tokens": tokens or _tokens(inp=0),
        "parts": parts or [],
        "tool_calls": tool_calls or [],
        "model_id": model_id,
        "session_title": "",
        "summary": summary,
        "is_compaction_checkpoint": False,
        "message_type": "",
    }


class EstimateTests(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_four_chars_per_token(self):
        self.assertEqual(estimate_tokens(_chars(10)), 10)

    def test_compact_counts(self):
        self.assertEqual(format_token_count(76800), "76.8k")
        self.assertEqual(format_token_count(200_000), "200k")


class BucketTests(unittest.TestCase):
    def test_synthetic_text_is_system(self):
        steps = [
            _step(0, role="user", parts=[
                {"type": "text", "text": _chars(20), "synthetic": True},
            ]),
            _step(1, tokens=_tokens(inp=20), parts=[]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertEqual(buckets["system"], 20)
        self.assertEqual(buckets["conversation"], 0)

    def test_system_role_is_system(self):
        steps = [
            _step(0, role="system", parts=[
                {"type": "text", "text": _chars(12)},
            ]),
            _step(1, tokens=_tokens(inp=12), parts=[]),
        ]
        self.assertEqual(context_usage_breakdown(steps)["buckets"]["system"], 12)

    def test_user_and_assistant_text_are_conversation(self):
        steps = [
            _step(0, role="user", parts=[
                {"type": "text", "text": _chars(8)},
            ]),
            _step(1, tokens=_tokens(inp=18), parts=[
                {"type": "text", "text": _chars(6)},
                {"type": "reasoning", "text": _chars(4)},
            ]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertEqual(buckets["conversation"], 18)
        self.assertEqual(buckets["system"], 0)

    def test_regular_tool_output_bucket(self):
        steps = [
            _step(1, tokens=_tokens(inp=30), tool_calls=[{
                "tool_name": "Read",
                "input": {"file_path": "a.py"},
                "output": _chars(24),
            }]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertEqual(buckets["tool_outputs"], 24)
        self.assertGreater(buckets["conversation"], 0)

    def test_skill_call_is_skills(self):
        steps = [
            _step(1, tokens=_tokens(inp=40), tool_calls=[{
                "tool_name": "Skill",
                "input": {"skill": "my-skill", "args": _chars(10)},
                "output": _chars(15),
            }]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertGreater(buckets["skills"], 0)
        self.assertEqual(buckets["tool_outputs"], 0)
        self.assertEqual(buckets["conversation"], 0)

    def test_mcp_call_claude_code_format(self):
        steps = [
            _step(1, tokens=_tokens(inp=200), tool_calls=[{
                "tool_name": "mcp__github__list_issues",
                "input": {"repo": "foo/bar"},
                "output": _chars(30),
            }]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertGreater(buckets["mcp"], 0)
        self.assertEqual(buckets["conversation"], 0)
        self.assertEqual(buckets["tool_outputs"], 0)

    def test_mcp_call_opencode_format(self):
        steps = [
            _step(1, tokens=_tokens(inp=200), tool_calls=[{
                "tool_name": "kernel-test-runner_run_test",
                "input": {"test_file": "test.py"},
                "output": _chars(50),
            }]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertGreater(buckets["mcp"], 0)
        self.assertEqual(buckets["conversation"], 0)
        self.assertEqual(buckets["tool_outputs"], 0)

    def test_non_mcp_tool_still_conversation(self):
        steps = [
            _step(1, tokens=_tokens(inp=200), tool_calls=[{
                "tool_name": "bash",
                "input": {"command": "ls -la"},
                "output": _chars(20),
            }]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertEqual(buckets["mcp"], 0)
        self.assertGreater(buckets["conversation"], 0)
        self.assertEqual(buckets["tool_outputs"], 20)

    def test_spawn_input_is_subagents(self):
        self.assertIn("Task", SPAWN_TOOL_NAMES)
        prompt = _chars(30)
        payload = {"description": "explore", "prompt": prompt}
        steps = [
            _step(1, tokens=_tokens(inp=200), tool_calls=[{
                "tool_name": "Task",
                "input": payload,
                "output": _chars(5),
            }]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertEqual(
            buckets["subagents"],
            estimate_tokens(json.dumps(payload, ensure_ascii=False)),
        )
        self.assertEqual(buckets["tool_outputs"], 5)
        self.assertEqual(buckets["conversation"], 0)

    def test_spawn_via_subagent_type(self):
        steps = [
            _step(1, tokens=_tokens(inp=20), tool_calls=[{
                "tool_name": "custom_spawn",
                "input": {"subagent_type": "explore", "prompt": _chars(16)},
                "output": "",
            }]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertGreater(buckets["subagents"], 0)
        self.assertEqual(buckets["conversation"], 0)

    def test_rules_and_mcp_from_raw(self):
        steps = [_step(1, tokens=_tokens(inp=500), parts=[
            {"type": "text", "text": _chars(10)},
        ])]
        buckets = context_usage_breakdown(
            steps,
            raw={
                "metadata": {
                    "rules": "Always use types. " + _chars(20),
                    "mcpServers": [{"name": "github", "tools": _chars(30)}],
                },
            },
        )["buckets"]
        self.assertGreater(buckets["rules"], 0)
        self.assertGreater(buckets["mcp"], 0)

    def test_summary_turn_is_summarized_conversation(self):
        steps = [
            _step(1, tokens=_tokens(inp=80), summary=True, parts=[
                {"type": "text", "text": _chars(25)},
            ]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        self.assertEqual(buckets["summarized"], 25)
        self.assertEqual(buckets["conversation"], 0)

    def test_tool_schemas_on_raw(self):
        schema = [{"name": "Read", "description": _chars(40)}]
        steps = [_step(1, tokens=_tokens(inp=100), parts=[
            {"type": "text", "text": _chars(10)},
        ])]
        buckets = context_usage_breakdown(
            steps, raw={"metadata": {"tools": schema}},
        )["buckets"]
        self.assertEqual(buckets["tools"], estimate_tokens(
            json.dumps(schema, ensure_ascii=False)
        ))

    def test_unattributed_when_occupancy_exceeds_estimate(self):
        steps = [
            _step(1, tokens=_tokens(inp=10_000), parts=[
                {"type": "text", "text": _chars(10)},
            ]),
        ]
        result = context_usage_breakdown(steps)
        self.assertEqual(result["occupancy"], 10_000)
        self.assertEqual(result["buckets"]["conversation"], 10)
        self.assertEqual(result["buckets"]["unattributed"], 9_990)

    def test_residual_row_is_not_labeled_unattributed(self):
        steps = [
            _step(1, tokens=_tokens(inp=10_000), parts=[
                {"type": "text", "text": _chars(10)},
            ]),
        ]
        result = context_usage_breakdown(steps)
        self.assertEqual(result["buckets"]["unattributed"], 9_990)
        html = format_context_usage_html(result)
        self.assertIn("Harness system definitions (not included in log)", html)
        self.assertNotIn("Unattributed", html)
        self.assertNotIn("System + tools", html)
        self.assertNotIn("Estimated from logged text", html)
        self.assertNotIn("harness overhead", html)

    def test_scale_when_estimate_exceeds_occupancy(self):
        steps = [
            _step(1, tokens=_tokens(inp=10), parts=[
                {"type": "text", "text": _chars(40)},
            ]),
        ]
        result = context_usage_breakdown(steps)
        self.assertTrue(result["scaled"])
        accounted = sum(
            v for k, v in result["buckets"].items() if k != "unattributed"
        )
        self.assertEqual(accounted, 10)
        self.assertEqual(result["buckets"]["unattributed"], 0)


class WindowAndCompactionTests(unittest.TestCase):
    def test_window_percentages(self):
        steps = [
            _step(1, tokens=_tokens(inp=20_000), model_id="claude-sonnet-4",
                  parts=[{"type": "text", "text": _chars(20_000)}]),
        ]
        result = context_usage_breakdown(steps)
        self.assertEqual(result["window_limit"], 200_000)
        self.assertEqual(result["loaded_pct"], 10.0)
        from trajviz.insight.context_usage import usage_segments
        conv = next(r for r in usage_segments(result) if r["key"] == "conversation")
        self.assertEqual(conv["window_pct"], 10.0)
        self.assertEqual(conv["loaded_pct"], 100.0)

    def test_override_window_limit(self):
        steps = [
            _step(1, tokens=_tokens(inp=20_000), model_id="claude-sonnet-4",
                  parts=[{"type": "text", "text": _chars(20_000)}]),
        ]
        result = context_usage_breakdown(steps, window_limit=256_000)
        self.assertEqual(result["window_limit"], 256_000)
        self.assertEqual(result["loaded_pct"], 7.8)

    def test_unknown_model_defaults_to_128k(self):
        steps = [
            _step(1, tokens=_tokens(inp=12_800), model_id="mystery-model",
                  parts=[{"type": "text", "text": _chars(12_800)}]),
        ]
        result = context_usage_breakdown(steps)
        self.assertEqual(result["window_limit"], 128_000)
        self.assertEqual(result["loaded_pct"], 10.0)

    def test_compaction_drops_pre_compaction_text(self):
        steps = [
            _step(0, tokens=_tokens(inp=4_000), parts=[
                {"type": "text", "text": "BEFORE_COMPACTION " + _chars(50)},
            ]),
            _step(1, role="user", parts=[
                {"type": "compaction", "summary": "kept notes"},
            ]),
            _step(2, tokens=_tokens(inp=8_000), parts=[
                {"type": "text", "text": "AFTER_PEAK " + _chars(20)},
            ]),
        ]
        buckets = context_usage_breakdown(steps)["buckets"]
        after_tokens = estimate_tokens("AFTER_PEAK " + _chars(20))
        self.assertEqual(buckets["conversation"], after_tokens)
        self.assertEqual(buckets["summarized"], estimate_tokens("kept notes"))

    def test_all_agents_uses_highest_peak(self):
        steps = [
            _step(0, tokens=_tokens(inp=1_000),
                  parts=[{"type": "text", "text": _chars(5)}]),
            _step(1, session_id="child", agent="explore", is_sub_agent=True,
                  tokens=_tokens(inp=9_000),
                  parts=[{"type": "text", "text": _chars(40)}]),
        ]
        result = context_usage_breakdown(steps, agent_key=PRESSURE_ALL_AGENTS)
        self.assertEqual(result["agent_id"], "child")
        self.assertEqual(result["occupancy"], 9_000)
        self.assertEqual(result["buckets"]["conversation"], 40)

        main = context_usage_breakdown(steps, agent_key=PRESSURE_MAIN_AGENT)
        self.assertEqual(main["occupancy"], 1_000)
        self.assertEqual(main["buckets"]["conversation"], 5)

    def test_opencode_compaction_summary_on_current_window(self):
        """Peak is pre-compaction; the stored summary is the following turn."""
        sid = "ses_explore"
        steps = [
            _step(0, session_id=sid, is_sub_agent=True, agent="explore",
                  tokens=_tokens(inp=50_000),
                  parts=[{"type": "text", "text": "BEFORE " + _chars(40)}]),
            _step(1, role="user", session_id=sid, is_sub_agent=True, agent="explore",
                  parts=[{"type": "compaction", "summary": ""}]),
            _step(2, session_id=sid, is_sub_agent=True, agent="compaction",
                  tokens=_tokens(inp=8_000), summary=True,
                  parts=[{"type": "text", "text": "OBJECTIVE " + _chars(30)}]),
            _step(3, session_id=sid, is_sub_agent=True, agent="explore",
                  tokens=_tokens(inp=9_000),
                  parts=[{"type": "text", "text": "AFTER " + _chars(10)}]),
        ]
        steps[2]["mode"] = "compaction"
        result = context_usage_breakdown(steps)
        self.assertEqual(result["occupancy"], 9_000)
        self.assertEqual(result["peak_occupancy"], 50_000)
        self.assertGreater(result["buckets"]["summarized"], 0)
        self.assertEqual(
            result["buckets"]["summarized"],
            estimate_tokens("OBJECTIVE " + _chars(30)),
        )
        after_tokens = estimate_tokens("AFTER " + _chars(10))
        self.assertEqual(result["buckets"]["conversation"], after_tokens)
        html = format_context_usage_html(result)
        self.assertIn("Summarized conversation", html)
        self.assertNotIn("Estimated from logged text", html)
        self.assertNotIn("harness overhead", html)
        self.assertNotIn("current window after compaction", html)

    def test_pre_compaction_snapshot_keeps_before_text(self):
        sid = "ses_explore"
        steps = [
            _step(0, session_id=sid, is_sub_agent=True, agent="explore",
                  tokens=_tokens(inp=50_000),
                  parts=[{"type": "text", "text": "BEFORE " + _chars(40)}]),
            _step(1, role="user", session_id=sid, is_sub_agent=True, agent="explore",
                  parts=[{"type": "compaction", "summary": ""}]),
            _step(2, session_id=sid, is_sub_agent=True, agent="compaction",
                  tokens=_tokens(inp=8_000), summary=True,
                  parts=[{"type": "text", "text": "OBJECTIVE " + _chars(30)}]),
            _step(3, session_id=sid, is_sub_agent=True, agent="explore",
                  tokens=_tokens(inp=9_000),
                  parts=[{"type": "text", "text": "AFTER " + _chars(10)}]),
        ]
        steps[2]["mode"] = "compaction"
        result = context_usage_breakdown(steps, snapshot_step=0)
        self.assertEqual(result["occupancy"], 50_000)
        self.assertEqual(result["step"], 0)
        self.assertEqual(
            result["buckets"]["conversation"],
            estimate_tokens("BEFORE " + _chars(40)),
        )
        self.assertEqual(result["buckets"]["summarized"], 0)
        self.assertNotIn("AFTER", format_context_usage_html(result))
        self.assertEqual(
            usage_snapshot_choices(steps, agent_key=PRESSURE_ALL_AGENTS),
            [("Current window", SNAPSHOT_CURRENT)],
        )
        choices = usage_snapshot_choices(steps, agent_key=sid)
        self.assertEqual(choices[0], ("Current window", SNAPSHOT_CURRENT))
        self.assertIn("0", [value for _label, value in choices])
        self.assertIsNone(parse_usage_snapshot(SNAPSHOT_CURRENT))
        self.assertEqual(parse_usage_snapshot("0"), 0)


class HtmlTests(unittest.TestCase):
    def test_html_omits_empty_categories(self):
        steps = [
            _step(1, tokens=_tokens(inp=50_000), model_id="claude-opus-4",
                  parts=[{"type": "text", "text": _chars(100)}]),
        ]
        html = format_context_pressure_html(
            {"points": [], "events": [], "window_limit": 200_000,
             "agents": [], "peak_occupancy": 50_000, "peak_pct": 25.0},
            steps=steps,
            agent_key=PRESSURE_ALL_AGENTS,
        )
        self.assertIn("Conversations", html)
        self.assertIn("Harness system definitions (not included in log)", html)
        self.assertNotIn("Unattributed", html)
        self.assertIn("% window", html)
        self.assertIn("% loaded", html)
        self.assertIn("% full", html)
        self.assertIn("ctx-usage-bar", html)
        for label in (
            "System prompt",
            "Skills",
            "Tool definitions",
            "Rules",
            "MCP",
            "Subagent definitions",
            "Summarized conversation",
            "Tool outputs",
        ):
            self.assertNotIn(label, html)

    def test_html_shows_subagents_and_summarized_when_present(self):
        steps = [
            _step(0, role="user", parts=[
                {"type": "compaction", "summary": "earlier work " + _chars(12)},
            ]),
            _step(1, tokens=_tokens(inp=400), tool_calls=[{
                "tool_name": "Task",
                "input": {"prompt": _chars(40)},
                "output": "",
            }]),
        ]
        html = format_context_usage_html(context_usage_breakdown(steps))
        self.assertIn("Subagent definitions", html)
        self.assertIn("Summarized conversation", html)
        self.assertNotIn("Skills", html)
        self.assertNotIn("System prompt", html)

    def test_html_hides_unattributed_when_zero(self):
        steps = [
            _step(1, tokens=_tokens(inp=20), parts=[
                {"type": "text", "text": _chars(20)},
            ]),
        ]
        html = format_context_usage_html(context_usage_breakdown(steps))
        self.assertNotIn("Unattributed", html)
        self.assertNotIn("not included in log", html)
        self.assertNotIn("Harness system definitions", html)
        self.assertIn("Conversations", html)
        self.assertNotIn("Skills", html)
        self.assertIn("% window", html)
        self.assertNotIn("window limit unknown", html)

    def test_unknown_window_uses_128k_column(self):
        steps = [
            _step(1, tokens=_tokens(inp=80), model_id="mystery-model",
                  parts=[{"type": "text", "text": _chars(20)}]),
        ]
        html = format_context_usage_html(context_usage_breakdown(steps))
        self.assertIn("% window", html)
        self.assertIn("% loaded", html)
        self.assertIn("128k", html)
