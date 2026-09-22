"""Delegation wall-clock is reported, and absent timing reads n/a — never 0.

Excluding spawn/delegation calls from tool-execution time is correct (the
child's own steps carry its work), but Claude Code stamps timing ONLY on the
spawn call. Dropping it there left every tool-timing metric at a hard 0 on
~21% of the real claude_code corpus — up to 1,862s (41% of wall-clock)
reported as "0s", including in the brief handed to the LLM analyst.
"""

from __future__ import annotations

import unittest

from trajviz.insight.assistant import build_analysis_brief
from trajviz.insight.formatting import format_behavioral_md
from trajviz.insight.metrics import build_message_metrics, compute_metrics


def _assistant_step(index, *, duration, tool_calls=None, is_sub_agent=False):
    calls = tool_calls or []
    step = {
        "index": index,
        "role": "assistant",
        "duration": duration,
        "parts": [],
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "tokens": {"total": 1, "input": 0, "output": 1, "reasoning": 0,
                   "cache_read": 0, "cache_write": 0},
        "finish": "stop",
        "time_created_ms": 0,
        "time_completed_ms": int(duration * 1000),
    }
    if is_sub_agent:
        step["is_sub_agent"] = True
    return step


def _metrics(steps):
    return compute_metrics(steps, {}, build_message_metrics(steps))


class DelegationOnlyTimingTests(unittest.TestCase):
    """A trajectory whose only timed call is the delegation (the CC shape)."""

    def setUp(self):
        self.steps = [
            _assistant_step(
                0,
                duration=100.0,
                tool_calls=[
                    {"tool_name": "Agent", "status": "success",
                     "metadata": {"totalDurationMs": 45_379}},
                    # Claude Code records no timing for ordinary tool calls.
                    {"tool_name": "Bash", "status": "success"},
                ],
            )
        ]
        self.metrics = _metrics(self.steps)

    def test_tool_timing_is_none_not_zero(self):
        for key in ("tool_time_total", "tool_wait_share", "avg_tool_duration",
                    "p95_tool_duration", "max_tool_duration",
                    "tool_time_fraction"):
            self.assertIsNone(self.metrics[key], key)

    def test_delegated_wall_clock_is_preserved(self):
        self.assertAlmostEqual(self.metrics["delegation_time_total"], 45.379, places=2)
        self.assertEqual(self.metrics["delegated_call_count"], 1)

    def test_panel_says_no_per_tool_timing_not_wrong_format(self):
        md = format_behavioral_md(self.metrics)
        # Scope the assertion to the Tool timing chip: the phrase legitimately
        # appears on the cache chip, which this fixture also lacks data for.
        chip = md[md.index("Tool timing"):][:400]
        self.assertIn("no per-tool timing in this export", chip)
        self.assertNotIn("not available for this format", chip)
        self.assertIn("Delegated", md)
        self.assertIn("45.38", md)

    def test_llm_brief_reports_na_rather_than_zero_seconds(self):
        brief = build_analysis_brief(self.steps, {})
        self.assertIn("p95_tool_duration: n/a", brief)
        self.assertIn("max_tool_duration: n/a", brief)
        self.assertIn("avg_tool_duration: n/a", brief)
        self.assertIn("tool_wait_pct: n/a", brief)
        self.assertNotIn("p95_tool_duration: 0s", brief)
        self.assertNotIn("tool_wait_pct: 0%", brief)


class MeasuredToolTimingUnaffectedTests(unittest.TestCase):
    """The spawn exclusion itself must stay intact where children are timed."""

    def test_spawn_excluded_but_real_tool_time_still_reported(self):
        parent = _assistant_step(
            0,
            duration=100.0,
            tool_calls=[
                {"tool_name": "task", "status": "success", "duration_ms": 80_000},
                {"tool_name": "bash", "status": "success", "duration_ms": 5_000},
            ],
        )
        child = _assistant_step(
            1,
            duration=80.0,
            is_sub_agent=True,
            tool_calls=[{"tool_name": "read", "status": "success", "duration_ms": 10_000}],
        )
        m = _metrics([parent, child])
        # Unchanged from the PR's intended behaviour: 5s + 10s, not 85s.
        self.assertEqual(m["tool_time_total"], 15.0)
        self.assertEqual(m["max_tool_duration"], 10.0)
        # ...and the delegation is now visible instead of silently discarded.
        self.assertEqual(m["delegation_time_total"], 80.0)

    def test_no_tool_calls_at_all_still_reads_not_available(self):
        m = _metrics([_assistant_step(0, duration=10.0)])
        self.assertIsNone(m["tool_time_total"])
        self.assertIsNone(m["delegation_time_total"])
        self.assertIn("not available for this format", format_behavioral_md(m))


if __name__ == "__main__":
    unittest.main()
