"""Parallel tool calls must not be double-counted, and a collapsed generation
window must not contribute output tokens against zero seconds.

Agents issue tool calls in parallel inside one step. Summing per-call
durations over-counts the overlap, which both inflates Tool time and drives
the output tok/s denominator to zero -- at which point the step's tokens are
credited to no time at all and the headline rate runs away.
"""

from __future__ import annotations

import unittest

from trajviz.insight.metrics import (
    build_message_metrics,
    compute_metrics,
    generation_seconds,
    non_spawn_tool_seconds,
)


def _step(index, *, duration, tool_calls=None, output_tokens=100):
    calls = tool_calls or []
    return {
        "index": index,
        "role": "assistant",
        "duration": duration,
        "parts": [],
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "finish": "stop",
        "tokens": {"total": output_tokens, "input": 0, "output": output_tokens,
                   "reasoning": 0, "cache_read": 0, "cache_write": 0},
        "time_created_ms": 0,
        "time_completed_ms": int(duration * 1000),
    }


def _call(name, *, start_ms=None, end_ms=None, duration_ms=None):
    tc = {"tool_name": name, "status": "success"}
    if start_ms is not None:
        tc["time_start"] = start_ms
        tc["time_end"] = end_ms
    if duration_ms is not None:
        tc["duration_ms"] = duration_ms
    return tc


class ParallelToolWindowTests(unittest.TestCase):
    def test_overlapping_calls_count_their_union_not_their_sum(self):
        # Three calls issued together, each 8s, overlapping into one 10s window.
        step = _step(0, duration=20.0, tool_calls=[
            _call("bash", start_ms=0, end_ms=8_000),
            _call("bash", start_ms=1_000, end_ms=9_000),
            _call("bash", start_ms=2_000, end_ms=10_000),
        ])
        # Sum would be 24s -- more than the union and more than a 20s step.
        self.assertAlmostEqual(non_spawn_tool_seconds(step), 10.0, places=3)

    def test_disjoint_calls_still_add_up(self):
        step = _step(0, duration=20.0, tool_calls=[
            _call("bash", start_ms=0, end_ms=3_000),
            _call("bash", start_ms=10_000, end_ms=14_000),
        ])
        self.assertAlmostEqual(non_spawn_tool_seconds(step), 7.0, places=3)

    def test_touching_windows_are_one_run(self):
        step = _step(0, duration=20.0, tool_calls=[
            _call("bash", start_ms=0, end_ms=5_000),
            _call("bash", start_ms=5_000, end_ms=9_000),
        ])
        self.assertAlmostEqual(non_spawn_tool_seconds(step), 9.0, places=3)

    def test_unstamped_durations_are_summed_as_before(self):
        step = _step(0, duration=20.0, tool_calls=[
            _call("bash", duration_ms=4_000),
            _call("bash", duration_ms=6_000),
        ])
        self.assertAlmostEqual(non_spawn_tool_seconds(step), 10.0, places=3)

    def test_generation_window_reflects_the_union(self):
        step = _step(0, duration=20.0, tool_calls=[
            _call("bash", start_ms=0, end_ms=8_000),
            _call("bash", start_ms=1_000, end_ms=9_000),
        ])
        gen_s, tool_wait_s = generation_seconds(step)
        self.assertAlmostEqual(tool_wait_s, 9.0, places=3)
        self.assertAlmostEqual(gen_s, 11.0, places=3)


class CollapsedGenerationWindowTests(unittest.TestCase):
    def test_step_with_no_generation_window_is_left_out_of_the_rate(self):
        # The tool stamps span the entire step: there is no measurable
        # generation time, so crediting 100 output tokens to it is meaningless.
        step = _step(0, duration=10.0, output_tokens=100,
                     tool_calls=[_call("bash", start_ms=0, end_ms=10_000)])
        m = compute_metrics([step], {}, build_message_metrics([step]))
        self.assertEqual(m["output_throughput_timed_steps"], 0)
        self.assertIsNone(m["output_tokens_per_sec"])
        # The omission is disclosed rather than silent.
        self.assertTrue(m["output_throughput_incomplete"])
        self.assertEqual(m["output_throughput_total_steps"], 1)

    def test_normal_step_still_produces_a_rate(self):
        step = _step(0, duration=10.0, output_tokens=100,
                     tool_calls=[_call("bash", start_ms=0, end_ms=5_000)])
        m = compute_metrics([step], {}, build_message_metrics([step]))
        self.assertEqual(m["output_throughput_timed_steps"], 1)
        self.assertEqual(m["output_tokens_per_sec"], 20.0)


if __name__ == "__main__":
    unittest.main()
