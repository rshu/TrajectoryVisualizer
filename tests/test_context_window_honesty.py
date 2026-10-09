"""A pressure percentage must never be computed against a window the data disproves.

When an export neither declares its context window nor names a model in the
window table, TrajViz assumes 128,000 tokens. GLM-5 and GPT-5.4 runs, which the
table does not know, fell back to that default even when they contain a single
request larger than 128,000 tokens that the provider accepted. The default was physically
impossible for them, and the dashboard still reported a peak pressure above
100% as if it were a measurement.

The contract pinned here:

* a window that a single request exceeds is UNKNOWN when it was assumed
  (the default) or looked up (the model table) — no percentage is shown and a
  note says why; a window the export declares, or one the user typed, is kept;
* a percentage computed against the default is disclosed as assumed;
* the editable window field starts empty unless the window was detected, so the
  load-time ``.change`` round trip cannot turn the default into a "user
  override" and bring the impossible percentage back;
* "premature compaction" is not judged against an unknown window.
"""

from __future__ import annotations

import unittest

from trajviz.insight.context_usage import (
    DEFAULT_CONTEXT_WINDOW_LIMIT,
    context_pressure_series,
    context_pressure_stats,
    detect_premature_compactions,
    resolve_context_window,
)
from trajviz.insight.formatting import format_context_pressure_html


def _step(i: int, occupancy: int, *, model: str = "glm-5", agent: str = "") -> dict:
    return {
        "index": i, "role": "assistant", "agent": agent, "model_id": model,
        "tokens": {"total": occupancy + 100, "input": occupancy, "output": 100,
                   "cache_read": 0, "cache_write": 0},
        "tool_calls": [], "parts": [],
    }


class ResolveWindowTests(unittest.TestCase):
    def test_an_assumed_window_that_a_request_exceeds_is_unknown(self):
        steps = [_step(0, 40_000), _step(1, 175_000)]
        limit, source, note = resolve_context_window(steps)
        self.assertIsNone(limit)
        self.assertEqual(source, "unknown")
        self.assertIn("175,000", note)
        self.assertIn(f"{DEFAULT_CONTEXT_WINDOW_LIMIT:,}", note)

    def test_an_assumed_window_that_holds_is_reported_as_the_default(self):
        limit, source, note = resolve_context_window([_step(0, 50_000)])
        self.assertEqual((limit, source, note), (DEFAULT_CONTEXT_WINDOW_LIMIT, "default", ""))

    def test_a_model_table_window_that_a_request_exceeds_is_unknown(self):
        limit, source, _ = resolve_context_window([_step(0, 250_000, model="claude-opus-4-6")])
        self.assertIsNone(limit)
        self.assertEqual(source, "unknown")

    def test_a_declared_window_is_kept_even_when_exceeded(self):
        # The export states its window; a peak above it is an accounting defect
        # to fix where the occupancy is computed, not a reason to drop the window.
        raw = {"metadata": {"context_window_limit": 258_400}}
        limit, source, _ = resolve_context_window([_step(0, 300_000)], raw)
        self.assertEqual((limit, source), (258_400, "declared"))

    def test_a_user_override_always_wins(self):
        limit, source, _ = resolve_context_window([_step(0, 175_000)], override=128_000)
        self.assertEqual((limit, source), (128_000, "override"))

    def test_exceedance_is_judged_across_all_agents_not_the_selected_one(self):
        steps = [_step(0, 10_000, agent=""), _step(1, 175_000, agent="child")]
        series = context_pressure_series(steps, agent_key="__main__")
        self.assertIsNone(series["window_limit"])


class PresentationTests(unittest.TestCase):
    def test_unknown_window_shows_no_percentage_and_says_why(self):
        series = context_pressure_series([_step(0, 175_000)])
        stats = context_pressure_stats(series)
        self.assertIsNone(stats["peak_pct"])
        self.assertEqual(stats["window_source"], "unknown")
        html_out = format_context_pressure_html(series)
        self.assertNotIn("136", html_out)
        self.assertIn("window unknown", html_out.lower())

    def test_percentage_against_the_default_is_labelled_assumed(self):
        series = context_pressure_series([_step(0, 64_000)])
        stats = context_pressure_stats(series)
        self.assertEqual(stats["peak_pct"], 50.0)
        self.assertIn("assumed", format_context_pressure_html(series).lower())

    def test_detected_window_is_not_labelled_assumed(self):
        series = context_pressure_series([_step(0, 64_000, model="claude-opus-4-6")])
        self.assertNotIn("assumed", format_context_pressure_html(series).lower())


class PrematureCompactionTests(unittest.TestCase):
    def test_low_occupancy_is_not_judged_against_an_unknown_window(self):
        steps = [
            _step(0, 175_000),
            {"index": 1, "role": "compaction", "is_compaction_checkpoint": True, "agent": "",
             "message_type": "compaction", "tokens": {}, "tool_calls": [], "parts": []},
            _step(2, 5_000),
        ]
        for flag in detect_premature_compactions(steps):
            self.assertTrue(flag["grew"], flag)
            self.assertIsNone(flag["before_pct"])


if __name__ == "__main__":
    unittest.main()
