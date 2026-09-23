"""Token counts and shares must stay inside their own domain.

OpenCode derives `input` by subtracting the cache read from the prompt size.
Against a provider whose `input_tokens` is already cache-exclusive (the
Anthropic-style endpoints) that subtracts twice, so 82 real corpus files carry
a NEGATIVE raw input. TrajViz summed it (Input: -3,175,801) and divided by it
(cache_read / total = 25,386.2%, rendered with a GREEN "strong cache reuse"
verdict). Both symptoms are the same record; these tests pin the domain rules
that stop either from reaching a reader as fact.
"""

from __future__ import annotations

import unittest

from trajviz.insight.analytics import compute_step_analytics
from trajviz.insight.formatting import format_behavioral_md, format_performance_md, wall_clock_fmt
from trajviz.insight.metrics import (
    build_message_metrics,
    compute_agent_summary,
    compute_health_verdict,
    compute_metrics,
)
from trajviz.insight.parser import cache_read_share, usable_token_count


def _step(index, *, total, inp, output=0, cache_read=0, cache_write=0, agent=""):
    return {
        "index": index,
        "role": "assistant",
        "duration": 1.0,
        "parts": [],
        "tool_calls": [],
        "tool_call_count": 0,
        "finish": "stop",
        "agent": agent,
        "time_created_ms": index * 1000,
        "time_completed_ms": index * 1000 + 1000,
        "tokens": {"total": total, "input": inp, "output": output,
                   "reasoning": 0, "cache_read": cache_read, "cache_write": cache_write},
    }


def _metrics(steps):
    return compute_metrics(steps, {}, build_message_metrics(steps))


# The real shape: total == input + output + cache_read, with input negative.
CORRUPT = [_step(i, total=1909, inp=-10739, output=248, cache_read=12400) for i in range(3)]
HEALTHY = [_step(i, total=10000, inp=2000, output=500, cache_read=7500) for i in range(3)]


class DomainHelpers(unittest.TestCase):
    def test_usable_token_count_rejects_impossible_values(self):
        for bad in (-1, -10739, float("nan"), float("inf"), float("-inf"), True, "5", None):
            with self.subTest(value=bad):
                self.assertIsNone(usable_token_count(bad))
        self.assertEqual(usable_token_count(0), 0)
        self.assertEqual(usable_token_count(12400), 12400)

    def test_cache_share_is_none_when_it_would_not_be_a_fraction(self):
        self.assertIsNone(cache_read_share(12400, 1909))   # the real corrupt record
        self.assertIsNone(cache_read_share(5, 0))
        self.assertIsNone(cache_read_share(-1, 10))
        self.assertEqual(cache_read_share(7500, 10000), 0.75)
        self.assertEqual(cache_read_share(10, 10), 1.0)    # 100% is legal


class CorruptRecordsDoNotProduceNumbers(unittest.TestCase):
    def setUp(self):
        self.m = _metrics(CORRUPT)

    def test_cache_ratio_is_withheld_and_the_count_published(self):
        self.assertIsNone(self.m["avg_cache_ratio"])
        self.assertEqual(self.m["cache_ratio_unusable_steps"], 3)

    def test_negative_input_is_not_summed(self):
        self.assertEqual(self.m["input_tokens"], 0)
        self.assertEqual(self.m["input_tokens_unusable_steps"], 3)
        self.assertGreaterEqual(self.m["tokens"]["input"], 0)

    def test_out_in_ratio_is_withheld_rather_than_divided_by_a_floor(self):
        # max(1, 0) used to turn this into a confident 9184.0.
        self.assertIsNone(self.m["output_input_ratio"])

    def test_health_verdict_refuses_to_rate_it(self):
        v = [x for x in compute_health_verdict(self.m, compute_step_analytics(CORRUPT))
             if x["metric"] == "Cache Efficiency"][0]
        self.assertEqual(v["label"], "N/A")
        self.assertNotEqual(v["status"], "good")
        self.assertIn("3", v["detail"])

    def test_per_step_analytics_agree_with_the_session(self):
        self.assertTrue(all(a["cache_ratio"] is None for a in compute_step_analytics(CORRUPT)))

    def test_agent_summary_never_reports_a_share_above_100(self):
        for a in compute_agent_summary(CORRUPT, {}):
            pct = a["cache_efficiency_pct"]
            if pct is not None:
                self.assertLessEqual(pct, 100)

    def test_no_surface_renders_a_literal_none_or_a_negative(self):
        import re

        md = format_behavioral_md(self.m) + format_performance_md(self.m, wall_clock_fmt(self.m)[1])
        text = re.sub(r"<[^>]+>", " ", md)
        self.assertNotIn("None", text)
        self.assertNotRegex(text, r"-[\d,]{4,}")


class HealthyRecordsAreUnaffected(unittest.TestCase):
    def test_a_well_formed_session_still_reports_every_number(self):
        m = _metrics(HEALTHY)
        self.assertEqual(m["avg_cache_ratio"], 75.0)
        self.assertEqual(m["cache_ratio_unusable_steps"], 0)
        self.assertEqual(m["input_tokens"], 6000)
        self.assertEqual(m["input_tokens_unusable_steps"], 0)
        self.assertEqual(m["output_input_ratio"], 0.25)
        v = [x for x in compute_health_verdict(m, compute_step_analytics(HEALTHY))
             if x["metric"] == "Cache Efficiency"][0]
        self.assertEqual(v["status"], "good")

    def test_a_session_with_no_token_data_still_loads(self):
        # The guard the whole suite previously missed: avg_cache_ratio is None
        # here too, and `avg_cache > 0` would raise TypeError.
        steps = [_step(0, total=0, inp=0)]
        m = _metrics(steps)
        self.assertIsNotNone(format_behavioral_md(m))

    def test_an_empty_session_still_loads(self):
        m = _metrics([])
        self.assertIsNotNone(format_behavioral_md(m))


if __name__ == "__main__":
    unittest.main()
