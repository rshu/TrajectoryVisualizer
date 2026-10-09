"""Token segments must add up to the reported total, whatever the export's schema.

Formats disagree about how their token fields nest, and so do models inside
one format:

* Claude Code: input, cache_read, cache_write and output are disjoint;
* Codex: cache_read is INSIDE input and reasoning is INSIDE output;
* OpenCode/Opus: everything disjoint, including reasoning;
* OpenCode/GLM: reasoning is INSIDE output.

Each surface had hard-coded one of these: the token chart's default branch
assumed nested reasoning and dropped cache_write (Claude Code bars fell short of
the total), its OpenCode branch assumed disjoint reasoning (GLM bars overshot
it), the per-agent chart subtracted reasoning from Opus output that never
contained it, and context occupancy ignored cache_write (a Claude Code first
turn, which writes its whole system prompt to the cache, read as a fraction of
its prompt). ``token_segments`` picks, per step, the reading whose implied total
matches the step's own total, and every surface draws from it.
"""

from __future__ import annotations

import unittest

from trajviz.insight.charts.usage import build_agent_token_chart, build_token_chart
from trajviz.insight.context_usage import step_context_occupancy
from trajviz.insight.parser import infer_non_cache_input, token_segments

# (name, total, input, output, reasoning, cache_read, cache_write)
SHAPES = [
    ("claude_code first turn", 15_304, 4, 300, 0, 0, 15_000),
    ("claude_code later turn", 41_410, 10, 400, 0, 40_000, 1_000),
    ("codex (cache in input, reasoning in output)", 100_800, 100_000, 800, 300, 96_000, 0),
    ("opencode opus (all disjoint)", 51_350, 1_000, 200, 150, 50_000, 0),
    ("opencode glm (reasoning in output)", 11_146, 10_866, 216, 176, 64, 0),
]


def _step(i, total, inp, out, rea, cr, cw):
    return {"index": i, "role": "assistant", "agent": "", "model_id": "m",
            "tokens": {"total": total, "input": inp, "output": out, "reasoning": rea,
                       "cache_read": cr, "cache_write": cw},
            "tool_calls": [], "parts": []}


def _bar_sum(fig) -> float:
    return sum(sum(v for v in (t.y or []) if isinstance(v, (int, float))) for t in fig.data)


class TokenSegmentTests(unittest.TestCase):
    def test_segments_add_up_to_the_reported_total_for_every_schema(self):
        for name, *vals in SHAPES:
            with self.subTest(name):
                seg = token_segments(*vals)
                parts = seg["fresh"] + seg["cache_read"] + seg["cache_write"] + seg["output"] + seg["reasoning"]
                self.assertEqual(parts, vals[0])

    def test_fresh_input_matches_the_previous_reading_where_it_was_right(self):
        # Claude Code and Codex keep the fresh value they always had.
        self.assertEqual(infer_non_cache_input(41_410, 10, 400, 0, 40_000, 1_000), 10)
        self.assertEqual(infer_non_cache_input(100_800, 100_000, 800, 300, 96_000), 4_000)

    def test_nested_reasoning_does_not_flip_the_input_reading(self):
        # GLM: adding reasoning on top of an output that already contains it
        # used to make "input includes cache" the closer fit, subtracting the
        # cache read from an input that never contained it.
        self.assertEqual(infer_non_cache_input(11_146, 10_866, 216, 176, 64), 10_866)

    def test_no_breakdown_treats_the_whole_total_as_fresh(self):
        self.assertEqual(token_segments(5_000, 0, 0, 0, 0, 0)["fresh"], 5_000)


class OccupancyTests(unittest.TestCase):
    def test_cache_write_is_part_of_the_prompt(self):
        occ = step_context_occupancy(_step(0, 15_304, 4, 300, 0, 0, 15_000))
        self.assertEqual(occ["occupancy"], 15_004)
        self.assertEqual(occ["fresh"], 15_004)
        self.assertEqual(occ["cache_read"], 0)

    def test_codex_occupancy_is_unchanged(self):
        occ = step_context_occupancy(_step(0, 100_800, 100_000, 800, 300, 96_000, 0))
        self.assertEqual(occ["occupancy"], 100_000)


class ChartTests(unittest.TestCase):
    def test_token_chart_bars_sum_to_the_total_in_every_branch(self):
        for fmt in (None, "ccsession", "codex", "opencode"):
            for name, *vals in SHAPES:
                with self.subTest(fmt=fmt, shape=name):
                    fig = build_token_chart([_step(0, *vals)], format=fmt)
                    self.assertEqual(_bar_sum(fig), vals[0])

    def test_agent_chart_bars_sum_to_each_agents_total(self):
        for name, total, inp, out, rea, cr, cw in SHAPES:
            with self.subTest(name):
                agent = {"label": "main", "total_tokens": total, "input_tokens": inp, "output_tokens": out,
                         "reasoning_tokens": rea, "cache_read_tokens": cr, "cache_write_tokens": cw}
                self.assertEqual(_bar_sum(build_agent_token_chart([agent])), total)


class TokensTooltipTests(unittest.TestCase):
    def test_the_tooltip_does_not_promise_a_sum_that_omits_cache_write(self):
        """The card's total includes cache write; the tooltip must not say otherwise."""
        from trajviz.insight.help import HELP_TEXT

        text = HELP_TEXT["tokens"]
        self.assertIn("cache write", text)
        self.assertNotIn("input + output + reasoning + cache read.", text)


if __name__ == "__main__":
    unittest.main()
