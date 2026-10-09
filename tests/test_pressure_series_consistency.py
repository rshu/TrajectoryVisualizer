"""Context-pressure series: what the chart draws must agree with what it counts.

Regression tests for review findings against an earlier context-pressure
implementation, re-checked against ``context_usage``. Five already held. Six
did not, and are fixed here:

* a turn that reports no usage was plotted as occupancy 0, a false cliff;
* a window with no live turn still produced compaction events and a dropdown
  entry that the chart could not draw;
* the second and later compactions of one window took their hover values from
  the splice's own artifacts instead of from the real neighbouring turns;
* signals of one compaction were split when another session's turns fell
  between them, or when three signals formed a chain;
* a Claude Code sub-agent's compaction checkpoint was booked to the main agent.
"""

from __future__ import annotations

import unittest

from trajviz.insight.charts.activity import build_context_pressure_chart
from trajviz.insight.context_usage import (
    context_pressure_series,
    context_pressure_stats,
    detect_compaction_events,
    pressure_agent_choices,
)


def _tokens(*, total, inp, out=0, reasoning=0, cache_read=0, cache_write=0):
    return {"total": total, "input": inp, "output": out, "reasoning": reasoning,
            "cache_read": cache_read, "cache_write": cache_write}


def _step(idx, *, role="assistant", agent="", is_sub_agent=False, session_id="",
          tokens=None, parts=None, summary=False, model_id=""):
    return {
        "index": idx, "role": role, "agent": agent, "is_sub_agent": is_sub_agent,
        "session_id": session_id, "tokens": tokens or _tokens(total=0, inp=0),
        "parts": parts or [], "tool_calls": [], "summary": summary,
        "is_compaction_checkpoint": False, "message_type": "", "model_id": model_id,
        "session_title": "",
    }


def _bare_step(idx, *, role="assistant", agent="", tokens=None, is_compaction_checkpoint=False):
    """Claude Code-shaped step: no ``session_id`` / ``is_sub_agent`` keys at all."""
    step = {"index": idx, "role": role, "agent": agent,
            "tokens": tokens or _tokens(total=0, inp=0), "parts": [], "tool_calls": []}
    if is_compaction_checkpoint:
        step["is_compaction_checkpoint"] = True
    return step


def _occ_steps(occs, session_id=""):
    return [_step(i, session_id=session_id, tokens=_tokens(total=o, inp=o)) for i, o in enumerate(occs)]


def _interleaved(*, summary_on_resume):
    return [
        _step(0, session_id="s", tokens=_tokens(total=100_000, inp=100_000)),
        _step(1, role="user", session_id="s", parts=[{"type": "compaction"}]),
        _step(2, session_id="sub", tokens=_tokens(total=5_000, inp=5_000)),
        _step(3, session_id="sub", tokens=_tokens(total=6_000, inp=6_000)),
        _step(4, session_id="s", summary=summary_on_resume, tokens=_tokens(total=20_000, inp=20_000)),
    ]


def _drawn_markers(series: dict) -> int:
    fig = build_context_pressure_chart([], series=series)
    return sum(len(t.x or []) for t in fig.data if str(t.name or "").endswith(" compact")
               and getattr(t, "mode", "") == "markers")


class DetectionTests(unittest.TestCase):
    def test_delta_token_traces_produce_no_phantom_compactions(self):
        self.assertEqual(detect_compaction_events(_occ_steps([5_000, 800, 900, 6_000, 700, 800])), [])

    def test_moderate_dip_that_recovers_stays_suppressed(self):
        self.assertEqual(detect_compaction_events(_occ_steps([10_000, 11_000, 6_500, 10_500, 11_000])), [])

    def test_cc_subagent_compaction_is_booked_to_the_sub_agent(self):
        steps = [
            _bare_step(0, tokens=_tokens(total=150_000, inp=150_000)),
            _bare_step(1, role="user", agent="agent_xyz", is_compaction_checkpoint=True),
            _bare_step(2, agent="agent_xyz", tokens=_tokens(total=12_000, inp=12_000)),
            _bare_step(3, tokens=_tokens(total=151_000, inp=151_000)),
        ]
        msgs = [e for e in detect_compaction_events(steps) if e["kind"] == "compaction_message"]
        self.assertEqual([e["agent"] for e in msgs], ["agent_xyz"])

    def test_non_string_agent_value_does_not_crash_the_dropdown(self):
        step = {"index": 0, "role": "assistant", "agent": 42, "session_id": "s1",
                "tokens": _tokens(total=100, inp=100), "parts": [], "tool_calls": []}
        self.assertTrue(any(value == "s1" for _, value in pressure_agent_choices([step])))


class CoalescingTests(unittest.TestCase):
    def test_another_sessions_turns_do_not_split_one_compaction(self):
        series = context_pressure_series(_interleaved(summary_on_resume=True))
        self.assertEqual(context_pressure_stats(series)["compaction_count"], 1)
        self.assertEqual(_drawn_markers(series), 1)
        points = next(a["points"] for a in series["agents"] if a["agent_id"] == "s")
        self.assertTrue(all(p["occupancy"] <= 20_000 for p in points if p["step"] >= 4))

    def test_an_interleaved_drop_without_summary_counts_once(self):
        series = context_pressure_series(_interleaved(summary_on_resume=False))
        self.assertEqual(context_pressure_stats(series)["compaction_count"], 1)

    def test_three_adjacent_signals_are_one_compaction(self):
        steps = [
            _step(0, session_id="s", tokens=_tokens(total=100_000, inp=100_000)),
            _step(1, role="user", session_id="s", parts=[{"type": "compaction"}]),
            _step(2, session_id="s", summary=True, tokens=_tokens(total=20_000, inp=20_000)),
            _step(3, role="user", session_id="s", parts=[{"type": "step_finish", "name": "compress-done"}]),
        ]
        series = context_pressure_series(steps)
        self.assertEqual(context_pressure_stats(series)["compaction_count"], 1)
        self.assertEqual(_drawn_markers(series), 1)

    def test_compactions_separated_by_a_live_turn_stay_separate(self):
        steps = [
            _step(0, session_id="s", tokens=_tokens(total=100_000, inp=100_000)),
            _step(1, role="user", session_id="s", parts=[{"type": "compaction"}]),
            _step(2, session_id="s", summary=True, tokens=_tokens(total=20_000, inp=20_000)),
            _step(3, session_id="s", tokens=_tokens(total=90_000, inp=90_000)),
            _step(4, role="user", session_id="s", parts=[{"type": "compaction"}]),
            _step(5, session_id="s", summary=True, tokens=_tokens(total=15_000, inp=15_000)),
        ]
        series = context_pressure_series(steps)
        self.assertEqual(context_pressure_stats(series)["compaction_count"], 2)
        self.assertEqual(_drawn_markers(series), 2)


class SeriesTests(unittest.TestCase):
    def test_zero_usage_turns_do_not_plot_as_cliffs(self):
        steps = [
            _step(0, tokens=_tokens(total=10_000, inp=10_000)),
            _step(1),  # aborted turn, no usage reported
            _step(2, tokens=_tokens(total=11_000, inp=11_000)),
        ]
        points = context_pressure_series(steps)["agents"][0]["points"]
        self.assertEqual([p["occupancy"] for p in points], [10_000, 11_000])
        self.assertEqual([p["local_turn"] for p in points], [1, 3])

    def test_a_window_without_live_turns_is_neither_counted_nor_offered(self):
        steps = [
            _step(0, tokens=_tokens(total=5_000, inp=5_000)),
            _step(1, role="user", session_id="ses_x", parts=[{"type": "compaction"}]),
            _step(2, tokens=_tokens(total=5_100, inp=5_100)),
        ]
        series = context_pressure_series(steps)
        self.assertEqual(series["events"], [])
        self.assertEqual(context_pressure_stats(series)["compaction_count"], 0)
        self.assertEqual([a["agent_id"] for a in series["agents"]], [""])
        self.assertNotIn("ses_x", [v for _, v in pressure_agent_choices(steps)])

    def test_an_agent_with_only_zero_usage_turns_is_not_offered(self):
        steps = [
            _step(0, tokens=_tokens(total=5_000, inp=5_000)),
            _step(1, session_id="ses_y"),
        ]
        self.assertNotIn("ses_y", [v for _, v in pressure_agent_choices(steps)])
        self.assertEqual([a["agent_id"] for a in context_pressure_series(steps)["agents"]], [""])

    def test_window_limit_dropped_when_peak_exceeds_it(self):
        steps = [_step(0, model_id="claude-sonnet-4-5", tokens=_tokens(total=400_000, inp=400_000))]
        series = context_pressure_series(steps)
        self.assertIsNone(series["window_limit"])
        self.assertIsNone(context_pressure_stats(series)["peak_pct"])

    def test_second_compaction_splice_uses_its_own_neighbours(self):
        steps = [
            _step(0, session_id="s", tokens=_tokens(total=100_000, inp=100_000)),
            _step(1, role="user", session_id="s", parts=[{"type": "compaction"}]),
            _step(2, session_id="s", summary=True, tokens=_tokens(total=20_000, inp=20_000)),
            _step(3, session_id="s", tokens=_tokens(total=90_000, inp=90_000)),
            _step(4, role="user", session_id="s", parts=[{"type": "compaction"}]),
            _step(5, session_id="s", summary=True, tokens=_tokens(total=15_000, inp=15_000)),
        ]
        points = context_pressure_series(steps)["agents"][0]["points"]
        for p in points:
            self.assertEqual(p["fresh"] + p["cache_read"], p["occupancy"], p)


if __name__ == "__main__":
    unittest.main()
