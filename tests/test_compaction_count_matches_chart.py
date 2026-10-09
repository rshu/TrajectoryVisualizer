"""The Compactions chip counts what the chart draws.

The chart coalesces adjacent same-agent compaction signals into one event
(``coalesce_compaction_events``) before drawing them, but the chip counted the
raw signals, so the two could disagree on the same trajectory ("Compactions 2"
above a chart showing one). Both now coalesce per agent.
"""

from __future__ import annotations

import unittest

from trajviz.insight.context_usage import context_pressure_stats


def _event(step: int, agent: str = "", before: int = 100_000, after: int = 20_000) -> dict:
    return {"step": step, "agent": agent, "kind": "occupancy_drop",
            "occupancy_before": before, "occupancy_after": after, "dropped": before - after}


def _stats(events: list[dict]) -> dict:
    return context_pressure_stats({"agents": [], "events": events, "window_limit": 200_000})


class CompactionCountTests(unittest.TestCase):
    def test_adjacent_signals_of_one_agent_count_once(self):
        self.assertEqual(_stats([_event(5), _event(6)])["compaction_count"], 1)

    def test_separate_compactions_count_separately(self):
        self.assertEqual(_stats([_event(5), _event(40)])["compaction_count"], 2)

    def test_signals_are_not_merged_across_agents(self):
        self.assertEqual(_stats([_event(5, "a"), _event(6, "b")])["compaction_count"], 2)

    def test_largest_drop_is_taken_from_the_merged_event(self):
        # step 5 drops 100k -> 60k; step 6 continues to 20k: one 80k compaction.
        stats = _stats([_event(5, before=100_000, after=60_000), _event(6, before=60_000, after=20_000)])
        self.assertEqual(stats["largest_drop"], 80_000)


if __name__ == "__main__":
    unittest.main()
