"""Golden unit test for the [S] structural detector "redundant-search".

Operational definition (cross_cutting.detect_redundant_search):
  Repeated nearly-identical SEARCH query within a short window.
Thresholds (catalog): window_steps=10, min_duplicates=2. No gating;
requires_semantic_labels=False, so a bare DetectorContext() suffices.

The detector normalizes each SEARCH query (args.pattern|query|regex|target,
lowercased + whitespace-collapsed), keeps a per-query sliding window of step
indices, drops indices older than `window_steps`, and fires once the live
window reaches `min_duplicates` occurrences. On firing it resets that query's
window.
"""

from __future__ import annotations

import unittest

from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY
from trajectory_visualizer.core.detection import DetectorContext


def act(i, atype, tool="", target="", effect="unknown", args=None, status=""):
    return {
        "step_index": i,
        "action_type": atype,
        "tool": tool,
        "target": target,
        "effect_label": effect,
        "args": args or {},
        "status": status,
    }


class TestRedundantSearch(unittest.TestCase):
    def test_trigger_two_identical_searches_in_window(self):
        # Two SEARCH steps with the same normalized query, 1 step apart
        # (well within window_steps=10) -> hits min_duplicates=2 -> 1 firing.
        steps = [
            act(0, "SEARCH", tool="grep", target="foo", args={"pattern": "TODO"}),
            act(1, "SEARCH", tool="grep", target="foo", args={"pattern": "TODO"}),
        ]
        dets = DETECTOR_REGISTRY["redundant-search"](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "redundant-search")
        self.assertEqual(firing.span, (0, 1))

    def test_near_miss_duplicate_outside_window_fires_zero(self):
        # Same query repeated, but the two occurrences are >window_steps (10)
        # apart: index 0 and index 11. When the 2nd is processed the 1st has
        # already aged out of the window (11 - 0 = 11 > 10), so the live window
        # holds only 1 occurrence -> below min_duplicates -> ZERO firings.
        steps = [act(0, "SEARCH", tool="grep", target="foo", args={"pattern": "TODO"})]
        steps += [act(i, "REASON") for i in range(1, 11)]
        steps.append(
            act(11, "SEARCH", tool="grep", target="foo", args={"pattern": "TODO"})
        )
        dets = DETECTOR_REGISTRY["redundant-search"](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
