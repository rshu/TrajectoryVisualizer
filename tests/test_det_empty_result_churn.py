"""Golden unit test for the structural [S] detector "empty-result-churn".

Operational definition (Phase 1): >=3 consecutive SEARCH steps return zero
matches. Threshold min_consecutive_empty=3, no gating, no semantic labels.
"""

from __future__ import annotations

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY

DETECTOR_ID = "empty-result-churn"


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


class TestEmptyResultChurn(unittest.TestCase):
    def test_trigger_exactly_one_firing(self):
        # Three consecutive SEARCH steps that each return an empty output ->
        # one firing spanning the whole streak (0,2).
        steps = [
            act(i, "SEARCH", tool="grep", target="foo", args={"output": ""})
            for i in range(3)
        ]
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, DETECTOR_ID)
        self.assertEqual(firing.span, (0, 2))

    def test_near_miss_two_empty_searches_zero_firings(self):
        # Only two consecutive empty searches: just under the threshold of 3.
        steps = [
            act(i, "SEARCH", tool="grep", target="foo", args={"output": ""})
            for i in range(2)
        ]
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
