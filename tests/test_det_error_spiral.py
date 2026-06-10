"""Golden unit test for the structural [S] detector "error-spiral".

Operational definition (appendix_catalog.tex, Phase 5):
"Same (tool, error_signature) pair recurs >=3 times with no observable change
in approach." Threshold: min_recurrences=3, ungated, no semantic labels needed.
"""

from __future__ import annotations

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY


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


class TestErrorSpiral(unittest.TestCase):
    def test_trigger_exactly_one_firing(self):
        # Three failed bash invocations sharing the same (tool, error_signature)
        # pair, with no intervening write to reset the recurrence tracker.
        steps = [
            act(
                i,
                "COMMAND",
                tool="bash",
                target="pytest",
                effect="failed",
                args={"stderr": "ImportError: cannot import name foo"},
            )
            for i in range(3)
        ]
        dets = DETECTOR_REGISTRY["error-spiral"](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "error-spiral")
        self.assertEqual(firing.span, (0, 2))
        self.assertEqual(firing.evidence["recurrences"], 3)

    def test_near_miss_two_recurrences_zero_firings(self):
        # Only two recurrences of the same signature -> just under the
        # min_recurrences=3 threshold -> no firing.
        steps = [
            act(
                i,
                "COMMAND",
                tool="bash",
                target="pytest",
                effect="failed",
                args={"stderr": "ImportError: cannot import name foo"},
            )
            for i in range(2)
        ]
        dets = DETECTOR_REGISTRY["error-spiral"](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
