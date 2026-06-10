"""Golden unit test for the structural [S] "search-loop" detector.

Operational definition (appendix_catalog.tex, Phase 1):
    ">=4 consecutive SEARCH/FILE_READ steps with no FILE_WRITE or validation
    COMMAND in between."

Threshold (from catalog): min_consecutive_steps = 4. No gating; no semantic
labels required. The detector consumes a list of CANONICAL-ACTION dicts.
"""

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


class SearchLoopDetectorTest(unittest.TestCase):
    def test_trigger_exactly_one_firing(self):
        # Four consecutive SEARCH/FILE_READ steps with no FILE_WRITE or
        # validation COMMAND in between -> exactly one search-loop firing.
        steps = [
            act(0, "SEARCH", tool="grep", target="foo"),
            act(1, "FILE_READ", tool="read", target="a.py"),
            act(2, "SEARCH", tool="grep", target="bar"),
            act(3, "FILE_READ", tool="read", target="b.py"),
        ]
        dets = DETECTOR_REGISTRY["search-loop"](steps, DetectorContext())

        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "search-loop")
        self.assertEqual(firing.span, (0, 3))

    def test_near_miss_zero_firings(self):
        # Only three consecutive SEARCH/FILE_READ steps -> just under the
        # min_consecutive_steps=4 threshold -> zero firings.
        steps = [
            act(0, "SEARCH", tool="grep", target="foo"),
            act(1, "FILE_READ", tool="read", target="a.py"),
            act(2, "SEARCH", tool="grep", target="bar"),
        ]
        dets = DETECTOR_REGISTRY["search-loop"](steps, DetectorContext())

        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
