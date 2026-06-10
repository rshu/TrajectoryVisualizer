"""Golden unit test for the structural [S] detector "edit-thrash".

Operational definition (implement.py / appendix_catalog.tex, Phase 3):
"Same file is written >=3 times within a short window with oscillating
(non-monotonic) changes." Implementation: oscillation is observed as at least
one write in the window having effect_label == "reverted".

Thresholds (catalog): min_writes=3, window_steps=10. No gating;
requires_semantic_labels=False.
"""

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


class TestEditThrash(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # Three writes to the SAME file within window_steps=10 (indices 0,1,2),
        # with one write reverted -> oscillation precondition satisfied.
        steps = [
            act(0, "FILE_WRITE", tool="edit", target="a.py", effect="survived"),
            act(1, "FILE_WRITE", tool="edit", target="a.py", effect="reverted"),
            act(2, "FILE_WRITE", tool="edit", target="a.py", effect="survived"),
        ]
        dets = DETECTOR_REGISTRY["edit-thrash"](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "edit-thrash")
        self.assertEqual(firing.span, (0, 2))

    def test_near_miss_no_oscillation_fires_zero(self):
        # Three writes to the same file within window, but NONE reverted:
        # the oscillation precondition is missing -> zero firings.
        steps = [
            act(0, "FILE_WRITE", tool="edit", target="a.py", effect="survived"),
            act(1, "FILE_WRITE", tool="edit", target="a.py", effect="survived"),
            act(2, "FILE_WRITE", tool="edit", target="a.py", effect="survived"),
        ]
        dets = DETECTOR_REGISTRY["edit-thrash"](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
