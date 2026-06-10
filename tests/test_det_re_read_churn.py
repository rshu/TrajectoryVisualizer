"""Golden unit test for the structural [S] detector "re-read-churn".

Operational definition (Phase 1): same file is read >= min_reads (3) times within
window_steps (10) with no intervening write to that file.
Thresholds (from catalog): min_reads=3, window_steps=10. No gating, no semantic labels.
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


class ReReadChurnDetectorTest(unittest.TestCase):
    def test_trigger_exactly_one_firing(self):
        # 3 reads of the same file at indices 0,1,2 — all within window_steps=10,
        # meets min_reads=3, no intervening write -> exactly one firing, span (0,2).
        steps = [
            act(i, "FILE_READ", tool="read", target="src/foo.py")
            for i in range(3)
        ]
        dets = DETECTOR_REGISTRY["re-read-churn"](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "re-read-churn")
        self.assertEqual(firing.span, (0, 2))

    def test_near_miss_zero_firings(self):
        # Only 2 reads of the same file -> below min_reads=3 -> zero firings.
        steps = [
            act(i, "FILE_READ", tool="read", target="src/foo.py")
            for i in range(2)
        ]
        dets = DETECTOR_REGISTRY["re-read-churn"](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
