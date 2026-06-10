"""Golden unit test for the [S] structural detector "validation-avoidance".

Detector logic (trajectory_visualizer/insight/detectors/validate.py,
detect_validation_avoidance):
  implement_count = # FILE_WRITE steps
  validate_count  = # validation COMMAND steps (target contains pytest/lint/...)
  - implement_count == 0                          -> no firing
  - validate_count == 0  and implement_count >= 1 -> fire (mode="no-validation")
  - implement_count / validate_count > 5.0        -> fire (mode="ratio")

Threshold (catalog): implement_to_validate_ratio = 5.0. No gating; no semantic
labels required.
"""

from __future__ import annotations

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY

DETECTOR_ID = "validation-avoidance"


def act(i, atype, tool="", target="", effect="unknown", args=None, status=""):
    """Build one canonical-action dict (the verified [S]-detector input shape)."""
    return {
        "step_index": i,
        "action_type": atype,
        "tool": tool,
        "target": target,
        "effect_label": effect,
        "args": args or {},
        "status": status,
    }


class ValidationAvoidanceDetectorTest(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # Three FILE_WRITE edits and ZERO validation commands -> "no-validation".
        steps = [
            act(i, "FILE_WRITE", tool="write", target="src/foo.py")
            for i in range(3)
        ]
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())

        self.assertEqual(len(dets), 1, f"expected exactly one firing, got {dets}")
        firing = dets[0]
        self.assertEqual(firing.detector_id, DETECTOR_ID)
        self.assertEqual(firing.span, (0, 2))
        self.assertEqual(firing.evidence["mode"], "no-validation")
        self.assertEqual(firing.evidence["implement_count"], 3)
        self.assertEqual(firing.evidence["validate_count"], 0)

    def test_near_miss_does_not_fire(self):
        # Five FILE_WRITE edits with ONE validation command: ratio = 5.0, which is
        # NOT strictly > 5.0, and validate_count > 0, so neither branch fires.
        steps = [
            act(i, "FILE_WRITE", tool="write", target="src/foo.py")
            for i in range(5)
        ]
        steps.append(act(5, "COMMAND", tool="bash", target="pytest tests/"))
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())

        self.assertEqual(len(dets), 0, f"expected zero firings, got {dets}")


if __name__ == "__main__":
    unittest.main()
