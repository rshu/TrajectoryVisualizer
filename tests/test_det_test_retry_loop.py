"""Golden unit test for the structural [S] detector 'test-retry-loop'.

Detector logic (validate.py::detect_test_retry_loop):
  - Only COMMAND steps that are BOTH a validation command (target contains a
    substring like 'pytest') AND failed (effect_label == 'failed') are counted.
  - Steps are grouped by key = (target.strip(), error_signature), where
    error_signature = '<tool>:<first-line-of-error>' built from args
    (stderr/error/message/output) or status.
  - A FILE_WRITE step clears all retry counters (a relevant intervening edit).
  - retries = (count for that key) - 1; fires once per key when retries >= 2
    (default threshold min_retries=2), i.e. >= 3 identical failed validations
    with no intervening edit.
"""

import unittest

from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY
from trajectory_visualizer.core.detection import DetectorContext


def act(i, atype, tool="", target="", effect="unknown", args=None, status=""):
    return {"step_index": i, "action_type": atype, "tool": tool, "target": target,
            "effect_label": effect, "args": args or {}, "status": status}


class TestTestRetryLoopDetector(unittest.TestCase):
    DETECTOR_ID = "test-retry-loop"

    def test_trigger_exactly_one_firing(self):
        # Three identical FAILED validation commands, same target + same error
        # signature, with NO intervening FILE_WRITE. retries = 3 - 1 = 2 >= 2.
        steps = [
            act(i, "COMMAND", tool="bash", target="pytest tests/test_foo.py",
                effect="failed", args={"stderr": "AssertionError: 1 != 2"})
            for i in range(3)
        ]
        dets = DETECTOR_REGISTRY[self.DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "test-retry-loop")
        self.assertEqual(firing.span, (0, 2))
        self.assertEqual(firing.evidence["retries"], 2)

    def test_near_miss_zero_firings(self):
        # Only TWO identical failed validation commands: retries = 2 - 1 = 1,
        # which is < min_retries (2). Nothing fires.
        steps = [
            act(i, "COMMAND", tool="bash", target="pytest tests/test_foo.py",
                effect="failed", args={"stderr": "AssertionError: 1 != 2"})
            for i in range(2)
        ]
        dets = DETECTOR_REGISTRY[self.DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
