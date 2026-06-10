"""Golden unit test for the [S] detector "unsupported-completion-claim".

Detector logic (trajectory_visualizer/insight/detectors/report.py):
  1. Find the last FILE_WRITE.
  2. If any step AFTER the last write is a passing validation COMMAND
     (validation substring, not failed) -> zero firings.
  3. The final REASON/assistant text must contain a completion cue
     (fixed|done|resolved|implemented|completed) -> one firing.

Ungated: empty thresholds, no gating, no semantic labels required.
"""

import unittest

from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY
from trajectory_visualizer.core.detection import DetectorContext

DETECTOR_ID = "unsupported-completion-claim"


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


class TestUnsupportedCompletionClaim(unittest.TestCase):
    def test_trigger_exactly_one_firing(self):
        # Edit a file, then claim "fixed" with NO passing validation afterward.
        steps = [
            act(0, "FILE_WRITE", tool="edit", target="src/foo.py", effect="survived"),
            act(1, "REASON", target="The bug is now fixed.",
                args={"text": "The bug is now fixed."}),
        ]
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, DETECTOR_ID)
        # span = (last_write, len(steps)-1) = (0, 1)
        self.assertEqual(firing.span, (0, 1))
        self.assertEqual(firing.evidence["cue"], "fixed")

    def test_near_miss_passing_validation_after_write(self):
        # Same completion cue, but a PASSING validation command runs after the
        # last write -> precondition broken -> zero firings.
        steps = [
            act(0, "FILE_WRITE", tool="edit", target="src/foo.py", effect="survived"),
            act(1, "COMMAND", tool="bash", target="pytest tests/",
                effect="survived", status="ok"),
            act(2, "REASON", target="The bug is now fixed.",
                args={"text": "The bug is now fixed."}),
        ]
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
