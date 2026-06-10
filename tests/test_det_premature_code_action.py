"""Golden unit test for the [S] structural detector "premature-code-action".

Operational definition (Phase 0): the first source-code FILE_WRITE occurs
before any repository FILE_READ or SEARCH. The detector is ungated and uses no
thresholds or semantic labels.
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


DETECTOR_ID = "premature-code-action"


class TestPrematureCodeAction(unittest.TestCase):
    def test_trigger_write_before_any_read_or_search(self):
        # A REASON step (neither read nor search) then a FILE_WRITE: the first
        # write occurs before any FILE_READ / SEARCH -> exactly one firing.
        steps = [
            act(0, "REASON", tool="think", target="plan the fix"),
            act(1, "FILE_WRITE", tool="edit", target="src/app.py"),
        ]
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, DETECTOR_ID)
        self.assertEqual(firing.span, (0, 1))

    def test_near_miss_read_before_write_fires_zero(self):
        # A FILE_READ precedes the FILE_WRITE: the agent looked before leaping,
        # so the precondition is missing -> zero firings.
        steps = [
            act(0, "FILE_READ", tool="read", target="src/app.py"),
            act(1, "FILE_WRITE", tool="edit", target="src/app.py"),
        ]
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
