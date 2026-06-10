"""Golden unit test for the structural [S] detector "verification-skip".

Operational definition (report.py / appendix_catalog.tex, Phase 6):
"The final 5 steps before session end contain no validation COMMAND after the
last source FILE_WRITE."

Detector facts (from catalog + source):
  - thresholds: {"tail_window_steps": 5}; no gating; no semantic labels.
  - Find last FILE_WRITE; if none, return []. Otherwise scan
    steps[max(last_write+1, len(steps)-5):] for any validation COMMAND
    (is_validation_command: a COMMAND whose target contains e.g. "pytest").
    A validation command in that window => no firing; otherwise one firing.

Trigger and near-miss differ by exactly one precondition: whether a `pytest`
validation COMMAND follows the last FILE_WRITE.
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


class TestVerificationSkip(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # Last FILE_WRITE at step 1, followed only by a non-validation COMMAND
        # (`ls`) and a REASON step -> no validation command after the last
        # write -> exactly one verification-skip firing.
        steps = [
            act(0, "FILE_READ", tool="read", target="a.py"),
            act(1, "FILE_WRITE", tool="edit", target="a.py"),
            act(2, "COMMAND", tool="bash", target="ls -la"),
            act(3, "REASON", target="looks good"),
        ]
        dets = DETECTOR_REGISTRY["verification-skip"](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "verification-skip")
        # Span runs from the last write to the final step.
        self.assertEqual(firing.span, (1, 3))

    def test_near_miss_fires_zero(self):
        # Identical shape, except the post-write COMMAND is a validation
        # invocation (`pytest`) -> the detector sees validation in the tail
        # after the last write -> zero firings.
        steps = [
            act(0, "FILE_READ", tool="read", target="a.py"),
            act(1, "FILE_WRITE", tool="edit", target="a.py"),
            act(2, "COMMAND", tool="bash", target="pytest tests/"),
            act(3, "REASON", target="all tests pass"),
        ]
        dets = DETECTOR_REGISTRY["verification-skip"](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
