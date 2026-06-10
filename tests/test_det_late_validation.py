"""Golden unit test for the [S] structural detector "late-validation".

Operational definition (validate.py / appendix_catalog.tex, Phase 4):
"No validation COMMAND fires until after >=N implement (FILE_WRITE) steps,
with no incremental checks in between."

Threshold (from catalog): min_implement_steps_before_validate = 10.
No gating, no semantic-label requirement.

The detector counts FILE_WRITE steps (implement_count) and records the first
write index. When the first validation COMMAND (target containing a validation
substring such as "pytest") appears, it fires iff implement_count >= 10. A
validation that arrives with fewer prior writes returns zero (validated early).
"""

from __future__ import annotations

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY


def act(i, atype, tool="", target="", effect="unknown", args=None, status=""):
    """Build one canonical-action dict (the [S] detector input shape)."""
    return {
        "step_index": i,
        "action_type": atype,
        "tool": tool,
        "target": target,
        "effect_label": effect,
        "args": args or {},
        "status": status,
    }


class TestLateValidationDetector(unittest.TestCase):
    DETECTOR_ID = "late-validation"

    def test_trigger_exactly_one_firing(self):
        # 10 FILE_WRITE (implement) steps == threshold, then a validation
        # COMMAND (pytest). implement_count == 10 >= 10 -> fires once.
        steps = [
            act(i, "FILE_WRITE", tool="edit", target=f"src/mod_{i}.py")
            for i in range(10)
        ]
        steps.append(act(10, "COMMAND", tool="bash", target="pytest tests/"))

        dets = DETECTOR_REGISTRY[self.DETECTOR_ID](steps, DetectorContext())

        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "late-validation")
        # Span runs from the first implement step to the late validation.
        self.assertEqual(firing.span, (0, 10))
        self.assertEqual(
            firing.evidence["implement_steps_before_first_validation"], 10
        )

    def test_near_miss_zero_firings(self):
        # One fewer implement step (9 < 10): the validation arrives early
        # enough, so the detector must NOT fire.
        steps = [
            act(i, "FILE_WRITE", tool="edit", target=f"src/mod_{i}.py")
            for i in range(9)
        ]
        steps.append(act(9, "COMMAND", tool="bash", target="pytest tests/"))

        dets = DETECTOR_REGISTRY[self.DETECTOR_ID](steps, DetectorContext())

        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
