"""Golden unit test for the [H] semantic detector "semantic-plan-stall".

Operational definition (appendix_catalog.tex, [H]):
    ">=5 plan-phase steps without any implement step (depends on semantic labeler)."

The detector reads only DetectorContext.labels — a dict {step_index: {"phase": ...}}.
It counts consecutive plan-phase steps; when an implement-phase step is reached with
plan_count >= min_plan_steps (default 5), it fires once with span
(first_plan_idx, implement_idx - 1). Fewer than 5 plan steps before implement -> 0.
"""

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY

DETECTOR_ID = "semantic-plan-stall"


class SemanticPlanStallTest(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # 5 consecutive plan-phase steps (idx 0..4), then an implement step (idx 5).
        # plan_count == 5 >= min_plan_steps (5) -> exactly one firing, span (0, 4).
        labels = {i: {"phase": "plan", "action": "planning"} for i in range(5)}
        labels[5] = {"phase": "implement", "action": "implement_runtime_logic"}

        ctx = DetectorContext(labels=labels)
        firings = DETECTOR_REGISTRY[DETECTOR_ID]([], ctx)

        self.assertEqual(len(firings), 1)
        self.assertEqual(firings[0].detector_id, DETECTOR_ID)
        self.assertEqual(firings[0].span, (0, 4))
        self.assertEqual(firings[0].evidence["plan_phase_steps"], 5)

    def test_near_miss_four_plan_steps_fires_zero(self):
        # Only 4 plan-phase steps (idx 0..3) before the implement step (idx 4).
        # plan_count == 4 < min_plan_steps (5) -> zero firings.
        labels = {i: {"phase": "plan", "action": "planning"} for i in range(4)}
        labels[4] = {"phase": "implement", "action": "implement_runtime_logic"}

        ctx = DetectorContext(labels=labels)
        firings = DETECTOR_REGISTRY[DETECTOR_ID]([], ctx)

        self.assertEqual(firings, [])


if __name__ == "__main__":
    unittest.main()
