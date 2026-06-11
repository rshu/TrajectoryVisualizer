"""Golden unit test for the [H] semantic detector "premature-implementation".

Operational definition: the first implement-phase step precedes any plan-phase
step (depends on the semantic labeler). The detector reads only
DetectorContext.labels (dict {step_index: {"phase": ...}}) and ignores `steps`.
"""

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY

DETECTOR_ID = "premature-implementation"


class PrematureImplementationTest(unittest.TestCase):
    def _run(self, labels):
        detector = DETECTOR_REGISTRY[DETECTOR_ID]
        return detector([], DetectorContext(labels=labels))

    def test_trigger_implement_before_plan(self):
        # First implement-phase step (index 0) precedes the first plan-phase
        # step (index 2) -> exactly one firing.
        labels = {
            0: {"phase": "implement", "action": "implement_runtime_logic"},
            1: {"phase": "understand", "action": "read_file"},
            2: {"phase": "plan", "action": "planning"},
            3: {"phase": "implement", "action": "implement_runtime_logic"},
        }
        firings = self._run(labels)
        self.assertEqual(len(firings), 1)
        self.assertEqual(firings[0].detector_id, DETECTOR_ID)
        self.assertEqual(firings[0].span, (0, 0))
        self.assertEqual(firings[0].evidence["first_implement_step"], 0)
        self.assertEqual(firings[0].evidence["first_plan_step"], 2)

    def test_near_miss_plan_before_implement(self):
        # First plan-phase step (index 0) precedes the first implement-phase
        # step (index 2) -> proper ordering, zero firings.
        labels = {
            0: {"phase": "plan", "action": "planning"},
            1: {"phase": "understand", "action": "read_file"},
            2: {"phase": "implement", "action": "implement_runtime_logic"},
        }
        firings = self._run(labels)
        self.assertEqual(len(firings), 0)


if __name__ == "__main__":
    unittest.main()
