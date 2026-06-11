"""Golden unit test for the [H] semantic detector "debug-wo-hypothesis".

Operational definition: >=3 debug-reproduction steps within the debug phase
without any root-cause-analysis step (depends on semantic labeler).

The detector reads only DetectorContext.labels, a mapping
{step_index: {"phase": ..., "action": ...}}. For steps whose phase == "debug":
  - reproduce actions: debug_reproduce / reproduce_failure / run_test
  - root-cause actions: root_cause_analysis / diagnose / hypothesize
It fires once when reproduce_count >= min_reproduce_steps (=3) and no
root-cause action is present.
"""

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY


class TestDebugWithoutHypothesis(unittest.TestCase):
    def test_trigger_fires_exactly_once(self) -> None:
        # 3 debug-phase reproduce steps, NO root-cause action -> exactly one firing.
        labels = {
            0: {"phase": "debug", "action": "debug_reproduce"},
            1: {"phase": "debug", "action": "run_test"},
            2: {"phase": "debug", "action": "reproduce_failure"},
        }
        firings = DETECTOR_REGISTRY["debug-wo-hypothesis"](
            [], DetectorContext(labels=labels)
        )
        self.assertEqual(len(firings), 1)
        self.assertEqual(firings[0].detector_id, "debug-wo-hypothesis")
        self.assertEqual(firings[0].span, (0, 2))
        self.assertEqual(firings[0].evidence["reproduce_steps"], 3)
        self.assertFalse(firings[0].evidence["root_cause_observed"])

    def test_near_miss_two_reproduce_steps_fires_zero(self) -> None:
        # Only 2 reproduce steps (< min_reproduce_steps=3) -> zero firings.
        labels = {
            0: {"phase": "debug", "action": "debug_reproduce"},
            1: {"phase": "debug", "action": "run_test"},
            2: {"phase": "debug", "action": "implement_runtime_logic"},
        }
        firings = DETECTOR_REGISTRY["debug-wo-hypothesis"](
            [], DetectorContext(labels=labels)
        )
        self.assertEqual(len(firings), 0)

    def test_near_miss_root_cause_present_fires_zero(self) -> None:
        # 3 reproduce steps but a root-cause action present -> suppressed, zero firings.
        labels = {
            0: {"phase": "debug", "action": "debug_reproduce"},
            1: {"phase": "debug", "action": "run_test"},
            2: {"phase": "debug", "action": "reproduce_failure"},
            3: {"phase": "debug", "action": "root_cause_analysis"},
        }
        firings = DETECTOR_REGISTRY["debug-wo-hypothesis"](
            [], DetectorContext(labels=labels)
        )
        self.assertEqual(len(firings), 0)


if __name__ == "__main__":
    unittest.main()
