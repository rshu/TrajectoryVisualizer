"""Golden unit test for the [H] semantic detector 'prompt-skim'.

prompt-skim fires once when, after the first REASON turn boundary, the agent
never re-references the user prompt (no label action in
{reread_prompt, reference_prompt, requote_prompt}). Building a fixture that
has such a reread action after the first REASON step suppresses the firing.

The detector reads BOTH the step objects (via h.action_type, to find the first
REASON step that marks the end of the first turn) AND context.labels (to look
for a reread action on later step indices).
"""

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY


def _steps():
    """Steps: a REASON turn at index 0, then two tool actions."""
    return [
        {"action_type": "REASON"},
        {"action_type": "EDIT"},
        {"action_type": "RUN_TEST"},
    ]


class TestPromptSkim(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # labels non-empty; no reread action after the first REASON (idx 0).
        labels = {
            0: {"phase": "understand", "action": "read_prompt"},
            1: {"phase": "implement", "action": "implement_runtime_logic"},
            2: {"phase": "validate", "action": "run_tests"},
        }
        ctx = DetectorContext(labels=labels)
        firings = DETECTOR_REGISTRY["prompt-skim"](_steps(), ctx)

        self.assertEqual(len(firings), 1)
        self.assertEqual(firings[0].detector_id, "prompt-skim")
        # span starts at the first REASON index (0) and ends at last step (2).
        self.assertEqual(firings[0].span, (0, 2))

    def test_near_miss_reread_after_turn_fires_zero(self):
        # Identical, except step index 2 carries a reread-prompt action,
        # which suppresses the firing.
        labels = {
            0: {"phase": "understand", "action": "read_prompt"},
            1: {"phase": "implement", "action": "implement_runtime_logic"},
            2: {"phase": "understand", "action": "reread_prompt"},
        }
        ctx = DetectorContext(labels=labels)
        firings = DETECTOR_REGISTRY["prompt-skim"](_steps(), ctx)

        self.assertEqual(len(firings), 0)


if __name__ == "__main__":
    unittest.main()
