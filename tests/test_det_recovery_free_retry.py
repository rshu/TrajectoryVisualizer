"""Golden unit test for the [S] structural detector "recovery-free-retry".

Operational definition (debug.py): a FAILED action is immediately retried with
the SAME tool and the SAME args at the next step (no intervening inspection,
edit, or parameter change). Each qualifying adjacent (i, i+1) pair fires once.

The detector is ungated: no thresholds, no tool/config/capability gating, and
requires_semantic_labels is False.
"""

from __future__ import annotations

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


class RecoveryFreeRetryTest(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # Two adjacent steps: first FAILED, second an identical retry
        # (same tool, same args). Exactly one (0,1) pair qualifies.
        steps = [
            act(0, "COMMAND", tool="bash", target="pytest -q",
                args={"command": "pytest -q"}, effect="failed"),
            act(1, "COMMAND", tool="bash", target="pytest -q",
                args={"command": "pytest -q"}, effect="unknown"),
        ]
        dets = DETECTOR_REGISTRY["recovery-free-retry"](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "recovery-free-retry")
        self.assertEqual(firing.span, (0, 1))

    def test_near_miss_fires_zero(self):
        # Same tool + same args, but the first step did NOT fail
        # (effect_label="survived" -> is_failed is False), so no retry pattern.
        steps = [
            act(0, "COMMAND", tool="bash", target="pytest -q",
                args={"command": "pytest -q"}, effect="survived"),
            act(1, "COMMAND", tool="bash", target="pytest -q",
                args={"command": "pytest -q"}, effect="unknown"),
        ]
        dets = DETECTOR_REGISTRY["recovery-free-retry"](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
