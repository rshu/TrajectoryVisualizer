"""Golden unit test for the [H] semantic detector "phase-oscillation".

Operational definition (appendix_catalog.tex, [H]):
">=3 transitions between the same two phases within a 6-step window".

The detector reads ONLY ``DetectorContext.labels`` (a dict
``{step_index: {"phase": ...}}``) and ignores the ``steps`` argument.
Thresholds: min_transitions=3, window_steps=6.
"""

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY


def _run(labels):
    detector = DETECTOR_REGISTRY["phase-oscillation"]
    return detector([], DetectorContext(labels=labels))


class PhaseOscillationDetectorTest(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # Six consecutive labels oscillating between the SAME two phases
        # (plan <-> debug): transitions = plan->debug, debug->plan,
        # plan->debug, debug->plan, plan->debug = 5 on pair {plan,debug},
        # which is >= min_transitions (3) within the 6-step window.
        phases = ["plan", "debug", "plan", "debug", "plan", "debug"]
        labels = {i: {"phase": p} for i, p in enumerate(phases)}

        detections = _run(labels)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].detector_id, "phase-oscillation")
        self.assertEqual(detections[0].span, (0, 5))
        self.assertEqual(
            sorted(detections[0].evidence["phase_pair"]), ["debug", "plan"]
        )

    def test_near_miss_distinct_pairs_fires_zero(self):
        # Same number of transitions, but each transition is on a DIFFERENT
        # phase pair, so no single two-phase pair reaches min_transitions (3).
        # transitions: plan->debug, debug->plan, plan->validate, validate->plan,
        # plan->report -> per-pair max count = 1 < 3 -> ZERO firings.
        phases = ["plan", "debug", "plan", "validate", "plan", "report"]
        labels = {i: {"phase": p} for i, p in enumerate(phases)}

        detections = _run(labels)

        self.assertEqual(len(detections), 0)


if __name__ == "__main__":
    unittest.main()
