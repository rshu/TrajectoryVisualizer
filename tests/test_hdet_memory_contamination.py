"""Golden unit test for the [H] semantic detector "memory-contamination".

Operational definition (appendix_catalog.tex, [H]):
"At session end, the agent writes incorrect or outdated information into a
persistent memory file. Requires LLM-judge or human annotation."

The detector (``memory_contamination.detect``) reads BOTH the ``steps`` and
``DetectorContext.labels``:

  * It scans only the TAIL of the trajectory (last 10 steps).
  * For each step it requires a FILE_WRITE action (``action_type == "FILE_WRITE"``)
    whose target basename (lowercased) is a known persistent-memory filename
    (e.g. ``CLAUDE.md``, ``AGENTS.md``, ``.cursorrules`` -- see
    ``intake._MEMORY_FILE_NAMES``).
  * The "incorrect/outdated" judgement comes from the per-step label:
    ``labels[i]["action"] == "memory_contamination"`` OR
    ``labels[i]["judge_memory_write"] == "incorrect"``.

It returns at most ONE PatternDetection (early ``return`` on first match).
The catalog declares no numeric thresholds for this detector.
"""

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY


def _run(steps, labels):
    detector = DETECTOR_REGISTRY["memory-contamination"]
    return detector(steps, DetectorContext(labels=labels))


class MemoryContaminationDetectorTest(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # A tail FILE_WRITE to a persistent-memory file (CLAUDE.md) that the
        # judge labelled as contaminating -> exactly one firing.
        steps = [
            {"action_type": "FILE_WRITE", "target": "src/feature.py"},
            {"action_type": "FILE_WRITE", "target": "/repo/CLAUDE.md"},
        ]
        labels = {
            0: {"phase": "implement", "action": "implement_runtime_logic"},
            1: {"phase": "report", "action": "memory_contamination"},
        }

        detections = _run(steps, labels)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].detector_id, "memory-contamination")
        self.assertEqual(detections[0].span, (1, 1))
        self.assertEqual(detections[0].evidence["memory_file"], "/repo/CLAUDE.md")

    def test_near_miss_correct_memory_write_fires_zero(self):
        # Identical structure -- a tail FILE_WRITE to CLAUDE.md -- but the judge
        # found the written content CORRECT (no "memory_contamination" action and
        # judge_memory_write != "incorrect"). The structural pre-condition is met
        # yet the intent/judgement condition is not -> ZERO firings.
        steps = [
            {"action_type": "FILE_WRITE", "target": "src/feature.py"},
            {"action_type": "FILE_WRITE", "target": "/repo/CLAUDE.md"},
        ]
        labels = {
            0: {"phase": "implement", "action": "implement_runtime_logic"},
            1: {"phase": "report", "action": "write_memory",
                "judge_memory_write": "correct"},
        }

        detections = _run(steps, labels)

        self.assertEqual(len(detections), 0)


if __name__ == "__main__":
    unittest.main()
