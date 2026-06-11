"""Golden unit test for the [H] detector 'semantic-fruitless-exploration'.

Operational definition (appendix_catalog.tex, [H]):
  ">=5 code-reading steps where >=4 files never appear in subsequent
   implement steps (depends on semantic labeler)."

The detector (semantic_fruitless_exploration.py) counts a step as a code-read
when EITHER context.labels[i]["action"] == "code_reading", OR the step is a
FILE_READ with context.labels[i]["phase"] == "understand". Each code-read needs
a non-empty target. A target is "unused" when it never appears as the target of
a FILE_WRITE step labeled phase=="implement". Thresholds (catalog):
  min_code_reads = 5, min_unused_files = 4.
"""

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.semantic import (
    semantic_fruitless_exploration as det,
)


def _read_step(path: str) -> dict:
    """A canonical-action-shaped FILE_READ step targeting `path`."""
    return {"action_type": "FILE_READ", "target": path, "tool": "read"}


class SemanticFruitlessExplorationTest(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # Five understand-phase reads of five DISTINCT files, none of which are
        # ever written in an implement step => 5 code reads (>=5) and 5 unused
        # files (>=4). Exactly one firing.
        files = [f"src/mod_{n}.py" for n in range(5)]
        steps = [_read_step(p) for p in files]
        labels = {i: {"phase": "understand", "action": "code_reading"} for i in range(5)}

        ctx = DetectorContext(labels=labels)
        firings = det.detect(steps, ctx)

        self.assertEqual(len(firings), 1)
        f = firings[0]
        self.assertEqual(f.detector_id, "semantic-fruitless-exploration")
        self.assertEqual(f.span, (0, 4))
        self.assertEqual(f.evidence["code_read_steps"], 5)
        self.assertEqual(len(f.evidence["unused_files"]), 5)

    def test_near_miss_fires_zero(self):
        # Only FOUR understand-phase reads -> below min_code_reads (5).
        # All other preconditions identical, so the single missing read is the
        # sole reason it does not fire.
        files = [f"src/mod_{n}.py" for n in range(4)]
        steps = [_read_step(p) for p in files]
        labels = {i: {"phase": "understand", "action": "code_reading"} for i in range(4)}

        ctx = DetectorContext(labels=labels)
        firings = det.detect(steps, ctx)

        self.assertEqual(firings, [])


if __name__ == "__main__":
    unittest.main()
