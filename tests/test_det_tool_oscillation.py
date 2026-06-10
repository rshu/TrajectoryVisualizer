"""Golden unit test for the [S] structural detector "tool-oscillation".

Operational definition (cross_cutting.detect_tool_oscillation):
  "Repeated FILE_READ -> FILE_WRITE -> FILE_READ loops on the same file/range
  with no progress."

The detector tracks per-file R/W history and counts R-W-R cycles. When a cycle
matches at history index j it advances j += 2, so the trailing R of one cycle is
reused as the leading R of the next. It fires when cycles >= min_cycles (=2).
No gating, no semantic labels required (catalog: thresholds={'min_cycles': 2},
gating=(), requires_semantic_labels=False).

  TRIGGER : R W R W R on the SAME file -> cycle at j=0 (R,W,R), advance to j=2
            (R,W,R) -> 2 cycles -> exactly one firing, span (0,4).
  NEAR-MISS: R W R on the same file -> only 1 cycle (< min_cycles=2) -> 0 firings.
"""

from __future__ import annotations

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


class TestToolOscillation(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # R W R W R on the same file -> 2 R-W-R cycles (the middle R is shared).
        seq = ["FILE_READ", "FILE_WRITE", "FILE_READ", "FILE_WRITE", "FILE_READ"]
        steps = [act(i, t, target="src/a.py") for i, t in enumerate(seq)]

        dets = DETECTOR_REGISTRY["tool-oscillation"](steps, DetectorContext())

        self.assertEqual(len(dets), 1, f"expected exactly one firing, got {dets}")
        firing = dets[0]
        self.assertEqual(firing.detector_id, "tool-oscillation")
        self.assertEqual(firing.span, (0, 4))
        self.assertEqual(firing.evidence["cycles"], 2)
        self.assertEqual(firing.evidence["file"], "src/a.py")

    def test_near_miss_fires_zero(self):
        # R W R is only a single cycle, one short of min_cycles=2 -> no firing.
        seq = ["FILE_READ", "FILE_WRITE", "FILE_READ"]
        steps = [act(i, t, target="src/a.py") for i, t in enumerate(seq)]

        dets = DETECTOR_REGISTRY["tool-oscillation"](steps, DetectorContext())

        self.assertEqual(len(dets), 0, f"expected zero firings, got {dets}")


if __name__ == "__main__":
    unittest.main()
