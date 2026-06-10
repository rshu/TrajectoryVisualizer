"""Golden unit test for the [S] detector "plan-less-execution".

Operational definition (plan.py / appendix_catalog.tex, Phase 2):
  ">=5 FILE_WRITE steps but zero TodoWrite/planning calls."
Threshold: min_file_writes = 5 (default DetectorContext).

The detect function gates only on (a) FILE_WRITE count >= 5 and
(b) absence of any planning-tool step (tool name in the planning set
{todowrite, taskcreate, taskupdate, tasklist, plan, update_plan}).
It does NOT consult context.tool_exposure at runtime, so a bare
DetectorContext() is sufficient.
"""

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


class TestPlanLessExecution(unittest.TestCase):
    def test_trigger_exactly_one_firing(self):
        # 5 FILE_WRITE steps (== threshold), zero planning steps.
        steps = [
            act(i, "FILE_WRITE", tool="Edit", target=f"src/mod{i}.py")
            for i in range(5)
        ]
        dets = DETECTOR_REGISTRY["plan-less-execution"](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "plan-less-execution")
        # span runs from first to last FILE_WRITE (inclusive).
        self.assertEqual(firing.span, (0, 4))
        self.assertEqual(firing.evidence["file_writes"], 5)
        self.assertEqual(firing.evidence["planning_calls"], 0)

    def test_near_miss_under_threshold(self):
        # 4 FILE_WRITE steps (one below threshold), still zero planning -> no fire.
        steps = [
            act(i, "FILE_WRITE", tool="Edit", target=f"src/mod{i}.py")
            for i in range(4)
        ]
        dets = DETECTOR_REGISTRY["plan-less-execution"](steps, DetectorContext())
        self.assertEqual(len(dets), 0)

    def test_near_miss_planning_present(self):
        # 5 FILE_WRITE steps but a planning (TodoWrite) call exists -> no fire,
        # because the pattern requires ZERO planning calls.
        steps = [
            act(i, "FILE_WRITE", tool="Edit", target=f"src/mod{i}.py")
            for i in range(5)
        ]
        steps.append(act(5, "REASON", tool="TodoWrite", args={"todos": ["a"]}))
        dets = DETECTOR_REGISTRY["plan-less-execution"](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
