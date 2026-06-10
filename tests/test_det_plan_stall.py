"""Golden unit test for the structural [S] detector "plan-stall".

plan-stall fires when there are >= min_plan_steps (catalog default 5) consecutive
planning/TodoWrite actions with NO intervening implement-phase step (FILE_WRITE or
phase=implement). A planning step is identified purely by its `tool` name being in
_PLANNING_TOOLS ({todowrite, taskcreate, taskupdate, tasklist, plan, update_plan}).

The detector (detect_plan_stall) does not itself enforce the catalog's tool-gating;
gating is evaluated separately by the runner via DetectorContext.gating_satisfied.
We still expose a planning tool in the context so the fixture is faithful to a real
run where the gate would be satisfied.
"""

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


# A DetectorContext whose tool_exposure satisfies the catalog tool-gate
# (a planning tool is present). Firing itself depends only on detector logic.
CTX = DetectorContext(tool_exposure=frozenset({"todowrite", "read"}))


class TestPlanStall(unittest.TestCase):
    def test_trigger_exactly_one_firing(self):
        # 5 planning (TodoWrite) actions, no implement step anywhere -> terminal stall.
        steps = [
            act(i, "REASON", tool="todowrite", args={"todos": [f"task-{i}"]})
            for i in range(5)
        ]
        dets = DETECTOR_REGISTRY["plan-stall"](steps, CTX)
        self.assertEqual(len(dets), 1, f"expected exactly 1 firing, got {dets}")
        firing = dets[0]
        self.assertEqual(firing.detector_id, "plan-stall")
        self.assertEqual(firing.span, (0, 4))
        self.assertEqual(firing.evidence["plan_steps_before_implement"], 5)

    def test_near_miss_zero_firings(self):
        # Only 4 planning actions: one under the min_plan_steps=5 threshold.
        steps = [
            act(i, "REASON", tool="todowrite", args={"todos": [f"task-{i}"]})
            for i in range(4)
        ]
        dets = DETECTOR_REGISTRY["plan-stall"](steps, CTX)
        self.assertEqual(len(dets), 0, f"expected 0 firings, got {dets}")


if __name__ == "__main__":
    unittest.main()
