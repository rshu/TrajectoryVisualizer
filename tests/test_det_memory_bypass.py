"""Golden unit test for the structural [S] detector "memory-bypass".

memory-bypass (Phase 0, intake.py) is config-gated: it requires a designated
memory/instruction file (e.g. CLAUDE.md) to be present in the workspace
(DetectorContext.workspace_files). It fires when that file is never read
before the first source-code FILE_WRITE.
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


class TestMemoryBypass(unittest.TestCase):
    def test_trigger_exactly_one_firing(self):
        # CLAUDE.md exists in the workspace but is never read; the agent goes
        # straight to a source-code write -> memory bypass.
        ctx = DetectorContext(workspace_files=frozenset({"CLAUDE.md"}))
        steps = [
            act(0, "SEARCH", tool="grep", target="def foo"),
            act(1, "FILE_READ", tool="read", target="src/foo.py"),
            act(2, "FILE_WRITE", tool="write", target="src/foo.py"),
        ]
        dets = DETECTOR_REGISTRY["memory-bypass"](steps, ctx)
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "memory-bypass")
        self.assertEqual(firing.span, (0, 2))

    def test_near_miss_zero_firings(self):
        # Same workspace + write, but CLAUDE.md IS read before the first
        # code write -> precondition for bypass is not met -> zero firings.
        ctx = DetectorContext(workspace_files=frozenset({"CLAUDE.md"}))
        steps = [
            act(0, "FILE_READ", tool="read", target="CLAUDE.md"),
            act(1, "FILE_READ", tool="read", target="src/foo.py"),
            act(2, "FILE_WRITE", tool="write", target="src/foo.py"),
        ]
        dets = DETECTOR_REGISTRY["memory-bypass"](steps, ctx)
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
