"""Golden unit test for the [S] structural detector "shell-over-tool".

Detector logic (trajectory_visualizer/insight/detectors/cross_cutting.py):
- Capability-gated: context.tool_exposure must contain BOTH a shell tool
  ({bash, shell, execute_command, terminal}) AND a structured-read tool
  ({read, grep, glob}). Otherwise zero firings.
- For each shell step (tool in the shell set) whose target's first word is a
  read/search command ({cat, head, tail, less, more, grep, rg, ag, find,
  fgrep, egrep}), it emits one PatternDetection with span (i, i).
- No numeric thresholds.
"""

import unittest

from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY
from trajectory_visualizer.core.detection import DetectorContext


def act(i, atype, tool="", target="", effect="unknown", args=None, status=""):
    return {"step_index": i, "action_type": atype, "tool": tool, "target": target,
            "effect_label": effect, "args": args or {}, "status": status}


class ShellOverToolDetectorTest(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # One shell step that does a `cat` read, while the session exposes BOTH
        # a shell tool and a structured-read tool -> exactly one firing.
        steps = [
            act(0, "COMMAND", tool="bash", target="cat src/module.py"),
        ]
        ctx = DetectorContext(tool_exposure=frozenset({"bash", "read"}))
        dets = DETECTOR_REGISTRY["shell-over-tool"](steps, ctx)

        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "shell-over-tool")
        self.assertEqual(firing.span, (0, 0))

    def test_near_miss_no_structured_tool_exposed_fires_zero(self):
        # Same shell `cat` step, but the session does NOT expose a structured
        # read tool (only the shell). Capability gate fails -> zero firings.
        steps = [
            act(0, "COMMAND", tool="bash", target="cat src/module.py"),
        ]
        ctx = DetectorContext(tool_exposure=frozenset({"bash"}))
        dets = DETECTOR_REGISTRY["shell-over-tool"](steps, ctx)

        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
