"""Golden unit test for the [S] detector 'edit-without-inspection'.

Operational definition (implement.py): the FIRST FILE_WRITE to a file fires when
that file has no prior FILE_READ and is not among any prior SEARCH step's matched
paths (args.matches/output/result, as a list of file-path strings).
"""

import unittest

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors import DETECTOR_REGISTRY

DETECTOR_ID = "edit-without-inspection"


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


class TestEditWithoutInspection(unittest.TestCase):
    def test_trigger_one_firing(self):
        # A FILE_WRITE to src/foo.py with no prior read or search hit on it.
        steps = [
            act(0, "SEARCH", tool="grep", target="bar", args={"matches": ["src/other.py"]}),
            act(1, "FILE_WRITE", tool="edit", target="src/foo.py", effect="survived"),
        ]
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, DETECTOR_ID)
        self.assertEqual(firing.span, (1, 1))
        self.assertEqual(firing.evidence["file"], "src/foo.py")

    def test_near_miss_inspected_before(self):
        # The same file is READ before the first write -> inspected -> zero firings.
        steps = [
            act(0, "FILE_READ", tool="read", target="src/foo.py"),
            act(1, "FILE_WRITE", tool="edit", target="src/foo.py", effect="survived"),
        ]
        dets = DETECTOR_REGISTRY[DETECTOR_ID](steps, DetectorContext())
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
