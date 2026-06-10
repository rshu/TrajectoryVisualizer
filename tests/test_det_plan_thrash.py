"""Golden unit test for the structural [S] detector "plan-thrash".

Operational definition (plan.py / appendix_catalog.tex, Phase 2):
"Repeated TodoWrite rewrites with high item-set turnover and no downstream
execution." Implementation: collect consecutive planning-tool snapshots into the
current plan block (a FILE_WRITE / phase=implement step resets the block, since
the paper requires "no downstream execution"); fire when the trailing block has
>= min_rewrites snapshots AND mean pairwise item-set turnover
(symmetric_difference / union) >= min_item_turnover.

Thresholds (catalog): min_rewrites=3, min_item_turnover=0.5.
Gating: ('tool-gated',) -> needs a planning tool in tool_exposure;
requires_semantic_labels=False. Note: the detect function keys off the step's
tool name (is_planning) and does not itself read tool_exposure, but we still set
a faithful tool-gated context here.
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


# Faithful tool-gated context: scaffold exposes a structured planning tool.
CTX = DetectorContext(tool_exposure=frozenset({"todowrite"}))


class TestPlanThrash(unittest.TestCase):
    def test_trigger_fires_exactly_once(self):
        # Three TodoWrite rewrites (rewrites=3 >= min_rewrites=3), no FILE_WRITE
        # or implement step in between (block never resets -> "no execution"),
        # and pairwise item sets are fully disjoint so every turnover ratio is
        # 1.0 -> mean_turnover = 1.0 >= min_item_turnover=0.5.
        steps = [
            act(0, "REASON", tool="todowrite", args={"todos": ["a"]}),
            act(1, "REASON", tool="todowrite", args={"todos": ["b"]}),
            act(2, "REASON", tool="todowrite", args={"todos": ["c"]}),
        ]
        dets = DETECTOR_REGISTRY["plan-thrash"](steps, CTX)
        self.assertEqual(len(dets), 1)
        firing = dets[0]
        self.assertEqual(firing.detector_id, "plan-thrash")
        self.assertEqual(firing.span, (0, 2))

    def test_near_miss_too_few_rewrites_fires_zero(self):
        # Only TWO planning snapshots (< min_rewrites=3). Turnover is still 1.0
        # (disjoint sets), so the ONLY unmet precondition is the rewrite count
        # -> zero firings.
        steps = [
            act(0, "REASON", tool="todowrite", args={"todos": ["a"]}),
            act(1, "REASON", tool="todowrite", args={"todos": ["b"]}),
        ]
        dets = DETECTOR_REGISTRY["plan-thrash"](steps, CTX)
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
