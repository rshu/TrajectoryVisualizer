"""The same trajectory must always produce the same plan ordering.

`compute_plan_metrics` collected plan item contents into a `set[str]` and
iterated it, so the order followed Python's per-process string-hash
randomisation. `charts/activity.py` then sorted with a NON-TOTAL key (every
never-started item mapped to the identical `(1, 0)`) and Python's sort is
stable, so tied rows inherited the set order: the Plan Progress Timeline's
y-axis rows, and which stalled items the Overview names, changed between runs
on the identical file. 116 of the 502 corpus trajectories with >=2 plan items
were affected. A figure in a paper has to be reproducible from its input.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from trajviz.insight.charts.activity import build_plan_timeline_chart
from trajviz.insight.patterns import compute_plan_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent

# Written in an order a set would scramble; the plan is read in this order.
_CONTENTS = [
    "Reproduce the bug locally",
    "Identify the root cause",
    "Implement the fix",
    "Run the existing tests",
    "Update the changelog",
]


def _history(started: set[int] = frozenset(), completed: set[int] = frozenset()):
    """One plan snapshot per step, all five items present from the start."""
    snapshots = []
    for step in range(3):
        items = []
        for i, content in enumerate(_CONTENTS):
            if i in completed:
                status = "completed"
            elif i in started:
                status = "in_progress"
            else:
                status = "pending"
            items.append({"content": content, "status": status})
        snapshots.append({"step": step, "items": items})
    return snapshots


class PlanOrderIsInsertionOrdered(unittest.TestCase):
    def test_items_follow_first_appearance_not_hash_order(self):
        pm = compute_plan_metrics(_history())
        self.assertEqual([i["content"] for i in pm["items"]], _CONTENTS)

    def test_order_is_stable_across_processes_with_different_hash_seeds(self):
        """The real failure mode: a fresh interpreter, a different hash seed."""
        script = textwrap.dedent(
            """
            import json, sys
            from trajviz.insight.patterns import compute_plan_metrics
            history = json.loads(sys.argv[1])
            pm = compute_plan_metrics(history)
            print(json.dumps([i["content"] for i in pm["items"]]))
            """
        )
        payload = json.dumps(_history())
        seen = set()
        for seed in ("0", "1", "2", "3"):
            proc = subprocess.run(
                [sys.executable, "-c", script, payload],
                cwd=str(REPO_ROOT),
                env={"PYTHONPATH": str(REPO_ROOT), "PYTHONHASHSEED": seed,
                     "PATH": "/usr/bin:/bin"},
                capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            seen.add(proc.stdout.strip())
        self.assertEqual(len(seen), 1, f"plan order varies by hash seed: {seen}")
        self.assertEqual(json.loads(seen.pop()), _CONTENTS)


class PlanTimelineRowOrderIsTotal(unittest.TestCase):
    """The chart must not inherit ordering from its caller."""

    def _rows(self, items):
        fig = build_plan_timeline_chart(
            _history(),
            {"items": items, "stalled": [], "phases": [], "plan_resets": 0,
             "total_items": len(items)},
            dark=False,
        )
        rows = []
        for tr in fig.data:
            y = getattr(tr, "y", None)
            if y is not None:
                rows.extend(list(y))
        return rows

    def test_never_started_items_render_in_first_appearance_order(self):
        # Every one of these maps to the same (1, 0) under the old key, so the
        # tie order was whatever the upstream set iteration happened to give.
        # The index tiebreak makes it first-appearance, top to bottom. Plotly
        # draws the last-added bar topmost, so trace order is the reverse.
        items = [{"content": c, "start_step": None, "end_step": None,
                  "duration_steps": None} for c in _CONTENTS]
        self.assertEqual(self._rows(items), list(reversed(_CONTENTS)))

    def test_items_tied_on_start_step_render_in_first_appearance_order(self):
        items = [{"content": c, "start_step": 4, "end_step": None,
                  "duration_steps": None} for c in _CONTENTS]
        self.assertEqual(self._rows(items), list(reversed(_CONTENTS)))

    def test_rendering_the_same_items_twice_gives_the_same_rows(self):
        items = [{"content": c, "start_step": None, "end_step": None,
                  "duration_steps": None} for c in _CONTENTS]
        self.assertEqual(self._rows(items), self._rows(items))

    def test_chronological_order_still_wins_over_the_tiebreak(self):
        items = [
            {"content": "late", "start_step": 9, "end_step": None, "duration_steps": None},
            {"content": "early", "start_step": 1, "end_step": None, "duration_steps": None},
        ]
        rows = self._rows(items)
        # Earliest start renders at the top; plotly draws the last-added bar
        # topmost, so "early" must come after "late" in trace order.
        self.assertEqual(rows.index("early") > rows.index("late"), True)


if __name__ == "__main__":
    unittest.main()
