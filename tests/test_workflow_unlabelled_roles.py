"""No step may be unreachable from the Workflow filters.

Filtering requires a role label (User or Assistant), and labels were assigned
from an allow-list: ``user`` -> User; ``assistant`` and the agent-protocol
roles ``task``/``system``/``compaction`` -> Assistant. Any other role got
neither, so it matched no filter combination at all — including the default
view. Codex records its harness instructions as a ``developer`` message, so on
every Codex run (200 of 200 sampled) that step was counted in the step total
but could never be displayed. Anything that is not a human turn now belongs to
the agent side.
"""

from __future__ import annotations

import unittest

from trajviz.insight.presenters.workflow import FILTER_CHIPS_DEFAULT, filter_workflow_steps


def _step(i: int, role: str) -> dict:
    return {"index": i, "role": role, "parts": [{"type": "text", "text": "x"}], "tool_calls": [],
            "tool_call_count": 0}


class UnlabelledRoleTests(unittest.TestCase):
    def test_every_step_is_visible_in_the_default_view(self):
        steps = [_step(0, "developer"), _step(1, "user"), _step(2, "assistant"), _step(3, "?"), _step(4, "tool")]
        self.assertEqual(filter_workflow_steps(steps, FILTER_CHIPS_DEFAULT), [0, 1, 2, 3, 4])

    def test_non_human_roles_filter_with_the_assistant(self):
        steps = [_step(0, "developer"), _step(1, "user"), _step(2, "assistant")]
        self.assertEqual(filter_workflow_steps(steps, ["Assistant"]), [0, 2])
        self.assertEqual(filter_workflow_steps(steps, ["User"]), [1])


if __name__ == "__main__":
    unittest.main()
