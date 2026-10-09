"""Context-pressure agent labels use the best name any of the agent's steps carries.

``_agent_pressure_label`` stopped at the first step of the agent: if that step
had neither an agent name nor a session title, it gave up and showed the raw
session-id prefix (``ses_…``), even when a later step of the same agent was
titled. It also assumed both fields were strings.
"""

from __future__ import annotations

import unittest

from trajviz.insight.context_usage import _agent_pressure_label


def _step(session: str, *, agent=None, title=None) -> dict:
    # An untagged child session (OpenCode does not always tag isSubAgent): the
    # generic naming path, not tagged_subagent_display_label's.
    step = {"index": 0, "role": "assistant", "session_id": session, "is_sub_agent": False,
            "tokens": {}, "tool_calls": [], "parts": []}
    if agent is not None:
        step["agent"] = agent
    if title is not None:
        step["session_title"] = title
    return step


class AgentPressureLabelTests(unittest.TestCase):
    def test_a_later_titled_step_names_the_agent(self):
        steps = [_step("ses_abcdef123456"), _step("ses_abcdef123456", title="Explore insight")]
        self.assertEqual(_agent_pressure_label("ses_abcdef123456", steps), "Explore insight")

    def test_non_string_fields_do_not_raise(self):
        steps = [_step("ses_abcdef123456", agent={"name": "x"}, title=["y"])]
        self.assertEqual(_agent_pressure_label("ses_abcdef123456", steps), "ses_abcd…")

    def test_without_any_name_the_id_prefix_is_used(self):
        steps = [_step("ses_abcdef123456")]
        self.assertEqual(_agent_pressure_label("ses_abcdef123456", steps), "ses_abcd…")


if __name__ == "__main__":
    unittest.main()
