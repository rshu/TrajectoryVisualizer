"""Tool Outcome Timeline: success/failure markers, colored by agent when multi-agent."""

from __future__ import annotations

import unittest

from trajviz.insight.charts.swimlanes import build_tool_outcome_timeline
from trajviz.insight.palette import SESSION_COLORS, TOOL_OUTCOME_COLORS
from trajviz.insight.tool_failure import tool_call_failed


def _step(
    index: int,
    *,
    agent: str = "",
    tools: list[tuple[str, str] | tuple[str, str, str]] | None = None,
) -> dict:
    tool_calls = []
    for entry in tools or []:
        name, status = entry[0], entry[1]
        tc: dict = {"tool_name": name, "status": status}
        if len(entry) == 3:
            tc["input"] = {"command": entry[2]}
        tool_calls.append(tc)
    return {
        "index": index,
        "role": "assistant",
        "agent": agent,
        "is_sub_agent": bool(agent),
        "session_id": agent or "main",
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "tokens": {"total": 0},
    }


class ToolOutcomeTimelineTests(unittest.TestCase):
    def test_single_agent_uses_outcome_colors(self):
        fig = build_tool_outcome_timeline([
            _step(0, tools=[("Read", "success"), ("Bash", "error")]),
        ])
        names = [t.name for t in fig.data if getattr(t, "name", None)]
        self.assertEqual(names, ["Success", "Failure"])
        self.assertEqual(fig.layout.title.text, "Tool Outcome Timeline")
        by_name = {t.name: t for t in fig.data}
        self.assertEqual(by_name["Success"].marker.color, TOOL_OUTCOME_COLORS["success"])
        self.assertEqual(by_name["Failure"].marker.color, TOOL_OUTCOME_COLORS["failure"])

    def test_bash_calls_use_shell_command_labels(self):
        fig = build_tool_outcome_timeline([
            _step(0, tools=[
                ("Bash", "success", "git status"),
                ("Bash", "error", "python3 tools/run_eval.py"),
                ("Read", "success"),
            ]),
        ])
        y_values = set()
        for trace in fig.data:
            y_values.update(trace.y)
        self.assertIn("git", y_values)
        self.assertIn("run_eval.py", y_values)
        self.assertIn("Read", y_values)
        self.assertNotIn("Bash", y_values)
        self.assertNotIn("python3", y_values)

    def test_customdata_carries_step_index_for_workflow_jump(self):
        fig = build_tool_outcome_timeline([
            _step(3, tools=[("Read", "success"), ("Bash", "error", "git status")]),
        ])
        self.assertEqual(fig.layout.clickmode, "event")
        by_name = {t.name: t for t in fig.data}
        self.assertEqual(list(by_name["Success"].customdata), [3])
        self.assertEqual(list(by_name["Failure"].customdata), [3])

    def test_multi_agent_customdata_is_step_index(self):
        steps = [
            _step(2, tools=[("Read", "success")]),
            _step(5, agent="explore", tools=[("Bash", "error", "npm test")]),
        ]
        fig = build_tool_outcome_timeline(steps)
        named = [
            t for t in fig.data
            if t.name and t.name not in ("Success (circle)", "Failure (x)")
        ]
        explore = next(t for t in named if t.name == "explore")
        self.assertEqual(list(explore.customdata), [5])
        # Hover includes the outcome plus the call's command.
        self.assertEqual(list(explore.hovertext), ["Failure<br>npm test"])

    def test_multi_agent_colors_by_agent_and_shapes_by_outcome(self):
        steps = [
            _step(0, tools=[("Read", "success")]),
            _step(1, agent="explore", tools=[("Grep", "success"), ("Bash", "error")]),
        ]
        fig = build_tool_outcome_timeline(steps)
        self.assertEqual(fig.layout.title.text, "Tool Outcome Timeline by Agent")
        named = [
            t for t in fig.data
            if t.name and t.name not in ("Success (circle)", "Failure (x)")
        ]
        self.assertEqual({t.name for t in named}, {"main", "explore"})
        self.assertEqual(
            {t.marker.color for t in named},
            {SESSION_COLORS[0], SESSION_COLORS[1]},
        )
        self.assertEqual(
            [t.name for t in fig.data if t.name in ("Success (circle)", "Failure (x)")],
            ["Success (circle)", "Failure (x)"],
        )
        explore = next(t for t in named if t.name == "explore")
        symbols = list(explore.marker.symbol) if isinstance(explore.marker.symbol, (list, tuple)) else [explore.marker.symbol]
        self.assertIn("x", symbols)
        widths = list(explore.marker.line.width)
        colors = list(explore.marker.line.color)
        self.assertEqual(widths, [0, 2])  # Grep success, Bash failure
        self.assertEqual(colors[0], "rgba(0,0,0,0)")
        self.assertEqual(colors[1], TOOL_OUTCOME_COLORS["failure"])

    def test_single_agent_failure_has_red_border(self):
        fig = build_tool_outcome_timeline([
            _step(0, tools=[("Read", "success"), ("Bash", "error")]),
        ])
        by_name = {t.name: t for t in fig.data}
        self.assertEqual(by_name["Failure"].marker.line.width, 2)
        self.assertEqual(by_name["Failure"].marker.line.color, TOOL_OUTCOME_COLORS["failure"])
        success_line = by_name["Success"].marker.line
        self.assertTrue(
            success_line is None
            or success_line.width in (None, 0)
            or success_line.color in (None, "rgba(0,0,0,0)")
        )

    def test_only_agents_with_tools_count_as_multi(self):
        # Second agent exists in the trajectory but has no tool calls.
        steps = [
            _step(0, tools=[("Read", "success"), ("Bash", "error")]),
            _step(1, agent="explore", tools=[]),
        ]
        fig = build_tool_outcome_timeline(steps)
        self.assertEqual(fig.layout.title.text, "Tool Outcome Timeline")
        names = [t.name for t in fig.data]
        self.assertEqual(names, ["Success", "Failure"])

    def test_empty_trajectory(self):
        fig = build_tool_outcome_timeline([])
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 0)


class SharedFailurePredicateTests(unittest.TestCase):
    """The timeline must agree with every other surface about which calls failed.

    It was the last consumer classifying outcomes with its own inline test
    (``error`` set or ``status == "error"``). OpenCode marks the invocation
    ``completed`` whatever the command's exit code, so a ``bash`` call that
    exited non-zero was drawn as a success while Tool Success, the Issues list
    and the failure patterns counted it as a failure. It now asks
    ``tool_failure.tool_call_failed``.
    """

    @staticmethod
    def _call(name: str, *, status: str = "completed", exit_code: int | None = 0) -> dict:
        meta = {} if exit_code is None else {"exit": exit_code}
        return {"tool_name": name, "status": status, "input": {"command": "pytest"},
                "output": "", "metadata": meta, "error": None}

    @staticmethod
    def _steps(calls: list[dict]) -> list[dict]:
        return [{"index": i, "role": "assistant", "agent": "", "tool_calls": [c], "parts": []}
                for i, c in enumerate(calls)]

    def test_a_completed_call_with_a_nonzero_exit_is_drawn_as_a_failure(self):
        call = self._call("bash", exit_code=1)
        self.assertTrue(tool_call_failed(call))
        names = {t.name for t in build_tool_outcome_timeline(self._steps([call])).data}
        self.assertTrue(any("fail" in (n or "").lower() for n in names), names)
        self.assertFalse(any("success" in (n or "").lower() for n in names), names)

    def test_drawn_failures_equal_the_shared_predicate_on_every_call(self):
        calls = [self._call("bash", exit_code=0), self._call("bash", exit_code=2),
                 self._call("read", status="error"), self._call("edit", status="Timeout", exit_code=None),
                 self._call("grep", exit_code=None)]
        fig = build_tool_outcome_timeline(self._steps(calls))
        drawn = sum(len(t.x or []) for t in fig.data if "fail" in (t.name or "").lower())
        self.assertEqual(drawn, sum(tool_call_failed(c) for c in calls))


if __name__ == "__main__":
    unittest.main()
