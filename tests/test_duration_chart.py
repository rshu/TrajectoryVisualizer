"""Step Duration chart: system (scaffold) vs tool (agentic) error series."""

from __future__ import annotations

import unittest

from trajviz.insight.charts.usage import build_duration_chart
from trajviz.insight.palette import DURATION_ERROR_COLORS
from trajviz.insight.step_errors import step_error_kind


def _tc(name: str, *, status: str = "error", exit_code: int | None = None) -> dict:
    meta = {} if exit_code is None else {"exit": exit_code}
    return {"tool_name": name, "status": status, "metadata": meta}


def _step(
    *,
    duration: float = 1.0,
    error_count: int = 0,
    finish: str = "",
    tool_calls: list[dict] | None = None,
) -> dict:
    return {
        "duration": duration,
        "error_count": error_count,
        "finish": finish,
        "tool_calls": tool_calls or [],
        "tokens": {"total": 0},
    }


class DurationErrorKindTests(unittest.TestCase):
    def test_scaffold_search_and_read_failures_are_system(self):
        self.assertEqual(step_error_kind(_step(tool_calls=[_tc("Grep")])), "system")
        self.assertEqual(step_error_kind(_step(tool_calls=[_tc("Read")])), "system")

    def test_bash_failures_are_tool(self):
        self.assertEqual(step_error_kind(_step(tool_calls=[_tc("Bash")])), "tool")
        self.assertEqual(step_error_kind(_step(tool_calls=[_tc("bash")])), "tool")
        self.assertEqual(step_error_kind(_step(tool_calls=[_tc("BashCommand")])), "tool")

    def test_agentic_workflow_failures_are_tool(self):
        self.assertEqual(step_error_kind(_step(tool_calls=[_tc("skill")])), "tool")
        self.assertEqual(step_error_kind(_step(tool_calls=[_tc("Task")])), "tool")
        self.assertEqual(step_error_kind(_step(tool_calls=[_tc("mcp__jira__create")])), "tool")
        self.assertEqual(step_error_kind(_step(tool_calls=[_tc("TodoWrite")])), "tool")

    def test_mixed_failures_prefer_tool(self):
        self.assertEqual(
            step_error_kind(_step(tool_calls=[_tc("Grep"), _tc("skill")])),
            "tool",
        )

    def test_provider_abort_is_system(self):
        self.assertEqual(step_error_kind(_step(finish="error")), "system")

    def test_error_count_fallback_without_tool_calls(self):
        self.assertEqual(step_error_kind(_step(error_count=1)), "tool")

    def test_normal_when_neither(self):
        self.assertIsNone(
            step_error_kind(_step(finish="stop", tool_calls=[_tc("Grep", status="success")])),
        )


class DurationChartSeriesTests(unittest.TestCase):
    def test_splits_system_and_tool_into_separate_traces(self):
        steps = [
            _step(duration=1.0),
            _step(duration=2.0, tool_calls=[_tc("Grep")]),
            _step(duration=3.0, tool_calls=[_tc("Bash")]),
        ]
        fig = build_duration_chart(steps)
        names = [t.name for t in fig.data if getattr(t, "name", None)]
        self.assertEqual(names, ["Normal", "System Error", "Tool Error"])

        by_name = {t.name: t for t in fig.data if getattr(t, "name", None)}
        self.assertEqual(list(by_name["Normal"].x), [0])
        self.assertEqual(list(by_name["System Error"].x), [1])
        self.assertEqual(list(by_name["Tool Error"].x), [2])
        self.assertEqual(by_name["System Error"].marker.color, DURATION_ERROR_COLORS["system"])
        self.assertEqual(by_name["Tool Error"].marker.color, DURATION_ERROR_COLORS["tool"])
        self.assertEqual(list(by_name["Normal"].customdata), [[0, ""]])
        self.assertEqual(list(by_name["System Error"].customdata), [[1, ""]])
        self.assertEqual(list(by_name["Tool Error"].customdata), [[2, ""]])
        self.assertEqual(fig.layout.legend.itemclick, "toggle")
        self.assertEqual(fig.layout.legend.itemdoubleclick, "toggleothers")

    def test_omits_empty_error_series(self):
        fig = build_duration_chart([_step(duration=1.0, tool_calls=[_tc("Grep")])])
        names = [t.name for t in fig.data if getattr(t, "name", None)]
        self.assertEqual(names, ["Normal", "System Error"])

    def test_subtracts_spawn_wait_from_bars(self):
        """Parent blocked on task should not show the child's full wall-clock."""
        steps = [
            _step(
                duration=100.0,
                tool_calls=[
                    {"tool_name": "task", "status": "success", "duration_ms": 90_000},
                    {"tool_name": "Read", "status": "success", "duration_ms": 500},
                ],
            ),
            _step(duration=5.0, tool_calls=[_tc("Grep", status="success")]),
            _step(
                duration=60.0,
                tool_calls=[
                    # Parallel tasks: take max (50s), not sum.
                    {"tool_name": "task", "status": "success", "duration_ms": 40_000},
                    {"tool_name": "task", "status": "success", "duration_ms": 50_000},
                ],
            ),
        ]
        fig = build_duration_chart(steps)
        by_name = {t.name: t for t in fig.data if getattr(t, "name", None)}
        normal = by_name["Normal"]
        self.assertEqual(list(normal.x), [0, 1, 2])
        self.assertAlmostEqual(normal.y[0], 10.0)  # 100 - 90
        self.assertAlmostEqual(normal.y[1], 5.0)
        self.assertAlmostEqual(normal.y[2], 10.0)  # 60 - 50
        self.assertEqual(normal.customdata[0][1], " (excl. task wait)")
        self.assertEqual(normal.customdata[1][1], "")
        self.assertEqual(normal.customdata[2][1], " (excl. task wait)")


if __name__ == "__main__":
    unittest.main()
