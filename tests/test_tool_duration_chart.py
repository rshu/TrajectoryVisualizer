"""Tool Call Duration chart: per-call stacks with step hover."""

from __future__ import annotations

import unittest

from trajviz.insight.charts.usage import build_tool_duration_chart
from trajviz.insight.palette import SESSION_COLORS


def _step(
    index: int | None,
    *,
    agent: str = "",
    tools: list[dict] | None = None,
) -> dict:
    tool_calls = tools or []
    step = {
        "role": "assistant",
        "agent": agent,
        "is_sub_agent": bool(agent),
        "session_id": agent or "main",
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "tokens": {"total": 0},
    }
    if index is not None:
        step["index"] = index
    return step


def _tc(name: str, *, ms: float | None = None, command: str | None = None) -> dict:
    tc: dict = {"tool_name": name, "status": "success"}
    if ms is not None:
        tc["duration_ms"] = ms
    if command is not None:
        tc["input"] = {"command": command}
    return tc


class ToolDurationChartTests(unittest.TestCase):
    def test_empty_when_no_timing(self):
        fig = build_tool_duration_chart([
            _step(0, tools=[_tc("Read"), _tc("Bash", command="git status")]),
        ])
        self.assertEqual(len(fig.data), 0)
        self.assertIn("timing", fig.layout.annotations[0].text.lower())

    def test_stacks_each_call_with_step_in_hover(self):
        fig = build_tool_duration_chart([
            _step(0, tools=[
                _tc("Read", ms=1000),
                _tc("Bash", ms=5000, command="git status"),
            ]),
            _step(3, tools=[
                _tc("Bash", ms=2000, command="python3 tools/run_eval.py"),
                _tc("Read", ms=500),
            ]),
        ])
        self.assertEqual(fig.layout.title.text, "Tool Call Duration")
        self.assertEqual(fig.layout.barmode, "overlay")
        self.assertEqual(fig.layout.clickmode, "event")
        self.assertEqual(len(fig.data), 4)

        by_tool: dict[str, list] = {}
        for trace in fig.data:
            by_tool.setdefault(trace.y[0], []).append(trace)

        self.assertEqual(set(by_tool), {"Read", "git", "run_eval.py"})
        read = by_tool["Read"]
        self.assertEqual(len(read), 2)
        bases = sorted(float(t.base[0]) for t in read)
        self.assertEqual(bases, [0.0, 1.0])
        steps = {int(t.customdata[0]) for t in read}
        self.assertEqual(steps, {0, 3})
        for trace in fig.data:
            self.assertIn("Step %{customdata}", trace.hovertemplate)
            # Bake segment secs into the template — %{x} with base is cumulative end.
            self.assertRegex(trace.hovertemplate, r"<br>\d+\.\d+s")
            self.assertNotIn("%{x", trace.hovertemplate)

    def test_windows_python_exe_labels_script_basename(self):
        fig = build_tool_duration_chart([
            _step(0, tools=[
                _tc(
                    "Bash",
                    ms=3000,
                    command=(
                        r"D:\install\py\python.exe "
                        r"C:\Users\x\.config\opencode\skills\checker_master\main.py"
                    ),
                ),
            ]),
        ])
        self.assertEqual({t.y[0] for t in fig.data}, {"main.py"})

    def test_cmd_c_labels_inner_command(self):
        fig = build_tool_duration_chart([
            _step(0, tools=[
                _tc("Bash", ms=3000, command='cmd /c "timeout /t 300 /nobreak"'),
            ]),
        ])
        self.assertEqual({t.y[0] for t in fig.data}, {"timeout"})

    def test_powershell_command_labels_inner_cmdlet(self):
        fig = build_tool_duration_chart([
            _step(0, tools=[
                _tc(
                    "Bash",
                    ms=3000,
                    command='powershell -Command "Start-Sleep -Seconds 300"',
                ),
            ]),
        ])
        self.assertEqual({t.y[0] for t in fig.data}, {"start-sleep"})

    def test_missing_index_falls_back_to_enumerate(self):
        steps = [
            _step(None, tools=[_tc("Read", ms=1000)]),
            _step(None, tools=[_tc("Read", ms=500)]),
        ]
        fig = build_tool_duration_chart(steps)
        steps_seen = sorted(int(t.customdata[0]) for t in fig.data)
        self.assertEqual(steps_seen, [0, 1])

    def test_colors_segments_by_agent(self):
        steps = [
            _step(0, tools=[_tc("Read", ms=2000)]),
            _step(1, agent="explore", tools=[_tc("Grep", ms=4000)]),
        ]
        fig = build_tool_duration_chart(steps)
        self.assertEqual(fig.layout.title.text, "Tool Call Duration by Agent")
        named = [t for t in fig.data if t.showlegend]
        self.assertEqual({t.name for t in named}, {"main", "explore"})
        self.assertEqual(
            {t.marker.color for t in fig.data},
            {SESSION_COLORS[0], SESSION_COLORS[1]},
        )

    def test_excludes_spawn_tool_wall_clock(self):
        """task/Agent duration is child wall-clock — omit so it doesn't dominate."""
        fig = build_tool_duration_chart([
            _step(0, tools=[
                _tc("task", ms=300_000),
                _tc("Read", ms=1000),
            ]),
            _step(1, tools=[
                _tc("Agent", ms=120_000),
                _tc("Bash", ms=2000, command="git status"),
            ]),
        ])
        labels = {t.y[0] for t in fig.data}
        self.assertEqual(labels, {"Read", "git"})
        self.assertNotIn("task", labels)
        self.assertNotIn("Agent", labels)


if __name__ == "__main__":
    unittest.main()
