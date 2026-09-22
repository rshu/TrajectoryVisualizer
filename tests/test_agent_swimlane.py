"""Agent swimlane: human user prompts occupy their own lane."""

from __future__ import annotations

import unittest

from trajviz.insight.charts.swimlanes import (
    USER_SWIMLANE_LABEL,
    build_agent_swimlane_chart,
)
from trajviz.insight.palette import USER_SWIMLANE_COLOR


def _step(
    index: int,
    *,
    role: str = "assistant",
    agent: str = "",
    is_sub_agent: bool = False,
    session_id: str = "main",
    tokens: int = 0,
    tool_call_count: int = 0,
    parts: list[dict] | None = None,
) -> dict:
    return {
        "index": index,
        "role": role,
        "agent": agent,
        "is_sub_agent": is_sub_agent,
        "session_id": session_id,
        "tokens": {"total": tokens},
        "tool_call_count": tool_call_count,
        "parts": parts or [],
    }


def _trace_bases(trace) -> list[int]:
    base = trace.base
    if base is None:
        return [0]
    if isinstance(base, (int, float)):
        return [int(base)]
    return [int(v) for v in base]


def _trace_widths(trace) -> list[int]:
    x = trace.x
    if x is None:
        return []
    return [int(v) for v in x]


def _lane_names(fig) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for trace in fig.data:
        name = str(getattr(trace, "name", "") or "")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


class AgentSwimlaneTests(unittest.TestCase):
    def test_user_and_main_lanes_without_subagents(self):
        fig = build_agent_swimlane_chart([
            _step(0, role="user", parts=[{"type": "text", "text": "fix it"}]),
            _step(1, tokens=12, tool_call_count=1),
        ])
        names = _lane_names(fig)
        self.assertIn(USER_SWIMLANE_LABEL, names)
        self.assertIn("main", names)
        user = next(t for t in fig.data if t.name == USER_SWIMLANE_LABEL)
        self.assertEqual(_trace_bases(user), [0])
        self.assertEqual(_trace_widths(user), [1])
        self.assertEqual(user.marker.color, USER_SWIMLANE_COLOR)
        main = next(t for t in fig.data if t.name == "main")
        self.assertEqual(_trace_bases(main), [1])
        self.assertEqual(list(fig.layout.yaxis.categoryarray), ["main", USER_SWIMLANE_LABEL])

    def test_task_and_compaction_stay_off_user_lane(self):
        fig = build_agent_swimlane_chart([
            _step(0, role="user", parts=[{"type": "text", "text": "please explore"}]),
            _step(1, tokens=4),
            _step(
                2,
                role="user",
                agent="explore",
                is_sub_agent=True,
                session_id="child",
                parts=[{"type": "text", "text": "Explore the repo"}],
            ),
            _step(3, agent="explore", is_sub_agent=True, session_id="child", tokens=8),
            _step(
                4,
                role="user",
                agent="explore",
                is_sub_agent=True,
                session_id="child",
                parts=[{"type": "compaction", "summary": "prior work"}],
            ),
        ])
        names = _lane_names(fig)
        self.assertEqual(names.count(USER_SWIMLANE_LABEL), 1)
        self.assertTrue(any(n.startswith("sub ") or n == "explore" for n in names))
        user_spans = []
        non_user_spans = []
        for t in fig.data:
            spans = [
                (base, base + width - 1)
                for base, width in zip(_trace_bases(t), _trace_widths(t), strict=True)
            ]
            if t.name == USER_SWIMLANE_LABEL:
                user_spans.extend(spans)
            else:
                non_user_spans.extend(spans)
        self.assertEqual(user_spans, [(0, 0)])
        covered = {i for start, end in non_user_spans for i in range(start, end + 1)}
        self.assertIn(2, covered)
        self.assertIn(4, covered)

    def test_empty_when_no_steps(self):
        fig = build_agent_swimlane_chart([])
        self.assertEqual(len(fig.data), 0)
        self.assertIn("No agent activity", fig.layout.annotations[0].text)

    def test_segment_customdata_is_first_step_for_workflow_jump(self):
        fig = build_agent_swimlane_chart([
            _step(0, role="user", parts=[{"type": "text", "text": "hi"}]),
            _step(1, tokens=4),
            _step(2, tokens=5),
            _step(5, tokens=6),
        ])
        self.assertEqual(fig.layout.clickmode, "event")
        by_name: dict[str, list] = {}
        for t in fig.data:
            by_name.setdefault(str(t.name), []).append(t)
        starts = sorted(int(t.customdata[0]) for t in by_name["main"])
        self.assertEqual(starts, [1, 5])
        self.assertEqual(list(by_name[USER_SWIMLANE_LABEL][0].customdata), [0])
