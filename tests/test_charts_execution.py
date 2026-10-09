"""Execute every public chart builder and assert the numbers it draws.

The chart package renders the quantities a reader takes away from the
dashboard (tokens per step, step duration, tool counts, tool duration, agent
lanes, context occupancy). A figure that *builds* proves nothing — these tests
assert that what the chart DRAWS reconciles with the quantity it claims to
show, and that the same tool call / step / interaction is never dropped or
double-drawn.

Everything here is built from literal step dicts so the suite stays hermetic;
the shapes mirror what ``parse_steps`` emits for the real formats.
"""

from __future__ import annotations

import math
import re
import unittest

from trajviz.insight.charts import (
    build_agent_swimlane_chart,
    build_agent_token_chart,
    build_context_pressure_chart,
    build_duration_chart,
    build_file_interaction_chart,
    build_label_phase_count_chart,
    build_label_phase_duration_chart,
    build_phase_count_comparison_chart,
    build_plan_timeline_chart,
    build_run_group_agent_timeline,
    build_skill_agent_chart,
    build_token_chart,
    build_tool_chart,
    build_tool_duration_chart,
    build_tool_outcome_timeline,
)
from trajviz.insight.metrics import (
    build_message_metrics,
    compute_metrics,
    step_duration_excluding_spawn,
)
from trajviz.insight.palette import (
    LABEL_PHASE_COLORS,
    SESSION_COLORS,
    TOKEN_COLORS,
    USER_SWIMLANE_COLOR,
)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

# Swimlane segment labels read "<n> steps, <t> tok" with grouped thousands.
_SEGMENT_TOKENS_RE = re.compile(r"([\d,]+) tok\b")


def tc(
    name: str,
    *,
    status: str = "completed",
    ms: float | None = None,
    command: str | None = None,
    file_path: str | None = None,
    skill: str | None = None,
    error: str | None = None,
) -> dict:
    """One tool call. ``ms`` stamps a duration window the chart can time."""
    call: dict = {"tool_name": name, "status": status}
    if ms is not None:
        call["time_start"] = 0
        call["time_end"] = ms
    inp: dict = {}
    if command is not None:
        inp["command"] = command
    if file_path is not None:
        inp["file_path"] = file_path
    if skill is not None:
        inp["skill"] = skill
    if inp:
        call["input"] = inp
    if error is not None:
        call["error"] = error
    return call


def step(
    index: int,
    *,
    role: str = "assistant",
    agent: str = "",
    session_id: str = "s-main",
    tool_calls: list[dict] | None = None,
    duration: float | None = 1.0,
    total: int = 0,
    inp: int = 0,
    output: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    reasoning: int = 0,
) -> dict:
    """A parsed step with a complete token dict, as the loaders emit."""
    calls = tool_calls or []
    return {
        "index": index,
        "role": role,
        "agent": agent,
        "session_id": session_id,
        "model_id": "claude-sonnet-4",
        "tokens": {
            "total": total,
            "input": inp,
            "output": output,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "reasoning": reasoning,
        },
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "duration": duration,
        "parts": [],
        "error_count": 0,
        "finish": "stop",
    }


def balanced(index: int, *, inp: int, output: int, cache_read: int = 0, reasoning: int = 0, **kw) -> dict:
    """A step in the schema the stacked token chart documents.

    ``total = fresh input + cache read + output``, with ``cache_read`` part of
    the reported ``input`` field and ``reasoning`` part of ``output`` — the
    shape Codex exports use.
    """
    return step(
        index,
        total=inp + cache_read + output,
        inp=inp + cache_read,
        output=output,
        cache_read=cache_read,
        reasoning=reasoning,
        **kw,
    )


def trace_values(fig, axis: str) -> list:
    out = []
    for tr in fig.data:
        vals = getattr(tr, axis, None)
        if vals is None:
            continue
        out.extend(vals if isinstance(vals, (list, tuple)) else [vals])
    return out


def assert_finite(testcase: unittest.TestCase, fig, label: str) -> None:
    """No NaN/Inf may reach a Plotly axis — it renders as a silent gap."""
    for axis in ("x", "y", "base"):
        for v in trace_values(fig, axis):
            if isinstance(v, float):
                testcase.assertTrue(
                    math.isfinite(v),
                    f"{label}: non-finite {axis} value {v!r}",
                )


# --------------------------------------------------------------------------
# Token Usage by Step
# --------------------------------------------------------------------------


class TokenChartTests(unittest.TestCase):
    """The stacked bar must equal the step's reported total token count.

    The chart's whole claim is that the segments decompose one number. If the
    stack and ``tokens["total"]`` diverge, the bar heights a reader compares
    across steps mean nothing.
    """

    def _stack_by_step(self, fig) -> dict[int, int]:
        per: dict[int, int] = {}
        for tr in fig.data:
            for x, y in zip(tr.x or [], tr.y or [], strict=True):
                per[x] = per.get(x, 0) + (y or 0)
        return per

    def test_disjoint_schema_stack_equals_step_total(self):
        steps = [
            balanced(0, inp=1000, output=200),
            balanced(1, inp=300, output=150, cache_read=5000, reasoning=50),
            balanced(2, inp=0, output=0),
        ]
        per = self._stack_by_step(build_token_chart(steps))
        for i, s in enumerate(steps):
            self.assertEqual(per.get(i, 0), s["tokens"]["total"], f"step {i} stack != total")

    def test_no_breakdown_falls_back_to_a_total_bar_equal_to_total(self):
        steps = [step(0, total=1234), step(1, total=99)]
        fig = build_token_chart(steps)
        names = [t.name for t in fig.data]
        self.assertEqual(names, ["Total"])
        self.assertEqual(list(fig.data[0].y), [1234, 99])

    def test_segments_are_never_negative(self):
        # A trace whose reported input is smaller than its cache read must not
        # produce a negative segment — plotly would stack it downward.
        steps = [step(0, total=209, inp=-140882, output=146, cache_read=140945)]
        fig = build_token_chart(steps, format="opencode")
        for y in trace_values(fig, "y"):
            self.assertGreaterEqual(y, 0)
        assert_finite(self, fig, "token/negative-input")

    def test_token_traces_use_the_shared_palette(self):
        steps = [balanced(0, inp=100, output=10, cache_read=20, reasoning=5)]
        colors = {t.name: t.marker.color for t in build_token_chart(steps).data}
        self.assertEqual(colors["Fresh Input"], TOKEN_COLORS["fresh_input"])
        self.assertEqual(colors["Cache Read"], TOKEN_COLORS["cache_read"])
        self.assertEqual(colors["Output"], TOKEN_COLORS["output"])
        self.assertEqual(colors["Reasoning"], TOKEN_COLORS["reasoning"])

    def test_every_drawn_series_carries_a_legend_name(self):
        steps = [balanced(0, inp=100, output=10, cache_read=20, reasoning=5)]
        for tr in build_token_chart(steps).data:
            self.assertTrue(tr.name, "a drawn token series has no legend label")


# --------------------------------------------------------------------------
# Step Duration
# --------------------------------------------------------------------------


class DurationChartTests(unittest.TestCase):
    """Bar heights must be the same seconds ``metrics`` attributes to the step."""

    def _bars_by_x(self, fig) -> dict[int, float]:
        per: dict[int, float] = {}
        for tr in fig.data:
            if tr.type != "bar":
                continue
            for x, y in zip(tr.x or [], tr.y or [], strict=True):
                per[x] = per.get(x, 0.0) + (y or 0.0)
        return per

    def test_bar_equals_step_duration_excluding_spawn(self):
        steps = [
            step(0, duration=4.0),
            step(1, duration=30.0, tool_calls=[tc("Task", ms=25_000), tc("Read")]),
            step(2, duration=2.5, tool_calls=[tc("Bash", ms=1_000)]),
        ]
        per = self._bars_by_x(build_duration_chart(steps))
        for i, s in enumerate(steps):
            self.assertAlmostEqual(per[i], step_duration_excluding_spawn(s), places=6)
        # The delegated 25s must not appear as a spike on the parent step.
        self.assertAlmostEqual(per[1], 5.0, places=6)

    def test_missing_duration_is_excluded_from_the_average_line(self):
        # A step with no recorded duration is not a zero-second step; folding
        # it into the mean would drag the average line toward zero.
        steps = [step(0, duration=10.0), step(1, duration=None), step(2, duration=20.0)]
        fig = build_duration_chart(steps)
        avg = [a for a in fig.layout.annotations if a.text and a.text.startswith("Avg:")]
        self.assertEqual(len(avg), 1)
        self.assertEqual(avg[0].text, "Avg: 15.0s")

    def test_non_finite_duration_never_reaches_the_axis(self):
        steps = [step(0, duration=float("nan")), step(1, duration=float("inf")), step(2, duration=3.0)]
        fig = build_duration_chart(steps)
        assert_finite(self, fig, "duration/non-finite")

    def test_error_kinds_split_into_their_own_named_series(self):
        steps = [
            step(0, duration=1.0),
            step(1, duration=1.0, tool_calls=[tc("Read", status="error")]),
            step(2, duration=1.0, tool_calls=[tc("Bash", status="error", command="pytest")]),
        ]
        by_name = {t.name: t for t in build_duration_chart(steps).data if t.type == "bar"}
        self.assertEqual(set(by_name), {"Normal", "System Error", "Tool Error"})
        self.assertEqual(list(by_name["Normal"].x), [0])
        self.assertEqual(list(by_name["System Error"].x), [1])
        self.assertEqual(list(by_name["Tool Error"].x), [2])

    def test_hover_customdata_carries_the_real_step_index(self):
        # The Workflow jump handler reads customdata[0]; positional order and
        # the step's own index diverge whenever a trajectory does not start
        # numbering at zero.
        steps = [step(7, duration=1.0), step(8, duration=2.0)]
        normal = next(t for t in build_duration_chart(steps).data if t.name == "Normal")
        self.assertEqual([cd[0] for cd in normal.customdata], [7, 8])


# --------------------------------------------------------------------------
# Tool Call Frequency / Duration
# --------------------------------------------------------------------------


class ToolChartTests(unittest.TestCase):
    """Bar lengths must account for every tool call, exactly once."""

    def test_total_bar_length_equals_metrics_tool_call_count(self):
        steps = [
            step(0, tool_calls=[tc("Read", file_path="/a.py"), tc("Bash", command="pytest -q")]),
            step(1, tool_calls=[tc("Read", file_path="/b.py"), tc("Grep"), tc("Bash", command="git status")]),
        ]
        metrics = compute_metrics(steps, {}, build_message_metrics(steps))
        drawn = sum(v for v in trace_values(build_tool_chart(steps), "x") if isinstance(v, (int, float)))
        self.assertEqual(drawn, metrics["tool_call_count"])

    def test_per_agent_stack_sums_to_the_per_tool_total(self):
        steps = [
            step(0, tool_calls=[tc("Read"), tc("Read")]),
            step(1, agent="explore", session_id="s-explore", tool_calls=[tc("Read"), tc("Grep")]),
        ]
        fig = build_tool_chart(steps)
        per_tool: dict[str, int] = {}
        for tr in fig.data:
            for y, x in zip(tr.y or [], tr.x or [], strict=True):
                per_tool[y] = per_tool.get(y, 0) + x
        self.assertEqual(per_tool["Read"], 3)
        self.assertEqual(per_tool["Grep"], 1)

    def test_agent_series_use_the_shared_session_palette(self):
        steps = [
            step(0, tool_calls=[tc("Read")]),
            step(1, agent="explore", session_id="s-explore", tool_calls=[tc("Grep")]),
        ]
        colors = {t.marker.color for t in build_tool_chart(steps).data}
        self.assertTrue(colors.issubset(set(SESSION_COLORS)), colors)


class ToolDurationChartTests(unittest.TestCase):
    """Each tool's bar must be the sum of that tool's timed calls."""

    def _row_totals(self, fig) -> dict[str, float]:
        rows: dict[str, float] = {}
        for tr in fig.data:
            rows[tr.y[0]] = rows.get(tr.y[0], 0.0) + tr.x[0]
        return rows

    def test_segments_sum_to_the_tools_measured_duration(self):
        steps = [
            step(0, tool_calls=[tc("Read", ms=250), tc("Bash", ms=4_000, command="pytest -q")]),
            step(1, tool_calls=[tc("Read", ms=750)]),
        ]
        rows = self._row_totals(build_tool_duration_chart(steps))
        self.assertAlmostEqual(rows["Read"], 1.0, places=6)
        self.assertAlmostEqual(rows["pytest"], 4.0, places=6)

    def test_delegation_wall_clock_is_not_charted_as_tool_time(self):
        # A `task` call's duration is the child agent's whole run; charting it
        # double-counts work the child's own calls already show.
        steps = [step(0, tool_calls=[tc("Task", ms=600_000), tc("Read", ms=500)])]
        rows = self._row_totals(build_tool_duration_chart(steps))
        self.assertEqual(set(rows), {"Read"})
        self.assertAlmostEqual(rows["Read"], 0.5, places=6)

    def test_untimed_calls_report_no_timing_instead_of_zero_bars(self):
        # "This export records no per-tool timing" is not "the tools took 0s".
        fig = build_tool_duration_chart([step(0, tool_calls=[tc("Read"), tc("Grep")])])
        self.assertEqual(len(fig.data), 0)
        messages = [a.text for a in fig.layout.annotations]
        self.assertTrue(any("timing" in (m or "") for m in messages), messages)

    def test_each_segment_names_the_step_it_ran_on(self):
        steps = [step(3, tool_calls=[tc("Read", ms=1_000)]), step(9, tool_calls=[tc("Read", ms=2_000)])]
        fig = build_tool_duration_chart(steps)
        self.assertEqual(sorted(t.customdata[0] for t in fig.data), [3, 9])


class ToolOutcomeTimelineTests(unittest.TestCase):
    """Every tool call must produce exactly one marker."""

    def _marker_count(self, fig) -> int:
        n = 0
        for tr in fig.data:
            xs = list(tr.x or [])
            if xs and xs[0] is None:  # dummy shape-legend rows
                continue
            n += len(xs)
        return n

    def test_marker_count_matches_the_number_of_tool_calls(self):
        steps = [
            step(0, tool_calls=[tc("Read"), tc("Bash", status="error", command="pytest")]),
            step(1, agent="explore", session_id="s-explore", tool_calls=[tc("Grep"), tc("Read")]),
        ]
        metrics = compute_metrics(steps, {}, build_message_metrics(steps))
        self.assertEqual(self._marker_count(build_tool_outcome_timeline(steps)), metrics["tool_call_count"])

    def test_marker_count_survives_a_single_agent_trajectory(self):
        steps = [step(0, tool_calls=[tc("Read"), tc("Read"), tc("Grep", status="error")])]
        self.assertEqual(self._marker_count(build_tool_outcome_timeline(steps)), 3)


# --------------------------------------------------------------------------
# Agent swimlane / run-group timeline
# --------------------------------------------------------------------------


class SwimlaneTests(unittest.TestCase):
    """Lane segments must tile the step axis and carry the steps' own tokens."""

    def test_segment_widths_cover_every_step_exactly_once(self):
        steps = [
            step(0, total=100),
            step(1, total=200, agent="explore", session_id="s-explore"),
            step(2, total=300),
            step(3, total=400),
        ]
        fig = build_agent_swimlane_chart(steps)
        covered = sum(t.x[0] for t in fig.data)
        self.assertEqual(covered, len(steps))

    def test_segment_labels_account_for_every_token(self):
        steps = [
            step(0, total=100_000),
            step(1, total=250_500, agent="explore", session_id="s-explore"),
            step(2, total=7),
        ]
        drawn = 0
        for tr in build_agent_swimlane_chart(steps).data:
            match = _SEGMENT_TOKENS_RE.search(tr.text)
            self.assertIsNotNone(match, tr.text)
            drawn += int(match.group(1).replace(",", ""))
        self.assertEqual(drawn, sum(s["tokens"]["total"] for s in steps))

    def test_human_prompts_get_their_own_lane_and_color(self):
        steps = [step(0, role="user", total=10), step(1, total=20)]
        fig = build_agent_swimlane_chart(steps)
        lanes = {tr.y[0]: tr.marker.color for tr in fig.data}
        self.assertIn("user", lanes)
        self.assertEqual(lanes["user"], USER_SWIMLANE_COLOR)
        self.assertTrue(set(lanes.values()) - {USER_SWIMLANE_COLOR} <= set(SESSION_COLORS))

    def test_customdata_is_the_first_step_of_the_segment(self):
        steps = [step(4, total=1), step(5, total=1), step(6, total=1)]
        fig = build_agent_swimlane_chart(steps)
        self.assertEqual([t.customdata[0] for t in fig.data], [4])

    def test_out_of_order_steps_still_cover_every_step(self):
        steps = [step(i, total=1) for i in (5, 3, 4, 0, 1, 2)]
        fig = build_agent_swimlane_chart(steps)
        self.assertEqual(sum(t.x[0] for t in fig.data), len(steps))


class RunGroupTimelineTests(unittest.TestCase):
    """One lane per loaded run; segments tile that run's steps."""

    def test_one_lane_per_run_and_segments_cover_each_run(self):
        runs = [
            {"label": "run-a", "steps": [step(0, total=1), step(1, total=1)]},
            {"label": "run-b", "steps": [step(0, total=1), step(1, total=1), step(2, total=1)]},
        ]
        fig = build_run_group_agent_timeline(runs)
        widths: dict[str, int] = {}
        for tr in fig.data:
            widths[tr.y[0]] = widths.get(tr.y[0], 0) + tr.x[0]
        self.assertEqual(widths, {"run-a": 2, "run-b": 3})

    def test_runs_with_no_steps_do_not_create_empty_lanes(self):
        runs = [{"label": "run-a", "steps": [step(0, total=1)]}, {"label": "empty", "steps": []}]
        fig = build_run_group_agent_timeline(runs)
        self.assertEqual({tr.y[0] for tr in fig.data}, {"run-a"})


# --------------------------------------------------------------------------
# Context pressure
# --------------------------------------------------------------------------


class ContextPressureChartTests(unittest.TestCase):
    """The stacked composition must reach the occupancy line it sits under."""

    STEPS = [
        step(0, total=120, inp=100, output=20),
        step(1, total=200, inp=50, output=30, cache_read=120),
        step(2, total=250, inp=40, output=10, cache_read=200),
    ]

    def test_fresh_plus_cache_equals_the_occupancy_line(self):
        fig = build_context_pressure_chart(self.STEPS)
        by_name = {t.name: list(t.y) for t in fig.data if t.name}
        for fresh, cache, occ in zip(
            by_name["Fresh input"], by_name["Cache read"], by_name["Occupancy"], strict=True
        ):
            self.assertEqual(fresh + cache, occ)

    def test_occupancy_line_is_not_the_fresh_input_fill_color(self):
        # After a compaction the stack is almost all fresh input; an occupancy
        # line in the same blue disappears into its own fill.
        fig = build_context_pressure_chart(self.STEPS)
        occ = next(t for t in fig.data if t.name == "Occupancy")
        self.assertNotEqual(occ.line.color, TOKEN_COLORS["fresh_input"])

    def test_occupancy_values_are_finite_and_non_negative(self):
        fig = build_context_pressure_chart(self.STEPS)
        assert_finite(self, fig, "pressure")
        occ = next(t for t in fig.data if t.name == "Occupancy")
        for v in occ.y:
            self.assertGreaterEqual(v, 0)


# --------------------------------------------------------------------------
# File interaction timeline
# --------------------------------------------------------------------------


class FileInteractionChartTests(unittest.TestCase):
    """Every recorded interaction gets a marker, on its own file row."""

    def test_marker_count_equals_interaction_count(self):
        interactions = [
            {"step": 0, "path": "/repo/a.py", "tool": "Read", "type": "read"},
            {"step": 1, "path": "/repo/b.py", "tool": "Write", "type": "write"},
            {"step": 2, "path": "/repo/a.py", "tool": "Grep", "type": "search"},
        ]
        fig = build_file_interaction_chart(interactions)
        drawn = sum(len(list(tr.x or [])) for tr in fig.data if not (tr.x and tr.x[0] is None))
        self.assertEqual(drawn, len(interactions))

    def test_distinct_paths_never_share_a_row(self):
        # Two long paths that shorten to the same label must stay separate
        # rows, or one file's history is silently merged into another's.
        long_a = "/home/dev/workspace/project/src/very/deep/tree/module_one/handler.py"
        long_b = "/home/dev/workspace/project/src/very/deep/other/module_two/handler.py"
        interactions = [
            {"step": 0, "path": long_a, "tool": "Read", "type": "read"},
            {"step": 1, "path": long_b, "tool": "Read", "type": "read"},
        ]
        rows = set(trace_values(build_file_interaction_chart(interactions), "y"))
        rows.discard(None)
        self.assertEqual(len(rows), 2, rows)

    def test_target_files_are_outlined_and_others_are_not(self):
        interactions = [
            {"step": 0, "path": "/repo/target.py", "tool": "Write", "type": "write"},
            {"step": 1, "path": "/repo/other.py", "tool": "Write", "type": "write"},
        ]
        fig = build_file_interaction_chart(interactions, {"/repo/target.py"})
        widths = [w for tr in fig.data for w in (tr.marker.line.width or [])]
        self.assertEqual(sorted(widths), [0, 2])


# --------------------------------------------------------------------------
# Skill sankey / plan timeline / agent tokens / label charts
# --------------------------------------------------------------------------


class SkillSankeyTests(unittest.TestCase):
    """Link values must account for every Skill invocation."""

    def test_link_values_sum_to_the_number_of_skill_calls(self):
        steps = [
            step(0, tool_calls=[tc("skill", skill="brainstorming"), tc("skill", skill="brainstorming")]),
            step(1, agent="explore", session_id="s-explore", tool_calls=[tc("skill", skill="dataviz")]),
            step(2, tool_calls=[tc("Read")]),
        ]
        sankey = build_skill_agent_chart(steps).data[0]
        self.assertEqual(sum(sankey.link.value), 3)

    def test_node_labels_carry_the_counts_they_aggregate(self):
        steps = [step(0, tool_calls=[tc("skill", skill="dataviz"), tc("skill", skill="dataviz")])]
        labels = list(build_skill_agent_chart(steps).data[0].node.label)
        self.assertIn("main (2)", labels)
        self.assertIn("dataviz (2)", labels)

    def test_no_skill_calls_renders_an_explanatory_empty_figure(self):
        fig = build_skill_agent_chart([step(0, tool_calls=[tc("Read")])])
        self.assertEqual(len(fig.data), 0)
        self.assertTrue(any(a.text for a in fig.layout.annotations))


class PlanTimelineTests(unittest.TestCase):
    """No todo item may be dropped from the plan timeline."""

    def test_every_plan_item_gets_a_bar(self):
        plan_metrics = {
            "items": [
                {"content": "Reproduce the bug", "start_step": 1, "end_step": 4},
                {"content": "Write the fix", "start_step": 5, "end_step": None},
                {"content": "Run the tests", "start_step": None, "end_step": 9},
                {"content": "Update the docs", "start_step": None, "end_step": None},
            ],
            "stalled": [],
        }
        fig = build_plan_timeline_chart([], plan_metrics)
        self.assertEqual(sum(1 for t in fig.data if t.type == "bar"), 4)

    def test_completed_bar_spans_start_to_end(self):
        plan_metrics = {"items": [{"content": "Fix it", "start_step": 2, "end_step": 7}], "stalled": []}
        bar = build_plan_timeline_chart([], plan_metrics).data[0]
        self.assertEqual(bar.base, 2)
        self.assertEqual(bar.x[0], 5)

    def test_bars_are_finite_for_items_with_no_timing(self):
        plan_metrics = {"items": [{"content": "Never started", "start_step": None, "end_step": None}], "stalled": []}
        assert_finite(self, build_plan_timeline_chart([], plan_metrics), "plan/no-timing")


class AgentTokenChartTests(unittest.TestCase):
    """Per-agent bars must be the summary numbers, not re-derived ones."""

    SUMMARIES = [
        {
            "label": "main",
            "total_tokens": 1300,
            "input_tokens": 1000,
            "output_tokens": 200,
            "reasoning_tokens": 50,
            "cache_read_tokens": 100,
        },
        {
            "label": "explore",
            "total_tokens": 400,
            "input_tokens": 300,
            "output_tokens": 100,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
        },
    ]

    def test_cache_read_bar_is_the_summarys_cache_read(self):
        fig = build_agent_token_chart(self.SUMMARIES)
        cache = next(t for t in fig.data if t.name == "Cache Read")
        self.assertEqual(dict(zip(cache.x, cache.y, strict=True)), {"main": 100, "explore": 0})

    def test_reasoning_is_not_counted_twice_against_output(self):
        fig = build_agent_token_chart(self.SUMMARIES)
        by_name = {t.name: dict(zip(t.x, t.y, strict=True)) for t in fig.data}
        self.assertEqual(by_name["Output"]["main"] + by_name["Reasoning"]["main"], 200)

    def test_one_cluster_per_agent(self):
        fig = build_agent_token_chart(self.SUMMARIES)
        for tr in fig.data:
            self.assertEqual(list(tr.x), ["main", "explore"])

    def test_no_agents_renders_an_explanatory_empty_figure(self):
        fig = build_agent_token_chart([])
        self.assertEqual(len(fig.data), 0)
        self.assertTrue(any(a.text for a in fig.layout.annotations))


class LabelChartTests(unittest.TestCase):
    """Label charts must plot the counts/durations they are handed."""

    def test_phase_counts_are_plotted_verbatim_in_canonical_order(self):
        counts = {"validate": 2, "understand": 5, "implement": 9}
        bar = build_label_phase_count_chart(counts).data[0]
        self.assertEqual(list(bar.x), ["understand", "implement", "validate"])
        self.assertEqual(list(bar.y), [5, 9, 2])

    def test_phase_colors_come_from_the_shared_taxonomy_palette(self):
        bar = build_label_phase_count_chart({"debug": 1, "report": 2}).data[0]
        self.assertEqual(
            list(bar.marker.color),
            [LABEL_PHASE_COLORS["debug"], LABEL_PHASE_COLORS["report"]],
        )

    def test_phase_durations_keep_one_decimal(self):
        bar = build_label_phase_duration_chart({"plan": 12.34}).data[0]
        self.assertEqual(list(bar.y), [12.3])

    def test_comparison_chart_pairs_both_sides_for_every_phase(self):
        fig = build_phase_count_comparison_chart({"plan": 3}, {"plan": 5, "debug": 2}, "ref", "cmp")
        by_name = {t.name: dict(zip(t.x, t.y, strict=True)) for t in fig.data}
        self.assertEqual(by_name["ref"], {"plan": 3, "debug": 0})
        self.assertEqual(by_name["cmp"], {"plan": 5, "debug": 2})


# --------------------------------------------------------------------------
# Degenerate shapes, light and dark
# --------------------------------------------------------------------------


def _one_agent() -> list[dict]:
    return [step(0, total=10, inp=8, output=2, tool_calls=[tc("Read", ms=100)])]


def _fifty_agents() -> list[dict]:
    return [step(i, agent=f"ag{i}", session_id=f"s{i}", total=10, inp=8, output=2) for i in range(50)]


def _long_run() -> list[dict]:
    return [balanced(i, inp=100 + i, output=10, tool_calls=[tc("Read", ms=100)]) for i in range(173)]


SHAPES = {
    "empty": [],
    "one_step": _one_agent(),
    "all_zero_tokens": [step(i) for i in range(4)],
    "no_duration": [step(i, duration=None, total=10, inp=8, output=2) for i in range(4)],
    "out_of_order": [step(i, total=5, inp=4, output=1) for i in (5, 3, 4, 0, 1, 2)],
    "duplicate_index": [step(0, total=5), step(0, total=5), step(1, total=5)],
    "fifty_agents": _fifty_agents(),
    "long_run_173": _long_run(),
}

BUILDERS = {
    "token": lambda s, d: build_token_chart(s, dark=d),
    "token_opencode": lambda s, d: build_token_chart(s, dark=d, format="opencode"),
    "duration": lambda s, d: build_duration_chart(s, dark=d),
    "tool": lambda s, d: build_tool_chart(s, dark=d),
    "tool_duration": lambda s, d: build_tool_duration_chart(s, dark=d),
    "skill": lambda s, d: build_skill_agent_chart(s, dark=d),
    "swimlane": lambda s, d: build_agent_swimlane_chart(s, dark=d),
    "tool_outcome": lambda s, d: build_tool_outcome_timeline(s, dark=d),
    "context_pressure": lambda s, d: build_context_pressure_chart(s, dark=d),
}


class DegenerateShapeTests(unittest.TestCase):
    """Real traces are ragged; no chart may crash or emit a non-finite value."""

    def test_every_builder_survives_every_shape_in_both_themes(self):
        for shape, steps in SHAPES.items():
            for name, build in BUILDERS.items():
                for dark in (False, True):
                    with self.subTest(shape=shape, chart=name, dark=dark):
                        fig = build(steps, dark)
                        assert_finite(self, fig, f"{name}/{shape}/dark={dark}")

    def test_dark_variant_repaints_the_figure_background(self):
        for name, build in BUILDERS.items():
            with self.subTest(chart=name):
                dark = build(_long_run(), True)
                self.assertEqual(dark.layout.paper_bgcolor, "rgba(0,0,0,0)")

    def test_fifty_agents_still_draw_one_lane_each(self):
        fig = build_agent_swimlane_chart(_fifty_agents())
        self.assertEqual(len({tr.y[0] for tr in fig.data}), 50)

    def test_long_run_keeps_one_bar_per_step(self):
        steps = _long_run()
        per = {}
        for tr in build_token_chart(steps).data:
            for x, y in zip(tr.x or [], tr.y or [], strict=True):
                per[x] = per.get(x, 0) + y
        self.assertEqual(len(per), len(steps))


if __name__ == "__main__":
    unittest.main()
