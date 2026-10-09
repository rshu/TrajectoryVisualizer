"""Every metric must name its own domain: its denominator, its time window,
and the predicate that decided an outcome.

These are not value tests. Each one pins a *contract* that a reader of the
metrics dict would otherwise have to reconstruct from the source:

- a tool call's outcome is decided by one predicate, not by a second status
  allow-list that happens to agree with it;
- the window a duration is MEASURED over is the same window the call was
  ADMITTED on, so parallel calls union correctly whichever stamp family the
  export used;
- an average and a median over "step tokens" share a denominator, or they are
  two different statistics wearing one name;
- rate metrics disclose whether they are per model-time or per wall-clock,
  because the two bases can differ by more than an order of magnitude.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

from trajviz.insight import metrics as metrics_mod
from trajviz.insight.metrics import (
    _compute_timing_metrics,
    build_message_metrics,
    compute_metrics,
    non_spawn_tool_seconds,
    tool_call_duration_ms,
)


def _step(index, *, role="assistant", duration=1.0, tool_calls=None, tokens=None,
          created_ms=None, completed_ms=None):
    calls = tool_calls or []
    tok = {"total": 0, "input": 0, "output": 0, "reasoning": 0,
           "cache_read": 0, "cache_write": 0}
    tok.update(tokens or {})
    return {
        "index": index,
        "role": role,
        "duration": duration,
        "parts": [],
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "finish": "stop",
        "agent": "",
        "model_id": "",
        "tokens": tok,
        "time_created_ms": index * 1000 if created_ms is None else created_ms,
        "time_completed_ms": (index * 1000 + int(duration * 1000)
                              if completed_ms is None else completed_ms),
    }


def _metrics(steps, raw=None):
    return compute_metrics(steps, raw or {}, build_message_metrics(steps))


def _if_chain_branch_bodies(node: ast.If) -> list[list[ast.stmt]]:
    """Bodies of every branch of ONE if/elif/else chain, in source order."""
    bodies: list[list[ast.stmt]] = []
    cur = node
    while True:
        bodies.append(cur.body)
        orelse = cur.orelse
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            cur = orelse[0]  # `elif` — same chain, keep walking
            continue
        if orelse:
            bodies.append(orelse)
        return bodies


class OneOutcomePredicate(unittest.TestCase):
    """`tool_failure.tool_call_failed` is the single source of truth for
    whether a tool call failed. A second status allow-list inside
    `_compute_tool_stats` is either redundant (two branches, one body) or a
    silent disagreement with the predicate. Neither is acceptable, and the
    structural check catches both without having to guess the status strings.
    """

    def test_no_branch_of_one_if_chain_repeats_another_branches_body(self):
        src = textwrap.dedent(inspect.getsource(metrics_mod._compute_tool_stats))
        tree = ast.parse(src)
        # An `elif` is the single If inside its parent's orelse; skip those as
        # chain heads so each chain is examined exactly once.
        continuations = {
            id(n.orelse[0])
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or id(node) in continuations:
                continue
            dumps = [
                "\n".join(ast.dump(stmt) for stmt in body)
                for body in _if_chain_branch_bodies(node)
            ]
            with self.subTest(line=node.lineno):
                self.assertEqual(
                    len(dumps), len(set(dumps)),
                    "two branches of one if-chain have identical bodies, so one "
                    "of them is dead: collapse it instead of implying a "
                    "distinction the code does not make",
                )

    def test_a_call_with_no_failure_signal_scores_success_whatever_its_status(self):
        # Behaviour lock, not a bug repro: this passes before and after the
        # collapse. It records the intent -- a call carrying no failure signal
        # is optimistically a success, so an in-flight or unstamped call does
        # not depress the rate.
        for status in ("?", "unknown", "", None, "pending", "running"):
            with self.subTest(status=status):
                m = _metrics([_step(0, tool_calls=[{"tool_name": "bash", "status": status}])])
                self.assertEqual(m["tool_success_rate"], 100.0)
                self.assertEqual(m["tool_fail"], 0)

    def test_a_failure_signal_outranks_a_blank_status(self):
        m = _metrics([_step(0, tool_calls=[
            {"tool_name": "bash", "status": "?", "error_type": "timeout"},
        ])])
        self.assertEqual(m["tool_success_rate"], 0.0)
        self.assertEqual(m["tool_fail"], 1)


class ToolWindowPrecedence(unittest.TestCase):
    """`non_spawn_tool_seconds` admits a call into its union using
    `tool_call_duration_ms`, which prefers the WIDE part-level window. It then
    has to measure that same window; measuring the narrower state-level pair
    instead made the gate and the measurement two different quantities, and
    sent part-stamped parallel calls down the naive-sum path.
    """

    def test_union_measures_the_same_window_the_gate_admitted(self):
        tc = {"tool_name": "bash", "status": "success",
              "time_created": 0, "time_updated": 30_000,
              "time_start": 10_000, "time_end": 20_000}
        step = _step(0, duration=40.0, tool_calls=[tc])
        self.assertAlmostEqual(non_spawn_tool_seconds(step), 30.0, places=3)
        self.assertAlmostEqual(
            non_spawn_tool_seconds(step), tool_call_duration_ms(tc) / 1000.0, places=3
        )

    def test_parallel_part_stamped_calls_are_counted_once(self):
        part = [{"tool_name": "bash", "status": "success",
                 "time_created": 0, "time_updated": 5_000} for _ in range(2)]
        state = [{"tool_name": "bash", "status": "success",
                  "time_start": 0, "time_end": 5_000} for _ in range(2)]
        # The two stamp families must give the same answer for the same shape.
        self.assertAlmostEqual(non_spawn_tool_seconds(_step(0, tool_calls=part)), 5.0, places=3)
        self.assertAlmostEqual(non_spawn_tool_seconds(_step(0, tool_calls=state)), 5.0, places=3)

    def test_the_union_spans_both_stamp_families(self):
        step = _step(0, duration=20.0, tool_calls=[
            {"tool_name": "bash", "status": "success", "time_created": 0, "time_updated": 5_000},
            {"tool_name": "bash", "status": "success", "time_start": 2_000, "time_end": 8_000},
        ])
        self.assertAlmostEqual(non_spawn_tool_seconds(step), 8.0, places=3)

    def test_a_call_with_no_window_still_contributes_its_length(self):
        # Claude Code stamps no window at all, only metadata.totalDurationMs.
        # There is no union to compute, so the duration is summed -- an upper
        # bound, and the one case the precedence fix cannot improve.
        step = _step(0, duration=20.0, tool_calls=[
            {"tool_name": "bash", "status": "success", "metadata": {"totalDurationMs": 4_000}},
            {"tool_name": "bash", "status": "success", "metadata": {"totalDurationMs": 4_000}},
        ])
        self.assertAlmostEqual(non_spawn_tool_seconds(step), 8.0, places=3)


class StepTokenDenominators(unittest.TestCase):
    """`avg_tokens_per_step` divides by ALL steps while `median_step_tokens`
    and `p95_step_tokens` are taken over assistant steps only, so the three
    "per step" figures are not comparable. The existing key keeps its
    denominator (its values have already been reported); the comparable one
    is a sibling.
    """

    def setUp(self):
        steps = []
        for i in range(3):
            steps.append(_step(2 * i, role="user", tokens={"total": 0}))
            steps.append(_step(2 * i + 1, tokens={"total": 1000, "output": 1000}))
        self.m = _metrics(steps)

    def test_avg_and_median_step_tokens_share_a_denominator(self):
        self.assertEqual(self.m["avg_tokens_per_assistant_step"], 1000)
        self.assertEqual(self.m["median_step_tokens"], 1000)
        self.assertEqual(self.m["p95_step_tokens"], 1000)

    def test_the_all_steps_average_is_still_published(self):
        # Continuity: callers already read this key, so it must not move.
        self.assertEqual(self.m["avg_tokens_per_step"], 500)
        self.assertEqual(self.m["assistant_steps"], 3)
        self.assertEqual(self.m["total_steps"], 6)

    def test_an_empty_session_reports_zero_for_both(self):
        m = _metrics([])
        self.assertEqual(m["avg_tokens_per_step"], 0)
        self.assertEqual(m["avg_tokens_per_assistant_step"], 0)

    def test_a_user_only_session_reports_zero_rather_than_dividing_by_none(self):
        m = _metrics([_step(0, role="user", tokens={"total": 0})])
        self.assertEqual(m["avg_tokens_per_assistant_step"], 0)


class LatencyMetricsNameWhatTheyMeasure(unittest.TestCase):
    """`time_to_first_token` is the first user message to the first assistant
    message COMPLETED -- a first-response latency. No supported export records
    token-level or streaming timing, so a true TTFT is not computable and the
    key must not be described as one.
    """

    def test_first_response_latency_measures_message_completion_not_a_first_token(self):
        steps = [
            _step(0, role="user", duration=0.0, created_ms=0, completed_ms=0),
            _step(1, duration=5.0, created_ms=10, completed_ms=5_010),
        ]
        t = _compute_timing_metrics(steps)
        self.assertEqual(t["time_to_first_token"], 5.01)
        # Explicitly NOT the assistant step's own creation offset: if this were
        # a time-to-first-token it would be 0.01, not the whole turn.
        self.assertNotEqual(t["time_to_first_token"], 0.01)

    def test_the_timing_contract_does_not_claim_a_token_level_measurement(self):
        doc = _compute_timing_metrics.__doc__ or ""
        self.assertNotIn("TTFT", doc)
        self.assertNotIn("TTLT", doc)
        self.assertIn("completed", doc)
        self.assertIn("latency", doc)


class RateMetricsDiscloseTheirBase(unittest.TestCase):
    """Two denominator families live in one dict: per-model-time (summed step
    durations) and per-session-wall-clock. They differ whenever work overlaps,
    at times by more than 10x, so the contract has to say which is which.
    """

    def test_the_metrics_contract_names_both_denominators(self):
        doc = compute_metrics.__doc__ or ""
        self.assertIn("summed step durations", doc)
        self.assertIn("wall-clock", doc)
        self.assertIn("tokens_per_second", doc)
        self.assertIn("tool_wait_share", doc)

    def test_tool_shares_stay_on_the_session_wall_clock(self):
        # Behaviour lock (passes today): the tripwire that fires if anyone
        # "unifies" the bases by moving tool shares onto summed step durations.
        # A parent blocked 60s on `task` overlaps its child's 60s of real tool
        # work, so summed durations (120s) exceed the 65s the session actually
        # took; only the wall-clock base keeps the share a share.
        steps = [
            _step(0, duration=60.0, created_ms=0, completed_ms=60_000, tool_calls=[
                {"tool_name": "task", "status": "success", "time_start": 0, "time_end": 60_000},
            ]),
            _step(1, duration=60.0, created_ms=5_000, completed_ms=65_000, tool_calls=[
                {"tool_name": "bash", "status": "success", "time_start": 5_000, "time_end": 65_000},
            ]),
        ]
        m = _metrics(steps)
        self.assertGreater(m["total_duration"], m["wall_clock"])
        self.assertLessEqual(m["tool_time_fraction"], 1.0)
        self.assertLessEqual(m["tool_wait_share"], 100.0)
        # And the base is the wall clock, not the summed durations.
        self.assertAlmostEqual(
            m["tool_time_fraction"], m["tool_time_total"] / m["wall_clock"], places=3
        )


if __name__ == "__main__":
    unittest.main()
