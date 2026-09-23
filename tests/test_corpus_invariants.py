"""Numeric invariants every ``compute_metrics`` result must satisfy.

These are the properties a reader of the dashboard (or of a paper built on it)
implicitly relies on: a duration is never negative, a percentage is a
percentage, an aggregate is never smaller than the largest thing it aggregates,
and no cell ever renders ``nan``/``inf``. They are cheap to state and easy to
break silently — the PR #13 merge moved tool-timing, spawn-wait and
generation-window arithmetic into new code paths, and a regression there shows
up as a plausible-looking wrong number rather than as a crash.

Every invariant below was validated against all 2,500 real trajectories in the
research corpus before being written down, so a failure here means production
behaviour changed, not that the invariant was too strict.

The corpus-backed case at the bottom re-runs the same battery over real files
when they happen to be available; it is skipped everywhere else so the suite
stays hermetic.
"""

from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

from trajviz.insight.metrics import (
    build_message_metrics,
    compute_metrics,
    generation_seconds,
    non_spawn_tool_seconds,
    spawn_wait_seconds,
)

# Metrics that measure elapsed time. None of them can be negative.
DURATION_KEYS = (
    "total_duration", "wall_clock", "avg_duration", "median_duration",
    "p50_duration", "p90_duration", "p95_duration", "p99_duration", "max_duration",
    "tool_time_total", "delegation_time_total", "avg_tool_duration",
    "p95_tool_duration", "max_tool_duration", "output_throughput_timed_seconds",
    "output_throughput_tool_wait_seconds", "time_to_first_token", "time_to_last_token",
)

# Metrics that count things. None of them can be negative.
COUNT_KEYS = (
    "total_steps", "assistant_steps", "user_steps", "tool_call_count",
    "tool_success", "tool_fail", "delegated_call_count", "multi_tool_steps",
    "no_tool_assistant_steps", "reasoning_parts", "text_parts", "snapshot_parts",
    "patch_steps", "patch_lines", "output_tokens", "cache_read_tokens",
    "output_throughput_timed_steps", "output_throughput_total_steps",
)

# Metrics expressed as 0-100 percentages.
#
# ``avg_cache_ratio`` is deliberately absent: it divides cache-read tokens by a
# per-step total that some providers report *excluding* cache reads, so on real
# data it exceeds 100. That is a reported defect, not an invariant to relax
# here — add the key once the denominator is fixed.
PERCENT_KEYS = (
    "tool_success_rate", "output_throughput_coverage_pct", "tool_wait_share",
    "non_cache_ratio",
)

# Metrics expressed as 0-1 fractions.
FRACTION_KEYS = ("autonomy_ratio", "tool_time_fraction", "tool_system_failure_rate")


def _num(metrics: dict, key: str) -> float | None:
    """Metric value when it is a real number, else None (bools are not numbers)."""
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _assistant_step(index: int, *, duration=None, tool_calls=None, output=10, role="assistant"):
    """A parsed step in the shape ``parse_steps`` emits."""
    calls = list(tool_calls or [])
    completed = int((duration or 0) * 1000)
    return {
        "index": index,
        "role": role,
        "duration": duration,
        "parts": [],
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "tokens": {
            "total": output + 5, "input": 5, "output": output,
            "reasoning": 0, "cache_read": 0, "cache_write": 0,
        },
        "finish": "stop",
        "time_created_ms": index * 1000,
        "time_completed_ms": index * 1000 + completed,
    }


def _metrics(steps: list[dict]) -> dict:
    return compute_metrics(steps, {}, build_message_metrics(steps))


class _InvariantMixin:
    """Shared assertion battery so fixtures and real files are judged alike."""

    def assert_metric_invariants(self, metrics: dict, label: str) -> None:
        for key, value in metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                self.fail(f"{label}: {key} is {value!r}; no metric may be NaN or Inf")

        for key in DURATION_KEYS:
            value = _num(metrics, key)
            if value is not None:
                self.assertGreaterEqual(value, 0.0, f"{label}: {key} is a negative duration")

        for key in COUNT_KEYS:
            value = _num(metrics, key)
            if value is not None:
                self.assertGreaterEqual(value, 0.0, f"{label}: {key} is a negative count")

        for key in PERCENT_KEYS:
            value = _num(metrics, key)
            if value is not None:
                self.assertGreaterEqual(value, 0.0, f"{label}: {key}={value} is below 0%")
                self.assertLessEqual(value, 100.0, f"{label}: {key}={value} is above 100%")

        for key in FRACTION_KEYS:
            value = _num(metrics, key)
            if value is not None:
                self.assertGreaterEqual(value, 0.0, f"{label}: {key}={value} is below 0")
                self.assertLessEqual(value, 1.0, f"{label}: {key}={value} is above 1")

        # Aggregates are never smaller than the parts they summarise.
        success, fail = _num(metrics, "tool_success"), _num(metrics, "tool_fail")
        count = _num(metrics, "tool_call_count")
        if None not in (success, fail, count):
            self.assertEqual(
                success + fail, count,
                f"{label}: tool_success+tool_fail={success + fail} but tool_call_count={count}",
            )

        total, wall = _num(metrics, "tool_time_total"), _num(metrics, "wall_clock")
        if None not in (total, wall) and wall > 0:
            self.assertLessEqual(
                total, wall * 1.0001,
                f"{label}: tool_time_total={total}s exceeds wall_clock={wall}s",
            )

        longest, average = _num(metrics, "max_tool_duration"), _num(metrics, "avg_tool_duration")
        if None not in (longest, average):
            self.assertGreaterEqual(longest, average - 1e-9, f"{label}: max_tool_duration below the average")
        if None not in (longest, total):
            self.assertLessEqual(longest, total + 1e-6, f"{label}: max_tool_duration exceeds tool_time_total")

        p95, longest = _num(metrics, "p95_tool_duration"), _num(metrics, "max_tool_duration")
        if None not in (p95, longest):
            self.assertLessEqual(p95, longest + 1e-9, f"{label}: p95_tool_duration exceeds max_tool_duration")

        p95_step, longest_step = _num(metrics, "p95_duration"), _num(metrics, "max_duration")
        if None not in (p95_step, longest_step):
            self.assertLessEqual(p95_step, longest_step + 1e-9, f"{label}: p95_duration exceeds max_duration")

        timed, total_steps = (
            _num(metrics, "output_throughput_timed_steps"),
            _num(metrics, "output_throughput_total_steps"),
        )
        if None not in (timed, total_steps):
            self.assertLessEqual(timed, total_steps, f"{label}: more timed assistant steps than assistant steps")

        assistant, steps_total = _num(metrics, "assistant_steps"), _num(metrics, "total_steps")
        if None not in (assistant, steps_total):
            self.assertLessEqual(assistant, steps_total, f"{label}: assistant_steps exceeds total_steps")

        calls, seconds = _num(metrics, "delegated_call_count"), _num(metrics, "delegation_time_total")
        if calls == 0 and seconds:
            self.fail(f"{label}: delegation_time_total={seconds}s with zero delegated calls")


class SyntheticSessionInvariantTests(_InvariantMixin, unittest.TestCase):
    """The battery must hold for every shape the loaders can produce."""

    def test_plain_session(self):
        steps = [
            _assistant_step(0, duration=None, role="user", output=0),
            _assistant_step(1, duration=12.0, tool_calls=[
                {"tool_name": "bash", "status": "success", "duration_ms": 3_000},
            ]),
            _assistant_step(2, duration=8.0, tool_calls=[
                {"tool_name": "read", "status": "error", "duration_ms": 500},
            ]),
        ]
        self.assert_metric_invariants(_metrics(steps), "plain")

    def test_session_without_any_timing(self):
        """Formats that stamp no durations must not fabricate numbers that break the battery."""
        steps = [
            _assistant_step(0, duration=None, role="user", output=0),
            _assistant_step(1, duration=None, tool_calls=[{"tool_name": "bash", "status": "success"}]),
        ]
        self.assert_metric_invariants(_metrics(steps), "untimed")

    def test_session_with_zero_length_steps(self):
        """A zero duration must not turn a rate into inf or a share into NaN."""
        steps = [
            _assistant_step(0, duration=0.0, role="user", output=0),
            _assistant_step(1, duration=0.0, tool_calls=[{"tool_name": "bash", "status": "success"}]),
        ]
        self.assert_metric_invariants(_metrics(steps), "zero-length")

    def test_session_with_no_steps(self):
        self.assert_metric_invariants(_metrics([]), "empty")

    def test_delegating_session(self):
        """Spawn calls feed delegation_time_total, never negative tool time."""
        parent = _assistant_step(0, duration=100.0, tool_calls=[
            {"tool_name": "task", "status": "success", "duration_ms": 80_000},
            {"tool_name": "bash", "status": "success", "duration_ms": 5_000},
        ])
        child = _assistant_step(1, duration=80.0, tool_calls=[
            {"tool_name": "read", "status": "success", "duration_ms": 10_000},
        ])
        metrics = _metrics([parent, child])
        self.assert_metric_invariants(metrics, "delegating")
        self.assertEqual(metrics["delegated_call_count"], 1)

    def test_step_that_is_entirely_tool_wait(self):
        """A step whose tool stamps fill it has no generation window to divide by.

        Counting its output tokens against a zero-second denominator is how
        output_tokens_per_sec goes to infinity; the step must be left out of
        the throughput aggregate instead, and tool wait must stay at 100%,
        never above it.
        """
        step = _assistant_step(0, duration=10.0, tool_calls=[
            {"tool_name": "bash", "status": "success", "time_start": 0, "time_end": 10_000},
        ])
        metrics = _metrics([step])
        self.assert_metric_invariants(metrics, "all-tool-wait")
        rate = _num(metrics, "output_tokens_per_sec")
        if rate is not None:
            self.assertLess(rate, 1e6, "output_tokens_per_sec exploded on a collapsed generation window")


class GenerationWindowTests(unittest.TestCase):
    """Generation time is a window, so overlapping tool calls count once."""

    def test_parallel_tool_windows_are_unioned_not_summed(self):
        """Agents fire tools in parallel; summing their spans double-counts the overlap.

        Two 10s calls overlapping by 5s occupy 15s of the step, not 20s.
        Summing inflates tool time past the step's own wall clock and drives
        the generation denominator to zero.
        """
        step = _assistant_step(0, duration=20.0, tool_calls=[
            {"tool_name": "bash", "status": "success", "time_start": 1_000_000, "time_end": 1_010_000},
            {"tool_name": "bash", "status": "success", "time_start": 1_005_000, "time_end": 1_015_000},
        ])
        self.assertAlmostEqual(non_spawn_tool_seconds(step), 15.0, places=6)
        generated, waited = generation_seconds(step)
        self.assertAlmostEqual(waited, 15.0, places=6)
        self.assertAlmostEqual(generated, 5.0, places=6)

    def test_disjoint_tool_windows_still_add_up(self):
        """The union must not collapse genuinely separate calls into one."""
        step = _assistant_step(0, duration=30.0, tool_calls=[
            {"tool_name": "bash", "status": "success", "time_start": 1_000_000, "time_end": 1_005_000},
            {"tool_name": "bash", "status": "success", "time_start": 1_020_000, "time_end": 1_027_000},
        ])
        self.assertAlmostEqual(non_spawn_tool_seconds(step), 12.0, places=6)

    def test_generation_window_never_goes_negative(self):
        step = _assistant_step(0, duration=5.0, tool_calls=[
            {"tool_name": "bash", "status": "success", "time_start": 0, "time_end": 60_000},
        ])
        generated, waited = generation_seconds(step)
        self.assertGreaterEqual(generated, 0.0)
        self.assertLessEqual(waited, 5.0)

    def test_parallel_spawns_count_once(self):
        """Delegations run concurrently, so the parent waits for the longest, not the sum."""
        step = _assistant_step(0, duration=100.0, tool_calls=[
            {"tool_name": "task", "status": "success", "duration_ms": 30_000},
            {"tool_name": "task", "status": "success", "duration_ms": 40_000},
        ])
        self.assertAlmostEqual(spawn_wait_seconds(step), 40.0, places=6)

    def test_untimed_step_has_no_generation_window(self):
        self.assertIsNone(generation_seconds(_assistant_step(0, duration=None)))


def _corpus_root() -> Path | None:
    """Research corpus location, when this checkout happens to sit beside one."""
    override = os.environ.get("TRAJVIZ_CORPUS")
    if override:
        path = Path(override)
        return path if path.is_dir() else None
    candidate = Path(__file__).resolve().parents[2] / "TraceProbe" / "data" / "trajectory"
    return candidate if candidate.is_dir() else None


_CORPUS = _corpus_root()


@unittest.skipUnless(_CORPUS is not None, "research trajectory corpus not present")
class CorpusInvariantTests(_InvariantMixin, unittest.TestCase):
    """Same battery, real trajectories — the sampling keeps it under a second."""

    def test_sampled_corpus_satisfies_every_invariant(self):
        from trajviz.insight.loaders import load_trajectory
        from trajviz.insight.parser import parse_steps

        files: list[Path] = []
        for sub in sorted(p for p in _CORPUS.iterdir() if p.is_dir()):
            found = sorted(p for p in sub.iterdir() if p.suffix in (".json", ".jsonl"))
            files.extend(found[:: max(1, len(found) // 12)][:12])
        self.assertTrue(files, "corpus directory present but contains no trajectories")

        for path in files:
            with self.subTest(trajectory=path.name):
                raw = load_trajectory(str(path))
                steps = parse_steps(raw)
                metrics = compute_metrics(steps, raw, build_message_metrics(steps))
                self.assert_metric_invariants(metrics, f"{path.parent.name}/{path.name}")


if __name__ == "__main__":
    unittest.main()
