"""The pattern module's predicates must agree with each other and say what they mean.

The first four defects below are all in `trajviz/insight/patterns.py` and were
verified against real exports as moving no reported metric. Two further classes
at the end of this file cover fixes that DO move one — the
recovery walk's failure definition and the validation-command vocabulary — and
each states its measured delta in its own docstring.

The four behaviour-preserving ones:

* The plan vocabulary was written twice. `_PLAN_TOOL_NAMES` matched exact
  spellings for the phase classifier, while `extract_plan_history` matched its
  own lowercase tuple, so `TodoUpdate` set the plan phase but produced no plan
  history and `todo_write` did the exact opposite. No real export observed emits either
  spelling (only `TodoWrite` and `todowrite` occur), so unifying them is
  behaviour-preserving on every reported figure and only fixes the latent case.
* `detect_phase_anomalies` called a span fraction `confidence`: it is the share
  of the trajectory the regressed phase covers, so it *falls* as the evidence
  grows. Nothing reads the key, so the honest name is free.
* `patterns.compute_autonomy_ratio` was dead, and disagreed with the live
  definition in `metrics.py` (which divides by user+assistant turns, not by all
  steps — real exports carry a third `developer` role, so the two differ on Codex).

The fourth, `_is_fruitless_step` treating a missing output as an empty one, is
characterised here rather than fixed: the fix would move `fruitless_streaks`
and the wasted-step total the Overview reports. See the comment at its
definition.
"""

from __future__ import annotations

import unittest

from trajviz.insight import patterns
from trajviz.insight.metrics import build_message_metrics, compute_metrics
from trajviz.insight.patterns import (
    _VALIDATION_COMMAND_PATTERNS,
    _is_validation_command,
    _step_has_success,
    classify_structural_phase,
    detect_fruitless_streaks,
    detect_phase_anomalies,
    extract_plan_history,
)


def _step(index: int, role: str = "assistant", calls: list[dict] | None = None, **extra) -> dict:
    """One parsed step, carrying only the keys the predicates under test read."""
    step = {
        "index": index,
        "role": role,
        "tool_calls": calls or [],
        "tool_call_count": len(calls or []),
        "tokens": {"total": 0, "input": 0, "output": 0,
                   "reasoning": 0, "cache_read": 0, "cache_write": 0},
        "parts": [],
        "finish": "stop",
    }
    step.update(extra)
    return step


def _plan_call(tool_name: str, *contents: str) -> dict:
    todos = [{"content": c, "status": "in_progress"} for c in contents]
    return {"tool_name": tool_name, "status": "success", "input": {"todos": todos}, "output": "ok"}


class PlanToolVocabularyIsShared(unittest.TestCase):
    """The phase classifier and extract_plan_history must accept the same names."""

    def test_todoupdate_produces_plan_history(self):
        """Set the plan phase but drop the snapshot: the original disagreement."""
        steps = [_step(0, calls=[_plan_call("TodoUpdate", "write the fix")])]
        self.assertEqual(classify_structural_phase(steps[0]), "plan")
        history = extract_plan_history(steps)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["items"], [{"content": "write the fix", "status": "in_progress"}])

    def test_snake_case_todo_write_sets_the_plan_phase(self):
        """The disagreement in the other direction: history but no plan phase."""
        steps = [_step(0, calls=[_plan_call("todo_write", "write the fix")])]
        self.assertEqual(classify_structural_phase(steps[0]), "plan")
        self.assertEqual(len(extract_plan_history(steps)), 1)

    def test_lowercase_spellings_are_recognised(self):
        """A future export may lowercase the name; one vocabulary, case-folded."""
        for name in ("todoupdate", "tasklist", "enterplanmode", "taskcreate"):
            with self.subTest(name=name):
                self.assertEqual(classify_structural_phase(_step(0, calls=[_plan_call(name)])), "plan")

    def test_plan_mode_tool_without_todos_adds_no_snapshot(self):
        """Guard against over-collection: unifying the names must not invent plans."""
        steps = [_step(0, calls=[{"tool_name": "EnterPlanMode", "status": "success", "input": {}}])]
        self.assertEqual(classify_structural_phase(steps[0]), "plan")
        self.assertEqual(extract_plan_history(steps), [])


class PhaseAnomalySpanIsNotConfidence(unittest.TestCase):
    def test_phase_anomaly_reports_span_not_confidence(self):
        steps = [_step(i) for i in range(10)]
        phases = [
            {"name": "validate", "start_idx": 0, "end_idx": 4},
            {"name": "implement", "start_idx": 5, "end_idx": 9},
        ]
        anomalies = detect_phase_anomalies(steps, phases)
        self.assertEqual(len(anomalies), 1)
        entry = anomalies[0]
        self.assertEqual(entry["span_fraction"], 0.5)
        self.assertNotIn("confidence", entry)
        # The prose already described it correctly; keep the two in step.
        self.assertIn("50.0% of trajectory", entry["explanation"])

    def test_span_fraction_grows_with_the_regression_not_with_certainty(self):
        """Pins why the old name was wrong: a longer regression scores higher."""
        short = detect_phase_anomalies(
            [_step(i) for i in range(10)],
            [{"name": "validate", "start_idx": 0, "end_idx": 7},
             {"name": "implement", "start_idx": 8, "end_idx": 9}],
        )
        self.assertEqual(short[0]["span_fraction"], 0.2)


class AutonomyRatioHasOneDefinition(unittest.TestCase):
    def test_patterns_no_longer_defines_autonomy_ratio(self):
        self.assertFalse(
            hasattr(patterns, "compute_autonomy_ratio"),
            "patterns.compute_autonomy_ratio is dead and uses a different denominator "
            "than the live metrics.py definition",
        )

    def test_live_definition_stays_a_fraction_with_a_third_role(self):
        """Codex exports carry `developer` steps; the live formula ignores them."""
        steps = [
            _step(0, role="user"),
            _step(1, role="assistant"),
            _step(2, role="developer"),
            _step(3, role="assistant"),
        ]
        ratio = compute_metrics(steps, {}, build_message_metrics(steps))["autonomy_ratio"]
        self.assertTrue(0.0 <= ratio <= 1.0)
        self.assertEqual(ratio, round(2 / 3, 4))


class FruitlessSearchTreatsMissingOutputAsEmpty(unittest.TestCase):
    """Characterisation, NOT a fix: `fruitless_streaks` is a reported number.

    `_is_fruitless_step` cannot distinguish "the search returned nothing" from
    "the call never resolved, so nothing was recorded" — both read as empty.
    `parse_steps` always materialises `output`, so the only real exposure is an
    unresolved or failed search call (cursor's `status: unknown`, three opencode
    `status: error` steps). Tightening the predicate would shorten streaks and
    the wasted-step total the Overview reports, so it stays, pinned here.
    """

    def test_unresolved_search_calls_still_count_as_fruitless(self):
        steps = [
            _step(i, calls=[{"tool_name": "Grep", "status": "unknown", "input": {"pattern": "x"}}])
            for i in range(3)
        ]
        streaks = detect_fruitless_streaks(steps)
        self.assertEqual([s["length"] for s in streaks], [3])

    def test_failed_search_calls_still_count_as_fruitless(self):
        steps = [
            _step(i, calls=[{"tool_name": "Grep", "status": "error", "input": {"pattern": "x"}}])
            for i in range(3)
        ]
        self.assertEqual([s["length"] for s in detect_fruitless_streaks(steps)], [3])


class RecoveryWalkSharesOneFailureDefinition(unittest.TestCase):
    """`_step_has_success` must agree with every other failure surface.

    The local copy this replaces was case-sensitive and consulted the exit code
    only when the status was blank. OpenCode marks a bash invocation
    ``completed`` regardless of its exit status, so a failed command read as a
    *recovery* here while `tool_failure.tool_call_failed` — and therefore
    tool_success_rate, the cluster labels and the Workflow badges — called it a
    failure. PR #13 corrected that reading everywhere except this walk.

    This MOVES a reported value: `recovery_path` changes on OpenCode runs whose
    failed commands were read as recoveries, and the honest "no recovery found"
    count rises. claude_code and codex are bit-identical, because neither emits
    `metadata.exit`.
    """

    def test_a_completed_call_with_a_nonzero_exit_is_not_a_recovery(self):
        step = {"tool_calls": [{"tool_name": "bash", "status": "completed",
                                "metadata": {"exit": 1}}]}
        self.assertFalse(_step_has_success(step))

    def test_a_completed_call_with_a_zero_exit_is_a_recovery(self):
        step = {"tool_calls": [{"tool_name": "bash", "status": "completed",
                                "metadata": {"exit": 0}}]}
        self.assertTrue(_step_has_success(step))

    def test_status_matching_is_case_insensitive(self):
        for status in ("Error", "TIMEOUT", "Failed", "timed_out", "canceled"):
            with self.subTest(status=status):
                self.assertFalse(_step_has_success({"tool_calls": [{"status": status}]}))

    def test_an_error_field_alone_defeats_a_success_status(self):
        step = {"tool_calls": [{"status": "completed", "error": "boom"}]}
        self.assertFalse(_step_has_success(step))

    def test_a_step_with_no_tool_calls_is_not_a_recovery(self):
        # Load-bearing: detect_failure_patterns walks forward until a step
        # "has success", so a bare text step must not end the walk.
        self.assertFalse(_step_has_success({"tool_calls": []}))
        self.assertFalse(_step_has_success({}))

    def test_one_good_call_beside_a_failed_one_still_recovers(self):
        step = {"tool_calls": [{"status": "error"}, {"status": "success"}]}
        self.assertTrue(_step_has_success(step))


class ValidationPatternsAreCommandShaped(unittest.TestCase):
    """A validation pattern must name a tool, not an English word.

    `_is_validation_command` substring-matches the whole normalised command, so
    the bare entries "lint", "check" and "verify" fired on prose. The dominant
    false positive was a standalone word inside a quoted script body —
    `python -c "... # first verify the bug exists ..."` — which a word-boundary
    match would still have hit, so the verb has to stay attached to its tool.

    This MOVES a reported value: the validate phase shrinks and
    phase_regressions falls with it.
    """

    NOT_VALIDATION = (
        "git checkout -b feat/x",
        "cat docs/howto/deployment/checklist.txt",
        'python -c "import x  # check if attribute exists"',
        'python -c "# first verify the bug exists"',
        "grep -rn subprocess.check_call .",
        "python -c 'from astropy.io.fits import verify'",
        "ls source checkouts",
    )

    REAL_VALIDATION = (
        "python -m pytest tests/ -q",
        "ruff check trajviz",
        "cargo check",
        "npm run lint",
        "make lint",
        "golangci-lint run ./...",
        "pre-commit run --all-files",
        "git diff --check",
        "mypy trajviz",
        "go test ./...",
    )

    def test_prose_and_unrelated_commands_are_not_validation(self):
        for command in self.NOT_VALIDATION:
            with self.subTest(command=command):
                self.assertFalse(_is_validation_command(command))

    def test_real_validation_commands_still_match(self):
        for command in self.REAL_VALIDATION:
            with self.subTest(command=command):
                self.assertTrue(_is_validation_command(command))

    def test_no_pattern_is_a_bare_english_word(self):
        # The guard that keeps this class honest: a future bare verb would pass
        # the two tests above while reintroducing the whole defect class.
        bare = {"lint", "check", "verify", "test", "build", "run"}
        offenders = [p for p in _VALIDATION_COMMAND_PATTERNS if p in bare]
        self.assertEqual(offenders, [], f"bare English words match prose: {offenders}")


class StepCapsAgree(unittest.TestCase):
    """`patterns` keeps its own cap and cannot import `session` (import cycle).

    `session.build_loaded_session` truncates to `session.MAX_STEPS` before any
    detector runs, then `patterns` applies `_MAX_STEPS` again to the already-cut
    list. Harmless while they agree; if `patterns._MAX_STEPS` were ever the
    smaller of the two it would silently truncate a second time and every phase
    segment past that point would vanish with no warning anywhere. Asserted
    here because the dependency direction forbids a runtime check.
    """

    def test_patterns_cap_is_not_stricter_than_the_session_cap(self):
        from trajviz.insight.session import MAX_STEPS

        self.assertGreaterEqual(
            patterns._MAX_STEPS, MAX_STEPS,
            f"patterns._MAX_STEPS ({patterns._MAX_STEPS}) < session.MAX_STEPS "
            f"({MAX_STEPS}): detectors would re-truncate an already-truncated list",
        )


if __name__ == "__main__":
    unittest.main()
