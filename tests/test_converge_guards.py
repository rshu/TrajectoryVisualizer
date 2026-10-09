"""Regression tests for the 2026-10 Converge walkthrough gaps (C2-C9, D3).

Covers:
- C2: the empty-steps refusal names the real detected format instead of always
      degrading to 'unrecognised' (the key it read, ``_format``, has no writer).
- C3: harmful_ratio falls back to a count basis when the compared side carries
      no cost data, so "harmful extras with no token data" stops being
      indistinguishable from "no harmful extras".
- C4: pattern counting matches ``parent_type`` as well as ``type``, so the
      'write_retry' umbrella in PATTERN_DIRECTIONS is reachable.
- C5: the milestone name/label maps are single-sourced in milestones.py.
- C6: compute_anchor_analysis no longer advertises a custom_rules hook that no
      entry point can reach (classify_file keeps the real one).
- C8: the 'trajectory-converge' name cli.py advertises as its prog is a
      declared console script.
- C9: compare_segments' per-segment action lists are disjoint by step_index —
      the property that keeps re-entrant alignment sub-quadratic.
- D3 (converge half): effect labeling routes bare status strings through the
      shared tool_failure.status_failed predicate, so the alias statuses
      ('timeout', 'timed_out', 'canceled') are honoured.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from trajviz.converge import charts, milestones
from trajviz.converge.alignment import (
    build_comparison_report,
    compute_harmful_divergence,
)
from trajviz.converge.anchor import classify_file, compute_anchor_analysis
from trajviz.converge.canonical import (
    ActionCost,
    CanonicalAction,
    assign_effect_labels,
)
from trajviz.converge.divergence import count_patterns
from trajviz.converge.eval_layers import _extract_layer_metric
from trajviz.converge.intervention import PATTERN_DIRECTIONS, _count_pattern
from trajviz.converge.milestones import (
    extract_milestones,
    segment_by_milestones,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# C2: the refusal message names the format it actually detected
# ---------------------------------------------------------------------------

class EmptyStepsMessageNamesFormatTests(unittest.TestCase):
    """C2: ``raw.get("_format")`` has no writer, so the label always degraded."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.good = self.tmp / "good.json"
        self.good.write_text(json.dumps({"trajectory": [
            {"type": "assistant", "uuid": f"u{i}",
             "message": {"id": f"m{i}", "role": "assistant",
                         "content": [{"type": "text", "text": f"step {i}"}],
                         "usage": {"input_tokens": 10, "output_tokens": 5}}}
            for i in range(2)
        ]}))

    def test_recognised_format_is_named_in_the_refusal(self):
        # A Claude Code export the loader recognises but that yields no steps.
        empty_cc = self.tmp / "empty_cc.json"
        empty_cc.write_text(json.dumps(
            {"format": "ccsession-trajectory", "trajectory": []}))
        with self.assertRaises(ValueError) as ctx:
            build_comparison_report(str(empty_cc), str(self.good))
        msg = str(ctx.exception)
        self.assertIn("no steps parsed", msg)
        self.assertIn("Claude Code", msg)
        self.assertNotIn("unrecognised", msg)

    def test_non_trajectory_object_still_reports_unknown(self):
        not_a_trajectory = self.tmp / "unknown.json"
        not_a_trajectory.write_text('{"hello": "world"}')
        with self.assertRaises(ValueError) as ctx:
            build_comparison_report(str(not_a_trajectory), str(self.good))
        self.assertIn("detected format: unknown", str(ctx.exception))


# ---------------------------------------------------------------------------
# C3: harmful_ratio count fallback
# ---------------------------------------------------------------------------

def _cmd(step_index: int, effect_label: str, token_share: int = 0) -> CanonicalAction:
    return CanonicalAction(
        step_index=step_index,
        action_type="COMMAND",
        target="pytest",
        effect_label=effect_label,
        cost=ActionCost(token_share=token_share),
    )


class HarmfulRatioCountFallbackTests(unittest.TestCase):
    """C3: 0.0 was overloaded — "no harm" and "no cost data" looked identical."""

    def test_zero_cost_harmful_extras_use_a_count_basis(self):
        actions = [_cmd(i, "failed") for i in range(3)] + [_cmd(i, "survived") for i in (3, 4)]
        result = compute_harmful_divergence([0, 1, 2], actions)
        self.assertEqual(result["harmful_ratio"], 0.6)
        # harmful_cost keeps reporting the genuine (zero) token/latency totals.
        self.assertEqual(result["harmful_cost"], {"tokens": 0, "latency_ms": 0})

    def test_costed_path_is_unchanged(self):
        actions = [_cmd(i, "failed", token_share=100) for i in range(3)]
        result = compute_harmful_divergence([0, 1, 2], actions)
        self.assertEqual(result["harmful_ratio"], 1.0)
        self.assertEqual(result["harmful_cost"]["tokens"], 300)

    def test_fallback_cannot_manufacture_harm(self):
        actions = [_cmd(i, "survived") for i in range(4)]
        result = compute_harmful_divergence([0, 1], actions)
        self.assertEqual(result["harmful_ratio"], 0.0)


# ---------------------------------------------------------------------------
# C4: pattern counting honours parent_type
# ---------------------------------------------------------------------------

_WRITE_RETRY_PATTERNS = [
    {"type": "reverted_and_rewritten", "parent_type": "write_retry"},
    {"type": "iterative_refinement", "parent_type": "write_retry"},
]


class PatternCountingIsUnifiedTests(unittest.TestCase):
    """C4: intervention counted `type` only, so 'write_retry' always read 0."""

    def test_umbrella_parent_type_is_counted(self):
        report = {"patterns": list(_WRITE_RETRY_PATTERNS)}
        self.assertEqual(_count_pattern(report, "write_retry"), 2)

    def test_intervention_and_eval_layers_agree_on_every_direction_key(self):
        report = {"patterns": list(_WRITE_RETRY_PATTERNS)}
        for ptype in PATTERN_DIRECTIONS:
            self.assertEqual(
                _count_pattern(report, ptype),
                _extract_layer_metric(
                    {"source": "patterns", "pattern_type": ptype},
                    report["patterns"], None),
                f"{ptype} counted differently by intervention and eval_layers",
            )

    def test_count_patterns_matches_plain_type_too(self):
        self.assertEqual(
            count_patterns(_WRITE_RETRY_PATTERNS, "reverted_and_rewritten"), 1)
        self.assertEqual(count_patterns(_WRITE_RETRY_PATTERNS, "broad_exploration"), 0)


class PatternDeltasReachTheUmbrellaTests(unittest.TestCase):
    """C4 leg 2: compute_pattern_deltas never considered parent_type as a key."""

    def _paired(self):
        from trajviz.converge.batch import BatchResult
        from trajviz.converge.intervention import compute_pattern_deltas
        before = BatchResult(task_id="t1", report={"patterns": list(_WRITE_RETRY_PATTERNS)})
        after = BatchResult(task_id="t1", report={"patterns": [_WRITE_RETRY_PATTERNS[0]]})
        return compute_pattern_deltas([(before, after)])

    def test_write_retry_key_is_emitted(self):
        self.assertIn("write_retry", self._paired())

    def test_existing_keys_keep_their_frequencies(self):
        deltas = self._paired()
        self.assertEqual(deltas["reverted_and_rewritten"]["before_frequency"], 1.0)
        self.assertEqual(deltas["reverted_and_rewritten"]["after_frequency"], 1.0)
        self.assertEqual(deltas["iterative_refinement"]["before_frequency"], 1.0)
        self.assertEqual(deltas["iterative_refinement"]["after_frequency"], 0.0)


# ---------------------------------------------------------------------------
# C5: milestone labels are single-sourced
# ---------------------------------------------------------------------------

class MilestoneLabelsAreSingleSourcedTests(unittest.TestCase):
    """C5: charts.py and rendering.py each carried a private copy."""

    def test_charts_reuses_the_milestones_maps(self):
        self.assertIs(charts._MILESTONE_LABELS, milestones.MILESTONE_LABELS)
        self.assertIs(charts._MILESTONE_NAMES, milestones.MILESTONE_NAMES)

    def test_rendering_imports_the_shared_map(self):
        from trajviz.converge import rendering
        self.assertIs(rendering.MILESTONE_LABELS, milestones.MILESTONE_LABELS)

    def test_maps_cover_exactly_the_keys_extract_milestones_produces(self):
        produced = set(extract_milestones([]))
        self.assertEqual(set(milestones.MILESTONE_NAMES), produced)
        self.assertEqual(set(milestones.MILESTONE_LABELS), produced)

    def test_every_label_appears_in_the_rendered_milestone_table(self):
        from trajviz.converge.rendering import build_comparison_report_html
        full = {name: i for i, name in enumerate(milestones.MILESTONE_NAMES)}
        html = build_comparison_report_html(
            {"outcome": {}, "patterns": [], "ref_milestones": full, "cmp_milestones": full})
        for label in milestones.MILESTONE_LABELS.values():
            self.assertIn(label, html)


# ---------------------------------------------------------------------------
# C6: the unreachable custom_rules hook is gone; classify_file keeps its own
# ---------------------------------------------------------------------------

class AnchorCustomRulesHookTests(unittest.TestCase):
    """C6: compute_anchor_analysis -> classify_anchor_files was always None."""

    def test_compute_anchor_analysis_takes_no_rules_parameter(self):
        import inspect
        params = list(inspect.signature(compute_anchor_analysis).parameters)
        self.assertEqual(params, ["ref_actions", "cmp_actions", "anchor_files"])

    def test_classify_file_keeps_the_real_extension_point(self):
        self.assertEqual(
            classify_file("src/widget.py", [("**/widget.py", "generated")]),
            "generated")
        self.assertEqual(classify_file("src/widget.py"), "source")

    def test_file_classes_output_is_unchanged(self):
        analysis = compute_anchor_analysis(
            [], [], {"src/app.py", "tests/test_app.py", "api/schema.proto"})
        self.assertEqual(analysis["file_classes"],
                         {"source": 1, "generated": 0, "test": 1, "fixture": 0, "spec": 1})


# ---------------------------------------------------------------------------
# C8: the advertised console script exists
# ---------------------------------------------------------------------------

def _cli_prog() -> str | None:
    """The prog name ``cli.main()`` puts in its own --help / error output.

    Captured by spying on ArgumentParser rather than shelling out, so the test
    stays fast and needs no install. ``main()`` with no mode selected calls
    ``parser.error``, which raises SystemExit after the parser is built.
    """
    from trajviz.converge import cli

    captured: dict[str, str | None] = {}
    real_init = argparse.ArgumentParser.__init__

    def spy(self, *args, **kwargs):
        captured.setdefault("prog", kwargs.get("prog"))
        real_init(self, *args, **kwargs)

    with mock.patch.object(argparse.ArgumentParser, "__init__", spy), \
            mock.patch.object(sys, "argv", ["prog-under-test"]), \
            contextlib.redirect_stderr(io.StringIO()), \
            contextlib.suppress(SystemExit):
        cli.main()
    return captured.get("prog")


class ConvergeConsoleScriptTests(unittest.TestCase):
    """C8: cli.py advertised prog='trajectory-converge' with no such script."""

    def _scripts(self) -> dict[str, str]:
        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f).get("project", {}).get("scripts", {})

    def test_console_script_is_declared(self):
        self.assertEqual(self._scripts().get("trajectory-converge"),
                         "trajviz.converge.cli:main")

    def test_declared_target_is_importable_and_callable(self):
        import importlib
        self.assertTrue(callable(importlib.import_module("trajviz.converge.cli").main))

    def test_advertised_prog_is_a_declared_script(self):
        self.assertIn(_cli_prog(), self._scripts())


# ---------------------------------------------------------------------------
# C9: segment disjointness keeps re-entrant alignment sub-quadratic
# ---------------------------------------------------------------------------

class SegmentDisjointnessTests(unittest.TestCase):
    """C9: compare_segments re-enters align_trajectories per paired segment.

    The re-entry is additive rather than multiplicative only because the
    segments partition the actions. Pinned here instead of a wall-clock bound,
    which would be flaky in CI.
    """

    def test_segments_partition_actions_by_step_index(self):
        actions = [
            CanonicalAction(step_index=0, action_type="FILE_READ", target="/r/a.py",
                            effect_label="justified"),
            CanonicalAction(step_index=1, action_type="FILE_WRITE", target="/r/a.py",
                            effect_label="survived"),
            CanonicalAction(step_index=2, action_type="COMMAND", target="pytest",
                            effect_label="justified"),
            CanonicalAction(step_index=3, action_type="FILE_WRITE", target="/r/b.py",
                            effect_label="survived"),
        ]
        ms = extract_milestones(actions)
        segments = segment_by_milestones(actions, ms)
        seen: set[int] = set()
        for seg in segments:
            indices = {a.step_index for a in seg["actions"]}
            self.assertFalse(seen & indices, f"segment {seg['label']!r} overlaps an earlier one")
            seen |= indices


# ---------------------------------------------------------------------------
# D3 (converge half): bare status strings go through status_failed
# ---------------------------------------------------------------------------

class EffectLabelHonoursStatusAliasesTests(unittest.TestCase):
    """D3: canonical.py gated on a narrow 5-status tuple, not FAILURE_STATUSES."""

    def _label(self, status: str) -> str:
        a = CanonicalAction(step_index=0, action_type="COMMAND", target="pytest",
                            tool="Bash", status=status)
        steps = [{"index": 0, "tool_calls": [{"tool_name": "Bash", "status": status}]}]
        assign_effect_labels([a], steps, anchor_files={"zzz.py"})
        return a.effect_label

    def test_alias_statuses_are_failures(self):
        for status in ("timed_out", "canceled", "TIMEOUT", "Failure"):
            with self.subTest(status=status):
                self.assertEqual(self._label(status), "failed")

    def test_already_recognised_statuses_still_fail(self):
        for status in ("error", "failed", "failure", "cancelled", "timeout"):
            with self.subTest(status=status):
                self.assertEqual(self._label(status), "failed")

    def test_clean_status_is_not_a_failure(self):
        self.assertNotEqual(self._label("completed"), "failed")

    def test_error_text_fallback_fires_for_alias_statuses(self):
        # The failing call's output names a file; a later read of that file is
        # 'justified' by the error reference — but only if the alias status is
        # recognised as a failure at the error-text fallback.
        actions = [
            CanonicalAction(step_index=0, action_type="COMMAND", target="pytest",
                            tool="Bash", status="timed_out"),
            CanonicalAction(step_index=1, action_type="FILE_READ", target="/repo/mod.py",
                            tool="Read"),
        ]
        steps = [
            {"index": 0, "tool_calls": [{
                "tool_name": "Bash", "status": "timed_out",
                "output": 'Traceback: File "/repo/mod.py", line 3',
            }]},
            {"index": 1, "tool_calls": [{"tool_name": "Read",
                                         "input": {"file_path": "/repo/mod.py"}}]},
        ]
        assign_effect_labels(actions, steps, anchor_files={"zzz.py"})
        self.assertEqual(actions[1].effect_label, "justified")
        self.assertEqual(actions[1].effect_detail["reason"], "error_reference")


if __name__ == "__main__":
    unittest.main()
