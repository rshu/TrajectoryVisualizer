"""A trajectory that fails to load must never become a zero-valued comparison.

`build_comparison_report` is the file-path entry point used by the Converge
CLI, the Converge app, and `batch.run_batch`. It previously fed the `_error`
dict straight through, so a missing/corrupt/unparseable file produced a
confident report of all zeros: the CLI printed a full summary and exited 0,
and `run_batch` counted the task as a SUCCESS and folded its fabricated zeros
into cross-task aggregate statistics (one bad file in four moved the reported
mean alignment_f1 by -25%). The Insight UI's `run_comparison` has always
refused these with `ok: False`; this pins the same guard for the file path.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trajviz.converge.alignment import build_comparison_report
from trajviz.converge.batch import ManifestEntry, aggregate_reports, run_batch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _min_trajectory(path: Path, *, steps: int = 2) -> Path:
    """Smallest Claude Code-shaped export the loader accepts."""
    entries = []
    for i in range(steps):
        entries.append({
            "type": "assistant",
            "uuid": f"u{i}",
            "message": {
                "id": f"m{i}",
                "role": "assistant",
                "content": [{"type": "text", "text": f"step {i}"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        })
    path.write_text(json.dumps({"trajectory": entries}))
    return path


class BuildComparisonReportRejectsBadInput(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.good = _min_trajectory(self.tmp / "good.json")
        self.corrupt = self.tmp / "corrupt.json"
        self.corrupt.write_text("this is not json at all")

    def test_missing_file_raises_rather_than_reporting_zeros(self):
        with self.assertRaises(ValueError) as ctx:
            build_comparison_report(str(self.tmp / "nope.json"), str(self.good))
        self.assertIn("reference", str(ctx.exception))

    def test_corrupt_compared_file_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_comparison_report(str(self.good), str(self.corrupt))
        self.assertIn("compared", str(ctx.exception))

    def test_valid_json_that_is_not_a_trajectory_is_refused(self):
        # `load_trajectory` returns an unrecognised JSON OBJECT unchanged, with
        # no `_error` at all, so an `_error`-only guard lets it through and it
        # scores 0.0 on every metric. This is the door that reproduced the
        # full 25% batch-aggregate distortion after the first fix.
        not_a_trajectory = self.tmp / "unknown.json"
        not_a_trajectory.write_text('{"hello": "world"}')
        with self.assertRaises(ValueError) as ctx:
            build_comparison_report(str(not_a_trajectory), str(self.good))
        self.assertIn("no steps parsed", str(ctx.exception))

    def test_empty_json_array_is_refused(self):
        empty = self.tmp / "empty.json"
        empty.write_text("[]")
        with self.assertRaises(ValueError):
            build_comparison_report(str(self.good), str(empty))

    def test_two_good_files_still_produce_a_report(self):
        other = _min_trajectory(self.tmp / "good2.json", steps=3)
        report = build_comparison_report(str(self.good), str(other))
        self.assertIsInstance(report, dict)
        self.assertIn("alignment", report)


class BatchExcludesUnloadableTasks(unittest.TestCase):
    """The aggregate must count a bad task as a failure, not a zero-scoring run."""

    def test_corrupt_task_is_a_failure_and_leaves_metrics_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = _min_trajectory(tmp / "a.json", steps=2)
            b = _min_trajectory(tmp / "b.json", steps=3)
            corrupt = tmp / "bad.json"
            corrupt.write_text("{ not json")

            good_only = run_batch([ManifestEntry(task_id="t1", reference=str(a), compared=str(b))])
            with_bad = run_batch([
                ManifestEntry(task_id="t1", reference=str(a), compared=str(b)),
                ManifestEntry(task_id="bad", reference=str(corrupt), compared=str(b)),
            ])

            agg_good = aggregate_reports(good_only)
            agg_bad = aggregate_reports(with_bad)

            self.assertEqual(agg_bad["success_count"], 1)
            self.assertEqual(agg_bad["failure_count"], 1)
            # The bad task must not perturb the statistics of the good one.
            self.assertEqual(
                agg_bad["metrics"]["alignment_f1"]["mean"],
                agg_good["metrics"]["alignment_f1"]["mean"],
            )
            self.assertEqual(agg_bad["metrics"]["alignment_f1"]["count"], 1)
            errored = [r for r in with_bad if r.error]
            self.assertEqual(len(errored), 1)
            self.assertEqual(errored[0].task_id, "bad")

    def test_unrecognised_json_object_does_not_poison_the_aggregate(self):
        """The `_error`-free door: valid JSON that simply is not a trajectory."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = _min_trajectory(tmp / "a.json", steps=2)
            b = _min_trajectory(tmp / "b.json", steps=3)
            poison = tmp / "poison.json"
            poison.write_text('{"hello": "world"}')

            good_only = run_batch([ManifestEntry(task_id="t1", reference=str(a), compared=str(b))])
            with_poison = run_batch([
                ManifestEntry(task_id="t1", reference=str(a), compared=str(b)),
                ManifestEntry(task_id="poison", reference=str(poison), compared=str(b)),
            ])

            agg_good = aggregate_reports(good_only)
            agg_poison = aggregate_reports(with_poison)
            self.assertEqual(agg_poison["success_count"], 1)
            self.assertEqual(agg_poison["failure_count"], 1)
            self.assertEqual(
                agg_poison["metrics"]["alignment_f1"]["mean"],
                agg_good["metrics"]["alignment_f1"]["mean"],
            )


class ConvergeCliExitsNonZeroOnBadInput(unittest.TestCase):
    """A scripted corpus run must be able to detect a failed comparison."""

    def test_cli_reports_error_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            good = _min_trajectory(Path(td) / "g.json")
            proc = subprocess.run(
                [sys.executable, "-m", "trajviz.converge.cli",
                 str(Path(td) / "missing.json"), str(good), "--output", "summary"],
                cwd=str(REPO_ROOT),
                env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
                capture_output=True, text=True, timeout=180,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Could not load", proc.stderr)
        # It must not have printed a comparison summary as if it had succeeded.
        self.assertNotIn("Converge Comparison Summary", proc.stdout)


if __name__ == "__main__":
    unittest.main()
