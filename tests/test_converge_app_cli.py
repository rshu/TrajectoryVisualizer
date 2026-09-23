"""Smoke coverage for the Converge app, CLI, batch and intervention paths.

Converge is the project's second shipping Gradio app (`python -m
trajviz.converge`) plus a CLI. Its UI/CLI layer had no test coverage at all
(`app.py`, `cli.py`, `charts.py` were at 0%), so a refactor could break the
whole entry point without a single test failing. These are deliberately
shallow -- they assert the wiring holds and the reports carry real content,
not the exact numbers, which the converge algorithm tests already own.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _trajectory(path: Path, *, steps: int, tool: str = "Bash") -> Path:
    """A minimal Claude Code-shaped export with tool calls the aligner can see."""
    entries = []
    for i in range(steps):
        entries.append({
            "type": "assistant",
            "uuid": f"u{i}",
            "message": {
                "id": f"m{i}",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"step {i}"},
                    {"type": "tool_use", "id": f"t{i}", "name": tool,
                     "input": {"command": f"echo {i}"}},
                ],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        })
    path.write_text(json.dumps({"trajectory": entries}))
    return path


def _manifest(path: Path, tasks: list[tuple[str, Path, Path]]) -> Path:
    path.write_text(json.dumps([
        {"task_id": t, "reference": str(r), "compared": str(c)} for t, r, c in tasks
    ]))
    return path


class ConvergeAppConstructs(unittest.TestCase):
    def test_build_ui_constructs_without_error(self):
        from trajviz.converge.app import build_ui

        app = build_ui()
        self.assertIsNotNone(app)

    def test_file_pickers_only_offer_types_the_loader_accepts(self):
        # The loader rejects .log outright (loaders.py), so offering it in the
        # picker invites a file that can only fail.
        source = (REPO_ROOT / "trajviz" / "converge" / "app.py").read_text()
        self.assertNotIn('".log"', source)


class ConvergeCliRuns(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ref = _trajectory(self.tmp / "ref.json", steps=4)
        self.cmp = _trajectory(self.tmp / "cmp.json", steps=6)

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "trajviz.converge.cli", *args],
            cwd=str(REPO_ROOT),
            env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=300,
        )

    def test_summary_output_reports_both_sides(self):
        proc = self._run(str(self.ref), str(self.cmp), "--output", "summary")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Converge Comparison Summary", proc.stdout)
        # Confidence labelling is the honesty contract of this report.
        self.assertIn("Confidence:", proc.stdout)

    def test_json_output_is_parseable_and_carries_alignment(self):
        proc = self._run(str(self.ref), str(self.cmp), "--output", "json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        for key in ("alignment", "outcome", "confidence"):
            self.assertIn(key, report)

    def test_no_arguments_is_a_usage_error_not_a_crash(self):
        proc = self._run()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Provide ref_file and cmp_file", proc.stderr)


class ConvergeBatchAndIntervention(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.a = _trajectory(self.tmp / "a.json", steps=4)
        self.b = _trajectory(self.tmp / "b.json", steps=6)
        self.c = _trajectory(self.tmp / "c.json", steps=5, tool="Read")

    def test_batch_aggregates_every_task(self):
        from trajviz.converge.batch import aggregate_reports, parse_manifest, run_batch

        path = _manifest(self.tmp / "m.json",
                         [("t1", self.a, self.b), ("t2", self.a, self.c)])
        results = run_batch(parse_manifest(str(path)))
        self.assertEqual(len(results), 2)
        agg = aggregate_reports(results)
        self.assertEqual(agg["success_count"], 2)
        self.assertEqual(agg["failure_count"], 0)
        self.assertEqual(agg["metrics"]["alignment_f1"]["count"], 2)

    def test_manifest_rejects_a_missing_reference_up_front(self):
        from trajviz.converge.batch import parse_manifest

        path = _manifest(self.tmp / "bad.json",
                         [("t1", self.tmp / "nope.json", self.b)])
        with self.assertRaises(FileNotFoundError):
            parse_manifest(str(path))

    def test_intervention_pairs_tasks_and_reports_deltas(self):
        from trajviz.converge.batch import parse_manifest, run_batch
        from trajviz.converge.intervention import compute_metric_deltas, pair_tasks

        before = _manifest(self.tmp / "before.json", [("t1", self.a, self.b)])
        after = _manifest(self.tmp / "after.json", [("t1", self.a, self.c)])
        before_results = run_batch(parse_manifest(str(before)))
        after_results = run_batch(parse_manifest(str(after)))

        paired, before_only, after_only = pair_tasks(before_results, after_results)
        self.assertEqual(len(paired), 1)
        self.assertEqual((before_only, after_only), ([], []))

        deltas = compute_metric_deltas(paired)
        self.assertTrue(deltas, "intervention produced no metric deltas")
        for name, d in deltas.items():
            with self.subTest(metric=name):
                self.assertIn("delta", d)
                # A delta must be the difference of the two reported values.
                if isinstance(d.get("before"), (int, float)) and isinstance(d.get("after"), (int, float)):
                    self.assertAlmostEqual(d["delta"], d["after"] - d["before"], places=3)


if __name__ == "__main__":
    unittest.main()
