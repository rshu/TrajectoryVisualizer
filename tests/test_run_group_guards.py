"""The multi-run scorecard must not score a run it could not read.

Three ways a bad input used to pass as a real run, each closed here:

* an unrecognised JSON object comes back from the loader with no ``_error``
  and parses to zero steps; it was scored as a run with 0 steps, 0 tokens and
  0.0% tool success, then won every "lowest" flag. The comparison report has
  refused such a side since it was closed there; this is the same guard;
* a missing tool-success rate (no tool calls) was coerced to 0.0%, flagging
  the run worst for a number it never had;
* one run whose analysis raised took the whole scorecard down with it.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trajviz.insight import run_group
from trajviz.insight.run_group import build_run_group_scorecard

_CC = Path(__file__).resolve().parent / "fixtures" / "reference" / "data" / "trajectory" / "claude_code"


class RunGroupGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.good = sorted(str(p) for p in _CC.glob("*.json"))
        self.assertGreaterEqual(len(self.good), 2)

    def _bogus(self) -> str:
        path = os.path.join(self.tmp, "not_a_trajectory.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"hello": "world"}, handle)
        return path

    def test_a_file_with_no_steps_is_an_error_row_not_a_run(self):
        out = build_run_group_scorecard([self.good[0], self._bogus(), self.good[1]])
        bogus = next(r for r in out["rows"] if r["path"].endswith("not_a_trajectory.json"))
        self.assertTrue(bogus["error"])
        self.assertIn("no trajectory steps", bogus["error"])
        self.assertIsNone(bogus["steps"])

    def test_a_missing_success_rate_stays_missing(self):
        from trajviz.insight.parser import parse_steps

        raw = {"messages": [{"info": {"role": "assistant", "time": {"created": 1_000, "completed": 2_000},
                                      "tokens": {"total": 10, "input": 5, "output": 5}},
                             "parts": [{"type": "text", "text": "no tools used"}]}]}
        row = run_group.build_run_scorecard_row(raw, path="x.json", steps=parse_steps(raw))
        self.assertIsNone(row["tool_success_pct"])

    def test_one_run_failing_analysis_does_not_lose_the_others(self):
        real = run_group.build_run_scorecard_row

        def flaky(raw, **kwargs):
            # Fail the real analysis of one run, not the error row built for it.
            if "_error" not in raw and str(kwargs.get("path", "")).endswith(os.path.basename(self.good[1])):
                raise RuntimeError("analysis exploded")
            return real(raw, **kwargs)

        with mock.patch.object(run_group, "build_run_scorecard_row", side_effect=flaky):
            out = build_run_group_scorecard([self.good[0], self.good[1]])
        errors = [r for r in out["rows"] if r.get("error")]
        self.assertEqual(len(out["rows"]), 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("RuntimeError", errors[0]["error"])


if __name__ == "__main__":
    unittest.main()
