"""Regression tests for the analysis-layer fixes (metrics/patterns).

Each test pins one defect from the quality-review fix batch so it cannot
silently regress:

- B7:  command_success_rate must not count exit=None as a failure.
- B8:  _percentile must use nearest-rank (ceil), not a truncated index.
- B11: tool-selection anti-pattern detector must match lowercase 'bash'.
- B12: edit_precision must not count cancelled/timed-out edits as successes.
"""

import unittest

from trajviz.insight.metrics import (
    _compute_command_metrics,
    _percentile,
    compute_diagnostic_metrics,
)
from trajviz.insight.patterns import detect_tool_selection_antipatterns


def _step(idx, tool_calls, role="assistant"):
    """Minimal parsed-step dict accepted by the analysis functions."""
    return {
        "index": idx,
        "role": role,
        "agent": "main",
        "model_id": "",
        "duration": None,
        "finish": "",
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "parts": [],
        "tokens": {"total": 0, "input": 0, "output": 0,
                   "reasoning": 0, "cache_read": 0},
    }


class CommandMetricsExitNoneTests(unittest.TestCase):
    """B7: exit=None (cancelled/unfinished command) is not a failure."""

    def test_exit_none_is_not_a_failure(self):
        steps = [_step(0, [
            {"tool_name": "Bash", "status": "success", "metadata": {"exit": None}},
            {"tool_name": "Bash", "status": "success", "metadata": {"exit": 0}},
        ])]
        m = _compute_command_metrics(steps)
        self.assertEqual(m["command_call_count"], 2)
        self.assertEqual(m["command_failures"], 0)
        self.assertEqual(m["command_success_rate"], 1.0)

    def test_nonzero_exit_still_counts_as_failure(self):
        steps = [_step(0, [
            {"tool_name": "Bash", "status": "error", "metadata": {"exit": 1}},
            {"tool_name": "Bash", "status": "success", "metadata": {"exit": 0}},
        ])]
        m = _compute_command_metrics(steps)
        self.assertEqual(m["command_call_count"], 2)
        self.assertEqual(m["command_failures"], 1)
        self.assertEqual(m["command_success_rate"], 0.5)


class PercentileNearestRankTests(unittest.TestCase):
    """B8: nearest-rank ceil(q*n)-1, not the truncated int((n-1)*q)."""

    def test_p95_of_ten_values_is_the_max(self):
        vals = [float(v) for v in range(1, 11)]
        # Truncating index returned 9.0 (the ~p89); nearest-rank is 10.0.
        self.assertEqual(_percentile(vals, 0.95), 10.0)

    def test_p90_of_ten_values(self):
        vals = [float(v) for v in range(1, 11)]
        # ceil(0.9 * 10) - 1 = 8 -> 9.0
        self.assertEqual(_percentile(vals, 0.90), 9.0)

    def test_p50_of_four_values_is_lower_median(self):
        # ceil(0.5 * 4) - 1 = 1 -> second value
        self.assertEqual(_percentile([1.0, 2.0, 3.0, 4.0], 0.50), 2.0)

    def test_outlier_surfaces_in_p95(self):
        vals = [1.0] * 9 + [60.0]
        self.assertEqual(_percentile(vals, 0.95), 60.0)

    def test_p99_separates_from_p95_beyond_20_values(self):
        vals = [float(v) for v in range(1, 26)]  # n=25
        self.assertEqual(_percentile(vals, 0.95), 24.0)
        self.assertEqual(_percentile(vals, 0.99), 25.0)

    def test_boundary_guards_unchanged(self):
        vals = [3.0, 1.0, 2.0]
        self.assertEqual(_percentile(vals, 0.0), 1.0)   # q<=0 -> min
        self.assertEqual(_percentile(vals, 1.0), 3.0)   # q>=1 -> max


class ToolSelectionLowercaseBashTests(unittest.TestCase):
    """B11: OpenCode's lowercase 'bash' must be matched, like 'Bash'."""

    def test_lowercase_bash_is_flagged(self):
        steps = [
            _step(0, [{"tool_name": "bash",
                       "input": {"command": "cat /etc/hosts"}}]),
            _step(1, [{"tool_name": "Bash",
                       "input": {"command": "cat /etc/hosts"}}]),
        ]
        flagged = detect_tool_selection_antipatterns(steps)
        self.assertEqual(len(flagged), 2)
        self.assertEqual(sorted(f["step"] for f in flagged), [0, 1])

    def test_other_tools_are_not_flagged(self):
        steps = [_step(0, [{"tool_name": "Read",
                            "input": {"command": "cat /etc/hosts"}}])]
        self.assertEqual(detect_tool_selection_antipatterns(steps), [])


class EditPrecisionFailureStatusTests(unittest.TestCase):
    """B12: cancelled/timed-out edits are failures, not successes."""

    def _diag(self, statuses):
        steps = [_step(i, [{"tool_name": "Edit", "status": st, "input": {}}])
                 for i, st in enumerate(statuses)]
        return compute_diagnostic_metrics(steps, trajectory=[])

    def test_cancelled_edit_is_not_a_success(self):
        m = self._diag(["cancelled", "success"])
        self.assertEqual(m["edit_total"], 2)
        self.assertEqual(m["edit_success"], 1)
        self.assertEqual(m["edit_precision"], 50.0)

    def test_timeout_edits_yield_zero_precision(self):
        m = self._diag(["timeout", "timed_out"])
        self.assertEqual(m["edit_total"], 2)
        self.assertEqual(m["edit_success"], 0)
        self.assertEqual(m["edit_precision"], 0.0)

    def test_successful_edits_unaffected(self):
        m = self._diag(["success", "completed"])
        self.assertEqual(m["edit_total"], 2)
        self.assertEqual(m["edit_success"], 2)
        self.assertEqual(m["edit_precision"], 100.0)


if __name__ == "__main__":
    unittest.main()
