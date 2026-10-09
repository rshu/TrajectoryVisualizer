"""Shared tool_call_failed / error-kind predicates."""

from __future__ import annotations

import unittest

from trajviz.insight.tool_failure import (
    status_failed,
    step_has_success,
    tool_call_error_kind,
    tool_call_failed,
    step_error_kind,
)


class ToolFailureTests(unittest.TestCase):
    def test_status_union_is_case_insensitive(self):
        self.assertTrue(tool_call_failed({"status": "TIMED_OUT"}))
        self.assertTrue(tool_call_failed({"status": "Canceled"}))
        self.assertTrue(tool_call_failed({"status": "error"}))

    def test_exit_code_and_error_fields(self):
        self.assertTrue(tool_call_failed({"status": "completed", "metadata": {"exit": 1}}))
        self.assertFalse(tool_call_failed({"status": "completed", "metadata": {"exit": 0}}))
        self.assertTrue(tool_call_failed({"status": "completed", "error_type": "ENOENT"}))
        self.assertTrue(tool_call_failed({"status": "completed", "error": "boom"}))

    def test_error_kind_system_vs_tool(self):
        self.assertEqual(tool_call_error_kind({"tool_name": "Read"}), "system")
        self.assertEqual(tool_call_error_kind({"tool_name": "Bash"}), "tool")

    def test_step_error_kind_tool_wins(self):
        step = {
            "finish": "stop",
            "error_count": 0,
            "tool_calls": [
                {"tool_name": "Read", "status": "error"},
                {"tool_name": "Bash", "status": "error"},
            ],
        }
        self.assertEqual(step_error_kind(step), "tool")


class StatusFailedTests(unittest.TestCase):
    """The status-only predicate for callers holding a bare status string."""

    def test_whole_union_including_aliases(self):
        for status in ("error", "failed", "failure", "cancelled", "canceled",
                       "timeout", "timed_out", "TIMED_OUT", "Canceled"):
            with self.subTest(status=status):
                self.assertTrue(status_failed(status))

    def test_success_and_missing_statuses(self):
        for status in ("completed", "ok", "success", "", None):
            with self.subTest(status=status):
                self.assertFalse(status_failed(status))


class StepHasSuccessTests(unittest.TestCase):
    """The step-level recovery predicate shared with patterns.py."""

    def test_nonzero_exit_is_not_success(self):
        # OpenCode reports a failed shell call as "completed" + metadata.exit.
        step = {"tool_calls": [{"tool_name": "bash", "status": "completed", "metadata": {"exit": 1}}]}
        self.assertFalse(step_has_success(step))

    def test_alias_statuses_are_not_success(self):
        for status in ("timed_out", "canceled", "ERROR"):
            with self.subTest(status=status):
                self.assertFalse(step_has_success({"tool_calls": [{"status": status}]}))

    def test_error_field_only_is_not_success(self):
        self.assertFalse(step_has_success({"tool_calls": [{"status": "completed", "error": "boom"}]}))

    def test_mixed_step_is_success(self):
        step = {"tool_calls": [{"status": "error"}, {"status": "completed"}]}
        self.assertTrue(step_has_success(step))

    def test_no_tool_calls_is_not_success(self):
        # Pins the semantics patterns.detect_failure_patterns depends on: a bare
        # text step must not read as recovering from an error cluster.
        self.assertFalse(step_has_success({}))
        self.assertFalse(step_has_success({"tool_calls": []}))
        self.assertFalse(step_has_success({"tool_calls": None}))


if __name__ == "__main__":
    unittest.main()
