"""Shared tool_call_failed / error-kind predicates."""

from __future__ import annotations

import unittest

from trajviz.insight.tool_failure import (
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


if __name__ == "__main__":
    unittest.main()
