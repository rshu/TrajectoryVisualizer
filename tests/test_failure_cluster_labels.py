"""Failure clustering: peel Bash commands; tag system vs tool errors."""

import unittest

from trajviz.insight.diagnostics import cluster_errors
from trajviz.insight.patterns import detect_failure_patterns


def _bash(idx: int, command: str, *, exit_code: int = 1) -> dict:
    return {
        "index": idx,
        "role": "assistant",
        "tool_calls": [{
            "tool_name": "Bash",
            "input": {"command": command},
            "status": "error",
            "metadata": {"exit": exit_code},
            "output": f"exit {exit_code}",
        }],
    }


def _grep_fail(idx: int) -> dict:
    return {
        "index": idx,
        "role": "assistant",
        "tool_calls": [{
            "tool_name": "Grep",
            "input": {"pattern": "zzz"},
            "status": "error",
            "error": "No matches found",
            "output": "No matches found",
        }],
    }


class FailureClusterLabelTests(unittest.TestCase):
    def test_bash_clusters_by_base_command_not_generic_bash(self):
        steps = [
            _bash(1, "npm test"),
            _bash(2, "npm test"),
            _bash(3, "pytest -q"),
        ]
        clusters = cluster_errors(steps)
        labels = {(c["tool"], c["pattern"], c["error_class"]) for c in clusters}
        self.assertIn(("npm", "exit code 1", "tool"), labels)
        self.assertIn(("pytest", "exit code 1", "tool"), labels)
        self.assertFalse(any(c["tool"] == "Bash" for c in clusters))

    def test_scaffold_grep_is_system_class(self):
        clusters = cluster_errors([_grep_fail(1), _grep_fail(2)])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["error_class"], "system")
        self.assertEqual(clusters[0]["tool"], "Grep")

    def test_detect_failure_patterns_propagates_error_class(self):
        pats = detect_failure_patterns([_bash(1, "cargo test")])
        self.assertEqual(len(pats), 1)
        self.assertEqual(pats[0]["error_class"], "tool")
        self.assertTrue(pats[0]["cluster_label"].startswith("cargo:"))


if __name__ == "__main__":
    unittest.main()
