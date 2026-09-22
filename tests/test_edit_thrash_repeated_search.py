"""Unit tests for edit-thrash and repeated-search detectors."""

import unittest

from trajviz.insight.patterns import detect_edit_thrash, detect_repeated_searches


def _write(idx: int, path: str, *, status: str = "success") -> dict:
    return {
        "index": idx,
        "role": "assistant",
        "tool_calls": [{
            "tool_name": "Write",
            "input": {"file_path": path},
            "status": status,
            "output": "ok" if status == "success" else "failed",
        }],
    }


def _empty_grep(idx: int, pattern: str, path: str = "") -> dict:
    return {
        "index": idx,
        "role": "assistant",
        "tool_calls": [{
            "tool_name": "Grep",
            "input": {"pattern": pattern, "path": path},
            "status": "success",
            "output": "No matches found",
        }],
    }


class EditThrashTests(unittest.TestCase):
    def test_successful_optimize_loop_is_not_thrash(self):
        steps = [
            _write(1, "src/a.py"),
            _write(3, "src/a.py"),
            _write(5, "src/a.py"),
            _write(7, "src/a.py"),
        ]
        self.assertEqual(detect_edit_thrash(steps), [])

    def test_detects_retries_when_a_write_failed(self):
        steps = [
            _write(1, "src/a.py", status="error"),
            _write(3, "src/a.py"),
            _write(5, "src/a.py"),
            _write(20, "src/other.py"),
        ]
        thrash = detect_edit_thrash(steps)
        self.assertEqual(len(thrash), 1)
        self.assertEqual(thrash[0]["path"], "src/a.py")
        self.assertEqual(thrash[0]["count"], 3)
        self.assertEqual(thrash[0]["fail_count"], 1)
        self.assertEqual(thrash[0]["steps"], [1, 3, 5])

    def test_ignores_writes_spread_beyond_window(self):
        steps = [
            _write(1, "src/a.py", status="error"),
            _write(20, "src/a.py"),
            _write(40, "src/a.py"),
        ]
        self.assertEqual(detect_edit_thrash(steps), [])


class RepeatedSearchTests(unittest.TestCase):
    def test_detects_nonconsecutive_identical_empty_searches(self):
        steps = [
            _empty_grep(1, "missing_symbol"),
            {"index": 2, "role": "assistant", "tool_calls": [
                {"tool_name": "Read", "input": {"file_path": "x"}, "status": "success", "output": "hi"},
            ]},
            _empty_grep(4, "missing_symbol"),
            _empty_grep(7, "missing_symbol"),
        ]
        reps = detect_repeated_searches(steps)
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0]["count"], 3)
        self.assertEqual(reps[0]["steps"], [1, 4, 7])

    def test_skips_purely_consecutive_runs(self):
        # Fruitless streaks cover consecutive empties; skip here.
        steps = [
            _empty_grep(1, "x"),
            _empty_grep(2, "x"),
            _empty_grep(3, "x"),
        ]
        self.assertEqual(detect_repeated_searches(steps), [])


if __name__ == "__main__":
    unittest.main()
