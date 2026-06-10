"""Robustness of load_trajectory: wrong-shape JSON, oversized files, JSONL recovery."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from trajectory_visualizer.insight import loaders
from trajectory_visualizer.insight.loaders import load_trajectory


class LoaderRobustnessTests(unittest.TestCase):
    def _tmp(self, content: str, suffix: str = ".json") -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
        f.write(content)
        f.close()
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        return f.name

    def test_top_level_list_is_rejected_not_crashed(self):
        path = self._tmp(json.dumps([1, 2, 3]))
        out = load_trajectory(path)
        self.assertIn("_error", out)
        self.assertIn("top level", out["_error"].lower())

    def test_top_level_scalar_is_rejected(self):
        path = self._tmp(json.dumps("just a string"))
        out = load_trajectory(path)
        self.assertIn("_error", out)

    def test_oversized_file_rejected_before_read(self):
        path = self._tmp("{}")
        # Patch the cap below the file size to exercise the guard deterministically.
        original = loaders._MAX_FILE_BYTES
        loaders._MAX_FILE_BYTES = 0
        try:
            out = load_trajectory(path)
        finally:
            loaders._MAX_FILE_BYTES = original
        self.assertIn("_error", out)
        self.assertIn("too large", out["_error"].lower())

    def test_jsonl_recovers_from_one_bad_line(self):
        good = json.dumps({"type": "session_meta", "payload": {}})
        path = self._tmp(good + "\n{ this is not json\n", suffix=".jsonl")
        out = load_trajectory(path)
        # Should not be a hard error solely because of the one bad trailing line.
        self.assertNotIn("_error", out)
        self.assertIn("_load_warning", out)
        self.assertIn("Skipped 1", out["_load_warning"])

    def test_jsonl_all_bad_is_error(self):
        path = self._tmp("not json\nalso not json\n", suffix=".jsonl")
        out = load_trajectory(path)
        self.assertIn("_error", out)


if __name__ == "__main__":
    unittest.main()
