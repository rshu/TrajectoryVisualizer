"""Tests for batch-manifest path confinement (LFI prevention)."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from trajectory_visualizer.converge.batch import parse_manifest


class ManifestConfinementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        # Two in-tree trajectory files the manifest may legitimately reference.
        (self.base / "ref.json").write_text("{}", encoding="utf-8")
        (self.base / "cmp.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_manifest(self, entries):
        path = self.base / "manifest.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return str(path)

    def test_relative_in_tree_paths_resolve(self):
        m = self._write_manifest([
            {"task_id": "t1", "reference": "ref.json", "compared": "cmp.json"}
        ])
        entries = parse_manifest(m, confine=True)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].reference.endswith("ref.json"))

    def test_absolute_path_rejected_when_confined(self):
        m = self._write_manifest([
            {"task_id": "evil", "reference": "/etc/passwd", "compared": "cmp.json"}
        ])
        with self.assertRaises(ValueError) as cm:
            parse_manifest(m, confine=True)
        self.assertIn("absolute", str(cm.exception).lower())

    def test_dot_dot_escape_rejected_when_confined(self):
        # Create a file outside base_dir and try to traverse to it.
        outside = Path(self.tmp.name).parent / "outside_secret.json"
        outside.write_text("{}", encoding="utf-8")
        try:
            rel = os.path.join("..", outside.name)
            m = self._write_manifest([
                {"task_id": "evil", "reference": rel, "compared": "cmp.json"}
            ])
            with self.assertRaises(ValueError) as cm:
                parse_manifest(m, confine=True)
            self.assertIn("escape", str(cm.exception).lower())
        finally:
            outside.unlink(missing_ok=True)

    def test_absolute_path_allowed_when_trusted(self):
        # Trusted CLI mode (confine=False) keeps absolute paths working.
        abs_ref = self.base / "ref.json"
        m = self._write_manifest([
            {"task_id": "t1", "reference": str(abs_ref), "compared": "cmp.json"}
        ])
        entries = parse_manifest(m, confine=False)
        self.assertEqual(entries[0].reference, str(abs_ref))


if __name__ == "__main__":
    unittest.main()
