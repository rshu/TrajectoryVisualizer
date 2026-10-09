"""Exported reports: no silent overwrites, and a genuinely offline option.

* Writing into a directory derives the file name from the trajectory's stem,
  so two different trajectories sharing a stem — the same task from two
  harnesses, ``claude_code/x.json`` and ``codex/x.jsonl`` — wrote the same file
  and the second silently replaced the first. Each report now records the
  sha256 of the bytes it was built from; a derived name already holding a
  report of OTHER bytes is not overwritten. Regenerating the same trajectory
  still replaces its own report, and an explicit file destination is written
  as asked.
* The report loads plotly.js from its CDN (documented), so opening it makes a
  network request and the charts are blank offline. ``inline_plotly`` embeds
  the library instead.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from trajviz.insight.report import report_from_path

_CC = Path(__file__).resolve().parent / "fixtures" / "reference" / "data" / "trajectory" / "claude_code"


class ReportCollisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "reports")
        os.makedirs(self.out)
        self.src = sorted(_CC.glob("*.json"))[0]
        self.a = os.path.join(self.tmp, "a", self.src.name)
        self.b = os.path.join(self.tmp, "b", self.src.name)
        for target in (self.a, self.b):
            os.makedirs(os.path.dirname(target))
            shutil.copy(self.src, target)
        # Same stem, different bytes: a second run of "the same task".
        doc = json.loads(Path(self.b).read_text())
        doc.setdefault("metadata", {})["note"] = "a different run"
        Path(self.b).write_text(json.dumps(doc))

    def test_a_different_trajectory_with_the_same_stem_does_not_overwrite(self):
        first = report_from_path(self.a, self.out)
        first_html = Path(first).read_text()
        second = report_from_path(self.b, self.out)
        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).read_text(), first_html)

    def test_regenerating_the_same_trajectory_replaces_its_own_report(self):
        self.assertEqual(report_from_path(self.a, self.out), report_from_path(self.a, self.out))

    def test_an_explicit_file_destination_is_written_as_asked(self):
        dest = os.path.join(self.tmp, "chosen.html")
        report_from_path(self.a, dest)
        self.assertEqual(report_from_path(self.b, dest), dest)


class OfflineReportTests(unittest.TestCase):
    def test_inline_plotly_needs_no_network(self):
        src = str(sorted(_CC.glob("*.json"))[0])
        tmp = tempfile.mkdtemp()
        default_html = Path(report_from_path(src, os.path.join(tmp, "cdn.html"))).read_text()
        offline_html = Path(report_from_path(src, os.path.join(tmp, "offline.html"), inline_plotly=True)).read_text()
        self.assertIn('src="https://cdn.plot.ly', default_html)
        self.assertNotIn('src="https://cdn.plot.ly', offline_html)
        self.assertGreater(len(offline_html), len(default_html) + 1_000_000)


if __name__ == "__main__":
    unittest.main()
