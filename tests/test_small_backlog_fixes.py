"""Three small contracts from the review backlog.

* The Tool Call Duration axis says it sums per-call time. Parallel calls each
  count there, while the "Tool time" chip is wall-clock (the union of the call
  windows), so the two legitimately differ — the axis has to say which it is.
* A batch report keeps each pair's formats, success-detection method and
  confidence: it summarises pairwise reports and must not be less informative
  than them about whether a comparison can be trusted.
* The ICode export marker vouches for the dialect, not for the shape: a file
  carrying it without ``state.messages`` is not claimed as ICode.
"""

from __future__ import annotations

import unittest

from trajviz.converge.batch import BatchResult, build_batch_report
from trajviz.insight.charts.usage import build_tool_duration_chart
from trajviz.insight.formats.sniff import _looks_like_icode_object


class DurationAxisTests(unittest.TestCase):
    def test_axis_says_it_sums_per_call_time(self):
        step = {"index": 0, "role": "assistant", "agent": "", "parts": [],
                "tool_calls": [{"tool_name": "bash", "input": {"command": "ls"}, "status": "completed",
                                "time_start": 1_000, "time_end": 3_000, "metadata": {}}]}
        fig = build_tool_duration_chart([step])
        self.assertIn("summed", (fig.layout.xaxis.title.text or "").lower())


class BatchTrustFieldTests(unittest.TestCase):
    def test_per_task_entries_keep_formats_and_confidence(self):
        report = {
            "outcome": {"reference_success": True, "compared_success": False,
                        "reference_format": "Claude Code", "compared_format": "Codex CLI",
                        "success_detection": "heuristic (finish marker, not task correctness)"},
            "alignment": {}, "patterns": [], "anchor_mode": "self",
            "confidence": {"alignment": "heuristic", "outcome": "heuristic"},
        }
        results = [BatchResult(task_id="t1", report=report, error=None)]
        entry = build_batch_report("manifest.json", results, {}, {}, {}, {})["per_task"][0]
        self.assertEqual(entry["outcome"]["compared_format"], "Codex CLI")
        self.assertEqual(entry["outcome"]["reference_format"], "Claude Code")
        self.assertIn("heuristic", entry["outcome"]["success_detection"])
        self.assertEqual(entry["confidence"], report["confidence"])


class IcodeSniffTests(unittest.TestCase):
    MARK = {"_chrys_export": {"format": "chrys-expanded-session-v1"}}

    def test_marker_alone_does_not_claim_a_file(self):
        self.assertFalse(_looks_like_icode_object(dict(self.MARK, info={}, messages=[])))

    def test_marker_with_icode_shape_is_claimed(self):
        self.assertTrue(_looks_like_icode_object(dict(self.MARK, meta={}, state={"messages": []})))


if __name__ == "__main__":
    unittest.main()
