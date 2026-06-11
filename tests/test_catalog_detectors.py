"""Tests for the catalog-detector dashboard bridge (run all [S] detectors)."""

import unittest

from trajectory_visualizer.insight.catalog_detectors import (
    run_catalog_detectors,
    summarize,
    render_catalog_detectors_html,
    labels_from_labeled_json,
)


def _tc(name, inp, status="completed", output="ok"):
    return {"type": "tool_call", "tool_name": name, "input": inp,
            "output": output, "status": status, "tool_id": f"t-{name}"}


def _step(i, parts):
    return {"index": i, "role": "assistant",
            "tokens": {"total": 100, "input": 50, "output": 50},
            "duration": 1.0, "tool_calls": parts, "parts": parts}


class CatalogDetectorsTests(unittest.TestCase):
    def test_returns_record_per_S_detector(self):
        steps = [_step(0, [_tc("Read", {"file_path": "a.py"})])]
        results = run_catalog_detectors(steps)
        self.assertEqual(len(results), 20)  # all [S] detectors
        ids = {r["id"] for r in results}
        self.assertIn("search-loop", ids)
        self.assertIn("shell-over-tool", ids)
        for r in results:
            self.assertIn(r["status"], ("fired", "clear", "gated"))

    def test_search_loop_fires_on_consecutive_searches(self):
        steps = [_step(i, [_tc("Grep", {"pattern": f"q{i}"})]) for i in range(4)]
        results = run_catalog_detectors(steps)
        fired = {r["id"] for r in results if r["status"] == "fired"}
        self.assertIn("search-loop", fired)

    def test_gating_records_reason_when_unmet(self):
        # No planning tool exposed -> plan-stall is gated, not clear.
        steps = [_step(0, [_tc("Read", {"file_path": "a.py"})])]
        results = run_catalog_detectors(steps)
        plan_stall = next(r for r in results if r["id"] == "plan-stall")
        self.assertEqual(plan_stall["status"], "gated")
        self.assertIn("tool-gated", plan_stall["reason"])

    def test_summary_and_render(self):
        steps = [_step(i, [_tc("Grep", {"pattern": f"q{i}"})]) for i in range(4)]
        results = run_catalog_detectors(steps)
        s = summarize(results)
        self.assertEqual(s["fired"] + s["clear"] + s["gated"], 20)
        self.assertGreaterEqual(s["total_detections"], 1)
        html = render_catalog_detectors_html(results)
        self.assertIn("<table", html)
        self.assertIn("search", html.lower())

    def test_empty_result_churn_via_output_enrichment(self):
        # 3 consecutive searches whose OUTPUT is empty. canonicalize only carries
        # input, so this only fires because the bridge enriches args['output'].
        steps = [_step(i, [_tc("Grep", {"pattern": f"q{i}"}, output="")]) for i in range(3)]
        results = run_catalog_detectors(steps)
        fired = {r["id"] for r in results if r["status"] == "fired"}
        self.assertIn("empty-result-churn", fired)

    def test_h_detectors_gated_without_labels(self):
        # With the [H] band but no labels, semantic detectors are gated (could
        # not fire), not falsely clear.
        steps = [_step(0, [_tc("Read", {"file_path": "a.py"})])]
        results = run_catalog_detectors(steps, bands=("[S]", "[H]"))
        ids = {r["id"] for r in results}
        self.assertIn("semantic-plan-stall", ids)  # [H] present
        pss = next(r for r in results if r["id"] == "semantic-plan-stall")
        self.assertEqual(pss["status"], "gated")
        self.assertIn("requires_semantic_labels", pss["reason"])

    def test_h_detectors_fire_with_labels(self):
        # 5 plan-phase labels then implement -> semantic-plan-stall fires.
        data = {"steps": [{"phase": "plan", "action": "planning"} for _ in range(5)]
                + [{"phase": "implement", "action": "x"}]}
        labels = labels_from_labeled_json(data)
        self.assertEqual(len(labels), 6)
        steps = [_step(i, []) for i in range(6)]
        results = run_catalog_detectors(steps, labels=labels, bands=("[S]", "[H]"))
        fired = {r["id"] for r in results if r["status"] == "fired"}
        self.assertIn("semantic-plan-stall", fired)

    def test_render_empty(self):
        self.assertIn("Load a trajectory", render_catalog_detectors_html([]))


if __name__ == "__main__":
    unittest.main()
