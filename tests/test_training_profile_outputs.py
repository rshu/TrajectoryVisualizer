import json
import os
import unittest
from pathlib import Path

from trajectory_visualizer.insight.insight import _build_chart_outputs, _build_overview_outputs
from trajectory_visualizer.insight.analytics import compute_step_analytics
from trajectory_visualizer.insight.loaders import load_trajectory
from trajectory_visualizer.insight.metrics import build_message_metrics, compute_metrics, compute_health_verdict
from trajectory_visualizer.insight.parser import parse_steps


FIXTURE = Path(__file__).parent / "fixtures" / "training_conversation_minimal.json"
TOOL_FIXTURE = Path(__file__).parent / "fixtures" / "training_conversation_with_tools.json"

# Optional real-sample regression file; see TRAJECTORY_TRAINING_SAMPLES in
# test_training_conversation.py. Skips cleanly when the corpus is absent.
SAMPLES_ROOT = Path(
    os.environ.get(
        "TRAJECTORY_TRAINING_SAMPLES",
        "/Users/lxs/Documents/AI/trajectory-training-samples",
    )
)
REAL_AGENT_SAMPLE = SAMPLES_ROOT / "claude_agent_sdk_tools.json"


class TrainingProfileOutputTests(unittest.TestCase):
    def _load(self):
        raw = load_trajectory(str(FIXTURE))
        steps = parse_steps(raw)
        rows = build_message_metrics(steps)
        metrics = compute_metrics(steps, raw, rows)
        analytics = compute_step_analytics(steps)
        verdicts = compute_health_verdict(metrics, analytics)
        return raw, steps, rows, metrics, verdicts, analytics

    def test_training_overview_uses_training_copy_and_no_work_anomalies(self):
        raw, steps, rows, metrics, verdicts, analytics = self._load()

        out = _build_overview_outputs(
            steps, raw, metrics, rows, verdicts, "n/a", str(FIXTURE), trajectory_format="training_conversation"
        )

        self.assertIn("Training conversation", out["metrics_text"])
        self.assertIn("Plain Conversation", out["metrics_text"])
        self.assertIn("assistant turns", out["behavior_text"])
        self.assertEqual(out["anomaly_html"], "")
        self.assertIn("reasoning", out["per_message_text"].lower())

    @unittest.skipUnless(
        REAL_AGENT_SAMPLE.is_file(),
        f"training sample not found at {REAL_AGENT_SAMPLE}; set TRAJECTORY_TRAINING_SAMPLES",
    )
    def test_training_overview_shows_scaffold_label_for_real_agent_sample(self):
        tool_path = REAL_AGENT_SAMPLE
        raw = load_trajectory(str(tool_path))
        steps = parse_steps(raw)
        rows = build_message_metrics(steps)
        metrics = compute_metrics(steps, raw, rows)
        analytics = compute_step_analytics(steps)
        verdicts = compute_health_verdict(metrics, analytics)

        out = _build_overview_outputs(
            steps, raw, metrics, rows, verdicts, "n/a", str(tool_path), trajectory_format="training_conversation"
        )

        self.assertIn("Claude Agent SDK", out["metrics_text"])
        self.assertIn("Scaffold", out["kpi_html"])

    def test_training_chart_outputs_adapt_work_sections_instead_of_leaving_them_blank(self):
        raw, steps, rows, metrics, verdicts, analytics = self._load()
        phases = []
        agent_summaries = []

        out = _build_chart_outputs(
            steps, rows, phases, analytics, agent_summaries,
            dark=False,
            trajectory=raw.get("trajectory", []),
            trajectory_format="training_conversation",
        )

        self.assertIn("training", out["agent_cards_html"].lower())
        self.assertIn("reasoning", out["antipattern_html"].lower())
        self.assertGreater(len(out["tok_fig"].data), 0)
        self.assertGreater(len(out["dur_fig"].data), 0)
        self.assertGreater(len(out["context_growth_fig"].data), 0)

    def test_training_chart_outputs_use_real_tool_chart_when_tool_calls_exist(self):
        raw = load_trajectory(str(TOOL_FIXTURE))
        steps = parse_steps(raw)
        rows = build_message_metrics(steps)

        out = _build_chart_outputs(
            steps, rows, [], compute_step_analytics(steps), [],
            dark=False,
            trajectory=raw.get("trajectory", []),
            trajectory_format="training_conversation",
        )

        self.assertEqual(out["tl_fig"].layout.title.text, "Tool Call Frequency")
        tool_names = []
        for trace in out["tl_fig"].data:
            tool_names.extend(list(trace.y) if getattr(trace, "y", None) is not None else [])
        self.assertTrue(any(name in ("str_replace_editor", "execute_bash") for name in tool_names))


if __name__ == "__main__":
    unittest.main()
