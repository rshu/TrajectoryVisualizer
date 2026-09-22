import json
import unittest
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "codearts_minimal.json"


class CodeArtsTests(unittest.TestCase):
    def test_format_is_detected_before_generic_opencode(self):
        from trajviz.insight.loaders import detect_format

        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(detect_format(raw), "codearts")

        generic = {"info": {"id": "ses_open"}, "messages": []}
        self.assertEqual(detect_format(generic), "opencode")

    def test_loader_preserves_product_identity_hierarchy_and_token_totals(self):
        from trajviz.insight.loaders import detect_format, load_trajectory

        loaded = load_trajectory(str(FIXTURE))

        self.assertEqual(detect_format(loaded), "codearts")
        self.assertEqual(loaded["metadata"]["agent"], "codearts")
        self.assertEqual(loaded["metadata"]["generator_name"], "codearts")
        self.assertEqual(loaded["metadata"]["session_count"], 2)
        self.assertEqual(loaded["metadata"]["sub_agent_count"], 1)
        self.assertTrue(loaded["metadata"]["export_complete"])
        self.assertEqual(loaded["token_usage"]["total_tokens"], 150)
        self.assertEqual(loaded["token_usage"]["reasoning_tokens"], 15)
        self.assertEqual(loaded["token_usage"]["cache_read_tokens"], 35)

    def test_parser_preserves_message_part_and_session_fields(self):
        from trajviz.insight.loaders import load_trajectory
        from trajviz.insight.parser import parse_steps

        steps = parse_steps(load_trajectory(str(FIXTURE)))

        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["message_id"], "msg_user")
        self.assertTrue(steps[0]["parts"][1]["synthetic"])
        self.assertTrue(steps[0]["parts"][0]["metadata"]["isUserInput"])
        self.assertEqual(steps[1]["parts"][0]["time"]["end"], 1600)
        self.assertEqual(steps[1]["tool_calls"][0]["tool_id"], "call_task")
        self.assertEqual(steps[2]["parent_session_id"], "ses_root")
        self.assertEqual(steps[2]["session_depth"], 1)
        self.assertEqual(steps[2]["session_title"], "Explorer")
        self.assertTrue(steps[2]["is_sub_agent"])

    def test_real_user_text_wins_over_an_earlier_synthetic_reminder(self):
        from trajviz.insight.parser import _parse_parts

        _, _, _, _, preview = _parse_parts([
            {"type": "text", "text": "system reminder", "synthetic": True},
            {"type": "text", "text": "actual user request", "metadata": {"isUserInput": True}},
        ])

        self.assertEqual(preview, "actual user request")

    def test_codearts_keeps_all_five_token_legend_items(self):
        from trajviz.insight.charts import build_token_chart
        from trajviz.insight.loaders import load_trajectory
        from trajviz.insight.parser import parse_steps

        steps = parse_steps(load_trajectory(str(FIXTURE)))
        figure = build_token_chart(steps, format="codearts")

        self.assertEqual(
            [trace.name for trace in figure.data],
            ["Fresh Input", "Cache Read", "Output", "Reasoning", "Cache Write"],
        )
        # Root assistant step: 50 fresh + 20 cache + 20 output + 10 reasoning.
        self.assertEqual(sum(trace.y[1] for trace in figure.data), 100)

    def test_human_readable_format_label(self):
        from trajviz.insight.presenters import trajectory_format_label

        self.assertEqual(trajectory_format_label("codearts"), "CodeArts")


if __name__ == "__main__":
    unittest.main()
