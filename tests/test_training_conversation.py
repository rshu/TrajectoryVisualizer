import json
import os
import tempfile
import unittest
from pathlib import Path

from trajectory_visualizer.insight.loaders import detect_format, load_trajectory
from trajectory_visualizer.insight.metrics import validate_token_integrity
from trajectory_visualizer.insight.parser import parse_steps


FIXTURE = Path(__file__).parent / "fixtures" / "training_conversation_minimal.json"
TOOL_FIXTURE = Path(__file__).parent / "fixtures" / "training_conversation_with_tools.json"

# Optional regression corpus of real agent samples. Not committed (may contain
# proprietary content); point at it with TRAJECTORY_TRAINING_SAMPLES. Tests that
# need it skip cleanly when it is absent (e.g. CI / fresh clones).
SAMPLES_ROOT = Path(
    os.environ.get(
        "TRAJECTORY_TRAINING_SAMPLES",
        "/Users/lxs/Documents/AI/trajectory-training-samples",
    )
)


class TrainingConversationTests(unittest.TestCase):
    def test_detects_training_conversation_from_messages_without_trusting_meta_info(self):
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

        self.assertEqual(detect_format(raw), "training_conversation")

    def test_loads_training_conversation_with_capabilities_and_no_meta_labels(self):
        loaded = load_trajectory(str(FIXTURE))

        self.assertEqual(loaded.get("_analysis_profile"), "training")
        self.assertEqual(loaded.get("_source_format"), "openai_conversations")
        self.assertEqual(loaded.get("format"), "training-conversation")
        self.assertEqual(loaded.get("_capabilities"), {
            "has_timing": False,
            "has_tool_calls": False,
            "has_runtime_token_usage": False,
            "has_reasoning_content": True,
            "has_training_labels": False,
        })
        self.assertNotIn("data_label", json.dumps(loaded))
        self.assertNotIn("value_score", json.dumps(loaded))

    def test_parses_training_conversation_as_basic_steps_with_reasoning_part(self):
        loaded = load_trajectory(str(FIXTURE))
        steps = parse_steps(loaded)

        self.assertEqual([s["role"] for s in steps], ["system", "user", "assistant", "user", "assistant"])
        assistant = steps[2]
        self.assertIsNone(assistant["duration"])
        self.assertEqual(assistant["tool_calls"], [])
        self.assertTrue(assistant["has_reasoning"])
        self.assertGreater(assistant["tokens"]["estimated_total"], 0)
        self.assertTrue(assistant["tokens"]["estimated"])
        self.assertEqual([p["type"] for p in assistant["parts"]], ["reasoning", "text"])

    def test_training_estimated_tokens_do_not_trigger_zero_token_warning(self):
        loaded = load_trajectory(str(FIXTURE))
        steps = parse_steps(loaded)

        self.assertEqual(validate_token_integrity(steps), [])

    def test_jsonl_training_conversation_is_not_supported_in_v1(self):
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps(raw) + "\n")
            temp_path = f.name
        try:
            loaded = load_trajectory(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        self.assertIn("_error", loaded)
        self.assertIn("JSONL", loaded["_error"])

    def test_training_conversation_with_tool_calls_preserves_tool_structure(self):
        loaded = load_trajectory(str(TOOL_FIXTURE))
        steps = parse_steps(loaded)

        first_assistant = steps[2]
        tool_step = steps[3]
        second_assistant = steps[4]

        self.assertEqual(first_assistant["role"], "assistant")
        self.assertEqual(first_assistant["tool_call_count"], 1)
        self.assertEqual(first_assistant["tool_calls"][0]["tool_name"], "str_replace_editor")
        self.assertEqual(tool_step["role"], "tool")
        self.assertIn("files and directories", tool_step["text_preview"])
        self.assertEqual(second_assistant["tool_calls"][0]["tool_name"], "execute_bash")

    def test_infers_openhands_scaffold_from_system_prompt(self):
        loaded = load_trajectory(str(TOOL_FIXTURE))

        self.assertEqual(loaded["_training_scaffold"]["id"], "generic_tool_calling")

    @unittest.skipUnless(
        SAMPLES_ROOT.is_dir(),
        f"training samples not found at {SAMPLES_ROOT}; set TRAJECTORY_TRAINING_SAMPLES",
    )
    def test_infers_known_scaffolds_from_real_samples_without_meta_info(self):
        root = SAMPLES_ROOT
        expectations = {
            "openhands_agent_tools.json": "openhands",
            "claude_agent_sdk_tools.json": "claude_agent_sdk",
            "xml_command_line_scaffold.json": "xml_command_line",
            "plain_qa_fast.json": "plain_conversation",
        }
        for filename, scaffold_id in expectations.items():
            with self.subTest(filename=filename):
                loaded = load_trajectory(str(root / filename))
                self.assertEqual(loaded["_training_scaffold"]["id"], scaffold_id)
                self.assertNotIn("meta_info", json.dumps(loaded))


if __name__ == "__main__":
    unittest.main()
