import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import step_labeler_v2
from trajviz.insight.labels import aggregate_labels


class StepLabelerV2Tests(unittest.TestCase):
    def test_refuses_to_overwrite_input_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_path = Path(tmp) / "trajectory.json"
            original = '{"messages": []}\n'
            trajectory_path.write_text(original, encoding="utf-8")

            with (
                patch.object(step_labeler_v2, "load_all_steps") as load_steps,
                self.assertRaises(step_labeler_v2.OutputSafetyError),
            ):
                step_labeler_v2.label_trajectory(
                    str(trajectory_path),
                    str(trajectory_path),
                    base_url="https://example.invalid/v1",
                    api_key="test",
                    model="test-model",
                )

            load_steps.assert_not_called()
            self.assertEqual(
                trajectory_path.read_text(encoding="utf-8"), original
            )

    def test_atomic_write_preserves_existing_output_on_serialization_error(self):
        steps = [
            {
                "index": 0,
                "role": "user",
                "text_preview": "Please inspect this project",
                "tokens": {},
                "parts": [],
                "tool_calls": [],
            }
        ]

        def fail_after_partial_write(_data, handle, **_kwargs):
            handle.write('{"partial":')
            raise RuntimeError("serialization failed")

        with tempfile.TemporaryDirectory() as tmp:
            trajectory_path = Path(tmp) / "trajectory.json"
            trajectory_path.write_text('{"messages": []}\n', encoding="utf-8")
            output_path = Path(tmp) / "labeled.json"
            original_output = '{"existing": true}\n'
            output_path.write_text(original_output, encoding="utf-8")
            taxonomy_path = (
                Path(step_labeler_v2.__file__).resolve().parent
                / "TAXONOMY_REFERENCE.md"
            )

            with (
                patch.object(step_labeler_v2, "load_all_steps", return_value=steps),
                patch.object(
                    step_labeler_v2.json,
                    "dump",
                    side_effect=fail_after_partial_write,
                ),
                self.assertRaisesRegex(RuntimeError, "serialization failed"),
            ):
                step_labeler_v2.label_trajectory(
                    str(trajectory_path),
                    str(output_path),
                    base_url="https://example.invalid/v1",
                    api_key="test",
                    model="test-model",
                    taxonomy_path=str(taxonomy_path),
                )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"), original_output
            )
            self.assertEqual(list(Path(tmp).glob(".labeled.json.*.tmp")), [])

    def test_emits_every_index_and_defaults_user(self):
        steps = [
            {
                "index": 4,
                "raw_index": 0,
                "role": "user",
                "text_preview": "Please inspect this project",
                "tokens": {},
                "parts": [],
                "tool_calls": [],
            },
            {
                "index": 9,
                "raw_index": 1,
                "role": "assistant",
                "text_preview": "I will inspect the files",
                "tokens": {"total": 10},
                "parts": [{"type": "reasoning", "text": "Need inspect files"}],
                "tool_calls": [],
            },
            {
                "index": 12,
                "raw_index": 2,
                "role": "system",
                "text_preview": "system context",
                "tokens": {},
                "parts": [],
                "tool_calls": [],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "labeled.json"
            taxonomy_path = (
                Path(step_labeler_v2.__file__).resolve().parent
                / "TAXONOMY_REFERENCE.md"
            )
            with (
                patch.object(step_labeler_v2, "load_all_steps", return_value=steps),
                patch.object(
                    step_labeler_v2.v1,
                    "call_llm",
                    return_value='{"phase":"understand","action":"code_reading"}',
                ) as call_llm,
            ):
                step_labeler_v2.label_trajectory(
                    "input.json",
                    str(output_path),
                    base_url="https://example.invalid/v1",
                    api_key="test",
                    model="test-model",
                    taxonomy_path=str(taxonomy_path),
                )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(data["schema_version"], "trajectory_labels.v2")
        self.assertEqual([s["index"] for s in data["steps"]], [4, 9, 12])
        self.assertEqual([s["raw_index"] for s in data["steps"]], [0, 1, 2])
        self.assertEqual(len(data["steps"]), len(steps))
        self.assertEqual(
            (data["steps"][0]["phase"], data["steps"][0]["action"]),
            ("user", "user_prompt"),
        )
        self.assertEqual(data["steps"][0]["label_source"], "default")
        self.assertEqual(
            (data["steps"][1]["phase"], data["steps"][1]["action"]),
            ("understand", "code_reading"),
        )
        self.assertEqual(data["steps"][1]["label_source"], "llm")
        self.assertEqual(
            (data["steps"][2]["phase"], data["steps"][2]["action"]),
            ("unknown", "unknown"),
        )
        call_llm.assert_called_once()

    def test_empty_assistant_is_retained_without_llm_call(self):
        steps = [
            {
                "index": 0,
                "role": "assistant",
                "text_preview": "",
                "parts": [],
                "tool_calls": [],
                "tokens": {},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "labeled.json"
            taxonomy_path = (
                Path(step_labeler_v2.__file__).resolve().parent
                / "TAXONOMY_REFERENCE.md"
            )
            with (
                patch.object(step_labeler_v2, "load_all_steps", return_value=steps),
                patch.object(step_labeler_v2.v1, "call_llm") as call_llm,
            ):
                step_labeler_v2.label_trajectory(
                    "input.json",
                    str(output_path),
                    base_url="https://example.invalid/v1",
                    api_key="test",
                    model="test-model",
                    taxonomy_path=str(taxonomy_path),
                )
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(data["steps"]), 1)
        self.assertEqual(data["steps"][0]["index"], 0)
        self.assertEqual(data["steps"][0]["label_source"], "fallback")
        call_llm.assert_not_called()

    def test_v2_user_steps_do_not_distort_assistant_metrics_or_duplicate_timeline(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as trajectory:
            data = {
                "schema_version": "trajectory_labels.v2",
                "taxonomy_version": "v1",
                "model": "test-model",
                "trajectory_file": trajectory.name,
                "steps": [
                    {
                        "index": 0,
                        "role": "user",
                        "phase": "user",
                        "action": "user_prompt",
                        "duration_s": 0,
                    },
                    {
                        "index": 1,
                        "role": "assistant",
                        "phase": "understand",
                        "action": "code_reading",
                        "duration_s": 1,
                    },
                ],
            }
            source_steps = [{"index": 0, "role": "user"}]
            with (
                patch(
                    "trajviz.insight.loaders.load_trajectory",
                    return_value={},
                ),
                patch(
                    "trajviz.insight.parser.parse_steps",
                    return_value=source_steps,
                ),
            ):
                result = aggregate_labels(data)

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["classified"], 1)
        self.assertEqual(result["classification_rate"], 100.0)
        self.assertNotIn("user", result["phase_counts"])
        self.assertEqual([s["index"] for s in result["steps"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
