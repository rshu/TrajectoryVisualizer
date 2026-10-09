"""Tests for the Gradio-free load_session pipeline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from trajviz.insight.session import LoadError, LoadedSession, load_session

FIXTURE = Path(__file__).parent / "fixtures" / "codearts_minimal.json"


class LoadSessionTests(unittest.TestCase):
    def test_loads_fixture_as_loaded_session(self):
        result = load_session(str(FIXTURE))
        self.assertIsInstance(result, LoadedSession)
        self.assertEqual(result.format, "codearts")
        self.assertGreater(len(result.steps), 0)
        self.assertEqual(result.steps_total, len(result.steps))
        self.assertFalse(result.truncated)
        self.assertEqual(result.path, str(FIXTURE))
        self.assertFalse(result.show_root_cause)  # CodeArts is a noisy-error format

    def test_missing_file_is_not_found(self):
        result = load_session("/no/such/trajectory.json")
        self.assertIsInstance(result, LoadError)
        self.assertEqual(result.code, "not_found")

    def test_format_mismatch(self):
        result = load_session(str(FIXTURE), format_hint="ccsession")
        self.assertIsInstance(result, LoadError)
        self.assertEqual(result.code, "mismatch")
        self.assertEqual(result.selected, "ccsession")
        self.assertEqual(result.detected, "codearts")

    def test_truncation_when_max_steps_is_low(self):
        with patch("trajviz.insight.session.MAX_STEPS", 1):
            result = load_session(str(FIXTURE))
        self.assertIsInstance(result, LoadedSession)
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.steps), 1)
        self.assertGreater(result.steps_total, 1)

    def test_unknown_payload_without_hint(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"not": "a trajectory"}, fh)
            path = fh.name
        try:
            result = load_session(path)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertIsInstance(result, LoadError)
        self.assertEqual(result.code, "unknown")

    def test_packer_uses_named_slots(self):
        from trajviz.insight.ui.load import empty_load_outputs, load_slot_keys, pack_load_outputs

        result = load_session(str(FIXTURE))
        self.assertIsInstance(result, LoadedSession)
        packed = pack_load_outputs(result)
        empty = empty_load_outputs(detail="*No file selected or file not found.*")
        keys = load_slot_keys()
        self.assertEqual(set(packed), set(keys))
        self.assertEqual(set(empty), set(keys))
        self.assertIsInstance(packed["state_steps"], list)
        self.assertGreater(len(packed["state_steps"]), 0)
        self.assertIsInstance(packed["state_raw"], dict)
        self.assertTrue(packed["state_raw"])
        self.assertTrue(packed["overview_kpi_html"])
        self.assertTrue(packed["raw_json"])

    def test_duplicate_packer_keys_are_rejected(self):
        from trajviz.insight.ui.load import merge_packer_dicts

        with self.assertRaises(ValueError) as ctx:
            merge_packer_dicts([{"token_chart": None}, {"token_chart": None}])
        self.assertIn("token_chart", str(ctx.exception))

    def test_load_units_bind_pack_and_slots_to_the_same_module(self):
        from trajviz.insight.ui.load import LOAD_UNITS

        self.assertEqual(LOAD_UNITS[0].name, "shell")
        self.assertIsNone(LOAD_UNITS[0].module)
        tab_units = [unit for unit in LOAD_UNITS if unit.module is not None]
        self.assertEqual(
            [unit.name for unit in tab_units],
            ["upload", "overview_tab", "patterns_tab", "workflow_tab", "raw_tab"],
        )
        for unit in tab_units:
            self.assertTrue(callable(unit.module.pack_load))
            self.assertTrue(callable(unit.module.load_slots))
            self.assertIs(unit.pack, unit.module.pack_load)

    def test_merge_load_slots_rejects_missing_and_extra_tabs(self):
        from trajviz.insight.ui.load import LOAD_UNITS, merge_load_slots
        from trajviz.insight.ui.shared import SharedState

        shared = SharedState(
            state_steps=object(),
            state_raw=object(),
            state_dark=object(),
            state_analysis_brief=object(),
        )
        with self.assertRaises(ValueError) as missing:
            merge_load_slots(main_tabs=object(), shared=shared, refs={})
        missing_msg = str(missing.exception)
        self.assertIn("missing", missing_msg)
        self.assertIn("upload", missing_msg)
        self.assertIn("overview_tab", missing_msg)

        refs = {unit.module: object() for unit in LOAD_UNITS if unit.module is not None}
        refs[object()] = object()
        with self.assertRaises(ValueError) as extra:
            merge_load_slots(main_tabs=object(), shared=shared, refs=refs)
        self.assertIn("extra", str(extra.exception))
