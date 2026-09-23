"""A window the export declares beats any inference from the model id.

Codex states its own context window twice per export
(``task_started.payload.model_context_window`` and
``token_count.payload.info.model_context_window``, 258,400 in 500/500 corpus
files) and carries NO model id on any step. The loader discarded it, so every
Codex run fell back to ``DEFAULT_CONTEXT_WINDOW_LIMIT`` (128,000) and every
peak-pressure percentage came out 2.02x too high — flipping the verdict colour
on 163 of 500 files and reporting 70 of them as occupying more than 100% of a
window they never ran in.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trajviz.insight.context_usage import resolve_context_window_limit
from trajviz.insight.formats.codex import _codex_declared_context_window
from trajviz.insight.loaders import load_trajectory

DECLARED = 258_400


def _codex_jsonl(path: Path, *, declared: int | None = DECLARED) -> Path:
    events = [
        {"type": "session_meta", "payload": {"id": "s1", "cwd": "/w", "cli_version": "1"}},
    ]
    started = {"type": "event_msg", "payload": {"type": "task_started"}}
    if declared is not None:
        started["payload"]["model_context_window"] = declared
    events.append(started)
    events.append({
        "type": "response_item",
        "payload": {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}]},
    })
    path.write_text("\n".join(json.dumps(e) for e in events))
    return path


class DeclaredWindowIsExtracted(unittest.TestCase):
    def test_reads_the_task_started_declaration(self):
        events = [{"type": "event_msg",
                   "payload": {"type": "task_started", "model_context_window": DECLARED}}]
        self.assertEqual(_codex_declared_context_window(events), DECLARED)

    def test_reads_the_token_count_declaration(self):
        events = [{"type": "event_msg",
                   "payload": {"type": "token_count",
                               "info": {"model_context_window": DECLARED}}}]
        self.assertEqual(_codex_declared_context_window(events), DECLARED)

    def test_absent_or_nonsense_declarations_yield_none(self):
        for payload in ({}, {"model_context_window": 0}, {"model_context_window": -1},
                        {"model_context_window": True}, {"model_context_window": "big"},
                        {"info": "not a dict"}):
            with self.subTest(payload=payload):
                self.assertIsNone(_codex_declared_context_window([{"payload": payload}]))

    def test_malformed_events_do_not_raise(self):
        self.assertIsNone(_codex_declared_context_window([None, 5, "x", {"payload": None}]))


class DeclaredWindowSurvivesTheLoader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_metadata_carries_it_and_resolution_prefers_it(self):
        raw = load_trajectory(str(_codex_jsonl(self.tmp / "t.jsonl")))
        self.assertEqual(raw["metadata"]["context_window_limit"], DECLARED)
        self.assertEqual(resolve_context_window_limit([], raw), DECLARED)

    def test_a_user_set_limit_still_wins_over_the_declaration(self):
        raw = load_trajectory(str(_codex_jsonl(self.tmp / "t2.jsonl")))
        self.assertEqual(resolve_context_window_limit([], raw, override=500_000), 500_000)

    def test_without_a_declaration_it_falls_back_as_before(self):
        raw = load_trajectory(str(_codex_jsonl(self.tmp / "t3.jsonl", declared=None)))
        self.assertIsNone(raw["metadata"]["context_window_limit"])
        # Falls through to the model-id table / default rather than raising.
        self.assertIsNotNone(resolve_context_window_limit([], raw))


if __name__ == "__main__":
    unittest.main()
