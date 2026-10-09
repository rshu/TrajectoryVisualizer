"""One malformed line must not silently delete a DeepSeek-Harness sub-agent.

A DSH export merges each ``subagents/<id>/session.jsonl`` into the parent's
trajectory. Child logs went through the strict JSONL parser, whose interior
decode error is fatal — reasonable for the parent, which then fails visibly,
but for a child the failure was swallowed: the whole sub-agent vanished and the
trajectory still looked complete. A child log now keeps its valid events, and
the load carries a warning naming what was skipped, shown above the dashboard
and in the report header. Well-formed children are parsed exactly as before.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile

from trajviz.insight.loaders import load_trajectory
from trajviz.insight.parser import parse_steps
from trajviz.insight.presenters.overview import load_warnings_html
from trajviz.insight.session import LoadedSession, load_session

T0 = 1_787_623_647_000


def _evt(etype, seq, time, data):
    return {"type": etype, "seq": seq, "time": time, "data": data}


def _parent() -> list[dict]:
    return [
        {"type": "session", "version": 0, "id": "session-parent", "createdAt": T0, "cwd": "/w",
         "delegationDepth": 0, "agentPreset": "standard"},
        _evt("user/message", 1, T0 + 100, {"content": [{"type": "text", "text": "go"}],
                                           "source": {"kind": "user"}, "role": "user", "id": "u1"}),
        _evt("assistant/message", 2, T0 + 200, {"turn": 1, "step": 1, "message": {
            "role": "assistant", "content": [{"type": "text", "text": "delegating"}],
            "source": {"kind": "model", "provider": "p", "model": "m"}, "id": "a1"},
            "usage": {"inputTokens": 10, "outputTokens": 5}}),
    ]


def _child_lines() -> list[str]:
    events = [
        {"type": "session", "version": 0, "id": "child-1", "createdAt": T0 + 300, "cwd": "/w",
         "origin": "subagent", "parentSession": "session-parent", "delegationDepth": 1},
        _evt("user/message", 1, T0 + 400, {"content": [{"type": "text", "text": "explore"}],
                                           "source": {"kind": "user"}, "role": "user", "id": "cu"}),
        _evt("assistant/message", 2, T0 + 500, {"turn": 1, "step": 1, "message": {
            "role": "assistant", "content": [{"type": "text", "text": "child finding"}],
            "source": {"kind": "model", "provider": "p", "model": "m"}, "id": "ca"},
            "usage": {"inputTokens": 8, "outputTokens": 4}}),
    ]
    lines = [json.dumps(e) for e in events]
    lines.insert(2, '{"type": "assistant/message", "seq": 9, BROKEN')   # malformed interior line
    return lines


def _child_steps(raw: dict) -> list[dict]:
    return [s for s in parse_steps(raw) if s.get("session_id") == "child-1"]


class DshChildToleranceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _directory_export(self, child_lines: list[str]) -> str:
        root = os.path.join(self.tmp, "export")
        os.makedirs(os.path.join(root, "subagents", "child-1"))
        with open(os.path.join(root, "session.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(json.dumps(e) for e in _parent()) + "\n")
        with open(os.path.join(root, "subagents", "child-1", "session.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(child_lines) + "\n")
        return os.path.join(root, "session.jsonl")

    def test_a_malformed_child_line_keeps_the_rest_of_the_child_and_warns(self):
        raw = load_trajectory(self._directory_export(_child_lines()))
        self.assertNotIn("_error", raw)
        child = _child_steps(raw)
        self.assertTrue(any("child finding" in (s.get("text_preview") or "") for s in child))
        warnings = raw.get("_load_warnings") or []
        self.assertTrue(any("1 malformed line" in w for w in warnings), warnings)

    def test_the_warning_reaches_the_load_banner(self):
        result = load_session(self._directory_export(_child_lines()))
        self.assertIsInstance(result, LoadedSession)
        self.assertIn("malformed line", load_warnings_html(result))

    def test_a_well_formed_child_loads_without_a_warning(self):
        clean = [line for line in _child_lines() if "BROKEN" not in line]
        raw = load_trajectory(self._directory_export(clean))
        self.assertTrue(_child_steps(raw))
        self.assertFalse(raw.get("_load_warnings"))

    def test_zip_exports_get_the_same_tolerance(self):
        path = os.path.join(self.tmp, "export.zip")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("export/session.jsonl", "\n".join(json.dumps(e) for e in _parent()) + "\n")
            zf.writestr("export/subagents/child-1/session.jsonl", "\n".join(_child_lines()) + "\n")
        raw = load_trajectory(path)
        self.assertNotIn("_error", raw)
        self.assertTrue(any("child finding" in (s.get("text_preview") or "") for s in _child_steps(raw)))
        self.assertTrue(any("malformed line" in w for w in raw.get("_load_warnings") or []))


if __name__ == "__main__":
    unittest.main()
