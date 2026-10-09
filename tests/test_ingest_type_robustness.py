"""Malformed-but-plausible input must degrade, never raise.

``load_trajectory`` promises that any bytes on disk come back as a parsed
trajectory or a dict carrying ``_error``; ``load_session`` promises a
``LoadedSession`` or a ``LoadError``. Both promises were being kept site by site,
so they held only for the shapes someone had thought of. A seeded mutation
sweep over real exports — swap one to three values for a value of another type
— aborted 16% of loads across 40 distinct sites, and a few hundred kilobytes of
nested brackets raised ``RecursionError`` straight out of the loader.

Three layers now carry the contract, and each is pinned here:

* a boundary guard in ``load_trajectory`` (and in ``load_session``'s analysis
  stage), so a converter or analyser bug becomes an error message, not a crash;
* type normalisation of the step model in ``parse_steps``, so a wrong-typed
  field degrades to an empty value instead of breaking every consumer;
* containment of per-step detail rendering, so one odd step cannot blank the
  whole dashboard.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from trajviz.insight.loaders import load_trajectory
from trajviz.insight.session import LoadedSession, LoadError, load_session
from trajviz.insight.ui.load import pack_load_outputs

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _write(tmp: str, name: str, text: str) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _opencode_doc() -> dict:
    return {
        "info": {"id": "ses_1", "title": "t", "time": {"created": 1_000_000, "updated": 1_060_000}},
        "messages": [
            {"info": {"role": "user", "time": {"created": 1_000_000}, "sessionID": "ses_1"},
             "parts": [{"type": "text", "text": "fix the bug"}]},
            {"info": {"role": "assistant", "time": {"created": 1_001_000, "completed": 1_003_000},
                      "modelID": "m", "providerID": "p", "agent": "build", "finish": "tool-calls",
                      "sessionID": "ses_1",
                      "tokens": {"total": 400, "input": 200, "output": 150, "reasoning": 0,
                                 "cache": {"read": 50, "write": 0}}},
             "parts": [
                 {"type": "text", "text": "looking"},
                 {"type": "tool", "tool": "bash", "callID": "c1",
                  "state": {"status": "completed", "input": {"command": "ls"}, "output": "a.py",
                            "time": {"start": 1_001_500, "end": 1_002_000}, "metadata": {"exit": 0}}},
             ]},
        ],
    }


def _codex_lines() -> list[dict]:
    return [
        {"timestamp": "2026-01-01T00:00:00Z", "type": "session_meta",
         "payload": {"id": "s", "cwd": "/w", "originator": "codex_cli_rs", "cli_version": "0.1"}},
        {"timestamp": "2026-01-01T00:00:01Z", "type": "response_item",
         "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}},
        {"timestamp": "2026-01-01T00:00:02Z", "type": "response_item",
         "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}},
    ]


class LoaderNeverRaisesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_deeply_nested_json_is_an_error_not_a_recursion_crash(self):
        cases = {
            "deep-array.json": "[" * 200_000 + "]" * 200_000,
            "deep-object.json": '{"a":' * 200_000 + "1" + "}" * 200_000,
            "deep-event.jsonl": '{"type":"x","payload":' + "[" * 150_000 + "]" * 150_000 + "}\n",
        }
        for name, body in cases.items():
            with self.subTest(name):
                path = _write(self.tmp, name, body)
                raw = load_trajectory(path)
                self.assertIn("_error", raw)
                self.assertIsInstance(load_session(path), LoadError)

    def test_a_converter_failure_is_reported_not_raised(self):
        # Codex: a message whose `content` is a scalar used to be iterated.
        lines = _codex_lines()
        lines[1]["payload"]["content"] = 7
        path = _write(self.tmp, "c.jsonl", "\n".join(json.dumps(x) for x in lines))
        raw = load_trajectory(path)
        self.assertIsInstance(raw, dict)
        self.assertIsInstance(load_session(path), (LoadedSession, LoadError))

    def test_claude_code_with_mixed_type_indices_still_loads(self):
        src = _FIXTURES / "reference" / "data" / "trajectory" / "claude_code" / "django__django-11477.json"
        doc = json.loads(src.read_text())
        entries = doc.get("trajectory") or []
        self.assertGreater(len(entries), 2)
        entries[1]["index"] = "one"
        path = _write(self.tmp, "cc.json", json.dumps(doc))
        self.assertIsInstance(load_session(path), LoadedSession)


class StepModelTypeNormalisationTests(unittest.TestCase):
    """A wrong-typed field degrades to an empty value; the file still loads."""

    def test_wrong_typed_fields_load_and_render(self):
        doc = _opencode_doc()
        info = doc["messages"][1]["info"]
        info["finish"] = 123
        info["modelID"] = ["m"]
        info["agent"] = {"name": "build"}
        info["sessionDepth"] = "deep"
        info["time"]["created"] = -10**18                # absurd timestamp
        part = doc["messages"][1]["parts"][1]
        part["tool"] = ["bash"]
        part["state"]["input"] = 5
        part["state"]["output"] = True
        doc["messages"][1]["parts"][0]["text"] = False
        tmp = tempfile.mkdtemp()
        path = _write(tmp, "o.json", json.dumps(doc))

        result = load_session(path)
        self.assertIsInstance(result, LoadedSession)
        step = result.steps[1]
        for key in ("finish", "model_id", "agent", "text_preview"):
            self.assertIsInstance(step[key], str, key)
        tc = step["tool_calls"][0]
        self.assertIsInstance(tc["tool_name"], str)
        self.assertIsInstance(tc["output"], (str, dict, list))
        for dark in (False, True):
            pack_load_outputs(result, dark=dark)


class SeededMutationSweepTests(unittest.TestCase):
    """Regression net for the whole class: no mutation may raise out of the pipeline."""

    REPLACEMENTS = (None, 0, -1, 10**18, 1.5, "", "x", [], {}, [1, 2], {"k": "v"}, True)

    @staticmethod
    def _paths(obj, prefix=(), depth=0, out=None):
        out = [] if out is None else out
        if depth > 8:
            return out
        if isinstance(obj, dict):
            for key, value in obj.items():
                out.append(prefix + (key,))
                SeededMutationSweepTests._paths(value, prefix + (key,), depth + 1, out)
        elif isinstance(obj, list):
            for i, value in enumerate(obj[:5]):
                out.append(prefix + (i,))
                SeededMutationSweepTests._paths(value, prefix + (i,), depth + 1, out)
        return out

    def _sources(self) -> list[tuple[str, object, bool]]:
        sources: list[tuple[str, object, bool]] = [("opencode.json", _opencode_doc(), False),
                                                  ("codex.jsonl", _codex_lines(), True)]
        for name in ("icode_minimal.json", "cursor_minimal.json", "codearts_minimal.json"):
            sources.append((name, json.loads((_FIXTURES / name).read_text()), False))
        cc = _FIXTURES / "reference" / "data" / "trajectory" / "claude_code" / "django__django-11477.json"
        sources.append(("cc.json", json.loads(cc.read_text()), False))
        return sources

    def test_no_type_mutation_raises_out_of_load_or_render(self):
        rng = random.Random(20261009)
        tmp = tempfile.mkdtemp()
        failures: list[str] = []
        for name, base, jsonl in self._sources():
            trials = 6 if name == "cc.json" else 18
            for trial in range(trials):
                doc = json.loads(json.dumps(base))
                paths = self._paths(doc)
                for _ in range(rng.randint(1, 3)):
                    target = rng.choice(paths)
                    node = doc
                    try:
                        for key in target[:-1]:
                            node = node[key]
                        node[target[-1]] = rng.choice(self.REPLACEMENTS)
                    except (KeyError, IndexError, TypeError):
                        pass
                text = "\n".join(json.dumps(x) for x in doc) if jsonl else json.dumps(doc)
                path = _write(tmp, f"{trial}-{name}", text)
                stage = "load_trajectory"
                try:
                    raw = load_trajectory(path)
                    assert isinstance(raw, dict)
                    stage = "load_session"
                    result = load_session(path)
                    # Rendering is the expensive stage; sample it on every third
                    # trial so the sweep stays a couple of seconds.
                    if isinstance(result, LoadedSession) and trial % 3 == 0:
                        stage = "pack_load_outputs"
                        pack_load_outputs(result, dark=bool(trial % 2))
                except Exception as exc:  # noqa: BLE001 — the point is to catch anything
                    failures.append(f"{name} trial {trial} [{stage}]: {type(exc).__name__}: {exc}"[:200])
        self.assertEqual(failures, [], "\n".join(failures[:15]))


if __name__ == "__main__":
    unittest.main()
