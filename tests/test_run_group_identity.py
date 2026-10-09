"""Run-group columns must be attributable, and files must not be merged by guesswork.

* ``_unify_path_keys`` folds a relative path into an absolute one it is a
  suffix of. With several candidates it took whichever came first, so a
  relative ``__init__.py`` was credited to an arbitrary package. It now merges
  only an unambiguous alias.
* Run ids were the bare file stem plus a numeric suffix. Comparing one task
  across harnesses — the common case — gave columns like ``x`` / ``x-2``, and
  uploads arrive under temporary directories, so the folder that named the
  harness is gone. Colliding stems are now told apart by the detected format,
  then by the model.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from trajviz.insight.run_group import _unify_path_keys, build_run_group_scorecard

_CC = Path(__file__).resolve().parent / "fixtures" / "reference" / "data" / "trajectory" / "claude_code"


def _codex_rollout() -> str:
    ev = lambda ts, t, p: {"timestamp": f"2026-01-01T00:00:{ts:02d}Z", "type": t, "payload": p}  # noqa: E731
    rows = [
        ev(0, "session_meta", {"id": "s", "cwd": "/w"}),
        ev(1, "response_item", {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}),
        ev(2, "response_item", {"type": "function_call", "name": "exec_command", "call_id": "c1",
                                "arguments": json.dumps({"cmd": "ls"})}),
        ev(3, "response_item", {"type": "function_call_output", "call_id": "c1", "output": "a.py"}),
        ev(4, "response_item", {"type": "message", "role": "assistant",
                                "content": [{"type": "output_text", "text": "done"}]}),
        ev(5, "event_msg", {"type": "task_complete"}),
    ]
    return "\n".join(json.dumps(r) for r in rows)


class UnifyPathKeysTests(unittest.TestCase):
    def test_an_ambiguous_relative_path_is_not_merged(self):
        mapping = _unify_path_keys(["/r/pkg/__init__.py", "/r/other/__init__.py", "__init__.py"])
        self.assertEqual(mapping["__init__.py"], "__init__.py")

    def test_an_unambiguous_alias_still_merges(self):
        mapping = _unify_path_keys(["/r/src/u.py", "src/u.py"])
        self.assertEqual(mapping["src/u.py"], "/r/src/u.py")


class RunIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        src = sorted(_CC.glob("*.json"))[0]
        self.stem = src.stem
        self.cc = os.path.join(self.tmp, "a", src.name)
        os.makedirs(os.path.dirname(self.cc))
        shutil.copy(src, self.cc)
        self.codex = os.path.join(self.tmp, "b", f"{self.stem}.jsonl")
        os.makedirs(os.path.dirname(self.codex))
        with open(self.codex, "w", encoding="utf-8") as handle:
            handle.write(_codex_rollout())

    def test_same_task_across_harnesses_gets_attributable_ids(self):
        out = build_run_group_scorecard([self.cc, self.codex])
        ids = [r["run_id"] for r in out["rows"]]
        self.assertEqual(len(set(ids)), 2, ids)
        self.assertTrue(any("Claude Code" in i for i in ids), ids)
        self.assertTrue(any("Codex" in i for i in ids), ids)
        self.assertFalse(any(i.endswith("-2") for i in ids), ids)

    def test_distinct_stems_keep_their_plain_names(self):
        other = sorted(_CC.glob("*.json"))[1]
        out = build_run_group_scorecard([self.cc, str(other)])
        self.assertEqual(sorted(r["run_id"] for r in out["rows"]), sorted([self.stem, other.stem]))

    def test_a_baseline_without_a_source_path_keeps_uploads_aligned(self):
        from trajviz.insight.loaders import load_trajectory

        raw = load_trajectory(self.cc)
        raw.pop("_source_path", None)
        out = build_run_group_scorecard([self.codex], baseline_raw=raw)
        by_path = {r["path"]: r["run_id"] for r in out["rows"]}
        self.assertEqual(by_path["(overview)"], "overview")
        self.assertTrue(by_path[self.codex].startswith(self.stem), by_path)


if __name__ == "__main__":
    unittest.main()
