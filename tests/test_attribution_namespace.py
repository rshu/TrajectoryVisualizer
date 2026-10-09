"""Which DECAF verdict namespace the Attribution tab reads by default.

DECAF partitions its cached judge/arbiter verdicts by model slug, and stamps
every verdict with the evidence schema it was produced under. TrajViz trusts a
cached verdict only when that stamp matches DECAF's current ``SCHEMA_VERSION``
(``attribution._verdict_verifies``). So a default that names a model by hand
goes stale the moment DECAF re-baselines on another model or bumps its schema:
every verdict in the named namespace is refused and the tab silently runs with
both LLM layers disabled. That is what happened when the default was pinned to
``z-ai/glm-5.2`` and every glm verdict carried schema 11 under schema 20.

The default is therefore chosen by content: the namespace whose verdicts carry
the current schema. These tests drive the pure selector, so they run without
DECAF installed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trajviz.insight.attribution import _select_judge_model


def _verdict(root: Path, slug: str, agent: str, inst: str, *, model: str, schema: int) -> None:
    d = root / slug / agent
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{inst}.json").write_text(json.dumps({"model": model, "evidence_schema": schema, "verdict": {}}))


class SelectJudgeModelTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_picks_the_namespace_stamped_with_the_current_schema(self):
        # The stale default's namespace exists and is larger, but every verdict
        # in it is from an older schema and would be refused.
        for i in range(5):
            _verdict(self.root, "z-ai__glm-5.2", "claude_code", f"t-{i}", model="z-ai/glm-5.2", schema=11)
        for i in range(3):
            _verdict(self.root, "deepseek-v4-pro", "claude_code", f"t-{i}", model="deepseek-v4-pro", schema=20)
        self.assertEqual(_select_judge_model(self.root, 20), "deepseek-v4-pro")

    def test_returns_the_model_id_recorded_in_the_verdict_not_the_lossy_slug(self):
        # The slug maps "/" to "__"; the id must come from the record itself.
        _verdict(self.root, "z-ai__glm-5.2", "codex", "t", model="z-ai/glm-5.2", schema=20)
        self.assertEqual(_select_judge_model(self.root, 20), "z-ai/glm-5.2")

    def test_several_current_namespaces_prefer_the_one_with_most_verdicts(self):
        _verdict(self.root, "a-model", "codex", "t-1", model="a-model", schema=20)
        for i in range(2):
            _verdict(self.root, "b-model", "codex", f"t-{i}", model="b-model", schema=20)
        self.assertEqual(_select_judge_model(self.root, 20), "b-model")

    def test_no_current_namespace_returns_none(self):
        _verdict(self.root, "old", "codex", "t", model="old", schema=11)
        self.assertIsNone(_select_judge_model(self.root, 20))

    def test_missing_or_unreadable_cache_returns_none(self):
        self.assertIsNone(_select_judge_model(self.root / "absent", 20))
        bad = self.root / "broken" / "codex"
        bad.mkdir(parents=True)
        (bad / "t.json").write_text("{not json")
        self.assertIsNone(_select_judge_model(self.root, 20))


if __name__ == "__main__":
    unittest.main()
