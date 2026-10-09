"""The LLM features' reply language is a setting, with the documented default.

The AI sidebar and the Issues judge were hard-coded to reply in Simplified
Chinese inside an otherwise English dashboard, with no way to change it.
``ANALYZE_LANGUAGE`` now selects the language; leaving it unset keeps the
documented default and the exact prompts that implement it. Any other language
gets the same report structure with English headings, translated by the model.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from trajviz.insight import assistant, issue_judge


class AnalysisLanguageTests(unittest.TestCase):
    def test_default_is_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(assistant.system_prompt(), assistant.SYSTEM_PROMPT)
            self.assertEqual(assistant.auto_analysis_question(), assistant.AUTO_ANALYSIS_QUESTION)
            self.assertEqual(issue_judge._judge_language(), "Simplified Chinese (Mandarin)")

    def test_another_language_changes_prompt_and_first_question(self):
        with patch.dict(os.environ, {"ANALYZE_LANGUAGE": "English"}, clear=True):
            prompt = assistant.system_prompt()
            self.assertIn("Write the entire reply in English", prompt)
            self.assertIn("## Main performance bottlenecks", prompt)
            self.assertNotIn("主要性能瓶颈", prompt)
            self.assertNotIn("简体中文", prompt)
            self.assertTrue(assistant.auto_analysis_question().startswith("Analyse this run"))
            self.assertEqual(issue_judge._judge_language(), "English")

    def test_the_judge_prompt_names_the_configured_language(self):
        captured = {}

        def fake_chat(_cfg, system, messages):
            captured["system"], captured["user"] = system, messages[0]["content"]
            return '{"where": "x", "fix": "y", "confidence": "low"}'

        from pathlib import Path

        from trajviz.insight.presenters.issues import OverviewIssue
        from trajviz.insight.session import load_session

        fixture = sorted((Path(__file__).resolve().parent / "fixtures" / "reference" / "data" / "trajectory"
                          / "claude_code").glob("*.json"))[0]
        session = load_session(str(fixture))
        cfg = issue_judge.AnalysisLLMConfig(base_url="http://x", api_key="k", model="m", provider="openai",
                                            temperature=None, max_tokens=100, timeout=5, source="analyze")
        issue = OverviewIssue(kind="error", title="t", detail="d", steps=(1,), source_id="s")
        with patch.dict(os.environ, {"ANALYZE_LANGUAGE": "German"}, clear=True):
            issue_judge.judge_issue_fix(session, issue, config=cfg, chat_fn=fake_chat)
        self.assertIn("German", captured.get("system", ""))
        self.assertNotIn("{LANGUAGE}", captured.get("system", ""))
        self.assertIn("German", captured.get("user", ""))


if __name__ == "__main__":
    unittest.main()
