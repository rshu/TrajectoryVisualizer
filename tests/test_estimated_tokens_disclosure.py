"""Estimated token counts must be shown as estimates.

Cursor does not persist the tokens a model was billed for, so its converter
derives per-step tokens from the logged text (about four characters per token)
and records ``_capabilities.has_runtime_token_usage = False`` — a flag that
nothing read. The Tokens card and every token chart presented the estimate as
a measurement. The flag now drives a load warning (dashboard banner and report
header) and an "≈" on the Tokens card. Formats that record real usage are
unchanged.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from trajviz.insight.presenters.overview import build_overview_outputs, load_warnings_html
from trajviz.insight.session import LoadedSession, load_session

_FIX = Path(__file__).resolve().parent / "fixtures"


class EstimatedTokenDisclosureTests(unittest.TestCase):
    def _load(self, name: str) -> LoadedSession:
        result = load_session(str(_FIX / name))
        self.assertIsInstance(result, LoadedSession)
        return result

    def test_cursor_tokens_are_disclosed_as_estimates(self):
        session = self._load("cursor_minimal.json")
        self.assertGreater(session.metrics["tokens"]["total"], 0)
        self.assertIn("estimate", load_warnings_html(session).lower())
        self.assertIn("≈", build_overview_outputs(session)["kpi_html"])

    def test_formats_with_real_usage_carry_no_estimate_notice(self):
        session = self._load("codearts_minimal.json")
        self.assertNotIn("estimate", load_warnings_html(session).lower())
        self.assertNotIn("≈", build_overview_outputs(session)["kpi_html"])


if __name__ == "__main__":
    unittest.main()
