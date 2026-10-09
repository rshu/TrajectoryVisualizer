"""Report sections, chip theming, and chart-label uniqueness.

These cover the presentation layer of the HTML export rather than the numbers:
which panels survive into the document, whether the chips and tokens follow the
declared theme, and whether two long tool names can collapse into one chart row.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unittest

from trajviz.insight.charts._layout import _empty_figure, _truncate_chart_labels
from trajviz.insight.charts.swimlanes import build_tool_outcome_timeline
from trajviz.insight.charts.usage import build_tool_chart, build_tool_duration_chart
from trajviz.insight.formatting import _build_per_message_md, _metric_chip
from trajviz.insight.report import _figure_is_empty, _mixed_md_to_html, build_report_html
from trajviz.insight.session import LoadError, load_session
from trajviz.insight.styles import APP_CSS

# Two MCP-style ids that share their first 27 characters: both truncate to the
# same 30-char label, which Plotly then merges into a single y-axis category.
COLLIDING_TOOLS = (
    "mcp__playwright__browser_navigate_forward",
    "mcp__playwright__browser_navigate_back",
)


def _oc_session_raw() -> dict:
    """Minimal OpenCode trajectory: one tool call, no per-call timing."""
    return {
        "info": {
            "id": "ses_report_sections",
            "directory": "/home/user/proj",
            "time": {"created": 1_000, "updated": 4_000},
        },
        "messages": [
            {
                "info": {"role": "user", "time": {"created": 1000}},
                "parts": [{"type": "text", "text": "list the repo"}],
            },
            {
                "info": {
                    "role": "assistant",
                    "time": {"created": 2000, "completed": 3500},
                    "tokens": {
                        "total": 40,
                        "input": 20,
                        "output": 20,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
                "parts": [
                    {"type": "text", "text": "running ls"},
                    {
                        "type": "tool",
                        "tool": "bash",
                        "state": "completed",
                        "input": {"command": "ls"},
                        "output": "README.md\n",
                    },
                ],
            },
        ],
    }


def _step(index: int, tools: list[dict]) -> dict:
    return {
        "index": index,
        "role": "assistant",
        "agent": "",
        "is_sub_agent": False,
        "session_id": "main",
        "tool_calls": tools,
        "tool_call_count": len(tools),
        "tokens": {"total": 0},
    }


class _ReportFixture(unittest.TestCase):
    """Shared on-disk OpenCode session for the document-level assertions."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="trajviz-report-sections-")
        path = os.path.join(cls.tmp, "oc.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_oc_session_raw(), handle)
        result = load_session(path)
        assert not isinstance(result, LoadError), result
        cls.session = result

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class EmptyFigureSectionTests(_ReportFixture):
    """R3 — an explanatory empty state is a section worth exporting."""

    def test_message_only_figure_is_not_empty(self):
        self.assertFalse(_figure_is_empty(_empty_figure(300, "No tool calls recorded.")))

    def test_blank_placeholder_is_still_empty(self):
        self.assertTrue(_figure_is_empty(_empty_figure(380)))

    def test_untimed_tools_still_get_a_duration_panel(self):
        doc = build_report_html(self.session)
        self.assertIn("id='tool-call-duration'", doc)
        self.assertIn("No tool-call timing recorded", doc)


class ChipThemeTests(unittest.TestCase):
    """R4 — chips must take their colors from the theme tokens."""

    def test_plain_chip_uses_tokens(self):
        chip = _metric_chip("Steps", "12")
        self.assertIn("background:var(--ov-chip-bg", chip)
        self.assertIn("border:1px solid var(--ov-chip-border", chip)
        self.assertIn("color:var(--ov-chip-label", chip)
        self.assertIn("color:var(--ov-chip-value", chip)
        # The hexes may survive as var() fallbacks but never as bare values.
        self.assertNotIn("background:#f8fafc", chip)
        self.assertNotIn("color:#1e293b", chip)
        self.assertNotIn("color:#64748b", chip)

    def test_verdict_chip_uses_tokens(self):
        chip = _metric_chip("Tool-wait %", "71%", verdict="warn")
        self.assertIn("background:var(--ov-chip-warn-bg", chip)
        self.assertIn("var(--ov-chip-warn-border", chip)
        self.assertNotIn("background:#fffbeb", chip)

    def test_hint_uses_tokens(self):
        chip = _metric_chip("Avg cache %", "N/A", hint="not available for this format")
        self.assertIn("color:var(--ov-chip-hint", chip)
        self.assertNotIn("color:#94a3b8", chip)


class ThemeTokenScopeTests(_ReportFixture):
    """R5 — a declared theme must not depend on the reader's OS preference."""

    def test_dark_tokens_exist_as_a_class_block(self):
        self.assertRegex(APP_CSS, r"html\.tv-theme-dark\s*\{[^}]*--ov-bg:\s*#1a1b2e")

    def test_light_tokens_exist_as_a_class_block(self):
        self.assertRegex(APP_CSS, r"html\.tv-theme-light\s*\{[^}]*--ov-bg:\s*#f6f8fc")

    def test_dark_report_carries_the_dark_token_block(self):
        doc = build_report_html(self.session, dark=True)
        self.assertRegex(doc, r"html\.tv-theme-dark\s*\{[^}]*--ov-bg:\s*#1a1b2e")
        self.assertRegex(doc, r"html\.tv-theme-dark\s*\{[^}]*--ov-chip-bg:")

    def test_app_light_defaults_are_unchanged(self):
        """The Gradio app sets no theme class, so :root must still be light."""
        root = re.search(r":root\s*\{(.*?)\n\}", APP_CSS, re.DOTALL)
        self.assertIsNotNone(root)
        body = root.group(1)
        self.assertIn("color-scheme: light", body)
        self.assertIn("--ov-bg: #f6f8fc", body)
        self.assertIn("--ov-text: #0f172a", body)
        # The OS-preference override has to stay for the themeless dashboard.
        self.assertRegex(
            APP_CSS,
            r"@media \(prefers-color-scheme: dark\) \{\s*:root \{[^}]*--ov-bg: #1a1b2e",
        )


class ChartLabelUniquenessTests(unittest.TestCase):
    """R6 — truncation must not merge two tools into one chart row."""

    def test_helper_leaves_non_colliding_labels_alone(self):
        names = ["Read", "Bash", "a" * 40]
        self.assertEqual(_truncate_chart_labels(names), [names[0], names[1], "a" * 27 + "..."])

    def test_helper_disambiguates_a_collision(self):
        labels = _truncate_chart_labels(list(COLLIDING_TOOLS))
        self.assertEqual(len(set(labels)), 2)
        self.assertTrue(all(len(label) <= 30 for label in labels))

    def test_tool_chart_keeps_two_categories(self):
        steps = [
            _step(0, [{"tool_name": COLLIDING_TOOLS[0], "status": "success"}] * 3),
            _step(1, [{"tool_name": COLLIDING_TOOLS[1], "status": "success"}] * 2),
        ]
        fig = build_tool_chart(steps)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(len(set(fig.data[0].y)), 2)

    def test_tool_duration_chart_keeps_two_categories(self):
        steps = [
            _step(0, [{"tool_name": COLLIDING_TOOLS[0], "status": "success", "duration_ms": 1200}]),
            _step(1, [{"tool_name": COLLIDING_TOOLS[1], "status": "success", "duration_ms": 800}]),
        ]
        fig = build_tool_duration_chart(steps)
        self.assertEqual(len({trace.y[0] for trace in fig.data}), 2)
        self.assertEqual(len(set(fig.layout.yaxis.categoryarray)), 2)

    def test_tool_outcome_timeline_keeps_two_rows(self):
        steps = [
            _step(0, [{"tool_name": COLLIDING_TOOLS[0], "status": "success"}]),
            _step(1, [{"tool_name": COLLIDING_TOOLS[1], "status": "success"}]),
        ]
        fig = build_tool_outcome_timeline(steps)
        rows = {label for trace in fig.data for label in (trace.y or ()) if label}
        self.assertEqual(len(rows), 2)


class PerMessageEscapingTests(unittest.TestCase):
    """R1 — a newline in an untrusted scalar must not open a markup line."""

    ROW = {
        "index": 3,
        "role": "assistant",
        "agent": "sub\n<img src=x onerror=__tvxss(1)>",
        "finish": "stop",
        "duration": 1.0,
        "tokens_total": 10,
        "tokens_per_sec": 10.0,
        "cache_ratio": 0.0,
        "non_cache_tokens": 10,
        "tool_calls": 1,
    }

    def test_newline_in_agent_cannot_inject_markup(self):
        doc = _mixed_md_to_html(_build_per_message_md([dict(self.ROW)]))
        self.assertNotIn("<img", doc)
        self.assertIn("&lt;img", doc)
        self.assertEqual(doc.count("<tr>"), 2, "one header row plus exactly one body row")

    def test_pipe_in_agent_cannot_open_a_column(self):
        row = dict(self.ROW, agent="sub|agent")
        doc = _mixed_md_to_html(_build_per_message_md([row]))
        self.assertIn("sub|agent", doc)
        # 10 columns with an agent column; a raw pipe would push a cell out.
        self.assertEqual(doc.count("<td>"), 10)

    def test_chip_grid_markup_still_passes_through(self):
        md = "### Timing\n\n" + _metric_chip("Steps", "12") + "\n"
        doc = _mixed_md_to_html(md)
        self.assertIn("<div style='display:inline-flex", doc)

    def test_a_div_carrying_a_handler_is_not_passed_through(self):
        """The pass-through branch admits a chip grid, not any `<div>`.

        A trajectory string that still manages to start a line with `<div` must
        not be able to smuggle an event handler in with it.
        """
        doc = _mixed_md_to_html("<div onclick=\"__tvxss(1)\">x</div>\n")
        self.assertNotIn("<div onclick", doc)
        self.assertIn("&lt;div onclick=&quot;", doc)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
