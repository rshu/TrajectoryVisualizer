"""Untrusted trajectory strings must never reach a rendered surface as live markup.

Every string TrajViz displays comes out of a file the user opened: tool
arguments, shell command lines, file paths, model prose, error text, agent and
session names. A trajectory is a document, not a trusted input — it can be
shared, downloaded from a run harness, or written by a model that was itself
prompt-injected. If any of those strings reaches an HTML sink unescaped, opening
the file is enough to run the author's JavaScript, in the dashboard and in the
exported report a researcher mails around.

The sweep below pushes one trajectory whose every string field carries a tagged
payload through the real public entry points (``load_session`` →
``pack_load_outputs``, then each presenter) and asserts the payload never
survives as markup. It is keyed off the packer's own slot registry, so a tab
that adds a new rendered slot is swept automatically and an unclassified slot
fails ``test_every_load_slot_is_classified`` rather than silently escaping
review.
"""

import base64
import json
import os
import re
import shutil
import tempfile
import unittest

from trajviz.insight.presenters import __all__ as PRESENTER_EXPORTS
from trajviz.insight.presenters.issues import build_overview_issues_html
from trajviz.insight.presenters.overview import (
    build_diagnostics_outputs,
    build_overview_kpi_html,
    build_summary_outputs,
)
from trajviz.insight.presenters.patterns import (
    build_antipattern_html,
    render_failure_patterns_html,
    render_tool_sequences_html,
)
from trajviz.insight.presenters.workflow import (
    build_filtered_workflow_outputs,
    build_workflow_outputs,
)
from trajviz.insight.rendering import format_step_detail
from trajviz.insight.report import build_report_html
from trajviz.insight.session import LoadError, load_session
from trajviz.insight.ui.load import load_slot_keys, pack_load_outputs

# Distinct payload shapes, one per HTML context. Each must be inert wherever it
# lands: as element content, inside a single- or double-quoted attribute, and
# inside a string literal in emitted JavaScript.
TAG = "<img src=x onerror=__tvxss(1)>"
ATTR_DQ = '" onmouseover="__tvxss(2)'
ATTR_SQ = "' onmouseover='__tvxss(3)"
JS_BREAK = "');__tvxss(4);//"
PAYLOADS = {
    "element": TAG,
    "attribute-double-quoted": ATTR_DQ,
    "attribute-single-quoted": ATTR_SQ,
    "javascript-string": JS_BREAK,
}


def _pay(tag: str) -> str:
    """Inline payload carrying every context breakout, tagged by field name."""
    return f"ZZ{tag}ZZ{TAG}{ATTR_DQ}{ATTR_SQ}{JS_BREAK}"


def _pay_nl(tag: str) -> str:
    """Payload whose markup starts its own line.

    Renderers that treat a line beginning with ``<`` as pre-built HTML (the
    report's markdown/HTML mixer does) only leak when the untrusted value
    carries a newline, so the sweep has to include that shape.
    """
    return f"ZZ{tag}ZZ\n{TAG}\nZZ{tag}END"


def payload_trajectory() -> dict:
    """OpenCode-shaped export with a payload in every field a loader reads."""
    return {
        "info": {
            "id": _pay("SESSID"),
            "slug": _pay("SLUG"),
            "title": _pay("TITLE"),
            "directory": "/home/" + _pay("DIR"),
            "version": _pay("VERSION"),
            "time": {"created": 1_000_000, "updated": 1_060_000},
            "summary": {"additions": 1, "deletions": 1},
        },
        "messages": [
            {
                "info": {"role": "user", "time": {"created": 1_000_000},
                         "sessionID": "ses_root"},
                "parts": [{"type": "text", "text": _pay("USERTEXT")}],
            },
            {
                "info": {
                    "role": "assistant",
                    "time": {"created": 1_001_000, "completed": 1_003_000},
                    "modelID": _pay("MODELID"),
                    "providerID": _pay("PROVIDERID"),
                    "agent": _pay("AGENT"),
                    "mode": _pay("MODE"),
                    "finish": _pay("FINISH"),
                    "sessionID": "ses_root",
                    "sessionTitle": _pay("SESSTITLE"),
                    "path": {"cwd": _pay("CWD"), "root": _pay("ROOT")},
                    "tokens": {"total": 400, "input": 200, "output": 150,
                               "reasoning": 10, "cache": {"read": 30, "write": 20}},
                },
                "parts": [
                    {"type": "text", "text": _pay("ASSTTEXT")},
                    {"type": "reasoning", "text": _pay("REASONING")},
                    {
                        "type": "tool",
                        "tool": _pay("TOOLNAME"),
                        "callID": _pay("CALLID"),
                        "state": {
                            "status": "completed",
                            "title": _pay("TOOLTITLE"),
                            "input": {
                                "command": _pay("CMD"),
                                "file_path": "/tmp/" + _pay("FILEPATH"),
                                "filePath": "/tmp/" + _pay("FILEPATH2"),
                                "pattern": _pay("PATTERN"),
                                "description": _pay("TOOLDESC"),
                                "prompt": _pay("TOOLPROMPT"),
                                "url": "http://x/" + _pay("URL"),
                            },
                            "output": _pay("TOOLOUT"),
                            "time": {"start": 1_001_500, "end": 1_002_000},
                            "metadata": {"exit": 0, "title": _pay("METATITLE")},
                        },
                    },
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "call_err",
                        "state": {
                            "status": "error",
                            "title": _pay("ERRTITLE"),
                            "input": {"command": _pay("ERRCMD"),
                                      "description": _pay("ERRDESC")},
                            "output": _pay("ERROUT"),
                            "error": _pay("ERRFIELD"),
                            "time": {"start": 1_002_000, "end": 1_002_500},
                            "metadata": {"exit": 1},
                        },
                    },
                ],
            },
            {
                "info": {
                    "role": "assistant",
                    "time": {"created": 1_003_000, "completed": 1_005_000},
                    "modelID": _pay("MODELID2"),
                    "agent": _pay("SUBAGENT"),
                    "isSubAgent": True,
                    "sessionID": _pay_nl("CHILDSESS"),
                    "sessionTitle": _pay_nl("CHILDTITLE"),
                    "parentSessionID": "ses_root",
                    "sessionDepth": 1,
                    "tokens": {"total": 120, "input": 60, "output": 50,
                               "cache": {"read": 5, "write": 5}},
                },
                "parts": [
                    {"type": "text", "text": _pay("SUBTEXT")},
                    {
                        "type": "tool",
                        "tool": "edit",
                        "callID": "call_edit",
                        "state": {
                            "status": "completed",
                            "title": _pay("EDITTITLE"),
                            "input": {"filePath": "/repo/" + _pay("EDITFILE") + ".py",
                                      "oldString": _pay("OLDSTR"),
                                      "newString": _pay("NEWSTR")},
                            "output": _pay("EDITOUT"),
                            "time": {"start": 1_003_100, "end": 1_003_400},
                            "metadata": {},
                        },
                    },
                ],
            },
        ],
    }


# Slots that are not HTML sinks. Each exemption is a claim about how the value
# is consumed, not a convenience: if one stops holding, the payload becomes
# live. Anything not listed here is swept.
EXEMPT_SLOTS = {
    # gr.State: python objects handed back to callbacks, never inserted in the DOM.
    "state_steps": "gr.State",
    "state_raw": "gr.State",
    "state_analysis_brief": "gr.State (LLM prompt text)",
    # gr.Code renders into a CodeMirror text node.
    "raw_json": "gr.Code",
    # gr.Markdown defaults to sanitize_html=True (DOMPurify runs client-side).
    # The same strings are rendered WITHOUT sanitising by report.py — see
    # ReportInjectionTests.
    "behavior_md": "gr.Markdown(sanitize_html=True)",
    "hotspots_md": "gr.Markdown(sanitize_html=True)",
    "metrics_md": "gr.Markdown(sanitize_html=True)",
    "per_message_md": "gr.Markdown(sanitize_html=True)",
    # Plotly figures: plotly.js renders text through convertToTspans, which
    # drops every tag outside its own small whitelist.
    "agent_swimlane_chart": "plotly Figure",
    "agent_token_chart": "plotly Figure",
    "diag_file_chart": "plotly Figure",
    "diag_pressure_chart": "plotly Figure",
    "duration_chart": "plotly Figure",
    "plan_timeline_chart": "plotly Figure",
    "skill_chart": "plotly Figure",
    "token_chart": "plotly Figure",
    "tool_chart": "plotly Figure",
    "tool_duration_chart": "plotly Figure",
    "tool_outcome_chart": "plotly Figure",
    # Gradio binds these as component values (text / choices), not innerHTML.
    "diag_pressure_agent": "gr.Dropdown choices",
    "diag_usage_snapshot": "gr.Dropdown choices",
    "diag_window_limit": "gr.Number",
    "wf_filter_hidden": "gr.Textbox",
}

# Presenter exports that do not produce HTML, so the sweep does not drive them.
NON_HTML_PRESENTER_EXPORTS = {
    "ALL_FEATURE_FILTER", "DETAIL_PLACEHOLDER", "FEATURE_FILTERS",
    "FILTER_CHIPS_DEFAULT", "ROLE_FILTERS",
    "empty_plotly_fig", "build_chart_outputs", "build_label_ui_payload",
    "filter_workflow_steps", "raw_json_text", "trajectory_format_label",
}

# Presenter exports the sweep drives directly (in addition to everything
# pack_load_outputs reaches).
SWEPT_PRESENTER_EXPORTS = {
    "build_antipattern_html", "build_diagnostics_outputs",
    "build_filtered_workflow_outputs", "build_overview_issues_html",
    "build_overview_kpi_html", "build_overview_outputs", "build_summary_outputs",
    "build_workflow_outputs", "load_warnings_html",
    "render_failure_patterns_html", "render_tool_sequences_html",
}

# Presenter return keys that are Gradio component updates, not HTML — the same
# exemption `EXEMPT_SLOTS` records for the load slot they fill.
EXEMPT_PRESENTER_KEYS = (
    ".diag_pressure_dropdown",  # gr.Dropdown choices
    ".diag_pressure_chart",     # plotly Figure
    ".diag_file_chart",         # plotly Figure
)

_B64_BLOB = re.compile(r'data-b64="([A-Za-z0-9+/=]+)"')
_SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)


def _expand(text: str) -> str:
    """Append any base64 blob the page decodes back into the DOM.

    The Workflow tab ships every pre-rendered step-detail panel as a base64
    ``data-b64`` attribute and decodes it client-side, so the encoded bytes are
    an HTML sink even though the served string looks inert.
    """
    for blob in _B64_BLOB.findall(text):
        try:
            text += base64.b64decode(blob).decode("utf-8", "replace")
        except (ValueError, TypeError):  # pragma: no cover - malformed blob
            pass
    return text


def _strings(obj, path="", depth=0):
    """Yield ``(path, value)`` for every string reachable in a rendered payload."""
    if depth > 25:
        return
    if isinstance(obj, str):
        yield path, obj
        return
    if hasattr(obj, "to_plotly_json"):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _strings(value, f"{path}.{key}", depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from _strings(value, f"{path}[{i}]", depth + 1)
        return
    attrs = getattr(obj, "__dict__", None)
    if isinstance(attrs, dict) and attrs:
        for key, value in attrs.items():
            yield from _strings(value, f"{path}.{key}", depth + 1)


def find_leaks(label, rendered):
    """Return a report line for every payload that survived unescaped."""
    leaks = []
    for path, value in _strings(rendered, label):
        if any(key in path for key in EXEMPT_PRESENTER_KEYS):
            continue
        expanded = _expand(value)
        for kind, needle in PAYLOADS.items():
            if needle in expanded:
                at = expanded.index(needle)
                leaks.append(
                    f"{kind} payload survives at {path}: "
                    f"...{expanded[max(0, at - 120):at + len(needle) + 20]!r}"
                )
    return leaks


class InjectionSweepTests(unittest.TestCase):
    """Drive the real load path and assert nothing renders the payload live."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="trajviz-injection-")
        path = os.path.join(cls.tmp, "payload.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload_trajectory(), handle)
        result = load_session(path)
        if isinstance(result, LoadError):  # pragma: no cover - fixture must load
            raise AssertionError(f"payload fixture failed to load: {result.message}")
        cls.session = result
        cls.packed = pack_load_outputs(result)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_payload_actually_reaches_the_rendered_surfaces(self):
        """Guard: the sweep is worthless if the payload never got that far.

        Assert the tagged marker shows up (escaped) in the dashboard HTML, so a
        future refactor that stops rendering these fields turns this test red
        instead of leaving the rest of the file vacuously green.
        """
        blob = "".join(
            _expand(value) for key, value in self.packed.items()
            if key not in EXEMPT_SLOTS and isinstance(value, str)
        )
        # The content categories a trajectory viewer exists to show; if one stops
        # being rendered the sweep quietly stops covering it.
        for field in ("TOOLNAME", "CMD", "TOOLOUT", "FILEPATH",
                      "USERTEXT", "ASSTTEXT", "MODELID"):
            self.assertIn(f"ZZ{field}ZZ", blob, f"{field} no longer rendered anywhere")
        self.assertIn("&lt;img src=x onerror=", blob)

    def test_no_load_slot_renders_the_payload_as_markup(self):
        """Every HTML slot the dashboard fills on load must escape the payload."""
        leaks = []
        for dark in (False, True):
            packed = self.packed if not dark else pack_load_outputs(self.session, dark=True)
            for slot, value in packed.items():
                if slot in EXEMPT_SLOTS:
                    continue
                leaks += find_leaks(f"pack[dark={dark}].{slot}", value)
        self.assertEqual(leaks, [], "\n".join(leaks))

    def test_every_load_slot_is_classified(self):
        """A newly added load slot must be swept or explicitly exempted.

        `pack_load_outputs` is the one place every tab's load output is merged,
        so keying the sweep off its slot registry is what makes a new rendered
        surface impossible to add without a decision being recorded here.
        """
        unknown = sorted(set(EXEMPT_SLOTS) - load_slot_keys())
        self.assertEqual(unknown, [], f"exemptions for slots that no longer exist: {unknown}")
        swept = load_slot_keys() - set(EXEMPT_SLOTS)
        self.assertTrue(swept, "every load slot is exempted — the sweep covers nothing")

    def test_truncated_identifier_fields_are_still_escaped(self):
        """Short payloads must be escaped in fields the pipeline truncates.

        `_display_agent_label` caps agent labels at 20 chars (metrics.py), so
        the long sweep payload is cut before its markup is recognisable and the
        sweep cannot see whether that sink escapes. A real sub-agent title of
        16 characters survives truncation whole, and the Overview KPI card is
        on the default-visible tab of every load — so this sink needs a payload
        that fits.
        """
        from trajviz.insight.presenters.overview import _agent_steps_breakdown

        short = "<svg onload=x()>"          # 16 chars: under the 20-char cap
        self.assertLess(len(short), 20)
        html_out = _agent_steps_breakdown([
            {"label": short, "agent_id": "a1", "step_count": 2},
            {"label": "main", "agent_id": "", "step_count": 1},
        ])
        self.assertNotIn(short, html_out)
        self.assertIn("&lt;svg", html_out)

    def test_presenters_escape_the_payload(self):
        """Presenters called outside the load packer must escape it too.

        The report exporter and the Workflow filter callbacks reach these
        builders directly, so covering only `pack_load_outputs` would leave
        those entry points unswept.
        """
        session = self.session
        steps = session.steps
        surfaces = {
            "issues": build_overview_issues_html(session),
            "kpi": build_overview_kpi_html(session.metrics, session.wall_clock,
                                           verdicts=session.verdicts,
                                           message_rows=session.message_rows,
                                           agent_summaries=session.agent_summaries),
            "summary": build_summary_outputs(session),
            "diagnostics": build_diagnostics_outputs(session),
            "patterns.tool": render_tool_sequences_html(session.tool_sequences),
            "patterns.failure": render_failure_patterns_html(session.failure_patterns),
            "patterns.antipattern": build_antipattern_html(session),
            "workflow": build_workflow_outputs(steps),
            "workflow.filtered": build_filtered_workflow_outputs(
                steps, "Assistant,User,All", "",
            ),
        }
        leaks = []
        for label, rendered in surfaces.items():
            leaks += find_leaks(label, rendered)
        self.assertEqual(leaks, [], "\n".join(leaks))

    def test_every_presenter_export_is_classified(self):
        """Adding a presenter forces a swept/not-a-sink decision here."""
        unclassified = sorted(
            set(PRESENTER_EXPORTS) - SWEPT_PRESENTER_EXPORTS - NON_HTML_PRESENTER_EXPORTS
        )
        self.assertEqual(
            unclassified, [],
            "new presenter exports must be added to the sweep or marked "
            f"non-HTML: {unclassified}",
        )

    def test_step_detail_panels_escape_the_payload(self):
        """The Workflow detail panel renders the rawest content in the app.

        Tool arguments, tool output, diffs and model prose all land here, and it
        is injected into the page by script from a base64 blob rather than
        server-rendered, so nothing downstream will escape it for us.
        """
        leaks = []
        for step in self.session.steps:
            leaks += find_leaks(f"detail[{step.get('index')}]", format_step_detail(step))
        self.assertEqual(leaks, [], "\n".join(leaks))


class ReportInjectionTests(unittest.TestCase):
    """The exported standalone report has no client-side sanitiser at all.

    `gr.Markdown` strips markup in the dashboard, so the same string that is
    merely ugly on screen is executable in the file a researcher opens.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="trajviz-injection-report-")
        cls._body = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def report_body(self):
        """Build the report once and share it across the assertions."""
        if type(self)._body is None:
            type(self)._body = self._report_body(payload_trajectory())
        return type(self)._body

    def _report_body(self, traj):
        """Report HTML with the embedded plotly ``<script>`` payloads removed.

        Chart data rides in a JSON literal inside `<script>`; plotly's encoder
        escapes ``<`` and ``/`` there (so a trajectory string cannot close the
        tag), and the JS never evaluates it as markup. Stripping those blocks
        keeps the assertions about markup contexts, not about JSON escaping —
        the sanity check below is what holds plotly to that promise.
        """
        path = os.path.join(self.tmp, "report-src.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(traj, handle)
        result = load_session(path)
        self.assertNotIsInstance(result, LoadError)
        doc = build_report_html(result)
        self.assertEqual(
            doc.count("<script"), doc.count("</script>"),
            "a trajectory string closed or opened a <script> tag in the report",
        )
        return _SCRIPT_BLOCK.sub("", doc)

    def test_inline_markup_cannot_break_out_of_an_attribute(self):
        """Quote characters from a trajectory must not reach an attribute raw."""
        body = self.report_body()
        self.assertIn("ZZTOOLNAMEZZ", body, "payload never reached the report")
        self.assertNotIn(ATTR_DQ, body)
        self.assertNotIn(ATTR_SQ, body)

    @unittest.expectedFailure
    def test_agent_name_with_a_newline_cannot_inject_markup(self):
        """report.py:287 — a markdown line starting with ``<`` is emitted verbatim.

        A sub-agent's session id is written raw into the Per-Message table, so a
        newline in it ends the table row and the next line is handed through as
        pre-built HTML. Remove this decorator when `_mixed_md_to_html` escapes
        untrusted lines (or the metrics markdown escapes role/agent/finish).
        """
        body = self.report_body()
        self.assertNotIn(TAG, body)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
