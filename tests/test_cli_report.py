"""Contract of the headless entry point: ``python -m trajviz.insight --report``.

The dashboard is interactive and gets exercised by hand; the ``--report`` CLI is
the only way TrajViz is driven from a script, a Makefile or a batch job, so its
*process* behaviour is the contract:

* a caller that pipes stdout must receive the output path and nothing else;
* a caller that checks ``$?`` must see a clean ``SystemExit(1)`` on every ingest
  failure, with a human sentence on stderr instead of a Python traceback — a
  traceback means the failure was never considered, and it buries the message a
  script author needs;
* the exported file is handed to other people, so it must be a parseable
  document that reaches no third-party host beyond the Plotly bundle and does
  not carry the generating machine's filesystem layout inside it;
* generating a report must never phone home — the README sells TrajViz as
  offline analytics and the exported HTML embeds verbatim trajectory content
  (prompts, shell commands, file paths);
* the report reuses the dashboard's presenters, so the two must never be able
  to disagree about a number for the same trajectory.

Everything here is hermetic: the trajectory is synthesised in a temp directory,
and no test depends on the developer's corpus, home directory or network.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from unittest.mock import patch
from urllib.parse import urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Elements that never carry an end tag; the balance checker must not expect one.
_VOID_ELEMENTS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
)

# ``src=``/``href=`` values that point at an absolute http(s) URL.
_EXTERNAL_REF = re.compile(r"""(?:src|href)\s*=\s*["']?(https?://[^"'>\s]+)""", re.IGNORECASE)

_SECTION_ID = re.compile(r"<section[^>]*\sid='([a-z0-9-]+)'")

_TAGS = re.compile(r"<[^>]+>")
_SCRIPTS = re.compile(r"<script.*?</script>", re.DOTALL | re.IGNORECASE)

# Module-level fixture: one synthesised trajectory and one exported report,
# reused by every class below so the suite pays for a single CLI export.
_TMP: tempfile.TemporaryDirectory | None = None
TMP_DIR = ""
TRAJECTORY_PATH = ""
REPORT_PATH = ""
REPORT_HTML = ""
REPORT_RUN: subprocess.CompletedProcess | None = None


def _tiny_trajectory() -> dict:
    """A Claude Code session small enough to read, broad enough to exercise the report.

    Three assistant turns, one successful tool call and one failing one, so the
    KPI row, the tool tables, the failure patterns and the charts all have
    something real to render.
    """
    return {
        "format": "ccsession-trajectory",
        "format_version": "1.0",
        "session": {
            "session_id": "sess-cli-test",
            "model": "test-model",
            "working_directory": "/workspace/project",
        },
        "trajectory": [
            {
                "index": 0,
                "role": "user",
                "timestamp": "2026-01-01T00:00:00.000Z",
                "content": [{"type": "text", "text": "fix the failing test"}],
            },
            {
                "index": 1,
                "role": "assistant",
                "timestamp": "2026-01-01T00:00:05.000Z",
                "usage": {"input_tokens": 1000, "output_tokens": 250, "cache_read_input_tokens": 40},
                "content": [
                    {"type": "text", "text": "Reading the module."},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}},
                ],
            },
            {
                "index": 2,
                "role": "user",
                "timestamp": "2026-01-01T00:00:07.000Z",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "print(1)\n"}],
            },
            {
                "index": 3,
                "role": "assistant",
                "timestamp": "2026-01-01T00:00:12.000Z",
                "usage": {"input_tokens": 1300, "output_tokens": 120},
                "content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "pytest"}}],
            },
            {
                "index": 4,
                "role": "user",
                "timestamp": "2026-01-01T00:00:20.000Z",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t2",
                        "is_error": True,
                        "content": "ImportError: no module named x",
                    }
                ],
            },
            {
                "index": 5,
                "role": "assistant",
                "timestamp": "2026-01-01T00:00:25.000Z",
                "usage": {"input_tokens": 1500, "output_tokens": 60},
                "content": [{"type": "text", "text": "Fixed the import."}],
            },
        ],
        "sub_agents": [],
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke the module entry point as a real process, so exit codes are the real thing."""
    return subprocess.run(
        [sys.executable, "-m", "trajviz.insight", *args],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=REPO_ROOT),
        cwd=REPO_ROOT,
        timeout=180,
        check=False,
    )


def setUpModule() -> None:
    global _TMP, TMP_DIR, TRAJECTORY_PATH, REPORT_PATH, REPORT_HTML, REPORT_RUN
    _TMP = tempfile.TemporaryDirectory(prefix="trajviz-cli-test-")
    TMP_DIR = _TMP.name
    TRAJECTORY_PATH = os.path.join(TMP_DIR, "session.json")
    with open(TRAJECTORY_PATH, "w", encoding="utf-8") as handle:
        json.dump(_tiny_trajectory(), handle)
    REPORT_PATH = os.path.join(TMP_DIR, "report.html")
    REPORT_RUN = _run_cli("--report", TRAJECTORY_PATH, "-o", REPORT_PATH)
    if REPORT_RUN.returncode == 0:
        with open(REPORT_PATH, encoding="utf-8") as handle:
            REPORT_HTML = handle.read()


def tearDownModule() -> None:
    if _TMP is not None:
        _TMP.cleanup()


class _BalanceChecker(HTMLParser):
    """Collects unbalanced-tag problems so a report can be asserted parseable."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.open_tags: list[str] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag not in _VOID_ELEMENTS:
            self.open_tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_ELEMENTS:
            return
        if tag not in self.open_tags:
            self.problems.append(f"stray </{tag}>")
            return
        index = len(self.open_tags) - 1 - self.open_tags[::-1].index(tag)
        if index != len(self.open_tags) - 1:
            self.problems.append(f"</{tag}> closes over unclosed {self.open_tags[index + 1:]}")
        self.open_tags = self.open_tags[:index]


def _visible_text(document: str) -> str:
    """Rendered text of a report: markup and Plotly payloads removed."""
    return re.sub(r"\s+", " ", _TAGS.sub(" ", _SCRIPTS.sub(" ", document))).strip()


class ReportCliProcessContractTests(unittest.TestCase):
    """Exit status, stdout/stderr split and output placement of a successful export."""

    def test_help_exits_zero_and_documents_every_flag(self) -> None:
        """``--help`` is the only discovery surface a script author has."""
        result = _run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--report", "--output", "--dark", "--share", "--port", "--host"):
            self.assertIn(flag, result.stdout)

    def test_report_exits_zero_and_prints_only_the_output_path(self) -> None:
        """Scripts pipe stdout straight into the next command, so it must be the path alone."""
        self.assertEqual(REPORT_RUN.returncode, 0, REPORT_RUN.stderr)
        self.assertEqual(REPORT_RUN.stdout.strip(), REPORT_PATH)
        self.assertEqual(REPORT_RUN.stderr, "")

    def test_report_is_a_parseable_html_document_with_the_sections_it_claims(self) -> None:
        """The header advertises Overview, charts and Patterns; the body must contain them."""
        self.assertTrue(REPORT_HTML.startswith("<!DOCTYPE html>"))
        checker = _BalanceChecker()
        checker.feed(REPORT_HTML)
        checker.close()
        self.assertEqual(checker.problems, [], "report HTML is not well-formed")
        self.assertEqual(checker.open_tags, [], "report HTML left elements unclosed")
        ids = set(_SECTION_ID.findall(REPORT_HTML))
        for section in ("summary", "deep-dive", "tools", "diagnostics", "patterns"):
            self.assertIn(section, ids)

    def test_report_without_an_output_flag_prints_a_path_that_exists(self) -> None:
        """The help promises a temp file whose path is printed; a caller has nothing else to go on."""
        result = _run_cli("--report", TRAJECTORY_PATH)
        self.assertEqual(result.returncode, 0, result.stderr)
        printed = result.stdout.strip()
        try:
            self.assertTrue(os.path.isfile(printed), printed)
            self.assertEqual(os.path.basename(printed), "session-trajviz-report.html")
        finally:
            shutil.rmtree(os.path.dirname(printed), ignore_errors=True)

    def test_output_directory_receives_the_name_derived_from_the_trajectory(self) -> None:
        """``-o <dir>`` must place the report inside it, not create a file named after the dir."""
        destination = os.path.join(TMP_DIR, "reports")
        os.makedirs(destination, exist_ok=True)
        result = _run_cli("--report", TRAJECTORY_PATH, "-o", destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = os.path.join(destination, "session-trajviz-report.html")
        self.assertEqual(result.stdout.strip(), expected)
        self.assertTrue(os.path.isfile(expected))


class ReportCliFailureModeTests(unittest.TestCase):
    """Every ingest failure must be a clean exit 1 with a sentence on stderr.

    One case runs as a real subprocess to pin the process exit status; the rest
    drive ``main()`` in-process, where anything other than ``SystemExit`` — the
    exception escaping as a traceback — fails the test directly.
    """

    def _assert_clean_exit(self, path: str) -> str:
        from trajviz.insight.__main__ import main

        out, err = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", ["trajviz", "--report", path]),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(out.getvalue(), "", "a failed run must print no path for a caller to consume")
        self.assertTrue(err.getvalue().strip(), "a failed run must explain itself on stderr")
        return err.getvalue()

    def test_missing_file_fails_cleanly_as_a_process(self) -> None:
        """The commonest CLI mistake is a wrong path; it must not look like a crash."""
        result = _run_cli("--report", os.path.join(TMP_DIR, "absent.json"))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.strip())
        self.assertNotIn("Traceback (most recent call last)", result.stderr)

    def test_directory_input_fails_cleanly(self) -> None:
        """A directory that is not a DSH session is not loadable and must be reported, not raised."""
        plain_dir = os.path.join(TMP_DIR, "just-a-dir")
        os.makedirs(plain_dir, exist_ok=True)
        self._assert_clean_exit(plain_dir)

    def test_log_input_is_refused_with_an_explanation(self) -> None:
        """``.log`` support was removed; the refusal must name the reason, not raise."""
        log_path = os.path.join(TMP_DIR, "legacy.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("2026-01-01 event\n")
        self.assertIn(".log", self._assert_clean_exit(log_path))

    def test_unparseable_file_fails_cleanly(self) -> None:
        """Truncated or non-JSON input is routine in a corpus and must not traceback."""
        broken = os.path.join(TMP_DIR, "broken.json")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self._assert_clean_exit(broken)

    def test_undetectable_format_fails_cleanly(self) -> None:
        """Valid JSON of an unknown shape must be refused with a message, not a crash."""
        unknown = os.path.join(TMP_DIR, "unknown.json")
        with open(unknown, "w", encoding="utf-8") as handle:
            json.dump({"hello": "world"}, handle)
        self._assert_clean_exit(unknown)


class ReportOfflineGuaranteeTests(unittest.TestCase):
    """The exported file is a shareable artifact; nothing about it may leak or phone home."""

    def test_generating_a_report_makes_no_network_or_subprocess_call(self) -> None:
        """No trajectory content may leave the machine when a report is exported.

        Driven through the real ``__main__.main`` with a CPython audit hook armed
        *before* any trajviz import, so an egress anywhere on the path — including
        one added later by a new presenter or an LLM helper — fails the run. The
        child also proves the hook is live, so a green result can never be vacuous.
        """
        probe = os.path.join(TMP_DIR, "audited_run.py")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write(
                "import socket, sys\n"
                "BLOCKED = ('socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname',\n"
                "           'urllib.Request', 'http.client.connect', 'subprocess.Popen', 'os.system')\n"
                "def _hook(name, args):\n"
                "    if name in BLOCKED:\n"
                "        raise RuntimeError('EGRESS ATTEMPT: ' + name)\n"
                "sys.addaudithook(_hook)\n"
                "from trajviz.insight.__main__ import main\n"
                "sys.argv = ['trajviz', '--report', sys.argv[1], '-o', sys.argv[2]]\n"
                "main()\n"
                "try:\n"
                "    socket.getaddrinfo('blocked.invalid', 80)\n"
                "except RuntimeError:\n"
                "    print('HOOK-LIVE')\n"
                "else:\n"
                "    print('HOOK-DEAD')\n"
            )
        destination = os.path.join(TMP_DIR, "audited.html")
        result = subprocess.run(
            [sys.executable, probe, TRAJECTORY_PATH, destination],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=REPO_ROOT),
            cwd=REPO_ROOT,
            timeout=180,
            check=False,
        )
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout!r} stderr={result.stderr!r}")
        self.assertIn("HOOK-LIVE", result.stdout, "the audit hook was not armed; the result would be vacuous")
        self.assertTrue(os.path.isfile(destination))

    def test_report_reaches_no_third_party_host_other_than_the_plotly_bundle(self) -> None:
        """An exported report must not fetch fonts, images, trackers or scripts from anywhere else.

        Inlining the Plotly bundle (so the file is genuinely standalone) only
        removes references, so this invariant holds before and after that change.
        """
        for url in _EXTERNAL_REF.findall(REPORT_HTML):
            host = urlparse(url).netloc
            self.assertEqual(host, "cdn.plot.ly", f"report references a third-party host: {url}")
            self.assertTrue(url.endswith(".js"), f"unexpected CDN asset: {url}")

    def test_report_does_not_embed_the_generating_machines_paths(self) -> None:
        """Reports get emailed and attached; they must not carry the author's directory layout."""
        self.assertNotIn(TMP_DIR, REPORT_HTML)
        self.assertNotIn(REPO_ROOT, REPORT_HTML)
        # The trajectory's own name is intentionally shown; its location is not.
        self.assertIn("session.json", REPORT_HTML)


class ReportAgreesWithDashboardTests(unittest.TestCase):
    """The report and the live dashboard must never state different numbers."""

    @classmethod
    def setUpClass(cls) -> None:
        from trajviz.insight.session import LoadError, load_session

        session = load_session(TRAJECTORY_PATH)
        assert not isinstance(session, LoadError), session
        cls.session = session

    def test_every_number_the_dashboard_shows_also_appears_in_the_report(self) -> None:
        """Both surfaces claim to describe the same session; a divergence is a wrong number.

        The report is built from the same presenters as the Overview tab. If a
        future change repoints one of them, this catches the split before a
        researcher quotes two different figures for one trajectory.
        """
        from trajviz.insight.report import build_report_html
        from trajviz.insight.ui.overview_tab import pack_load

        dashboard = pack_load(self.session)
        report_text = _visible_text(build_report_html(self.session))

        for slot in ("metrics_md", "behavior_md", "hotspots_md", "per_message_md"):
            numbers = re.findall(r"\d[\d,.]*", _visible_text(dashboard[slot]))
            self.assertTrue(numbers, f"{slot} produced no numbers to compare")
            missing = [n for n in numbers if n not in report_text]
            self.assertEqual(missing, [], f"{slot}: report is missing dashboard numbers {missing[:5]}")

        kpi = dashboard["overview_kpi_html"]
        kpi_html = kpi["value"] if isinstance(kpi, dict) else kpi
        self.assertIn(_visible_text(kpi_html), report_text)

    def test_dark_and_light_reports_differ_only_in_theming(self) -> None:
        """``--dark`` is a rendering switch; it must not change a single reported value."""
        from trajviz.insight.report import build_report_html

        light = build_report_html(self.session, dark=False)
        dark = build_report_html(self.session, dark=True)
        self.assertEqual(_visible_text(light), _visible_text(dark))
        self.assertEqual(
            sorted(re.findall(r'"y":\s*\[[^\]]*\]', light)),
            sorted(re.findall(r'"y":\s*\[[^\]]*\]', dark)),
            "dark mode changed plotted data",
        )
        self.assertIn("content='light'", light)
        self.assertIn("content='dark'", dark)


class DashboardFlagWiringTests(unittest.TestCase):
    """``--host``/``--port``/``--share`` must reach Gradio unchanged."""

    def _launch_kwargs(self, argv: list[str]) -> dict:
        import trajviz.insight.insight as insight_module

        recorded: dict = {}

        class _Recorder:
            def launch(self, **kwargs: object) -> None:
                recorded.update(kwargs)

        from trajviz.insight.__main__ import main

        with (
            patch.object(insight_module, "build_ui", lambda: _Recorder()),
            patch.object(sys, "argv", ["trajviz", *argv]),
        ):
            main()
        return recorded

    def test_defaults_bind_to_loopback_without_a_public_link(self) -> None:
        """A bare launch must not expose the dashboard beyond the machine it runs on."""
        kwargs = self._launch_kwargs([])
        self.assertEqual(kwargs["server_name"], "127.0.0.1")
        self.assertEqual(kwargs["server_port"], 7860)
        self.assertFalse(kwargs["share"])

    def test_share_host_and_port_are_forwarded(self) -> None:
        """Silently dropping one of these would bind somewhere the operator did not ask for."""
        kwargs = self._launch_kwargs(["--share", "--host", "0.0.0.0", "--port", "8123"])
        self.assertEqual(kwargs["server_name"], "0.0.0.0")
        self.assertEqual(kwargs["server_port"], 8123)
        self.assertTrue(kwargs["share"])


if __name__ == "__main__":
    unittest.main()
