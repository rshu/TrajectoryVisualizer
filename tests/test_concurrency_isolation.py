"""Process-global state and concurrency isolation for the dashboard load path.

One Gradio process serves every browser session, so anything the load or export
path keeps at module level, in ``os.environ``, or at a shared filesystem path is
shared by every viewer at once. These tests pin the properties that make that
safe:

* a load is a pure function of the file — two viewers loading different
  trajectories at the same moment must each get their own numbers back, and a
  swap would silently publish one run's metrics under another run's name;
* loads are order-independent — loading B (or any other format) between two
  loads of A must reproduce A exactly, so a session's numbers cannot depend on
  what the previous viewer happened to open;
* the load path leaves no footprint in ``os.environ``, ``sys.path`` or any
  module-level table, so one viewer cannot reconfigure another's session;
* each HTML export lands in its own directory, so one viewer's report can never
  overwrite — or be served in place of — another's.

Fixtures are synthesized into a tmpdir, so nothing here needs the real corpus,
a network, or the developer's home directory.
"""

from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from trajviz.insight.loaders import load_trajectory
from trajviz.insight.rendering import _md_to_html_preview
from trajviz.insight.report import write_report_file
from trajviz.insight.session import LoadError, LoadedSession, load_session
from trajviz.insight.ui import upload
from trajviz.insight.ui.load import load_slot_keys, pack_load_outputs

_REPO_FIXTURES = Path(__file__).parent / "fixtures"
_THREADS = 4


def _write_jsonl(path: Path, events: list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def _write_pi(path: Path, *, input_tokens: int) -> None:
    """Minimal Pi event stream: a user turn, a tool-calling turn, a final answer."""

    def msg(role: str, ts: str, content: list, **extra) -> dict:
        message = {"role": role, "content": content, "timestamp": 1_787_535_033_022}
        message.update(extra)
        return {"type": "message", "id": f"id-{role}-{ts}", "timestamp": ts, "message": message}

    _write_jsonl(
        path,
        [
            {"type": "session", "version": 3, "id": "sess-pi", "timestamp": "2026-08-24T01:29:43.221Z", "cwd": "/p"},
            {
                "type": "model_change",
                "id": "m1",
                "timestamp": "2026-08-24T01:30:18.190Z",
                "provider": "zenmux",
                "modelId": "z-ai/glm-5.3",
            },
            msg("user", "2026-08-24T01:30:33.026Z", [{"type": "text", "text": "explore"}]),
            msg(
                "assistant",
                "2026-08-24T01:33:02.137Z",
                [{"type": "toolCall", "id": "call-1", "name": "bash", "arguments": {"command": "ls"}}],
                stopReason="toolUse",
                provider="zenmux",
                model="z-ai/glm-5.3",
                usage={"input": input_tokens, "output": 87, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 1},
            ),
            msg(
                "toolResult",
                "2026-08-24T01:33:03.369Z",
                [{"type": "text", "text": "ok"}],
                toolCallId="call-1",
                toolName="bash",
                isError=False,
            ),
            msg(
                "assistant",
                "2026-08-24T01:33:36.288Z",
                [{"type": "text", "text": "done"}],
                stopReason="stop",
                usage={"input": input_tokens + 10, "output": 782, "totalTokens": 2},
            ),
        ],
    )


def _write_dsh(path: Path, *, input_tokens: int) -> None:
    """Minimal DeepSeek-Harness event stream (parent session only, no subagents)."""

    def evt(etype: str, seq: int, time: int, data: dict) -> dict:
        return {"type": etype, "seq": seq, "time": time, "data": data}

    _write_jsonl(
        path,
        [
            {
                "type": "session",
                "version": 0,
                "id": "session-parent",
                "createdAt": 1_787_623_647_203,
                "cwd": "/p",
                "delegationDepth": 0,
                "agentPreset": "standard",
            },
            evt("request/context", 1, 1_787_623_647_211, {"provider": "deepseek-official", "model": "deepseek-v4"}),
            evt("session/title", 2, 1_787_623_647_212, {"title": "T"}),
            evt(
                "user/message",
                3,
                1_787_623_647_300,
                {"content": [{"type": "text", "text": "go"}], "source": {"kind": "user"}, "role": "user", "id": "u1"},
            ),
            evt(
                "assistant/message",
                5,
                1_787_623_647_400,
                {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool-call",
                                "id": "call-1",
                                "name": "bash",
                                "arguments": json.dumps({"command": "ls"}),
                            }
                        ],
                        "source": {"kind": "model", "provider": "deepseek-official", "model": "deepseek-v4"},
                        "id": "a1",
                    },
                    "usage": {
                        "inputTokens": input_tokens,
                        "outputTokens": 20,
                        "cacheReadTokens": 10,
                        "reasoningTokens": 5,
                    },
                },
            ),
            evt(
                "tool/call",
                6,
                1_787_623_647_401,
                {"turn": 1, "step": 1, "callId": "call-1", "name": "bash", "arguments": json.dumps({"command": "ls"})},
            ),
            evt(
                "tool/result",
                7,
                1_787_623_647_900,
                {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "source": {"kind": "tool", "callId": "call-1"},
                        "content": [
                            {
                                "type": "tool-result",
                                "toolCallId": "call-1",
                                "content": [{"type": "text", "text": "ok"}],
                                "isError": False,
                            }
                        ],
                        "role": "user",
                        "id": "res-1",
                    },
                },
            ),
        ],
    )


def _write_opencode(path: Path, *, title: str, assistant_steps: int, input_tokens: int) -> None:
    """Minimal but complete OpenCode session: one user turn, N tool-calling turns."""
    messages: list[dict] = [
        {
            "info": {
                "role": "user",
                "time": {"created": 1_700_000_000_000, "completed": 1_700_000_000_500},
                "id": "msg_u",
                "sessionID": "ses_iso",
            },
            "parts": [{"type": "text", "text": f"Task: {title}", "id": "prt_u"}],
        }
    ]
    for i in range(assistant_steps):
        created = 1_700_000_001_000 + i * 2_000
        messages.append(
            {
                "info": {
                    "role": "assistant",
                    "modelID": "model-x",
                    "providerID": "prov",
                    "agent": "build",
                    "time": {"created": created, "completed": created + 1_000},
                    "tokens": {
                        "total": input_tokens + i + 1,
                        "input": input_tokens,
                        "output": i + 1,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                    "finish": "tool-calls",
                    "id": f"msg_a{i}",
                    "sessionID": "ses_iso",
                },
                "parts": [
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": f"call{i}",
                        "id": f"prt{i}",
                        "state": {
                            "status": "completed",
                            "input": {"command": f"echo {i}"},
                            "output": "ok",
                            "time": {"start": created, "end": created + 800},
                            "metadata": {"exit": 0},
                        },
                    }
                ],
            }
        )
    payload = {
        "info": {
            "id": "ses_iso",
            "title": title,
            "version": "1.0",
            "time": {"created": 1_700_000_000_000, "updated": 1_700_000_001_000 + assistant_steps * 2_000},
        },
        "messages": messages,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _digest(path: str, *, dark: bool = False) -> str:
    """Stable fingerprint of everything the load handler hands the UI."""
    session = load_session(path)
    if isinstance(session, LoadError):
        return f"LoadError:{session.code}"
    packed = pack_load_outputs(session, dark=dark)
    flat: dict[str, str] = {}
    for key, value in packed.items():
        to_json = getattr(value, "to_json", None)  # Plotly figures
        flat[str(key)] = to_json() if callable(to_json) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(json.dumps(flat, sort_keys=True).encode()).hexdigest()


class _CorpusMixin:
    """Trajectories across every loader family, each with its own numbers."""

    tmp: str
    files: list[str]
    synthesized: list[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="trajviz-iso-")
        alpha = Path(cls.tmp) / "alpha_session.json"
        beta = Path(cls.tmp) / "beta_session.json"
        _write_opencode(alpha, title="alpha", assistant_steps=3, input_tokens=1_000)
        _write_opencode(beta, title="beta", assistant_steps=5, input_tokens=2_000)
        cls.synthesized = [str(alpha), str(beta)]
        cls.files = list(cls.synthesized)
        # The PR-added event-stream converters live in their own modules now, so
        # replay them alongside the object formats.
        pi = Path(cls.tmp) / "pi_session.jsonl"
        _write_pi(pi, input_tokens=1_728)
        dsh_dir = Path(cls.tmp) / "dsh_export"
        dsh_dir.mkdir()
        _write_dsh(dsh_dir / "session.jsonl", input_tokens=100)
        cls.files += [str(pi), str(dsh_dir / "session.jsonl")]
        for name in ("codearts_minimal.json", "cursor_minimal.json", "icode_minimal.json"):
            candidate = _REPO_FIXTURES / name
            if candidate.is_file():
                cls.files.append(str(candidate))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)


class ConcurrentLoadIsolationTests(_CorpusMixin, unittest.TestCase):
    def test_threads_loading_different_trajectories_keep_their_own_results(self):
        """Cross-contamination here would publish one run's metrics under another run's name.

        Eight browser tabs share one process; the load pipeline must behave as a
        pure function of the file it was given, not of whatever landed in the
        process most recently.
        """
        baseline = {path: _digest(path) for path in self.files}
        work = self.files * 2
        for _round in range(2):
            with futures.ThreadPoolExecutor(max_workers=_THREADS) as pool:
                got = list(pool.map(_digest, work))
            for path, digest in zip(work, got, strict=True):
                self.assertEqual(digest, baseline[path], f"concurrent load of {os.path.basename(path)} differs")

    def test_threads_get_the_metrics_of_the_trajectory_they_asked_for(self):
        """The published numbers, not just an opaque digest, must follow the input.

        The two synthesized runs are built with different step counts and token
        totals precisely so that a swapped result is detectable.
        """

        def load_numbers(path: str) -> tuple[str, tuple[int, int]]:
            session = load_session(path)
            self.assertIsInstance(session, LoadedSession)
            return path, (session.steps_total, int(session.metrics["input_tokens"]))

        serial = dict(load_numbers(path) for path in self.synthesized)
        self.assertEqual(len(set(serial.values())), len(serial), "fixtures are not distinguishable")

        targets = self.synthesized * 3
        with futures.ThreadPoolExecutor(max_workers=_THREADS) as pool:
            for path, numbers in pool.map(load_numbers, targets):
                self.assertEqual(numbers, serial[path], os.path.basename(path))

    def test_threads_loading_the_same_trajectory_agree(self):
        """Same input, many readers: any disagreement means shared mutable state."""
        path = self.files[0]
        baseline = _digest(path)
        with futures.ThreadPoolExecutor(max_workers=_THREADS) as pool:
            results = list(pool.map(_digest, [path] * (_THREADS * 2)))
        self.assertEqual(set(results), {baseline})


class LoadReentrancyTests(_CorpusMixin, unittest.TestCase):
    def test_reloading_a_after_b_reproduces_a(self):
        """A viewer who reopens a trajectory must see the same numbers as the first time."""
        first, second = self.files[0], self.files[1]
        before = _digest(first)
        _digest(second)
        self.assertEqual(_digest(first), before)

    def test_every_format_replays_identically_after_the_others(self):
        """Format converters now live in separate modules; none may carry state between loads."""
        first_pass = [_digest(path) for path in self.files]
        second_pass = [_digest(path) for path in self.files]
        for path, before, after in zip(self.files, first_pass, second_pass, strict=True):
            self.assertEqual(before, after, f"{os.path.basename(path)} changed on replay")

    def test_a_dark_render_does_not_change_the_next_light_render(self):
        """Theme is a per-request argument, not process state."""
        path = self.files[0]
        light = _digest(path)
        _digest(path, dark=True)
        self.assertEqual(_digest(path), light)

    def test_two_sessions_of_one_file_share_no_mutable_structure(self):
        """Each viewer gets its own parse, not a shared object graph.

        ``steps`` and ``raw`` are handed to per-session ``gr.State`` and read
        back by every later handler (workflow filter, utilization rebuild,
        export). If two loads of the same path returned the same objects, one
        viewer's tab could edit what another viewer is shown.
        """
        first, second = load_session(self.files[0]), load_session(self.files[0])
        self.assertIsInstance(first, LoadedSession)
        self.assertIsInstance(second, LoadedSession)
        self.assertIsNot(first.raw, second.raw)
        self.assertIsNot(first.steps, second.steps)
        if first.steps:
            self.assertIsNot(first.steps[0], second.steps[0])
        before = json.dumps(second.steps, sort_keys=True, default=str)
        first.steps.append({"index": 9_999, "role": "user"})
        first.raw["_injected"] = True
        self.assertEqual(json.dumps(second.steps, sort_keys=True, default=str), before)
        self.assertNotIn("_injected", second.raw)


class SharedRendererTests(unittest.TestCase):
    """The markdown/code renderer keeps one module-level Pygments formatter."""

    def test_concurrent_code_rendering_matches_serial_rendering(self):
        """One formatter object serves every viewer's Workflow cards at once.

        If it carried per-call state, two viewers rendering different code
        blocks simultaneously could be shown each other's highlighted output.
        """
        langs = ["python", "bash", "json", "javascript", "diff", "sql", "yaml", "text"]
        snippets = [
            f"prose {i}\n\n```{lang}\n{{'k': [1, 2, 3]}}\ndef f_{i}(x):\n    return x + {i}\n```\ntail {i}"
            for i, lang in enumerate(langs)
        ]
        expected = {snippet: _md_to_html_preview(snippet) for snippet in snippets}
        work = snippets * 4
        with futures.ThreadPoolExecutor(max_workers=_THREADS) as pool:
            got = list(pool.map(_md_to_html_preview, work))
        for snippet, rendered in zip(work, got, strict=True):
            self.assertEqual(rendered, expected[snippet])


class ProcessGlobalFootprintTests(_CorpusMixin, unittest.TestCase):
    def test_loading_trajectories_leaves_environment_and_sys_path_untouched(self):
        """A load that writes os.environ/sys.path would reconfigure every other viewer.

        ``attribution`` and ``llm_config`` deliberately do write process-global
        state, but only at import time and from ``build_ui`` — never from a load.
        """
        _digest(self.files[0])  # warm every lazy import before snapshotting
        env_before = dict(os.environ)
        path_before = list(sys.path)
        for path in self.files:
            _digest(path)
            _digest(path, dark=True)
        self.assertEqual(dict(os.environ), env_before)
        self.assertEqual(list(sys.path), path_before)

    def test_loading_trajectories_does_not_mutate_module_level_tables(self):
        """Tool-name sets, colour maps and label tables are shared by all sessions.

        If a load mutated one, the next viewer would be rendered against the
        previous viewer's data. Only ``ui.load._LOAD_SLOT_KEYS`` is allowed to
        change, and only once, from ``None`` to its memoized value.
        """
        load_slot_keys()  # settle the one legitimate memo
        _digest(self.files[0])

        def snapshot() -> dict[str, str]:
            out: dict[str, str] = {}
            for mod_name, module in list(sys.modules.items()):
                if not mod_name.startswith("trajviz."):
                    continue
                for attr, value in list(vars(module).items()):
                    if attr.startswith("__") or not isinstance(value, (dict, list, set, frozenset, tuple)):
                        continue
                    out[f"{mod_name}.{attr}"] = json.dumps(value, sort_keys=True, default=repr)
            return out

        before = snapshot()
        for path in self.files:
            _digest(path)
            _digest(path, dark=True)
        after = snapshot()
        drifted = [key for key in before.keys() & after.keys() if before[key] != after[key]]
        self.assertEqual(drifted, [])


class ExportIsolationTests(_CorpusMixin, unittest.TestCase):
    def setUp(self):
        self._saved_export_dir = upload._last_temp_export_dir
        self._made: list[str] = []

    def tearDown(self):
        upload._last_temp_export_dir = self._saved_export_dir
        for directory in self._made:
            shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _armed_path(update) -> str | None:
        value = update["value"] if isinstance(update, dict) else getattr(update, "value", None)
        return value if isinstance(value, str) else None

    def test_each_export_lands_in_its_own_directory_with_its_own_report(self):
        """Two viewers exporting at once must never be handed the same path.

        The export temp dir is process-global bookkeeping; a shared directory or
        filename would let one viewer's report overwrite another's, or be served
        in its place.
        """
        names = [os.path.basename(p) for p in self.synthesized]
        armed: list[str] = []
        for path in self.synthesized:
            raw = load_trajectory(path)
            update = upload.prepare_html_export(raw, None, False)
            armed_path = self._armed_path(update)
            self.assertIsNotNone(armed_path, f"export was not armed for {os.path.basename(path)}")
            self._made.append(os.path.dirname(armed_path))
            # The path handed back must be live at the moment it is handed back:
            # Gradio reads it in the same response to serve the download.
            self.assertTrue(os.path.isfile(armed_path), "armed export path does not exist")
            body = Path(armed_path).read_text(encoding="utf-8")
            own = os.path.basename(path)
            self.assertIn(own, body, "exported report does not name its own trajectory")
            for other in names:
                if other != own:
                    self.assertNotIn(other, body, "exported report mentions another viewer's trajectory")
            armed.append(armed_path)

        self.assertEqual(len(set(armed)), len(armed), "two exports shared a file path")
        self.assertEqual(
            len({os.path.dirname(p) for p in armed}), len(armed), "two exports shared a directory"
        )

    def test_a_failed_or_empty_load_disarms_the_export_button(self):
        """An un-loadable trajectory must not leave the previous viewer's report downloadable."""
        for payload in ({}, {"_error": "boom"}, "not-a-dict"):
            update = upload.prepare_html_export(payload, None, False)
            self.assertIsNone(self._armed_path(update), f"export stayed armed for {payload!r}")

    def test_only_a_directory_this_module_created_is_ever_registered_for_deletion(self):
        """The export bookkeeping deletes directories, so what it records matters.

        ``_replace_temp_export_dir`` records a directory and later ``rmtree``s
        it. Recording anything but a scratch directory this module itself made
        would turn a later export into data loss in a real directory.
        """
        upload._last_temp_export_dir = None
        raw = load_trajectory(self.synthesized[0])
        armed = self._armed_path(upload.prepare_html_export(raw, None, False))
        self.assertIsNotNone(armed)
        self._made.append(os.path.dirname(armed))
        recorded = upload._last_temp_export_dir
        self.assertIsNotNone(recorded)
        self.assertTrue(
            os.path.basename(recorded).startswith(upload._TEMP_EXPORT_PREFIX),
            f"export bookkeeping armed deletion of an unrelated directory: {recorded}",
        )

    def test_an_unrelated_directory_is_never_deleted_by_the_next_export(self):
        """A directory this module did not create must survive an export.

        A prefix check is the only thing standing between this bookkeeping and
        ``shutil.rmtree`` on a path the user cares about.
        """
        bystander = tempfile.mkdtemp(prefix="trajviz-user-keepsafe-")
        self._made.append(bystander)
        keepsake = Path(bystander) / "important.html"
        keepsake.write_text("do not delete me", encoding="utf-8")
        upload._last_temp_export_dir = bystander
        raw = load_trajectory(self.synthesized[0])
        armed = self._armed_path(upload.prepare_html_export(raw, None, False))
        self.assertIsNotNone(armed)
        self._made.append(os.path.dirname(armed))
        self.assertTrue(keepsake.is_file(), "an export deleted a directory it did not create")

    def test_concurrent_report_writes_land_in_separate_files(self):
        """``write_report_file`` picks its own destination on every call.

        The CLI and the dashboard both reach it; a shared default destination
        would let one report overwrite another mid-write.
        """
        raws = [load_trajectory(path) for path in self.synthesized]
        work = raws * 2
        with futures.ThreadPoolExecutor(max_workers=_THREADS) as pool:
            written = list(pool.map(write_report_file, work))
        self.assertEqual(len(set(written)), len(written), "two concurrent reports shared a path")
        for raw, path in zip(work, written, strict=True):
            self._made.append(os.path.dirname(path))
            own = os.path.basename(str(raw.get("_source_path", "")))
            self.assertIn(own, Path(path).read_text(encoding="utf-8"))


class AttributionRootIsolationTests(unittest.TestCase):
    """DECAF is configured through process globals; every request must set its own."""

    @classmethod
    def setUpClass(cls) -> None:
        from trajviz.insight import attribution

        if not attribution.DECAF_AVAILABLE:
            raise unittest.SkipTest("DECAF (awe) is not importable in this environment")
        cls.attribution = attribution

    def setUp(self):
        self._saved_env = os.environ.get("AWE_ARGUS_ROOT")
        self._roots = [tempfile.mkdtemp(prefix=f"trajviz-root{i}-") for i in range(_THREADS)]
        for root in self._roots:
            os.makedirs(os.path.join(root, "data", "requirements"), exist_ok=True)

    def tearDown(self):
        self.attribution.configure(self.attribution._DEFAULT_ROOT)
        if self._saved_env is None:
            os.environ.pop("AWE_ARGUS_ROOT", None)
        else:
            os.environ["AWE_ARGUS_ROOT"] = self._saved_env
        for root in self._roots:
            shutil.rmtree(root, ignore_errors=True)

    def test_concurrent_diagnoses_each_use_their_own_corpus_root(self):
        """``diagnose`` writes DECAF's process-global root, so it must not be inherited.

        Two viewers may point the Attribution tab at different corpus roots. If
        one request's root leaked into another's, a verdict would be grounded in
        a different corpus than the one the viewer named — and the answer would
        look authoritative anyway.
        """

        def run(root: str):
            result = self.attribution.diagnose(
                agent="claude_code", instance_id="trajviz_isolation_probe", argus_root=root
            )
            return root, result

        work = self._roots * 3
        with futures.ThreadPoolExecutor(max_workers=_THREADS) as pool:
            got = list(pool.map(run, work))
        for root, result in got:
            self.assertFalse(result.available)
            self.assertIsNotNone(result.reason)
            self.assertIn(str(Path(root).resolve()), result.reason, "diagnosis used another request's corpus root")
            for other in self._roots:
                if other != root:
                    self.assertNotIn(str(Path(other).resolve()), result.reason)


if __name__ == "__main__":
    unittest.main()
