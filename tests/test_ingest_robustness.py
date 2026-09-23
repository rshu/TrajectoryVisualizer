"""Ingest robustness: the never-raise contract of ``load_trajectory`` and metric invariants.

``run_group.build_run_group_scorecard`` calls ``load_trajectory`` with no
``try/except`` (``run_group.py:723``), so a loader that raises does not degrade
one row — it aborts the whole batch. The contract every test in this file
defends is therefore:

    for ANY bytes on disk, ``load_trajectory`` returns a dict that either
    carries ``_error`` (a clean, user-facing failure) or is a parsed
    trajectory. It never raises, never hangs, never consumes unbounded memory.

The second half defends ``compute_metrics``: numbers that reach a paper must be
finite, counts must not be negative, and a field whose name says *percent* must
be a percentage. These are asserted as properties over generated well-typed
payloads rather than pinned against today's output.

Tests marked ``@unittest.expectedFailure`` state a contract the code currently
breaks; each names the defect. **Remove the decorator when the defect is
fixed** — an unexpected success fails the run, which is the intended signal.
Nothing here pins the broken behaviour as correct.
"""

import json
import math
import os
import tempfile
import unittest
import zipfile

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from trajviz.insight.formats.parse import _parse_jsonl_events
from trajviz.insight.loaders import load_trajectory
from trajviz.insight.metrics import build_message_metrics, compute_metrics
from trajviz.insight.parser import parse_steps

# Metric keys whose name promises a non-negative quantity.
_NONNEG_NAME_HINTS = ("count", "total", "duration", "tokens", "seconds", "calls", "steps")


def _write_bytes(tmp, name, data):
    path = os.path.join(tmp, name)
    with open(path, "wb") as handle:
        handle.write(data if isinstance(data, bytes) else data.encode("utf-8"))
    return path


def _write_json(tmp, name, doc):
    return _write_bytes(tmp, name, json.dumps(doc))


def _codex_stream(extra=()):
    """A minimal, valid Codex rollout as a list of events."""
    return [
        {"type": "session_meta", "timestamp": "2026-01-05T12:00:00.000Z",
         "payload": {"id": "s1", "cwd": "/p/proj", "model": "gpt-5", "model_provider": "openai"}},
        {"type": "response_item", "timestamp": "2026-01-05T12:00:01.000Z",
         "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix it"}]}},
        {"type": "response_item", "timestamp": "2026-01-05T12:00:02.000Z",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "done"}]}},
        *extra,
    ]


def _jsonl_text(events):
    return "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)


def _opencode_doc(messages, info_time=None):
    return {
        "info": {"id": "s1", "time": info_time or {"created": 0, "updated": 10_000}},
        "messages": messages,
    }


def _opencode_message(*, index=0, role="assistant", tokens=None, parts=None, created=0, elapsed=1000):
    return {
        "role": role,
        "info": {
            "id": f"m{index}", "role": role,
            "time": {"created": created, "completed": created + elapsed},
            "tokens": tokens if tokens is not None else {
                "total": 100, "input": 40, "output": 60, "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        },
        "parts": parts if parts is not None else [],
    }


def _walk_numbers(obj, prefix=""):
    """Yield ``(dotted_key, value)`` for every scalar in a nested metrics dict."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk_numbers(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _walk_numbers(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _metrics_for(path):
    raw = load_trajectory(path)
    steps = parse_steps(raw)
    return compute_metrics(steps, raw, build_message_metrics(steps))


class LoadTrajectoryNeverRaises(unittest.TestCase):
    """Every hostile byte sequence must come back as a dict, not an exception.

    One unreadable file in an upload batch must cost that file's row, not the
    whole scorecard.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = self._dir.name
        self.addCleanup(self._dir.cleanup)

    def _assert_contract(self, path, hint=None):
        try:
            result = load_trajectory(path, hint)
        except Exception as exc:  # noqa: BLE001 - the contract is "no exception of any type"
            self.fail(f"load_trajectory({os.path.basename(path)!r}) raised {type(exc).__name__}: {exc}")
        self.assertIsInstance(result, dict, f"{path} returned {type(result).__name__}, not a dict")
        return result

    def test_degenerate_text_inputs_are_clean_errors_or_parses(self):
        """Empty, whitespace-only, and non-object JSON documents must not crash the loader."""
        cases = {
            "empty.json": b"",
            "empty.jsonl": b"",
            "whitespace.json": b"   \n\t\n  ",
            "whitespace.jsonl": b"   \n\t\n  ",
            "null.json": b"null",
            "true.json": b"true",
            "int.json": b"12345",
            "string.json": b'"hello"',
            "empty_array.json": b"[]",
            "empty_array.jsonl": b"[]",
            "empty_object.json": b"{}",
            "array_of_null.json": b"[null, null]",
            "array_of_string.json": b'["a", "b"]',
            "trailing_comma.json": b'{"info": {}, "messages": [],}',
            "lone_brace.json": b"{",
        }
        for name, data in cases.items():
            with self.subTest(case=name):
                self._assert_contract(_write_bytes(self.tmp, name, data))

    def test_encoding_hostility_is_a_clean_error(self):
        """A BOM must be tolerated; UTF-16 and invalid UTF-8 must degrade to ``_error``."""
        doc = json.dumps(_opencode_doc([_opencode_message()]))
        bom = self._assert_contract(_write_bytes(self.tmp, "bom.json", b"\xef\xbb\xbf" + doc.encode("utf-8")))
        self.assertNotIn("_error", bom, "a UTF-8 BOM is not a corrupt file")

        utf16 = self._assert_contract(_write_bytes(self.tmp, "utf16.json", doc.encode("utf-16")))
        self.assertIn("_error", utf16)

        bad = self._assert_contract(
            _write_bytes(self.tmp, "badutf8.json", b'{"info": {}, "messages": [], "x": "\xff\xfe"}')
        )
        self.assertIn("_error", bad)

    def test_wrong_types_in_every_structural_slot(self):
        """A field of the wrong JSON type must not take the loader down."""
        cases = {
            "messages_is_string.json": {"info": {}, "messages": "not a list"},
            "messages_of_null.json": {"info": {}, "messages": [None, None]},
            "parts_is_string.json": {"info": {}, "messages": [{"info": {}, "parts": "nope"}]},
            "parts_of_scalars.json": {"info": {}, "messages": [{"info": {}, "parts": [None, 1, "x"]}]},
            "info_is_null.json": {"info": {"time": None, "id": None},
                                  "messages": [{"info": None, "parts": []}]},
            "time_is_text.json": {"info": {"time": {"created": "yesterday"}}, "messages": []},
            "tokens_is_string.json": {"info": {}, "messages": [{"info": {"tokens": "lots"}, "parts": []}]},
            "cc_trajectory_is_string.json": {"format": "ccsession-trajectory", "trajectory": "nope"},
            "cc_trajectory_of_scalars.json": {"format": "ccsession-trajectory", "trajectory": [None, 3, "x"]},
            "cc_message_is_string.json": {"format": "ccsession-trajectory", "trajectory": [{"message": "str"}]},
        }
        for name, doc in cases.items():
            with self.subTest(case=name):
                self._assert_contract(_write_json(self.tmp, name, doc))

    def test_pathological_json_shapes(self):
        """Deep nesting and very large scalars must be handled without a crash."""
        deep = {"info": {}, "messages": []}
        cursor = deep
        for _ in range(200):
            nxt = {"a": None}
            cursor["x"] = nxt
            cursor = nxt
        self._assert_contract(_write_json(self.tmp, "deep.json", deep))

        big = {"info": {"id": "x" * 2_000_000}, "messages": []}
        self._assert_contract(_write_json(self.tmp, "bigstring.json", big))

        dup = b'{"info": {}, "messages": [], "messages": [1, 2, 3]}'
        self._assert_contract(_write_bytes(self.tmp, "dupkeys.json", dup))

    def test_truncated_documents_for_every_object_format(self):
        """A file cut off at 10/50/90% must produce ``_error`` or a parse — never a traceback."""
        sources = {
            "opencode.json": json.dumps(_opencode_doc([_opencode_message(index=i, created=i * 1000)
                                                       for i in range(20)])),
            "ccsession.json": json.dumps({
                "format": "ccsession-trajectory",
                "session": {"session_id": "s1", "working_directory": "/p"},
                "trajectory": [
                    {"index": i, "type": "assistant", "timestamp": "2026-01-05T12:00:00.000Z",
                     "message_id": f"m{i}", "message": {"role": "assistant", "content": [
                         {"type": "text", "text": "x" * 200}]}}
                    for i in range(20)
                ],
            }),
            "codex.jsonl": _jsonl_text(_codex_stream()),
        }
        for name, text in sources.items():
            data = text.encode("utf-8")
            for pct in (10, 50, 90):
                with self.subTest(case=name, pct=pct):
                    cut = data[: max(1, len(data) * pct // 100)]
                    self._assert_contract(_write_bytes(self.tmp, f"trunc{pct}_{name}", cut))

    def test_filesystem_hostility(self):
        """Missing files, directories, broken symlinks and unreadable files are ``_error``, not crashes."""
        missing = self._assert_contract(os.path.join(self.tmp, "does-not-exist.json"))
        self.assertIn("_error", missing)

        empty_dir = os.path.join(self.tmp, "adir.json")
        os.makedirs(empty_dir)
        self.assertIn("_error", self._assert_contract(empty_dir))

        link = os.path.join(self.tmp, "broken.json")
        os.symlink(os.path.join(self.tmp, "nowhere"), link)
        self.assertIn("_error", self._assert_contract(link))

        locked = _write_json(self.tmp, "locked.json", _opencode_doc([]))
        os.chmod(locked, 0)
        self.addCleanup(os.chmod, locked, 0o600)
        if os.access(locked, os.R_OK):
            self.skipTest("running as a user that ignores file permissions")
        self.assertIn("_error", self._assert_contract(locked))

    def test_session_directory_resolves_to_its_session_jsonl(self):
        """A DSH session *directory* must load its ``session.jsonl``, not error out."""
        sess = os.path.join(self.tmp, "sessdir")
        os.makedirs(sess)
        with open(os.path.join(sess, "session.jsonl"), "w", encoding="utf-8") as handle:
            handle.write('{"type": "session", "id": "s1", "createdAt": 0}\n{"type": "user/message"}\n')
        self.assertNotIn("_error", self._assert_contract(sess))

    def test_unsupported_extension_is_refused_by_name(self):
        """``.log`` is explicitly unsupported and must say so rather than parse garbage."""
        result = self._assert_contract(_write_bytes(self.tmp, "x.log", b"{}"))
        self.assertIn("_error", result)

    @unittest.expectedFailure
    def test_out_of_range_session_timestamp_is_a_clean_error(self):
        """A microsecond/nanosecond timestamp must not abort the batch.

        DEFECT (``trajviz/insight/formats/opencode.py:23-25``): ``info.time.created``
        / ``info.time.updated`` go straight into ``datetime.fromtimestamp``. A
        recorder that wrote microseconds instead of milliseconds raises
        ``ValueError: year 56664 is out of range`` out of ``load_trajectory``,
        which ``run_group`` does not catch. Clamp or reject the value instead.
        """
        doc = _opencode_doc([], info_time={"created": 0, "updated": 1_726_000_000_000_000})
        self._assert_contract(_write_json(self.tmp, "microseconds.json", doc))

    @unittest.expectedFailure
    def test_string_valued_token_counts_are_a_clean_error(self):
        """Token counts serialized as JSON strings must not abort the batch.

        DEFECT (``trajviz/insight/formats/opencode.py:146-147``, and the same
        pattern in ``codearts.py:101`` / ``claude_code.py:23``): the per-message
        token totals are accumulated with ``+=`` and no type check, so
        ``"tokens": {"input": "50"}`` raises ``TypeError: unsupported operand
        type(s) for +=: 'int' and 'str'`` out of ``load_trajectory``.
        """
        doc = _opencode_doc([_opencode_message(
            tokens={"total": "100", "input": "50", "output": "50", "cache": {"read": "0", "write": "0"}},
        )])
        self._assert_contract(_write_json(self.tmp, "string_tokens.json", doc))

    def test_non_finite_token_values_do_not_crash_the_pipeline(self):
        """``NaN``/``Infinity`` literals must not crash metrics computation.

        ``json.dump`` emits bare ``NaN`` / ``Infinity`` by default, so any
        upstream exporter written in Python can produce such a file. The loader
        accepts it, and these values used to reach ``statistics.median`` and
        raise ``ValueError: cannot convert float NaN to integer``. They are now
        dropped at parse time (``parser._finite_token``).
        """
        text = (
            '{"info": {"id": "s"}, "messages": [{"role": "assistant", "info": '
            '{"id": "m", "role": "assistant", "time": {"created": 1000, "completed": 2000}, '
            '"tokens": {"total": NaN, "input": NaN, "output": 1, "cache": {"read": 0, "write": 0}}}, '
            '"parts": []}]}'
        )
        path = _write_bytes(self.tmp, "nan.json", text)
        raw = self._assert_contract(path)
        if "_error" not in raw:
            steps = parse_steps(raw)
            compute_metrics(steps, raw, build_message_metrics(steps))


class JsonlLineSplitting(unittest.TestCase):
    """``formats/parse.py`` is the shared layer every JSONL load now goes through."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = self._dir.name
        self.addCleanup(self._dir.cleanup)

    def test_interior_corruption_is_reported_not_silently_dropped(self):
        """A corrupt first or middle line must fail loudly; silently skipping it loses steps."""
        events = _codex_stream()
        for position in (0, 1):
            with self.subTest(position=position):
                lines = _jsonl_text(events).splitlines()
                lines[position] = '{"type": "broken", '
                text = "\n".join(lines) + "\n"
                parsed, err = _parse_jsonl_events(text)
                self.assertIsNone(parsed)
                self.assertIsNotNone(err)
                self.assertIn(f"line {position + 1}", err)

    def test_corrupt_final_line_with_newline_is_an_error(self):
        """A terminated final line is complete, so a decode failure there is real corruption."""
        lines = _jsonl_text(_codex_stream()).splitlines()
        lines[-1] = '{"type": "broken", '
        parsed, err = _parse_jsonl_events("\n".join(lines) + "\n")
        self.assertIsNone(parsed)
        self.assertIsNotNone(err)

    def test_unterminated_final_line_is_treated_as_an_in_progress_append(self):
        """Rollouts are append-only: the last line without a newline may still be being written."""
        lines = _jsonl_text(_codex_stream()).splitlines()
        lines[-1] = '{"type": "broken", '
        parsed, err = _parse_jsonl_events("\n".join(lines))
        self.assertIsNone(err)
        self.assertEqual(len(parsed), len(_codex_stream()) - 1)

    def test_blank_and_crlf_lines_do_not_shift_the_event_stream(self):
        """Blank separators and Windows line endings must not add or drop events."""
        events = _codex_stream()
        text = "\r\n".join(json.dumps(e) for e in events) + "\r\n"
        parsed, err = _parse_jsonl_events(text)
        self.assertIsNone(err)
        self.assertEqual(len(parsed), len(events))

        padded = "\n\n" + "\n\n".join(json.dumps(e) for e in events) + "\n\n"
        parsed, err = _parse_jsonl_events(padded)
        self.assertIsNone(err)
        self.assertEqual(len(parsed), len(events))

    @unittest.expectedFailure
    def test_unicode_line_separators_inside_strings_do_not_split_events(self):
        """A valid JSONL event whose text contains U+2028 must still load.

        DEFECT (``trajviz/insight/formats/parse.py:21``): ``_parse_jsonl_events``
        splits with ``str.splitlines()``, which breaks on U+2028, U+2029 and
        U+0085 as well as ``\\n``. Those three are legal *unescaped* inside a
        JSON string, and serde_json (the Codex CLI writer) does not escape
        them, so one agent message containing a Unicode line separator turns a
        valid rollout into ``Invalid JSONL at line N: Unterminated string``.
        Split on ``\\n`` / ``\\r\\n`` only.
        """
        events = _codex_stream()
        events[2]["payload"]["content"][0]["text"] = "before\u2028after"
        text = _jsonl_text(events)
        # The file really is valid JSONL when split the way JSONL defines it.
        for line in text.split("\n"):
            if line.strip():
                json.loads(line)
        parsed, err = _parse_jsonl_events(text)
        self.assertIsNone(err, err)
        self.assertEqual(len(parsed), len(events))


class ZipIngestContract(unittest.TestCase):
    """The DSH zip branch reads members in memory and must stay bounded and safe."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = self._dir.name
        self.addCleanup(self._dir.cleanup)

    @staticmethod
    def _session_jsonl(session_id="s1"):
        return (
            json.dumps({"type": "session", "id": session_id, "createdAt": 0}) + "\n"
            + json.dumps({"type": "user/message", "body": {"text": "hi"}}) + "\n"
        )

    def _zip(self, name, members):
        path = os.path.join(self.tmp, name)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for member, payload in members.items():
                archive.writestr(member, payload)
        return path

    def _load(self, path):
        try:
            return load_trajectory(path)
        except Exception as exc:  # noqa: BLE001 - the contract is "no exception of any type"
            self.fail(f"load_trajectory({os.path.basename(path)!r}) raised {type(exc).__name__}: {exc}")

    def test_zip_without_session_jsonl_is_a_clean_error(self):
        """An arbitrary zip is not a DSH export and must say so."""
        result = self._load(self._zip("other.zip", {"readme.txt": "hello"}))
        self.assertIn("_error", result)

    def test_truncated_and_fake_archives_are_clean_errors(self):
        """``PK``-prefixed bytes that are not a zip must not escape as ``BadZipFile``."""
        good = self._zip("good.zip", {"session.jsonl": self._session_jsonl()})
        with open(good, "rb") as handle:
            data = handle.read()
        self.assertIn("_error", self._load(_write_bytes(self.tmp, "cut.zip", data[: len(data) // 2])))
        self.assertIn("_error", self._load(_write_bytes(self.tmp, "fake.zip", b"PKnot-a-zip")))

    def test_traversal_member_names_never_write_outside_the_archive(self):
        """Zip-slip guard: members are read in memory, so no ``../`` path may hit the disk."""
        outside = os.path.join(self.tmp, "evil.txt")
        path = self._zip("traversal.zip", {
            "session.jsonl": self._session_jsonl(),
            "../../evil.txt": "pwned",
            "subagents/../../../evil2/session.jsonl": '{"type": "session", "id": "c1", "createdAt": 0}\n',
        })
        before = sorted(os.listdir(self.tmp))
        self._load(path)
        self.assertFalse(os.path.exists(outside))
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.tmp), "evil.txt")))
        self.assertEqual(sorted(os.listdir(self.tmp)), before)

    def test_oversized_member_is_refused_without_decompressing_it(self):
        """A high-ratio bomb must be rejected on its declared size, not by expanding it."""
        path = os.path.join(self.tmp, "bomb.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr("session.jsonl", b"\0" * (64 * 1024 * 1024))
        self.assertLess(os.path.getsize(path), 1_000_000, "the bomb should be tiny on disk")
        result = self._load(path)
        self.assertIn("_error", result)

    def test_many_members_stay_bounded(self):
        """A zip with thousands of child sessions must load without scanning them all."""
        members = {"session.jsonl": self._session_jsonl()}
        for i in range(2000):
            members[f"subagents/c{i}/session.jsonl"] = self._session_jsonl(f"c{i}")
        result = self._load(self._zip("many.zip", members))
        self.assertIsInstance(result, dict)

    @unittest.expectedFailure
    def test_encrypted_member_is_a_clean_error(self):
        """A password-protected member must not abort the batch.

        DEFECT (``trajviz/insight/formats/dsh.py:962``, backlog item 8):
        ``archive.read(member)`` is guarded only against ``KeyError``, so
        ``zipfile`` raises ``RuntimeError: File 'session.jsonl' is encrypted,
        password required for extraction`` straight out of ``load_trajectory``.
        Catch ``RuntimeError`` (and ``zipfile.BadZipFile``) around the read.
        """
        plain = self._zip("plain.zip", {"session.jsonl": self._session_jsonl()})
        with open(plain, "rb") as handle:
            data = bytearray(handle.read())
        # Set the "encrypted" general-purpose flag in the local (+6) and
        # central-directory (+8) headers without actually encrypting anything.
        for magic, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
            i = 0
            while True:
                i = data.find(magic, i)
                if i < 0:
                    break
                data[i + offset] |= 0x01
                i += 4
        path = _write_bytes(self.tmp, "encrypted.zip", bytes(data))
        result = self._load(path)
        self.assertIn("_error", result)


class MetricInvariants(unittest.TestCase):
    """Numbers that reach a paper must be finite, non-negative, and in range."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.tmp = cls._dir.name

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _assert_invariants(self, metrics, doc):
        context = json.dumps(doc)[:400]
        for key, value in _walk_numbers(metrics):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            leaf = key.rsplit(".", 1)[-1].split("[")[0]
            if isinstance(value, float):
                self.assertFalse(math.isnan(value), f"{key} is NaN ({context})")
                self.assertFalse(math.isinf(value), f"{key} is infinite ({context})")
            if any(hint in leaf for hint in _NONNEG_NAME_HINTS):
                self.assertGreaterEqual(value, 0, f"{key} = {value} ({context})")
            if "percent" in leaf or leaf.endswith("_pct"):
                self.assertTrue(0 <= value <= 100, f"{key} = {value} is not a percentage ({context})")
        rate = metrics.get("tool_success_rate")
        if isinstance(rate, (int, float)):
            self.assertTrue(0 <= rate <= 100, f"tool_success_rate = {rate} ({context})")
        self.assertEqual(
            metrics["tool_call_count"],
            metrics["tool_success"] + metrics["tool_fail"],
            f"every tool call is either a success or a failure ({context})",
        )
        self.assertEqual(metrics["tool_call_count"], sum(metrics["tool_breakdown"].values()))

    @settings(max_examples=40, deadline=None, suppress_health_check=list(HealthCheck))
    @given(
        messages=st.lists(
            st.fixed_dictionaries({
                "role": st.sampled_from(["user", "assistant"]),
                "created": st.integers(min_value=0, max_value=10 ** 12),
                "elapsed": st.integers(min_value=0, max_value=10 ** 6),
                "total": st.integers(min_value=0, max_value=10 ** 6),
                "inp": st.integers(min_value=0, max_value=10 ** 6),
                "out": st.integers(min_value=0, max_value=10 ** 6),
                "cache_read": st.integers(min_value=0, max_value=10 ** 6),
                "cache_write": st.integers(min_value=0, max_value=10 ** 6),
                "tools": st.lists(
                    st.tuples(
                        st.sampled_from(["bash", "read", "edit", "task", "grep"]),
                        st.sampled_from(["completed", "error", "running", "pending"]),
                        st.integers(min_value=0, max_value=10 ** 6),
                    ),
                    max_size=3,
                ),
            }),
            max_size=6,
        ),
    )
    def test_well_typed_payloads_yield_sane_metrics(self, messages):
        """For non-negative, well-ordered inputs no metric may be NaN, negative, or >100%.

        A counterexample here means a researcher could publish a number that is
        arithmetically impossible for the trace it came from.
        """
        docs = []
        for i, spec in enumerate(messages):
            parts = [
                {"type": "tool", "tool": name, "id": f"t{i}-{j}",
                 "state": {"status": status, "input": {}, "output": "",
                           "time": {"start": spec["created"], "end": spec["created"] + dur}}}
                for j, (name, status, dur) in enumerate(spec["tools"])
            ]
            docs.append(_opencode_message(
                index=i, role=spec["role"], created=spec["created"], elapsed=spec["elapsed"],
                tokens={"total": spec["total"], "input": spec["inp"], "output": spec["out"],
                        "reasoning": 0,
                        "cache": {"read": spec["cache_read"], "write": spec["cache_write"]}},
                parts=parts,
            ))
        doc = _opencode_doc(docs)
        path = _write_json(self.tmp, "prop.json", doc)
        raw = load_trajectory(path)
        self.assertNotIn("_error", raw, raw.get("_error"))
        steps = parse_steps(raw)
        metrics = compute_metrics(steps, raw, build_message_metrics(steps))
        self.assertEqual(metrics["total_steps"], len(steps))
        self._assert_invariants(metrics, doc)

    def test_empty_and_single_step_trajectories_are_well_defined(self):
        """The degenerate sizes are where divide-by-zero guards are usually missed."""
        for label, messages in (("empty", []), ("one", [_opencode_message()])):
            with self.subTest(case=label):
                doc = _opencode_doc(messages)
                metrics = _metrics_for(_write_json(self.tmp, f"{label}.json", doc))
                self._assert_invariants(metrics, doc)

    @unittest.expectedFailure
    def test_cache_read_above_the_reported_total_cannot_exceed_100_percent(self):
        """``Avg cache %`` is a percentage and must never exceed 100.

        DEFECT (``trajviz/insight/metrics.py:382`` feeding ``:575``):
        ``cache_ratio = cache_read / tokens_total`` is unbounded, and OpenCode
        reports a ``total`` that excludes cache reads. 16 of the 2,500 real
        corpus trajectories therefore render ``Avg cache % = 25386.2%`` with a
        green "strong cache reuse" verdict
        (``opencode_opus/django__django-11555.json``). Clamp the ratio, or
        divide by a denominator that includes the cache read.
        """
        doc = _opencode_doc([
            _opencode_message(
                index=i, created=i * 1000,
                tokens={"total": 1909, "input": -10739, "output": 248, "reasoning": 0,
                        "cache": {"read": 12400, "write": 0}},
            )
            for i in range(3)
        ])
        metrics = _metrics_for(_write_json(self.tmp, "cache_over.json", doc))
        self.assertLessEqual(metrics["avg_cache_ratio"], 100)

    def test_token_totals_are_never_negative(self):
        """A token count is a count; a negative one must never reach a headline number.

        OpenCode subtracts the cache read from the prompt size, which
        double-subtracts against providers whose ``input_tokens`` is already
        cache-exclusive, so 82 real corpus files carry a negative raw
        ``input``. Those records used to be summed, showing an ``Input`` chip
        of ``-3,175,801``. They are now excluded from the sum and counted in
        ``input_tokens_unusable_steps`` so the surface can say so.
        """
        doc = _opencode_doc([
            _opencode_message(
                index=i, created=i * 1000,
                tokens={"total": 1909, "input": -10739, "output": 248, "reasoning": 0,
                        "cache": {"read": 12400, "write": 0}},
            )
            for i in range(3)
        ])
        metrics = _metrics_for(_write_json(self.tmp, "neg_tokens.json", doc))
        self.assertGreaterEqual(metrics["input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
