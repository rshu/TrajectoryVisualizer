"""Diagnostics/Issues predicate regressions: canonical failure detection, loud
session fields, and a scrubbed judge error banner.

Pins the four places where the Issues/Diagnostics surfaces used to disagree with
``trajviz.insight.tool_failure`` or hid a defect behind a defensive default.
"""

import dataclasses
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests

from trajviz.insight import diagnostics
from trajviz.insight.diagnostics import _error_pattern, identify_target_files
from trajviz.insight.issue_judge import iter_judge_overview_issues
from trajviz.insight.presenters.issues import (
    build_overview_issues_html,
    collect_overview_issues,
    rank_issues,
)
from trajviz.insight.session import LoadedSession

_SRC_ROOT = Path(__file__).resolve().parents[1] / "trajviz" / "insight"


def _step(**tc) -> dict:
    """One assistant step carrying a single tool call."""
    return {"index": 1, "role": "assistant", "tool_calls": [tc]}


class TargetFilePredicateTests(unittest.TestCase):
    """D1 — identify_target_files must not credit a failed write."""

    def test_failed_writes_are_not_targets(self):
        # One step per failure shape tool_call_failed recognises but the old
        # narrow ("error", "failed", "failure") set did not.
        steps = [
            _step(tool_name="Write", status="completed", metadata={"exit": 1},
                  input={"file_path": "/a/x.py"}),
            _step(tool_name="Edit", status="completed", error="String not found",
                  input={"file_path": "/a/y.py"}),
            _step(tool_name="Write", status="timeout", input={"file_path": "/a/z.py"}),
            _step(tool_name="Edit", status="cancelled", input={"file_path": "/a/c.py"}),
            _step(tool_name="Write", status="ERROR", input={"file_path": "/a/u.py"}),
            _step(tool_name="Write", status="completed", error_type="ENOENT",
                  input={"file_path": "/a/e.py"}),
        ]
        self.assertEqual(identify_target_files(steps), set())

    def test_clean_write_is_still_a_target(self):
        steps = [
            _step(tool_name="Write", status="completed", metadata={"exit": 0},
                  input={"file_path": "/a/ok.py"}),
            # OpenCode spells the field ``filePath``.
            _step(tool_name="Edit", status="completed", input={"filePath": "/a/ok2.py"}),
        ]
        self.assertEqual(identify_target_files(steps), {"/a/ok.py", "/a/ok2.py"})

    def test_patch_parts_are_unaffected(self):
        steps = [{
            "index": 1,
            "role": "assistant",
            "parts": [{"type": "patch", "files": ["a\\b.py"]}],
            "tool_calls": [],
        }]
        self.assertEqual(identify_target_files(steps), {"a/b.py"})


class ErrorPatternTests(unittest.TestCase):
    """D2 — a failure admitted by cluster_errors must get a real pattern."""

    def test_non_narrow_statuses_get_a_real_pattern(self):
        self.assertEqual(
            _error_pattern({"status": "timeout", "output": "Command timed out after 120s\nmore"}),
            "Command timed out after 120s",
        )
        self.assertEqual(_error_pattern({"status": "CANCELLED"}), "status: cancelled")
        self.assertEqual(_error_pattern({"status": "Failed"}), "status: failed")

    def test_error_type_is_the_last_resort(self):
        self.assertEqual(
            _error_pattern({"status": "completed", "error_type": "ENOENT"}), "ENOENT"
        )

    def test_existing_branches_keep_their_labels(self):
        # Branch order (error -> exit -> status -> error_type) must not change,
        # or clusters already-labelled data would be relabelled.
        self.assertEqual(_error_pattern({"status": "error", "output": "boom"}), "boom")
        self.assertEqual(_error_pattern({"metadata": {"exit": 2}}), "exit code 2")
        self.assertEqual(_error_pattern({"error": "nope\ntrace"}), "nope")
        self.assertEqual(
            _error_pattern({"status": "error", "error_type": "ENOENT", "output": "boom"}),
            "boom",
        )
        self.assertEqual(_error_pattern({"status": "completed"}), "unknown error")


class DeadHelperTests(unittest.TestCase):
    """X9 — classify_chain_steps had zero callers; it and its helper are gone."""

    def test_classify_chain_steps_is_removed(self):
        self.assertFalse(hasattr(diagnostics, "classify_chain_steps"))
        # _error_tool_target existed only to build that function's signatures.
        self.assertFalse(hasattr(diagnostics, "_error_tool_target"))

    def test_failure_chain_analysis_still_works(self):
        steps = [
            {"index": 1, "role": "assistant", "tool_calls": [{"tool_name": "Bash", "status": "error"}]},
            {"index": 2, "role": "assistant", "tool_calls": [{"tool_name": "Bash", "metadata": {"exit": 2}}]},
            {"index": 3, "role": "assistant", "tool_calls": [{"tool_name": "Bash", "status": "completed"}]},
        ]
        chains = diagnostics.detect_failure_chains(steps)
        self.assertEqual([(c["start"], c["end"]) for c in chains], [(1, 2)])
        self.assertEqual(diagnostics.compute_failure_chain_metrics(chains, 3)["longest_chain"], 2)


class SessionFieldAccessTests(unittest.TestCase):
    """D4 — Issues reads LoadedSession fields directly, so a rename is loud."""

    @staticmethod
    def _stub(**kwargs) -> SimpleNamespace:
        base = dict(
            steps=[],
            failure_patterns=[],
            failure_chains=[],
            fruitless_streaks=[],
            tool_selection=[],
            plan_metrics={},
            plan_history=[],
            edit_thrash=[],
            repeated_searches=[],
            phase_regressions=[],
            performance_bottlenecks=[],
            premature_compactions=[],
            file_interactions=[],
            format="",
        )
        base.update(kwargs)
        return SimpleNamespace(**base)

    def test_missing_session_field_fails_loudly(self):
        base = self._stub().__dict__
        for name in ("edit_thrash", "repeated_searches", "premature_compactions"):
            with self.subTest(field=name):
                stub = SimpleNamespace(**{k: v for k, v in base.items() if k != name})
                with self.assertRaises(AttributeError):
                    collect_overview_issues(stub)

    def test_no_defensive_getattr_on_session_remains(self):
        # The `or []` / `or ""` guards stay; the dead default must not, because
        # every name these modules read is a non-defaulted LoadedSession field.
        for rel in ("presenters/issues.py", "issue_judge.py"):
            text = (_SRC_ROOT / rel).read_text()
            self.assertEqual(
                re.findall(r"getattr\(\s*session\s*,", text), [], f"{rel} still guesses",
            )

    def test_every_session_name_read_is_a_dataclass_field(self):
        declared = {f.name for f in dataclasses.fields(LoadedSession)}
        for rel in ("presenters/issues.py", "issue_judge.py"):
            text = (_SRC_ROOT / rel).read_text()
            names = set(re.findall(r"\bsession\.([a-z_][a-z0-9_]*)", text))
            self.assertTrue(names, f"{rel} reads nothing off session?")
            self.assertEqual(names - declared, set(), f"{rel} reads a non-field")


class _Resp:
    """Minimal stand-in for the requests.Response an HTTPError carries."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class JudgeErrorScrubbingTests(unittest.TestCase):
    """D5 — judge failures must not leak the endpoint or a query-string key."""

    _URL = "https://api.example.test/v1/chat/completions?api-key=sk-SECRET123"

    def _run(self, chat_fn) -> list[str]:
        from trajviz.insight.llm_config import AnalysisLLMConfig

        cfg = AnalysisLLMConfig(
            base_url="https://api.example.test/v1",
            api_key="test-key",
            model="test-model",
            provider="openai",
            temperature=0.0,
            max_tokens=512,
            timeout=30,
            source="analyze",
        )
        session = SessionFieldAccessTests._stub(
            fruitless_streaks=[{"start_step": 4, "end_step": 6, "length": 3}],
            steps=[{"index": i, "role": "assistant", "tool_calls": []} for i in range(4, 7)],
        )
        issues = rank_issues(collect_overview_issues(session))
        self.assertEqual(len(issues), 1)
        snaps = list(iter_judge_overview_issues(
            session, issues, config=cfg, chat_fn=chat_fn, limit=1,
        ))
        return list(snaps[-1].errors)

    def test_http_error_scrubs_endpoint_and_key(self):
        def chat_fn(config, system, messages):
            raise requests.HTTPError(
                f"401 Client Error: Unauthorized for url: {self._URL}",
                response=_Resp(401),
            )

        errors = self._run(chat_fn)
        self.assertEqual(len(errors), 1)
        self.assertNotIn("sk-SECRET123", errors[0])
        self.assertNotIn("api.example.test", errors[0])
        self.assertIn("HTTP 401", errors[0])

        html = build_overview_issues_html(
            SessionFieldAccessTests._stub(),
            banner=f"Judge failed (1). First: {errors[0][:160]}",
        )
        self.assertNotIn("sk-SECRET123", html)
        self.assertNotIn("api.example.test", html)

    def test_requests_json_decode_error_is_scrubbed_too(self):
        # requests.JSONDecodeError is BOTH a RequestException and a ValueError,
        # so the ValueError passthrough must not claim it.
        def chat_fn(config, system, messages):
            raise requests.exceptions.JSONDecodeError("Expecting value", "", 0)

        errors = self._run(chat_fn)
        self.assertIn("Could not reach the analysis API", errors[0])

    def test_judge_parse_error_keeps_its_message(self):
        # Guard against over-scrubbing: this module's own ValueErrors are the
        # only debuggable signal left once HTTP text is gone.
        def chat_fn(config, system, messages):
            return "not json"

        errors = self._run(chat_fn)
        self.assertEqual(len(errors), 1)
        self.assertIn("Judge", errors[0])
        self.assertIn("JSON", errors[0])


if __name__ == "__main__":
    unittest.main()
