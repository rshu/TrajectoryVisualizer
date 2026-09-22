"""LLM Issues fix judge — pack, parse, render (mocked chat)."""

import unittest
from types import SimpleNamespace

from trajviz.insight.issue_judge import (
    judge_issue_fix,
    judge_overview_issues,
    iter_judge_overview_issues,
    pack_issue_judge_context,
    parse_issue_judgment,
)
from trajviz.insight.llm_config import AnalysisLLMConfig
from trajviz.insight.presenters.issues import (
    OverviewIssue,
    collect_overview_issues,
    rank_issues,
    render_overview_issues_html,
)


def _cfg() -> AnalysisLLMConfig:
    return AnalysisLLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        provider="openai",
        temperature=0.0,
        max_tokens=512,
        timeout=30,
        source="analyze",
    )


def _session(**kwargs):
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
        file_interactions=[],
        format="claude_code",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class ParseJudgmentTests(unittest.TestCase):
    def test_raw_json(self):
        j = parse_issue_judgment(
            '{"where":"CLAUDE.md","fix":"Stop Bash reads","also":"","confidence":"high"}'
        )
        self.assertEqual(j.where, "CLAUDE.md")
        self.assertEqual(j.fix, "Stop Bash reads")
        self.assertEqual(j.confidence, "high")

    def test_fenced_json(self):
        j = parse_issue_judgment(
            """Here you go:
```json
{"where": "Environment", "fix": "Fix permissions", "also": "not a prompt", "confidence": "medium"}
```
"""
        )
        self.assertEqual(j.where, "Environment")
        self.assertIn("permissions", j.fix)

    def test_rejects_missing_fix(self):
        with self.assertRaises(ValueError):
            parse_issue_judgment('{"where":"x","fix":"","also":"","confidence":"low"}')


class PackContextTests(unittest.TestCase):
    def test_includes_workflow_window_and_skills(self):
        session = _session(
            steps=[
                {
                    "index": 0,
                    "role": "user",
                    "text_preview": "Fix the auth bug in login.py",
                    "tool_calls": [],
                },
                {
                    "index": 4,
                    "role": "assistant",
                    "text_preview": "searching",
                    "tool_calls": [{
                        "tool_name": "Bash",
                        "input": {"command": "cat foo.py"},
                        "error_type": "exit 1",
                    }],
                },
                {
                    "index": 5,
                    "role": "assistant",
                    "text_preview": "retry",
                    "tool_calls": [{"tool_name": "Read", "input": {"file_path": "foo.py"}}],
                },
            ],
            file_interactions=[{
                "step": 3,
                "tool": "Skill",
                "path": "skill:search",
                "type": "skill",
            }],
        )
        issue = OverviewIssue(
            kind="antipattern",
            title="Bash-for-reading (1×)",
            detail="sed/cat",
            why="shell reads",
            steps=(4,),
            source_id="antipattern:bash_read",
        )
        packed = pack_issue_judge_context(session, issue)
        self.assertIn("Bash-for-reading", packed)
        self.assertIn('"index": 4', packed)
        self.assertIn("cat foo.py", packed)
        self.assertIn("skill:search", packed)
        self.assertIn("claude_code", packed)
        self.assertIn("user_task", packed)
        self.assertIn("Fix the auth bug in login.py", packed)


class JudgeMockTests(unittest.TestCase):
    def test_judge_issue_fix_uses_chat_fn(self):
        session = _session(steps=[{"index": 1, "role": "assistant", "tool_calls": []}])
        issue = OverviewIssue(
            kind="error",
            title="Bash: exit 1 (1×)",
            detail="fail",
            why="no recovery",
            steps=(1,),
            source_id="fail:0",
        )

        def chat_fn(config, system, messages):
            self.assertIn("fix judge", system.lower())
            self.assertIn("Simplified Chinese", system)
            self.assertIn("Simplified Chinese", messages[0]["content"])
            self.assertEqual(messages[0]["role"], "user")
            return (
                '{"where":"CLAUDE.md","fix":"Bash 读文件前先确认路径",'
                '"also":"","confidence":"high"}'
            )

        j = judge_issue_fix(session, issue, config=_cfg(), chat_fn=chat_fn)
        self.assertEqual(j.where, "CLAUDE.md")
        self.assertIn("Bash", j.fix)

    def test_judge_overview_attaches_judgment_and_html(self):
        session = _session(
            fruitless_streaks=[{
                "start_step": 4,
                "end_step": 6,
                "length": 3,
            }],
            steps=[
                {"index": i, "role": "assistant", "tool_calls": []}
                for i in range(4, 7)
            ],
        )
        issues = rank_issues(collect_overview_issues(session))
        self.assertEqual(len(issues), 1)

        def chat_fn(config, system, messages):
            return (
                '{"where":"CLAUDE.md","fix":"空搜索超过 N 次后停止",'
                '"also":"预先给出路径","confidence":"medium"}'
            )

        judged, errors = judge_overview_issues(
            session, issues, config=_cfg(), chat_fn=chat_fn,
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(judged[0].judgment)
        html = render_overview_issues_html(judged)
        self.assertIn(">Change<", html)
        self.assertIn(">Fix", html)
        self.assertNotIn("Show fix", html)
        self.assertIn("CLAUDE.md", html)
        self.assertIn("空搜索超过 N 次后停止", html)
        self.assertIn("with LLM fix", html)

    def test_iter_judge_yields_before_each_and_finished(self):
        session = _session(steps=[{"index": 1, "role": "assistant", "tool_calls": []}])
        issues = [
            OverviewIssue(
                kind="error",
                title=f"Issue {i}",
                detail="d",
                why="w",
                steps=(1,),
                source_id=f"t:{i}",
            )
            for i in range(2)
        ]
        calls = {"n": 0}

        def chat_fn(config, system, messages):
            calls["n"] += 1
            return (
                '{"where":"CLAUDE.md","fix":"修复",'
                '"also":"","confidence":"high"}'
            )

        snaps = list(iter_judge_overview_issues(
            session, issues, config=_cfg(), chat_fn=chat_fn,
        ))
        self.assertEqual(len(snaps), 3)  # before #1, before #2, finished
        self.assertFalse(snaps[0].finished)
        self.assertEqual(snaps[0].current, 1)
        self.assertEqual(snaps[0].current_title, "Issue 0")
        self.assertIsNone(snaps[0].issues[0].judgment)
        self.assertFalse(snaps[1].finished)
        self.assertEqual(snaps[1].current, 2)
        self.assertIsNotNone(snaps[1].issues[0].judgment)  # first done
        self.assertTrue(snaps[2].finished)
        self.assertEqual(calls["n"], 2)
        self.assertTrue(all(i.judgment for i in snaps[2].issues))

    def test_render_without_judgment_has_no_change_fix(self):
        html = render_overview_issues_html([
            OverviewIssue(
                kind="error",
                title="x",
                detail="y",
                why="z",
                steps=(1,),
            )
        ])
        self.assertNotIn(">Change<", html)
        self.assertNotIn(">Fix", html)


if __name__ == "__main__":
    unittest.main()
