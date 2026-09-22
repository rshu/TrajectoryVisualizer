"""Focused regressions for follow-up findings from the Insight review."""

import unittest

from trajviz.insight import patterns
from trajviz.insight.shell_cmd import primary_shell_command
from trajviz.insight.metrics import compute_health_verdict, compute_metrics


def _assistant_step(index: int, output_tokens: int, duration: float | None) -> dict:
    return {
        "index": index,
        "role": "assistant",
        "duration": duration,
        "parts": [],
        "tool_calls": [],
        "tool_call_count": 0,
        "tokens": {
            "total": output_tokens,
            "input": 0,
            "output": output_tokens,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        },
    }


def _bash_call(command: str, *, tool_name: str = "Bash") -> dict:
    return {
        "tool_name": tool_name,
        "input": {"command": command},
        "output": "",
        "status": "success",
    }


class OutputThroughputTests(unittest.TestCase):
    def test_partial_timing_uses_tokens_from_only_the_timed_steps(self):
        steps = [
            _assistant_step(0, output_tokens=100, duration=10.0),
            _assistant_step(1, output_tokens=900, duration=None),
        ]

        metrics = compute_metrics(steps, {})

        self.assertEqual(metrics["tokens"]["output"], 1000)
        self.assertEqual(metrics["output_tokens_per_sec"], 10.0)
        self.assertEqual(metrics["output_throughput_timed_tokens"], 100.0)
        self.assertEqual(metrics["output_throughput_timed_seconds"], 10.0)
        self.assertEqual(metrics["output_throughput_timed_steps"], 1)
        self.assertEqual(metrics["output_throughput_total_steps"], 2)
        self.assertEqual(metrics["output_throughput_coverage_pct"], 50.0)
        self.assertTrue(metrics["output_throughput_incomplete"])

        throughput = {
            verdict["metric"]: verdict
            for verdict in compute_health_verdict(metrics, [])
        }["Throughput"]
        self.assertEqual(throughput["status"], "bad")
        self.assertEqual(throughput["label"], "10.0 gen tok/s")
        self.assertIn("1/2 assistant steps with timing", throughput["detail"])

    def test_complete_timing_reports_full_coverage(self):
        metrics = compute_metrics([
            _assistant_step(0, output_tokens=100, duration=10.0),
            _assistant_step(1, output_tokens=400, duration=20.0),
        ], {})

        self.assertEqual(metrics["output_tokens_per_sec"], 16.7)
        self.assertEqual(metrics["output_throughput_coverage_pct"], 100.0)
        self.assertFalse(metrics["output_throughput_incomplete"])

    def test_spawn_wait_excluded_from_throughput_denominator(self):
        """Parent task wall-clock must not be added on top of the child's time."""
        parent = _assistant_step(0, output_tokens=100, duration=100.0)
        parent["tool_calls"] = [
            {"tool_name": "task", "status": "success", "duration_ms": 80_000},
        ]
        parent["tool_call_count"] = 1
        child = _assistant_step(1, output_tokens=400, duration=80.0)
        child["is_sub_agent"] = True

        metrics = compute_metrics([parent, child], {})

        # 100 output tokens kept; duration is 20s (parent excl. wait) + 80s child.
        self.assertEqual(metrics["output_throughput_timed_tokens"], 500.0)
        self.assertEqual(metrics["output_throughput_timed_seconds"], 100.0)
        self.assertEqual(metrics["output_tokens_per_sec"], 5.0)

    def test_tool_wait_excluded_from_throughput_denominator(self):
        """Long Bash/script waits must not look like near-zero gen throughput."""
        step = _assistant_step(0, output_tokens=100, duration=1210.0)
        step["tool_calls"] = [
            {"tool_name": "Bash", "status": "success", "duration_ms": 1_200_000},
        ]
        step["tool_call_count"] = 1

        metrics = compute_metrics([step], {})

        self.assertEqual(metrics["output_throughput_timed_seconds"], 10.0)
        self.assertEqual(metrics["output_throughput_tool_wait_seconds"], 1200.0)
        self.assertEqual(metrics["output_tokens_per_sec"], 10.0)

        throughput = {
            verdict["metric"]: verdict
            for verdict in compute_health_verdict(metrics, [])
        }["Throughput"]
        self.assertEqual(throughput["label"], "10.0 gen tok/s")
        self.assertIn("tool wait excluded", throughput["detail"])

    def test_overview_labels_generation_rate_as_output_throughput(self):
        # Import here because this UI module loads optional Gradio dependencies.
        from trajviz.insight.presenters import build_overview_kpi_html
        from trajviz.insight.formatting import (
            format_banner_html,
            format_performance_md,
        )

        metrics = {
            "total_steps": 2,
            "assistant_steps": 2,
            "user_steps": 0,
            "p95_duration": 10.0,
            "tokens": {"total": 1000},
            "tokens_per_second": 100.0,
            "output_tokens_per_sec": 10.0,
            "output_throughput_timed_steps": 1,
            "output_throughput_total_steps": 2,
            "output_throughput_incomplete": True,
            "output_throughput_tool_wait_seconds": 1200.0,
            "tool_success_rate": 0,
            "tool_call_count": 0,
        }

        html = build_overview_kpi_html(metrics, "10s")

        self.assertIn("10.0 gen tok/s", html)
        self.assertIn("excl. tools", html)
        self.assertIn("1/2 timed", html)
        self.assertIn(">Issues<", html)
        self.assertIn("none detected", html)
        self.assertNotIn("100.0 tok/s", html)
        self.assertNotIn("failed tool calls", html)

        banner = format_banner_html("trace.json", metrics, "10s")
        self.assertIn("10.0 gen tok/s", banner)
        self.assertNotIn("100.0 tok/s", banner)

        computed_metrics = compute_metrics([
            _assistant_step(0, output_tokens=100, duration=10.0),
            _assistant_step(1, output_tokens=900, duration=None),
        ], {})
        performance = format_performance_md(computed_metrics, "10s")
        self.assertIn("Total processed tok/sec", performance)
        self.assertIn("Median processed tok/sec", performance)

    def test_issues_kpi_card_after_tokens(self):
        from trajviz.insight.presenters import build_overview_kpi_html
        from trajviz.insight.presenters.issues import OverviewIssue

        metrics = {
            "total_steps": 3,
            "assistant_steps": 2,
            "user_steps": 1,
            "p95_duration": 1.0,
            "tokens": {"total": 10},
            "tool_success_rate": 100,
            "tool_call_count": 1,
        }
        issues = [
            OverviewIssue(kind="error", title="a", detail="", why="", steps=(1,)),
            OverviewIssue(kind="error", title="b", detail="", why="", steps=(2,)),
            OverviewIssue(kind="antipattern", title="c", detail="", why="", steps=(3,)),
        ]
        html = build_overview_kpi_html(metrics, "4s", issues=issues)
        tokens_at = html.find(">Tokens<")
        issues_at = html.find(">Issues<")
        tool_at = html.find(">Tool Success<")
        self.assertGreater(issues_at, tokens_at)
        self.assertGreater(tool_at, issues_at)
        self.assertIn("ov-kpi-breakdown", html)
        self.assertIn(">errors<", html)
        self.assertIn(">pattern<", html)
        self.assertIn(
            "<span class='ov-kpi-breakdown-count'>2</span>",
            html,
        )
        self.assertIn(
            "<span class='ov-kpi-breakdown-count'>1</span>",
            html,
        )
        self.assertIn("var(--ov-bad)", html)
        self.assertIn("var(--ov-warn)", html)
        self.assertIn("data-status='bad'", html)
        self.assertIn("ov-kpi-card--issues", html)
        self.assertIn("overview-issues", html)
        # Long detail stays on title/tooltip only — not a second visible subtitle line.
        self.assertIn("title='3 issues — review Overview Issues'", html)
        self.assertNotIn("margin-top:2px;'>3 issues — review Overview Issues", html)

    def test_steps_kpi_shows_agent_breakdown_not_error_verdict(self):
        from trajviz.insight.presenters import build_overview_kpi_html

        metrics = {
            "total_steps": 10,
            "assistant_steps": 9,
            "user_steps": 1,
            "p95_duration": 1.0,
            "tokens": {"total": 10},
            "tool_success_rate": 50,
            "tool_call_count": 4,
            "tool_fail": 3,
        }
        verdicts = [{
            "metric": "Errors",
            "status": "bad",
            "label": "3",
            "detail": "3 failed tool calls — agent may be struggling",
        }]
        agents = [
            {"label": "main", "step_count": 6},
            {"label": "explore", "step_count": 3},
        ]
        html = build_overview_kpi_html(
            metrics, "4s", verdicts=verdicts, agent_summaries=agents,
        )
        self.assertIn("ov-kpi-breakdown", html)
        self.assertIn("ov-kpi-breakdown-row", html)
        self.assertIn("ov-kpi-breakdown-swatch", html)
        self.assertIn(">main<", html)
        self.assertIn(">explore<", html)
        self.assertIn(
            "<span class='ov-kpi-breakdown-count'>6</span>",
            html,
        )
        self.assertIn(
            "<span class='ov-kpi-breakdown-count'>3</span>",
            html,
        )
        self.assertNotIn("failed tool calls", html)
        self.assertNotIn("agent may be struggling", html)


class WrappedShellSearchTests(unittest.TestCase):
    def test_common_wrappers_are_recognized_as_searches(self):
        commands = [
            "cd src && /usr/bin/rg needle .",
            "env LANG=C command grep -R needle .",
            "git -C repo grep needle",
            "printf '%s\\0' src | xargs -0 rg needle",
            "bash -lc 'cd src && find . -name *.py'",
            'cmd /c "rg needle ."',
            'powershell -Command "rg needle ."',
        ]

        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(patterns._is_search_call(_bash_call(command)))

    def test_primary_shell_command_peels_wrappers(self):
        cases = {
            "git status": "git",
            "FOO=1 BAR=2 npm test": "npm",
            "sudo -u kevin pytest -q": "pytest",
            "env LANG=C /usr/bin/python script.py": "script.py",
            "timeout 30 make build": "make",
            "bash -lc 'cd src && cargo test'": "cargo",
            "cd src && rg needle .": "rg",
            "cd only": "cd",
            "echo hello | grep hi": "echo",
            # Windows cmd.exe /c|/k — label the inner command, not cmd.
            'cmd /c "timeout /t 300 /nobreak"': "timeout",
            "cmd /c timeout /t 300 /nobreak": "timeout",
            'cmd.exe /c "ping -n 5 localhost"': "ping",
            'cmd /d /c "git status"': "git",
            'cmd /c "cd /d C:/foo && python.exe main.py"': "main.py",
            "timeout /t 300 /nobreak": "timeout",
            # PowerShell — label the -Command body or -File basename.
            'powershell -Command "Start-Sleep -Seconds 300"': "start-sleep",
            'powershell.exe -NoProfile -Command "Start-Sleep -Seconds 300"': "start-sleep",
            'pwsh -c "Get-ChildItem -Recurse"': "get-childitem",
            'powershell -c "git status"': "git",
            "powershell -File C:/scripts/run.ps1": "run.ps1",
            "PowerShell -Command { Start-Sleep -Seconds 1 }": "start-sleep",
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(primary_shell_command(command), expected)

    def test_primary_shell_command_names_python_scripts(self):
        cases = {
            "python3 tools/run_eval.py --fast": "run_eval.py",
            "python path/to/train.py": "train.py",
            "python3.12 -u scripts/foo.py arg": "foo.py",
            "python -m pytest -q": "pytest",
            "python3 -m http.server 8000": "http.server",
            "pypy3 -O bench.py": "bench.py",
            "python -c 'print(1)'": "python",
            "python3": "python3",
            "cd src && python3 ../bin/check.py": "check.py",
            # Windows interpreters: label by script/module, not python.exe.
            "python.exe script.py": "script.py",
            "pythonw.exe -u main.py": "main.py",
            "D:/install/py/python.exe C:/Users/x/.config/opencode/skills/checker_master/main.py":
                "main.py",
            r"D:\install\py\python.exe C:\Users\x\.config\opencode\skills\checker_master\main.py":
                "main.py",
            "C:/Python312/python.exe -m pytest -q": "pytest",
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(primary_shell_command(command), expected)

    def test_primary_shell_command_empty_or_malformed(self):
        self.assertIsNone(primary_shell_command(""))
        self.assertIsNone(primary_shell_command("   "))
        # Unclosed quote: fall back to first token rather than raising.
        self.assertEqual(primary_shell_command("rg 'unterminated"), "rg")

    def test_lowercase_bash_tool_name_is_recognized(self):
        self.assertTrue(patterns._is_search_call(
            _bash_call("FOO=1 ripgrep needle .", tool_name="bash")
        ))

    def test_commands_that_only_mention_search_tools_are_not_searches(self):
        commands = [
            "echo 'rg needle'",
            "python -c 'print(\"grep needle\")'",
            "printf '%s' 'find . -name *.py'",
        ]

        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(patterns._is_search_call(_bash_call(command)))

    def test_malformed_shell_quoting_is_conservative(self):
        self.assertFalse(patterns._is_search_call(_bash_call("rg 'unterminated")))

    def test_wrapped_empty_searches_form_a_fruitless_streak(self):
        steps = [
            {"index": 0, "role": "assistant", "tool_calls": [
                _bash_call("cd src && rg first ."),
            ]},
            {"index": 1, "role": "assistant", "tool_calls": [
                _bash_call("env LANG=C grep -R second ."),
            ]},
            {"index": 2, "role": "assistant", "tool_calls": [
                _bash_call("git grep third"),
            ]},
        ]

        streaks = patterns.detect_fruitless_streaks(steps)

        self.assertEqual(len(streaks), 1)
        self.assertEqual(streaks[0]["length"], 3)


class ReasoningTokensNaTests(unittest.TestCase):
    def test_unreported_reasoning_shows_na_not_fake_zero(self):
        from trajviz.insight.formats.claude_code import _cc_extract_usage, _cc_build_step
        from trajviz.insight.formatting import format_performance_md

        usage = _cc_extract_usage({"input_tokens": 10, "output_tokens": 5})
        self.assertNotIn("reasoning", usage)

        step = _cc_build_step([], role="assistant", usage=usage, timestamp_ms=1)
        self.assertNotIn("reasoning", step["tokens"])

        metrics = compute_metrics([step], {})
        self.assertFalse(metrics["reasoning_tokens_reported"])
        self.assertEqual(metrics["tokens"]["reasoning"], 0)

        html = format_performance_md(metrics, "1s")
        self.assertRegex(
            html,
            r"Reasoning</span><span[^>]*>N/A</span>",
        )

    def test_reported_zero_reasoning_still_shows_zero(self):
        from trajviz.insight.formatting import format_performance_md

        step = _assistant_step(0, output_tokens=50, duration=1.0)
        metrics = compute_metrics([step], {})
        self.assertTrue(metrics["reasoning_tokens_reported"])
        html = format_performance_md(metrics, "1s")
        self.assertRegex(
            html,
            r"Reasoning</span><span[^>]*>0</span>",
        )


if __name__ == "__main__":
    unittest.main()
