"""AI Trajectory Analysis sidebar: config, brief packing, chat (no live HTTP)."""

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from trajviz.insight import assistant, llm_config
from trajviz.insight.llm_config import (
    AnalysisLLMConfig,
    config_status_html,
    load_env_files,
    resolve_analysis_config,
    setup_help_text,
)


def _cfg(**overrides) -> AnalysisLLMConfig:
    base = dict(
        base_url="https://api.example.com/v1",
        api_key="sk-test",
        model="demo-model",
        provider="openai",
        temperature=0.2,
        max_tokens=2048,
        timeout=120,
        source="analyze",
    )
    base.update(overrides)
    return AnalysisLLMConfig(**base)


def _step(index, role="assistant", *, duration=1.0, tokens=10, tools=None, preview=""):
    tool_calls = tools or []
    return {
        "index": index,
        "role": role,
        "duration": duration,
        "finish": "",
        "text_preview": preview,
        "error_count": sum(1 for tc in tool_calls if tc.get("status") == "error" or tc.get("error_type")),
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "tokens": {
            "total": tokens,
            "input": tokens,
            "output": 0,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        },
        "parts": [],
    }


class AnalysisConfigTests(unittest.TestCase):
    def test_analyze_vars_win_over_label_vars(self):
        env = {
            "ANALYZE_BASE_URL": "https://analyze.example/v1",
            "ANALYZE_API_KEY": "sk-analyze",
            "ANALYZE_MODEL": "analyze-model",
            "ANALYZE_PROVIDER": "openai",
            "LABEL_BASE_URL": "https://label.example/v1",
            "LABEL_API_KEY": "sk-label",
            "LABEL_MODEL": "label-model",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = resolve_analysis_config()
        self.assertTrue(cfg.ready)
        self.assertEqual(cfg.base_url, "https://analyze.example/v1")
        self.assertEqual(cfg.api_key, "sk-analyze")
        self.assertEqual(cfg.model, "analyze-model")
        self.assertEqual(cfg.source, "analyze")

    def test_falls_back_to_label_vars(self):
        env = {
            "LABEL_BASE_URL": "https://label.example/v1",
            "LABEL_API_KEY": "sk-label",
            "LABEL_MODEL": "label-model",
            "LABEL_PROVIDER": "anthropic",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = resolve_analysis_config()
        self.assertTrue(cfg.ready)
        self.assertEqual(cfg.model, "label-model")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.source, "label")

    def test_missing_key_is_not_ready(self):
        env = {
            "ANALYZE_BASE_URL": "https://api.example.com/v1",
            "ANALYZE_MODEL": "demo",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = resolve_analysis_config()
            html = config_status_html(loaded_steps=0)
        self.assertFalse(cfg.ready)
        self.assertTrue(any("API_KEY" in name for name in cfg.missing))
        self.assertIn("Not configured", html)
        self.assertNotIn("sk-", html)
        self.assertIn("ANALYZE_API_KEY", html)

    def test_dotenv_is_fill_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("ANALYZE_MODEL=from-file\nANALYZE_API_KEY=sk-file\n", encoding="utf-8")
            previous = os.getcwd()
            env = os.environ.copy()
            env["ANALYZE_MODEL"] = "from-process"
            env.pop("ANALYZE_API_KEY", None)
            try:
                os.chdir(tmp)
                with (
                    patch.object(llm_config, "_REPO_ROOT", Path(tmp)),
                    patch.dict(os.environ, env, clear=True),
                ):
                    load_env_files()
                    self.assertEqual(os.environ["ANALYZE_MODEL"], "from-process")
                    self.assertEqual(os.environ["ANALYZE_API_KEY"], "sk-file")
            finally:
                os.chdir(previous)


class AnalysisBriefTests(unittest.TestCase):
    def test_brief_includes_task_health_bottlenecks_and_errors(self):
        steps = [
            _step(0, "user", duration=None, tokens=0, preview="Fix the flaky login test"),
            _step(1, duration=40.0, tokens=9000, tools=[
                {"tool_name": "Bash", "status": "error", "error_type": "missing_file",
                 "input": {"command": "cat /nope"}, "output": "No such file or directory"},
            ]),
            _step(2, duration=1.0, tokens=20, tools=[
                {"tool_name": "Read", "status": "success", "input": {"file_path": "app.py"}, "output": "ok"},
            ]),
        ]
        brief = assistant.build_analysis_brief(steps, {"metadata": {"title": "login-flake"}})
        self.assertIn("Fix the flaky login test", brief)
        self.assertIn("HEALTH", brief)
        self.assertIn("BOTTLENECKS", brief)
        self.assertIn("p95_tool_duration", brief)
        self.assertIn("tool_wait_pct", brief)
        self.assertIn("FILES", brief)
        self.assertIn("step 1", brief)
        self.assertIn("missing_file", brief)
        self.assertIn("ERROR_STEPS", brief)
        self.assertIn("login-flake", brief)

    def test_brief_clips_huge_tool_output(self):
        huge = "X" * 5000
        steps = [
            _step(0, tools=[{
                "tool_name": "Bash", "status": "error", "error_type": "tool_error",
                "input": {"command": "yes"}, "output": huge,
            }]),
        ]
        brief = assistant.build_analysis_brief(steps, {})
        self.assertNotIn(huge, brief)
        self.assertLess(len(brief), assistant._BRIEF_CHAR_LIMIT)


class AnalysisChatTests(unittest.TestCase):
    def test_no_trajectory_does_not_call_http(self):
        called = []

        def boom(*_a, **_k):
            called.append(1)
            raise AssertionError("should not call")

        history = assistant.answer_question("Where did it go wrong?", [], "", chat_fn=boom)
        self.assertEqual(called, [])
        self.assertEqual(history[0]["role"], "user")
        self.assertIn("请先加载", history[1]["content"])

    def test_missing_config_does_not_call_http(self):
        called = []
        cfg = _cfg(api_key="", source="missing")
        history = assistant.answer_question(
            "Bottlenecks?", [], "brief here", config=cfg,
            chat_fn=lambda *_a, **_k: called.append(1) or "nope",
        )
        self.assertEqual(called, [])
        self.assertIn(".env", history[-1]["content"])
        self.assertIn("ANALYZE_", history[-1]["content"])

    def test_chat_sends_brief_in_system_and_question_in_messages(self):
        captured = {}

        def fake_chat(config, system, messages):
            captured["config"] = config
            captured["system"] = system
            captured["messages"] = messages
            return "Step 4 is the bottleneck."

        cfg = _cfg()
        history = assistant.answer_question(
            "What is slow?",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            "## BOTTLENECKS\n- step 4: 40s",
            config=cfg,
            chat_fn=fake_chat,
        )
        self.assertIn("BOTTLENECKS", captured["system"])
        self.assertEqual(captured["messages"][-1], {"role": "user", "content": "What is slow?"})
        self.assertEqual(history[-1]["content"], "Step 4 is the bottleneck.")
        self.assertEqual(captured["config"].model, "demo-model")

    def test_http_error_does_not_leak_response_body(self):
        response = requests.models.Response()
        response.status_code = 401
        response._content = b'{"error":"sk-secret-should-not-appear"}'

        def boom(*_a, **_k):
            raise requests.HTTPError("401", response=response)

        history = assistant.answer_question(
            "Why?", [], "brief", config=_cfg(), chat_fn=boom,
        )
        self.assertIn("HTTP 401", history[-1]["content"])
        self.assertNotIn("sk-secret", history[-1]["content"])

    def test_setup_help_lists_missing_vars(self):
        text = setup_help_text(_cfg(api_key="", model="", base_url="", source="missing"))
        self.assertIn("ANALYZE_API_KEY", text)
        self.assertIn("LABEL_API_KEY", text)

    def test_openai_payload_uses_bearer_and_system(self):
        class FakeResp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(assistant.requests, "post", return_value=FakeResp()) as post:
            text = assistant.complete_chat(
                _cfg(), "SYS", [{"role": "user", "content": "q"}],
            )
        self.assertEqual(text, "ok")
        url = post.call_args[0][0]
        headers = post.call_args.kwargs["headers"]
        body = post.call_args.kwargs["json"]
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYS"})
        self.assertEqual(body["model"], "demo-model")

    def test_system_prompt_requires_simplified_chinese(self):
        self.assertIn("简体中文", assistant.SYSTEM_PROMPT)
        self.assertIn("主要性能瓶颈", assistant.SYSTEM_PROMPT)
        self.assertIn("p95_tool_duration", assistant.SYSTEM_PROMPT)
        self.assertIn("tool_wait_pct", assistant.SYSTEM_PROMPT)

    def test_auto_analysis_on_load_sends_chinese_question(self):
        captured = {}

        def fake_chat(_config, system, messages):
            captured["system"] = system
            captured["messages"] = messages
            return "步骤 1 是瓶颈。"

        steps = [
            _step(0, "user", duration=None, tokens=0, preview="Fix login"),
            _step(1, duration=12.0, tokens=100),
        ]
        brief, history = assistant.analyze_loaded_trajectory(
            steps, {"metadata": {"title": "run"}}, config=_cfg(), chat_fn=fake_chat,
        )
        self.assertIn("HEALTH", brief)
        self.assertEqual(captured["messages"][-1]["content"], assistant.AUTO_ANALYSIS_QUESTION)
        self.assertIn("简体中文", captured["system"])
        self.assertIn("主要性能瓶颈", captured["system"])
        self.assertEqual(history[-1]["content"], "步骤 1 是瓶颈。")
        self.assertTrue(history[-1]["content"])


class AnalysisUiTests(unittest.TestCase):
    def test_sidebar_is_wired_in_build_ui(self):
        from trajviz.insight.ui import sidebar

        source = inspect.getsource(sidebar)
        self.assertIn("🤖 AI Trajectory Analysis", source)
        self.assertIn("analyze_loaded_trajectory", source)
        self.assertIn("answer_question", source)
        self.assertNotIn("suggest_btns", source)
        self.assertNotIn("Where did it go wrong?", source)
        self.assertNotIn("analysis-panel-sub", source)
        self.assertNotIn("Simplified Chinese", source)
        from trajviz.insight.styles import APP_CSS
        self.assertIn("#analysis-sidebar .analysis-panel-title", APP_CSS)
        self.assertIn("text-align: center", APP_CSS)
        self.assertIn("color-scheme: light", APP_CSS)
        self.assertNotIn("resizable-chart .plotly-graph-div", APP_CSS)
        self.assertIn("max-height: none !important", APP_CSS)
        self.assertIn("position=\"right\"", source)
        self.assertIn("open=False", source)
        self.assertIn('width="max(480px, 25vw)"', source)
        self.assertIn('content: "AI Trajectory Analysis"', APP_CSS)
        self.assertIn("state_analysis_brief", source)
        self.assertIn("on_analysis_ask", source)
        # LLM first pass runs on sidebar expand (or load-while-open), not on every load.
        self.assertIn("on_sidebar_expand", source)
        self.assertIn("analysis_sidebar.expand", source)
        self.assertIn("state_sidebar_open", source)
        self.assertIn("on_trajectory_loaded", source)
