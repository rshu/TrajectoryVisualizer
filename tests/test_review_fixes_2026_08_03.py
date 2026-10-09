"""Regression tests for the 2026-08-03 project-review fixes.

Each test pins a specific defect found during the multi-round review so it does
not silently regress. Grouped by the module that was fixed.
"""

import json
import os
import tempfile
import unittest

from trajviz.insight.loaders import load_trajectory, detect_format
from trajviz.insight.parser import parse_steps
from trajviz.insight import rendering, formatting
from trajviz.insight.metrics import (
    compute_metrics, compute_health_verdict, effective_agent,
)
from trajviz.insight.analytics import compute_step_analytics
from trajviz.insight import patterns, diagnostics


def _codex_event(t, ts, payload):
    return {"type": t, "timestamp": ts, "payload": payload}


def _write(tmp, name, obj, jsonl=False):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as f:
        if jsonl:
            for e in obj:
                f.write(json.dumps(e) + "\n")
        else:
            json.dump(obj, f)
    return p


class CodexLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.events = [
            _codex_event("session_meta", "2026-01-05T12:00:00.000Z",
                         {"id": "s1", "cwd": "/p/proj", "cli_version": "1.2.3",
                          "model": "gpt-5", "model_provider": "openai"}),
            _codex_event("response_item", "2026-01-05T12:00:01.000Z",
                         {"type": "message", "role": "user",
                          "content": [{"type": "input_text", "text": "fix it"}]}),
            _codex_event("response_item", "2026-01-05T12:00:03.000Z",
                         {"type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "text": "ok"}]}),
            _codex_event("response_item", "2026-01-05T12:00:04.000Z",
                         {"type": "custom_tool_call", "call_id": "c1", "name": "exec_command",
                          "input": json.dumps({"cmd": "sed -i s/a/b/ app.py"})}),
            _codex_event("response_item", "2026-01-05T12:00:05.000Z",
                         {"type": "custom_tool_call_output", "call_id": "c1", "output": "done"}),
            _codex_event("event_msg", "2026-01-05T12:00:06.000Z",
                         {"type": "token_count", "info": {
                          "last_token_usage":
                              {"input_tokens": 1000, "cached_input_tokens": 800,
                               "output_tokens": 50, "reasoning_output_tokens": 20,
                               "total_tokens": 1070},
                          "total_token_usage":
                              {"input_tokens": 1000, "cached_input_tokens": 800,
                               "output_tokens": 50, "reasoning_output_tokens": 20,
                               "total_tokens": 1070}}}),
            _codex_event("event_msg", "2026-01-05T12:00:07.000Z", {"type": "task_complete"}),
        ]

    def _load(self):
        return load_trajectory(_write(self.tmp, "rollout.jsonl", self.events, jsonl=True))

    def test_detects_as_codex(self):
        self.assertEqual(detect_format(self._load()), "codex")

    def test_timestamps_are_epoch_ms(self):
        steps = parse_steps(self._load())
        for s in steps:
            self.assertIsInstance(s["time_created_ms"], int)

    def test_token_count_ingested(self):
        steps = parse_steps(self._load())
        self.assertEqual(steps[-1]["tokens"]["total"], 1070)
        m = compute_metrics(steps, self._load())
        self.assertEqual(m["tokens"]["total"], 1070)
        self.assertGreater(m["total_duration"], 0)

    def test_multiple_token_counts_are_accumulated_and_duplicates_ignored(self):
        extra_usage = _codex_event(
            "event_msg", "2026-01-05T12:00:06.500Z",
            {"type": "token_count", "info": {
                "last_token_usage": {
                    "input_tokens": 200, "cached_input_tokens": 150,
                    "output_tokens": 5, "reasoning_output_tokens": 2,
                    "total_tokens": 205,
                },
                "total_token_usage": {
                    "input_tokens": 1200, "cached_input_tokens": 950,
                    "output_tokens": 55, "reasoning_output_tokens": 22,
                    "total_tokens": 1275,
                },
            }},
        )
        duplicate_usage = _codex_event(
            "event_msg", "2026-01-05T12:00:06.750Z", extra_usage["payload"],
        )
        events = self.events[:-1] + [extra_usage, duplicate_usage, self.events[-1]]
        raw = load_trajectory(_write(self.tmp, "multi-usage.jsonl", events, jsonl=True))
        tokens = parse_steps(raw)[-1]["tokens"]

        self.assertEqual(tokens["total"], 1275)
        self.assertEqual(tokens["input"], 1200)
        self.assertEqual(tokens["output"], 55)
        self.assertEqual(tokens["reasoning"], 22)
        self.assertEqual(tokens["cache_read"], 950)

    def test_same_cumulative_with_different_last_usage_is_not_a_duplicate(self):
        parallel_usage = _codex_event(
            "event_msg", "2026-01-05T12:00:06.500Z",
            {"type": "token_count", "info": {
                "last_token_usage": {
                    "input_tokens": 9, "cached_input_tokens": 8,
                    "output_tokens": 1, "reasoning_output_tokens": 0,
                    "total_tokens": 10,
                },
                "total_token_usage": self.events[-2]["payload"]["info"]["total_token_usage"],
            }},
        )
        events = self.events[:-1] + [parallel_usage, self.events[-1]]
        raw = load_trajectory(_write(self.tmp, "parallel-usage.jsonl", events, jsonl=True))
        self.assertEqual(parse_steps(raw)[-1]["tokens"]["total"], 1080)

    def test_custom_tool_call_captured(self):
        steps = parse_steps(self._load())
        tools = [t["tool_name"] for s in steps for t in s["tool_calls"]]
        self.assertIn("Write", tools)  # sed -i classified as Write

    def test_modern_exec_input_preserves_and_classifies_command(self):
        modern_call = _codex_event(
            "response_item", "2026-01-05T12:00:05.100Z",
            {"type": "custom_tool_call", "call_id": "c2", "name": "exec",
             "input": 'const r = await tools.exec_command({"cmd":"rg -n needle src"});'},
        )
        modern_output = _codex_event(
            "response_item", "2026-01-05T12:00:05.200Z",
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "match"},
        )
        events = self.events[:-2] + [modern_call, modern_output] + self.events[-2:]
        steps = parse_steps(load_trajectory(
            _write(self.tmp, "modern-exec.jsonl", events, jsonl=True)))
        tool = next(t for s in steps for t in s["tool_calls"] if t["tool_id"] == "c2")
        self.assertEqual(tool["tool_name"], "Grep")
        self.assertEqual(tool["input"]["command"], "rg -n needle src")

    def test_modern_apply_patch_is_a_write_with_target(self):
        patch = "*** Begin Patch\n*** Update File: src/app.py\n@@\n-old\n+new\n*** End Patch"
        modern_call = _codex_event(
            "response_item", "2026-01-05T12:00:05.100Z",
            {"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch",
             "input": patch},
        )
        modern_output = _codex_event(
            "response_item", "2026-01-05T12:00:05.200Z",
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "Done!"},
        )
        events = self.events[:-2] + [modern_call, modern_output] + self.events[-2:]
        steps = parse_steps(load_trajectory(
            _write(self.tmp, "modern-patch.jsonl", events, jsonl=True)))
        tool = next(t for s in steps for t in s["tool_calls"] if t["tool_id"] == "c2")
        self.assertEqual(tool["tool_name"], "Write")
        self.assertEqual(tool["input"]["file_path"], "src/app.py")
        self.assertEqual(tool["input"]["patch"], patch)

    def test_non_shell_function_call_preserves_name_and_arguments(self):
        spawn = _codex_event(
            "response_item", "2026-01-05T12:00:05.100Z",
            {"type": "function_call", "call_id": "c2", "name": "spawn_agent",
             "arguments": json.dumps({"task_name": "review", "message": "inspect"})},
        )
        spawn_output = _codex_event(
            "response_item", "2026-01-05T12:00:05.200Z",
            {"type": "function_call_output", "call_id": "c2", "output": "queued"},
        )
        events = self.events[:-2] + [spawn, spawn_output] + self.events[-2:]
        steps = parse_steps(load_trajectory(
            _write(self.tmp, "spawn.jsonl", events, jsonl=True)))
        tool = next(t for s in steps for t in s["tool_calls"] if t["tool_id"] == "c2")
        self.assertEqual(tool["tool_name"], "spawn_agent")
        self.assertEqual(tool["input"]["task_name"], "review")

    def test_metadata_normalized(self):
        md = self._load()["metadata"]
        self.assertEqual(md["session_id"], "s1")
        self.assertEqual(md["directory_name"], "proj")
        self.assertEqual(md["model"], "gpt-5")

    def test_truncated_jsonl_does_not_reject_whole_file(self):
        p = os.path.join(self.tmp, "trunc.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for e in self.events[:3]:
                f.write(json.dumps(e) + "\n")
            f.write('{"type":"response_item","timestamp":"2026-')  # truncated
        raw = load_trajectory(p)
        self.assertNotIn("_error", raw)

    def test_malformed_interior_jsonl_is_rejected(self):
        p = os.path.join(self.tmp, "corrupt.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.events[0]) + "\n")
            f.write('{"type": definitely-not-json}\n')
            f.write(json.dumps(self.events[1]) + "\n")
        raw = load_trajectory(p)
        self.assertIn("_error", raw)
        self.assertIn("line 2", raw["_error"])

    def test_non_dict_first_line_errors_gracefully(self):
        p = os.path.join(self.tmp, "weird.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write('"just a string"\n')
        self.assertIn("_error", load_trajectory(p))

    def test_malformed_metadata_and_reasoning_entries_do_not_crash(self):
        events = [
            _codex_event("session_meta", "2026-01-05T12:00:00Z", None),
            _codex_event("response_item", "2026-01-05T12:00:01Z",
                         {"type": "reasoning", "summary": [None, {"text": "ok"}]}),
            _codex_event("event_msg", "2026-01-05T12:00:02Z", {"type": "task_complete"}),
        ]
        raw = load_trajectory(_write(self.tmp, "malformed-fields.jsonl", events, jsonl=True))
        self.assertNotIn("_error", raw)
        self.assertEqual(parse_steps(raw)[0]["text_preview"], "ok")


class ClaudeCodeLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._i = 0

    def _entry(self, mid, ts, text):
        e = {"role": "assistant", "message_id": mid, "index": self._i,
             "content": [{"type": "text", "text": text}],
             "usage": {"input_tokens": 10, "output_tokens": 5}}
        if ts is not None:
            e["timestamp"] = ts
        self._i += 1
        return e

    def _raw(self):
        self._i = 0
        return {
            "format": "ccsession-trajectory",
            "session": {"id": "s", "working_directory": "/p",
                        "started_at": "2026-01-05T12:00:00Z",
                        "ended_at": "2026-01-05T12:01:30Z"},
            "statistics": {},
            "trajectory": [
                self._entry("m0", "2026-01-05T12:00:00Z", "first"),
                self._entry("m1", "2026-01-05T12:00:10Z", "second"),
                self._entry("mNT", None, "no-timestamp"),
                self._entry("m2", "2026-01-05T12:01:20Z", "third"),
            ],
            "sub_agents": [{"agent_id": "sub1", "trajectory": [
                self._entry("sa0", "2026-01-05T12:00:30Z", "subagent")]}],
        }

    def _steps(self):
        return parse_steps(load_trajectory(_write(self.tmp, "cc.json", self._raw())))

    def test_missing_timestamp_step_not_sorted_to_front(self):
        steps = self._steps()
        self.assertTrue(steps[0]["text_preview"].startswith("first"))

    def test_subagent_duration_not_absorbing_main_idle(self):
        steps = self._steps()
        sub = next(s for s in steps if s["agent"] == "sub1")
        self.assertIsNone(sub["duration"])  # single sub-agent step, not 50s

    def test_main_step_duration_spans_next_main_step(self):
        steps = self._steps()
        m1 = next(s for s in steps if s["text_preview"].startswith("second"))
        self.assertEqual(m1["duration"], 70.0)  # 10s -> 80s, skipping interleaved sub-agent


class RobustnessTests(unittest.TestCase):
    def test_null_directory_does_not_crash(self):
        tmp = tempfile.mkdtemp()
        oc = {"info": {"id": "s", "directory": None, "time": {"created": 1000, "updated": 2000}},
              "messages": [{"info": {"role": "user", "time": {"created": 1000}}, "parts": []}]}
        raw = load_trajectory(_write(tmp, "n.json", oc))
        self.assertNotIn("_error", raw)

    def test_non_string_role_does_not_crash_renderers(self):
        steps = parse_steps({"messages": [{"role": 123, "parts": []},
                                          {"role": ["x"], "parts": []}]})
        rendering.render_workflow_html(steps)
        rendering.render_toc_sidebar(steps)
        rendering._format_step_header(steps[0])

    def test_explicit_zero_duration_is_not_backfilled(self):
        raw = {
            "messages": [{
                "info": {"role": "assistant", "time": {"created": 1000, "completed": 1000}},
                "parts": [{"type": "text", "text": "instant"}],
            }],
            "timing": {"finished_at": "1970-01-01T00:00:10Z"},
        }
        step = parse_steps(raw)[0]
        self.assertEqual(step["duration"], 0.0)
        self.assertEqual(step["time_completed_ms"], 1000)


class XssTests(unittest.TestCase):
    PAYLOAD = "<img src=x onerror=alert(1)>"

    def test_role_escaped_in_workflow_html(self):
        steps = parse_steps({"messages": [
            {"info": {"role": self.PAYLOAD, "time": {"created": 1, "completed": 2}},
             "parts": [{"type": "text", "text": "hi"}]}]})
        out = rendering.render_workflow_html(steps)
        self.assertNotIn("<img", out)
        self.assertIn("&lt;", out)

    def test_todowrite_content_escaped(self):
        out = rendering.build_antipattern_summary_html(
            [], [], {"stalled": [{"content": "</span>" + self.PAYLOAD, "start_step": 4}]}, 0)
        self.assertNotIn("<img", out)
        self.assertIn("&lt;", out)
        self.assertIn("tvGotoWorkflowStep(4)", out)
        self.assertIn("#4", out)

    def test_antipattern_summary_links_workflow_steps(self):
        out = rendering.build_antipattern_summary_html(
            [{"start_step": 10, "end_step": 12, "length": 3, "tools": ["Grep"]}],
            [{"step": 7, "command": "cat /tmp/x", "pattern": "cat"}],
            {"stalled": []},
            error_count=2,
            error_steps=[3, 9],
        )
        self.assertIn("Anti-pattern summary", out)
        self.assertIn("insight-step-link", out)
        for idx in (3, 7, 9, 10, 11, 12):
            self.assertIn(f"tvGotoWorkflowStep({idx})", out)
            self.assertIn(f"#{idx}", out)

    def test_metric_chip_escaped(self):
        chip = formatting._metric_chip(self.PAYLOAD, self.PAYLOAD)
        self.assertNotIn("<img", chip)


class MetricsTests(unittest.TestCase):
    def _verdicts(self, m):
        return {v["metric"]: v for v in compute_health_verdict(m, [])}

    def test_throughput_uses_output_rate_not_inflated_total(self):
        m = {"tokens": {"total": 918125}, "avg_cache_ratio": 98.0,
             "tokens_per_second": 3946.2, "output_tokens_per_sec": 48.3,
             "tool_call_count": 5, "tool_success_rate": 100, "tool_fail": 0}
        v = self._verdicts(m)["Throughput"]
        self.assertEqual(v["status"], "warn")
        self.assertIn("48.3", v["label"])
        self.assertIn("gen tok/s", v["label"])

    def test_verdicts_na_on_missing_token_data(self):
        m = {"tokens": {"total": 0}, "avg_cache_ratio": 0, "output_tokens_per_sec": None,
             "tool_call_count": 0, "tool_success_rate": 0, "tool_fail": 0}
        v = self._verdicts(m)
        self.assertEqual(v["Cache Efficiency"]["label"], "N/A")
        self.assertEqual(v["Throughput"]["label"], "N/A")

    def test_analytics_tool_time_metadata_fallback(self):
        step = {"index": 0, "role": "assistant", "duration": 10.0,
                "tokens": {"total": 100, "input": 100, "output": 0, "reasoning": 0, "cache_read": 0},
                "parts": [], "tool_call_count": 1, "error_count": 0, "has_reasoning": False,
                "text_preview": "", "finish": "", "model_id": "", "agent": "",
                "tool_calls": [{"tool_name": "Bash", "time_start": None, "time_end": None,
                                "duration_ms": None, "metadata": {"totalDurationMs": 8000}}]}
        sa = compute_step_analytics([step])[0]
        self.assertEqual(sa["tool_time_share"], 0.8)

    def test_effective_agent_named_main_maps_to_main(self):
        # CodeArts-style: named main agent with is_sub_agent present-and-false
        main = {"role": "assistant", "agent": "build", "is_sub_agent": False, "session_id": "root"}
        sub = {"role": "assistant", "agent": "Agent", "is_sub_agent": True, "session_id": "child"}
        self.assertEqual(effective_agent(main), "")
        self.assertEqual(effective_agent(sub), "child")
        # OpenCode "(subagent)" suffix
        self.assertEqual(effective_agent({"role": "assistant", "agent": "x (subagent)"}), "x (subagent)")


class PatternsDiagnosticsTests(unittest.TestCase):
    def _bash(self, i, cmd, out=""):
        return {"index": i, "role": "assistant",
                "tool_calls": [{"tool_name": "Bash", "input": {"command": cmd},
                                "output": out, "status": "success"}]}

    def test_silent_setup_bash_not_fruitless(self):
        setup = [self._bash(0, "mkdir -p build"), self._bash(1, "cp a b"),
                 self._bash(2, "chmod +x x"), self._bash(3, "git add -A")]
        self.assertEqual(patterns.detect_fruitless_streaks(setup), [])

    def test_empty_grep_via_bash_still_fruitless(self):
        s = [self._bash(0, "grep -rn foo ."), self._bash(1, "rg bar"), self._bash(2, "grep baz .")]
        streaks = patterns.detect_fruitless_streaks(s)
        self.assertTrue(streaks and streaks[0]["length"] == 3)

    def test_tool_sequences_non_overlapping(self):
        burst = [{"index": i, "role": "assistant",
                  "tool_calls": [{"tool_name": "Read", "input": {}, "status": "success"}]}
                 for i in range(4)]
        seqs = patterns.detect_tool_sequences(burst, min_freq=3)
        self.assertFalse([r for r in seqs if r["sequence"] == ["Read", "Read"]])

    def test_file_targeting_backslash_paths_match(self):
        def tc(tool, path):
            return {"tool_name": tool, "input": {"file_path": path}, "status": "success", "output": ""}
        steps = [
            {"index": 0, "role": "assistant", "tokens": {"total": 10}, "parts": [],
             "tool_calls": [tc("Read", r"C:\proj\app.py")]},
            {"index": 1, "role": "assistant", "tokens": {"total": 10}, "parts": [],
             "tool_calls": [tc("Edit", r"C:\proj\app.py")]},
        ]
        inter = diagnostics.extract_file_interactions(steps)
        targets = diagnostics.identify_target_files(steps)
        m = diagnostics.compute_file_targeting_metrics(inter, targets, len(steps))
        self.assertTrue(m["steps_to_first_touch"])

    def test_skill_tool_and_skill_md_are_skill_interactions(self):
        steps = [
            {"index": 0, "role": "assistant", "tokens": {"total": 10}, "parts": [],
             "tool_calls": [{"tool_name": "Skill", "input": {"skill": "create-hook"},
                             "status": "completed"}]},
            {"index": 1, "role": "assistant", "tokens": {"total": 10}, "parts": [],
             "tool_calls": [{"tool_name": "Read",
                             "input": {"file_path": "/home/user/.cursor/skills/canvas/SKILL.md"},
                             "status": "completed"}]},
            {"index": 2, "role": "assistant", "tokens": {"total": 10}, "parts": [],
             "tool_calls": [{"tool_name": "Read", "input": {"file_path": "/repo/app.py"},
                             "status": "completed"}]},
            {"index": 3, "role": "assistant", "tokens": {"total": 10}, "parts": [],
             "tool_calls": [{"tool_name": "Write",
                             "input": {"file_path": "/repo/.cursor/skills/new/SKILL.md"},
                             "status": "completed"}]},
        ]
        inter = diagnostics.extract_file_interactions(steps)
        by_step = {i["step"]: i for i in inter}
        self.assertEqual(by_step[0]["type"], "skill")
        self.assertEqual(by_step[0]["path"], "skill:create-hook")
        self.assertEqual(by_step[1]["type"], "skill")
        self.assertTrue(by_step[1]["path"].endswith("SKILL.md"))
        self.assertEqual(by_step[2]["type"], "read")
        self.assertEqual(by_step[3]["type"], "write")

        from trajviz.insight.charts import build_file_interaction_chart

        fig = build_file_interaction_chart(inter)
        names = {t.name for t in fig.data}
        self.assertIn("skill (star)", names)
        self.assertIn("read (circle)", names)
        self.assertIn("write (square)", names)
        skill_trace = next(t for t in fig.data if t.name == "skill (star)")
        self.assertEqual(skill_trace.marker.symbol, "star")
        self.assertIn("skill:create-hook", list(skill_trace.y))
        write_trace = next(t for t in fig.data if t.name == "write (square)")
        self.assertEqual(write_trace.marker.symbol, "square")

    def test_multi_agent_file_legend_explains_marker_shapes(self):
        from trajviz.insight.charts import build_file_interaction_chart

        interactions = [
            {"step": 0, "path": "/repo/a.py", "type": "read", "tool": "Read", "tokens": 1},
            {"step": 1, "path": "/repo/a.py", "type": "write", "tool": "Edit", "tokens": 1},
            {"step": 1, "path": "/repo", "type": "search", "tool": "Grep", "tokens": 1},
            {"step": 2, "path": "skill:create-hook", "type": "skill", "tool": "Skill", "tokens": 1},
        ]
        steps = [
            {"index": 0, "role": "assistant", "agent": "build", "is_sub_agent": False,
             "session_id": "ses_root", "tokens": {"total": 1}, "tool_call_count": 1},
            {"index": 1, "role": "assistant", "agent": "explore", "is_sub_agent": True,
             "session_id": "ses_ex", "tokens": {"total": 1}, "tool_call_count": 1},
            {"index": 2, "role": "assistant", "agent": "explore", "is_sub_agent": True,
             "session_id": "ses_ex", "tokens": {"total": 1}, "tool_call_count": 1},
        ]
        fig = build_file_interaction_chart(interactions, steps=steps)
        names = {t.name for t in fig.data if t.showlegend is not False}
        self.assertIn("read (circle)", names)
        self.assertIn("write (square)", names)
        self.assertIn("search (triangle)", names)
        self.assertIn("skill (star)", names)
        by_name = {t.name: t for t in fig.data}
        self.assertEqual(by_name["read (circle)"].marker.symbol, "circle")
        self.assertEqual(by_name["write (square)"].marker.symbol, "square")
        self.assertEqual(by_name["search (triangle)"].marker.symbol, "triangle-up")
        self.assertEqual(by_name["skill (star)"].marker.symbol, "star")

    def test_hotspot_inference_excludes_pre_step_idle(self):
        step = {"index": 0, "role": "assistant", "duration": 20.0,
                "tool_calls": [{"tool_name": "Task", "time_start": None, "time_end": None,
                                "duration_ms": 15000, "metadata": {}}]}
        d = diagnostics.decompose_hotspot_duration(step, idle_gap=8.0)
        self.assertEqual(d["inference_s"], 5.0)


class ConvergeTests(unittest.TestCase):
    def test_extract_base_command_multi_env(self):
        from trajviz.converge.canonical import _extract_base_command
        self.assertEqual(_extract_base_command("FOO=1 BAR=2 pytest"), "pytest")
        self.assertEqual(_extract_base_command("make test"), "make")

    def test_first_passing_validation_fires_for_validation_command(self):
        from trajviz.converge.canonical import CanonicalAction
        from trajviz.converge.milestones import extract_milestones
        acts = [CanonicalAction(step_index=5, action_type="COMMAND", target="pytest",
                                effect_label="justified",
                                effect_detail={"reason": "validation_command"})]
        self.assertEqual(extract_milestones(acts)["first_passing_validation"], 5)

    def test_recommendation_reports_minor_regressions(self):
        from trajviz.converge.intervention import generate_recommendation
        rec = generate_recommendation(
            {"latency": {"direction": "improved"}, "tokens": {"direction": "regressed"}}, [])
        self.assertNotIn("without regressions", rec)


if __name__ == "__main__":
    unittest.main()
