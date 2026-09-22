"""Tests for N-run scorecard + behavioral comparison."""

import json
import os
import tempfile
import unittest

from trajviz.converge.canonical import CanonicalAction
from trajviz.insight.charts import (
    bind_timeline_agents,
    build_run_group_agent_timeline,
    build_skill_agent_chart,
    build_tool_chart,
)
from trajviz.insight.rendering import render_workflow_html
from trajviz.insight.run_group import (
    build_behavioral_comparison,
    build_run_group_scorecard,
    build_run_group_scorecard_html,
    build_run_group_behavior_html,
    build_run_scorecard_row,
    default_run_label,
    extract_capability_usage,
    normalize_run_paths,
    _parse_skill_name,
)


def _minimal_raw(
    *,
    steps: int = 3,
    tokens_each: int = 100,
    finish: str = "stop",
    read_path: str = "a.py",
    extra_search: bool = False,
) -> dict:
    """Tiny synthetic trajectory for scorecard / behavior tests."""
    messages = []
    t = 1_000
    for i in range(steps):
        role = "user" if i % 2 == 0 else "assistant"
        info = {
            "role": role,
            "id": f"m{i}",
            "sessionID": "ses_test",
            "time": {"created": t, "completed": t + 500},
            "tokens": {
                "total": tokens_each if role == "assistant" else 0,
                "input": tokens_each if role == "assistant" else 0,
                "output": 0,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        }
        if role == "assistant":
            info["finish"] = finish
            info["agent"] = "build"
        parts: list[dict] = [{"type": "text", "text": f"msg {i}"}]
        if role == "assistant" and i == 1:
            parts.append(
                {
                    "type": "tool",
                    "tool": "read",
                    "callID": "c1",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": read_path},
                        "output": "ok",
                        "time": {"start": t, "end": t + 100},
                    },
                }
            )
            if extra_search:
                parts.append(
                    {
                        "type": "tool",
                        "tool": "grep",
                        "callID": "c2",
                        "state": {
                            "status": "completed",
                            "input": {"pattern": "TODO", "path": "src"},
                            "output": "hits",
                            "time": {"start": t, "end": t + 120},
                        },
                    }
                )
        messages.append({"info": info, "parts": parts})
        t += 1000
    return {
        "timing": {
            "started_at": "2024-01-01T00:00:00Z",
            "finished_at": "2024-01-01T00:01:00Z",
            "total_duration": 60,
        },
        "messages": messages,
    }


def _action(atype: str, target: str, step: int = 0) -> CanonicalAction:
    return CanonicalAction(step_index=step, action_type=atype, target=target, tool=atype)


def _steps_with_tools(calls: list[tuple[str, dict | None]]) -> list[dict]:
    """Minimal parsed steps carrying tool_calls for capability extraction."""
    tool_calls = []
    for _i, (name, inp) in enumerate(calls):
        tool_calls.append(
            {
                "tool_name": name,
                "input": inp or {},
                "status": "completed",
            }
        )
    return [
        {
            "index": 0,
            "role": "assistant",
            "tool_calls": tool_calls,
            "tokens": {"total": 10, "input": 10, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0},
        }
    ]


class CapabilityExtractionTests(unittest.TestCase):
    def test_skill_name(self):
        self.assertEqual(
            _parse_skill_name("Skill", {"skill": "create-rule"}),
            "create-rule",
        )
        self.assertEqual(
            _parse_skill_name("skill", {"name": "canvas"}),
            "canvas",
        )
        self.assertIsNone(_parse_skill_name("Bash", {"command": "ls"}))
        self.assertEqual(
            _parse_skill_name("invoke_skill", {"skill_name": "statusline"}),
            "statusline",
        )

    def test_extract_tools_and_skills(self):
        steps = _steps_with_tools(
            [
                ("Read", {"file_path": "a.py"}),
                ("mcp__slack__post_message", {"text": "hi"}),
                ("Skill", {"skill": "create-hook"}),
                ("Skill", {"skill": "create-hook"}),
                ("Bash", {"command": "ls"}),
            ]
        )
        usage = extract_capability_usage(steps)
        self.assertEqual(usage["tools"]["Read"], 1)
        self.assertEqual(usage["tools"]["Bash"], 1)
        self.assertEqual(usage["tools"]["Skill"], 2)
        self.assertEqual(usage["tools"]["mcp__slack__post_message"], 1)
        self.assertEqual(usage["skills"]["create-hook"], 2)
        self.assertNotIn("mcps", usage)


class NormalizePathsTests(unittest.TestCase):
    def test_list_and_dedupe(self):
        self.assertEqual(normalize_run_paths(["a.json", "b.json", "a.json"]), ["a.json", "b.json"])

    def test_single_and_none(self):
        self.assertEqual(normalize_run_paths("only.json"), ["only.json"])
        self.assertEqual(normalize_run_paths(None), [])

    def test_default_label(self):
        self.assertEqual(default_run_label("/tmp/claude-v2.json"), "claude-v2")


class ScorecardRowTests(unittest.TestCase):
    def test_row_from_raw(self):
        raw = _minimal_raw(steps=4, tokens_each=50)
        row = build_run_scorecard_row(raw, path="/tmp/run_a.json", label="run-a")
        self.assertEqual(row["label"], "run-a")
        self.assertIsNone(row["error"])
        self.assertTrue(row["finished"])
        self.assertEqual(row["steps"], 4)
        self.assertGreaterEqual(row["tokens"], 50)
        self.assertGreaterEqual(row["tool_calls"], 1)

    def test_error_row(self):
        row = build_run_scorecard_row({"_error": "boom"}, path="x.json")
        self.assertEqual(row["error"], "boom")
        self.assertIsNone(row["steps"])


class BehavioralComparisonTests(unittest.TestCase):
    def test_similarity_and_consensus(self):
        runs = [
            {
                "run_id": "a",
                "label": "A",
                "actions": [
                    _action("FILE_READ", "src/main.py", 1),
                    _action("FILE_WRITE", "src/main.py", 2),
                    _action("SEARCH", "bug@src", 3),
                ],
                "steps": _steps_with_tools(
                    [
                        ("Read", {}),
                        ("Skill", {"skill": "shared-skill"}),
                        ("mcp__github__list_issues", {}),
                    ]
                ),
            },
            {
                "run_id": "b",
                "label": "B",
                "actions": [
                    _action("FILE_READ", "src/main.py", 1),
                    _action("FILE_WRITE", "src/main.py", 2),
                    _action("FILE_READ", "other.py", 3),
                ],
                "steps": _steps_with_tools(
                    [
                        ("Read", {}),
                        ("Write", {}),
                        ("Skill", {"skill": "shared-skill"}),
                        ("Skill", {"skill": "only-b"}),
                        ("mcp__slack__post_message", {}),
                    ]
                ),
            },
            {
                "run_id": "c",
                "label": "C",
                "actions": [
                    _action("FILE_READ", "src/main.py", 1),
                    _action("FILE_WRITE", "src/main.py", 2),
                ],
                "steps": _steps_with_tools(
                    [
                        ("Read", {}),
                        ("Skill", {"skill": "shared-skill"}),
                    ]
                ),
            },
        ]
        behavior = build_behavioral_comparison(runs)
        self.assertEqual(behavior["baseline_run_id"], "a")
        self.assertEqual(behavior["similarity"]["a"]["a"], 1.0)
        self.assertGreater(behavior["similarity"]["a"]["c"], 0.5)
        self.assertTrue(any(r["type"] == "SEARCH" and r["target"] == "bug@src" for r in behavior["action_matrix"]))
        self.assertFalse(any(r["type"] in ("FILE_READ", "FILE_WRITE") for r in behavior["action_matrix"]))
        paths = {r["path"]: r for r in behavior["file_matrix"]}
        self.assertIn("src/main.py", paths)
        self.assertEqual(paths["src/main.py"]["kind"], "consensus")
        self.assertEqual(paths["src/main.py"]["cells"]["a"]["read"], 1)
        self.assertEqual(paths["src/main.py"]["cells"]["a"]["write"], 1)
        self.assertIn("other.py", paths)
        self.assertEqual(paths["other.py"]["kind"], "unique")
        self.assertEqual(paths["other.py"]["cells"]["b"]["read"], 1)
        self.assertEqual(paths["other.py"]["cells"]["a"]["read"], 0)
        action_by_key = {(r["type"], r["target"]): r for r in behavior["action_matrix"]}
        self.assertEqual(action_by_key[("SEARCH", "bug@src")]["kind"], "unique")
        self.assertTrue(action_by_key[("SEARCH", "bug@src")]["cells"]["a"]["present"])
        self.assertFalse(action_by_key[("SEARCH", "bug@src")]["cells"]["b"]["present"])

        tools = {r["key"]: r for r in behavior["tool_matrix"]}
        self.assertIn("Read", tools)
        self.assertEqual(tools["Read"]["kind"], "consensus")
        self.assertEqual(tools["Write"]["kind"], "unique")
        self.assertEqual(tools["mcp__github__list_issues"]["kind"], "unique")
        self.assertEqual(tools["mcp__slack__post_message"]["kind"], "unique")
        self.assertNotIn("mcp_matrix", behavior)
        skills = {r["key"]: r for r in behavior["skill_matrix"]}
        self.assertEqual(skills["shared-skill"]["kind"], "consensus")
        self.assertEqual(skills["only-b"]["kind"], "unique")

        # B and C have pattern slots vs baseline A
        self.assertIn("b", behavior["patterns_vs_baseline"])
        self.assertIn("c", behavior["patterns_vs_baseline"])
        html = build_run_group_scorecard_html(
            {
                "rows": [
                    {
                        "run_id": "a",
                        "label": "A",
                        "error": None,
                        "finished": True,
                        "steps": 1,
                        "wall_clock_s": 1,
                        "wall_clock_fmt": "1s",
                        "tokens": 1,
                        "tool_calls": 1,
                        "tool_success_pct": 100,
                        "peak_occupancy": 0,
                        "peak_pct": None,
                        "compactions": 0,
                        "format": "test",
                    },
                    {
                        "run_id": "b",
                        "label": "B",
                        "error": None,
                        "finished": True,
                        "steps": 2,
                        "wall_clock_s": 2,
                        "wall_clock_fmt": "2s",
                        "tokens": 2,
                        "tool_calls": 2,
                        "tool_success_pct": 100,
                        "peak_occupancy": 0,
                        "peak_pct": None,
                        "compactions": 0,
                        "format": "test",
                    },
                ],
                "behavior": behavior,
                "ok": True,
                "error": None,
            }
        )
        self.assertIn("File coverage", html)
        self.assertIn("Action coverage", html)
        self.assertIn("Tool coverage", html)
        self.assertNotIn("MCP coverage", html)
        self.assertIn("Skill coverage", html)
        self.assertIn("rg-badge-r", html)
        self.assertIn("rg-badge-w", html)
        self.assertIn("rg-atype-search", html)
        self.assertIn("other.py", html)
        self.assertIn("shared-skill", html)


class FileCoverageCountTests(unittest.TestCase):
    def test_read_write_counts_and_exclude_from_actions(self):
        runs = [
            {
                "run_id": "a",
                "label": "A",
                "actions": [
                    _action("FILE_READ", "x.py", 1),
                    _action("FILE_READ", "x.py", 2),
                    _action("FILE_READ", "x.py", 3),
                    _action("FILE_WRITE", "x.py", 4),
                    _action("FILE_WRITE", "x.py", 5),
                    _action("SEARCH", "todo", 6),
                ],
            },
            {
                "run_id": "b",
                "label": "B",
                "actions": [
                    _action("FILE_READ", "x.py", 1),
                    _action("COMMAND", "pytest", 2),
                ],
            },
        ]
        behavior = build_behavioral_comparison(runs)
        files = {r["path"]: r for r in behavior["file_matrix"]}
        self.assertEqual(files["x.py"]["cells"]["a"]["read"], 3)
        self.assertEqual(files["x.py"]["cells"]["a"]["write"], 2)
        self.assertEqual(files["x.py"]["cells"]["b"]["read"], 1)
        self.assertEqual(files["x.py"]["cells"]["b"]["write"], 0)
        types = {r["type"] for r in behavior["action_matrix"]}
        self.assertEqual(types, {"SEARCH", "COMMAND"})
        from trajviz.insight.run_group import _rw_badges

        self.assertIn("R×3", _rw_badges(files["x.py"]["cells"]["a"]))
        self.assertIn("W×2", _rw_badges(files["x.py"]["cells"]["a"]))
        self.assertEqual(_rw_badges(files["x.py"]["cells"]["b"]), _rw_badges({"read": 1, "write": 0}))
        self.assertIn(">R<", _rw_badges({"read": 1, "write": 0}))


class ScorecardGroupTests(unittest.TestCase):
    def test_requires_two_paths(self):
        result = build_run_group_scorecard([])
        self.assertFalse(result["ok"])
        html = build_run_group_scorecard_html(result)
        self.assertTrue(
            "two" in html.lower() or "overview" in html.lower(),
            html,
        )

    def test_overview_baseline_plus_one_upload(self):
        baseline = _minimal_raw(steps=4, tokens_each=200, read_path="shared.py")
        baseline["_source_path"] = "/tmp/overview-run.json"
        other = _minimal_raw(
            steps=4,
            tokens_each=80,
            read_path="shared.py",
            extra_search=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path_b = os.path.join(tmp, "alt-model.json")
            with open(path_b, "w", encoding="utf-8") as f:
                json.dump(other, f)
            # One upload is enough when Overview baseline is present
            result = build_run_group_scorecard(
                [path_b],
                baseline_raw=baseline,
            )
            self.assertTrue(result["ok"], result.get("error"))
            self.assertEqual(len(result["rows"]), 2)
            self.assertIn("baseline", result["rows"][0]["label"].lower())
            self.assertEqual(
                result["behavior"]["baseline_run_id"],
                result["rows"][0]["run_id"],
            )
            # Without baseline, one path is not enough
            alone = build_run_group_scorecard([path_b])
            self.assertFalse(alone["ok"])

    def test_two_temp_files_include_behavior(self):
        raw_a = _minimal_raw(steps=4, tokens_each=200, read_path="shared.py")
        raw_b = _minimal_raw(
            steps=4,
            tokens_each=80,
            read_path="shared.py",
            extra_search=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path_a = os.path.join(tmp, "model-a.json")
            path_b = os.path.join(tmp, "model-b.json")
            with open(path_a, "w", encoding="utf-8") as f:
                json.dump(raw_a, f)
            with open(path_b, "w", encoding="utf-8") as f:
                json.dump(raw_b, f)
            result = build_run_group_scorecard(
                [path_a, path_b],
                labels=["Model A", "Model B"],
            )
            self.assertTrue(result["ok"], result.get("error"))
            self.assertIsNotNone(result.get("behavior"))
            self.assertEqual(len(result.get("timeline_runs") or []), 2)
            behavior = result["behavior"]
            self.assertEqual(behavior["labels"]["model-a"], "Model A")
            self.assertIn("model-b", behavior["similarity"]["model-a"])
            html = build_run_group_scorecard_html(result, include_behavior=False)
            self.assertIn("Model A", html)
            self.assertNotIn("Behavioral similarity", html)
            behavior_html = build_run_group_behavior_html(result)
            self.assertIn("Behavioral similarity", behavior_html)
            self.assertIn("Consensus", behavior_html)
            fig = build_run_group_agent_timeline(result["timeline_runs"])
            self.assertGreater(len(fig.data), 0)
            self.assertIn("Agent timeline", fig.layout.title.text)

    def test_fixture_plus_copy(self):
        fixture = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "codearts_minimal.json",
        )
        if not os.path.isfile(fixture):
            self.skipTest("codearts_minimal.json missing")
        with tempfile.TemporaryDirectory() as tmp:
            twin = os.path.join(tmp, "codearts_twin.json")
            with open(fixture, encoding="utf-8") as src, open(twin, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            result = build_run_group_scorecard([fixture, twin])
            self.assertTrue(result["ok"], result.get("error"))
            self.assertIsNotNone(result.get("behavior"))
            # Identical twins → F1 near 1
            ids = result["behavior"]["run_ids"]
            self.assertEqual(len(ids), 2)
            f1 = result["behavior"]["similarity"][ids[0]][ids[1]]
            self.assertGreaterEqual(f1, 0.9)


class AgentTimelineChartTests(unittest.TestCase):
    def test_one_lane_per_run_colored_by_agent(self):
        # No is_sub_agent key → Claude-style: non-empty agent is a sub-agent id
        runs = [
            {
                "run_id": "a",
                "label": "Run A",
                "steps": [
                    {"index": 0, "role": "user", "agent": "", "tokens": {"total": 0}, "tool_call_count": 0},
                    {"index": 1, "role": "assistant", "agent": "", "tokens": {"total": 10}, "tool_call_count": 1},
                    {
                        "index": 2,
                        "role": "assistant",
                        "agent": "explore",
                        "tokens": {"total": 20},
                        "tool_call_count": 2,
                    },
                    {"index": 3, "role": "assistant", "agent": "explore", "tokens": {"total": 5}, "tool_call_count": 0},
                ],
            },
            {
                "run_id": "b",
                "label": "Run B",
                "steps": [
                    {"index": 0, "role": "assistant", "agent": "", "tokens": {"total": 8}, "tool_call_count": 1},
                    {"index": 1, "role": "assistant", "agent": "", "tokens": {"total": 8}, "tool_call_count": 0},
                ],
            },
        ]
        fig = build_run_group_agent_timeline(runs)
        # At least one bar segment per run; legend includes main
        self.assertGreaterEqual(len(fig.data), 2)
        y_vals = {t.y[0] for t in fig.data if t.y}
        self.assertEqual(y_vals, {"Run A", "Run B"})
        legend_names = {t.name for t in fig.data if t.showlegend}
        self.assertIn("main", legend_names)
        self.assertIn("explore", legend_names)

    def test_opencode_multi_session_splits_modes_and_explores(self):
        """OpenCode: is_sub_agent always false; split by session + agent mode."""
        runs = [
            {
                "run_id": "oc",
                "label": "OpenCode",
                "steps": [
                    {
                        "index": 0,
                        "role": "assistant",
                        "agent": "plan",
                        "session_id": "ses_root",
                        "is_sub_agent": False,
                        "tokens": {"total": 10},
                        "tool_call_count": 0,
                    },
                    {
                        "index": 1,
                        "role": "assistant",
                        "agent": "explore",
                        "session_id": "ses_ex1",
                        "is_sub_agent": False,
                        "tokens": {"total": 20},
                        "tool_call_count": 1,
                    },
                    {
                        "index": 2,
                        "role": "assistant",
                        "agent": "explore",
                        "session_id": "ses_ex2",
                        "is_sub_agent": False,
                        "tokens": {"total": 15},
                        "tool_call_count": 1,
                    },
                    {
                        "index": 3,
                        "role": "assistant",
                        "agent": "build",
                        "session_id": "ses_root",
                        "is_sub_agent": False,
                        "tokens": {"total": 30},
                        "tool_call_count": 2,
                    },
                ],
            }
        ]
        fig = build_run_group_agent_timeline(runs)
        legend_names = {t.name for t in fig.data if t.showlegend}
        self.assertIn("plan", legend_names)
        self.assertIn("build", legend_names)
        # Two explore sessions → disambiguated labels
        explore_labels = [n for n in legend_names if n.startswith("explore")]
        self.assertEqual(len(explore_labels), 2)
        self.assertEqual(len(fig.data), 4)

    def test_compaction_folds_into_explore_session(self):
        runs = [
            {
                "label": "Run",
                "steps": [
                    {
                        "index": 0,
                        "role": "assistant",
                        "agent": "explore",
                        "session_id": "ses_ex",
                        "is_sub_agent": False,
                        "tokens": {"total": 10},
                        "tool_call_count": 0,
                    },
                    {
                        "index": 1,
                        "role": "compaction",
                        "agent": "compaction",
                        "session_id": "ses_ex",
                        "is_sub_agent": False,
                        "tokens": {"total": 0},
                        "tool_call_count": 0,
                    },
                    {
                        "index": 2,
                        "role": "assistant",
                        "agent": "explore",
                        "session_id": "ses_ex",
                        "is_sub_agent": False,
                        "tokens": {"total": 5},
                        "tool_call_count": 0,
                    },
                    {
                        "index": 3,
                        "role": "assistant",
                        "agent": "build",
                        "session_id": "ses_root",
                        "is_sub_agent": False,
                        "tokens": {"total": 8},
                        "tool_call_count": 0,
                    },
                ],
            }
        ]
        fig = build_run_group_agent_timeline(runs)
        legend_names = {t.name for t in fig.data if t.showlegend}
        self.assertNotIn("compaction", legend_names)
        self.assertIn("explore", legend_names)
        self.assertIn("build", legend_names)
        # explore+compaction+explore should merge into one contiguous segment
        explore_segs = [t for t in fig.data if t.name == "explore"]
        self.assertEqual(len(explore_segs), 1)
        self.assertEqual(int(explore_segs[0].x[0]), 3)

    def test_empty_runs(self):
        fig = build_run_group_agent_timeline([])
        self.assertEqual(len(fig.data), 0)


def _opencode_step(
    index: int,
    *,
    agent: str,
    session_id: str,
    tool: str | None = None,
    tokens: int = 10,
) -> dict:
    step = {
        "index": index,
        "role": "assistant",
        "agent": agent,
        "session_id": session_id,
        "is_sub_agent": False,
        "duration": 1.0,
        "tokens": {
            "total": tokens,
            "input": tokens,
            "output": 0,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        },
        "text_preview": agent,
        "parts": [],
        "tool_calls": [],
        "tool_call_count": 0,
        "error_count": 0,
        "has_reasoning": False,
    }
    if tool:
        step["tool_calls"] = [{"tool_name": tool, "status": "completed"}]
        step["tool_call_count"] = 1
        step["parts"] = [{"type": "tool_call", "tool_name": tool}]
    return step


class TimelineIdentityTests(unittest.TestCase):
    def test_tool_chart_stacks_opencode_modes(self):
        steps = [
            _opencode_step(0, agent="plan", session_id="ses_root", tool="Read"),
            _opencode_step(1, agent="explore", session_id="ses_ex1", tool="Grep"),
        ]
        fig = build_tool_chart(steps)
        names = {t.name for t in fig.data}
        self.assertIn("plan", names)
        self.assertIn("explore", names)
        self.assertNotIn("main", names)
        plan = next(t for t in fig.data if t.name == "plan")
        explore = next(t for t in fig.data if t.name == "explore")
        idx = {name: i for i, name in enumerate(plan.y)}
        self.assertEqual(plan.x[idx["Read"]], 1)
        self.assertEqual(plan.x[idx["Grep"]], 0)
        self.assertEqual(explore.x[idx["Grep"]], 1)
        self.assertEqual(explore.x[idx["Read"]], 0)

    def test_tool_chart_splits_bash_into_shell_commands(self):
        def step(index: int, command: str) -> dict:
            s = _opencode_step(index, agent="build", session_id="ses_root")
            s["tool_calls"] = [{
                "tool_name": "Bash",
                "input": {"command": command},
                "status": "completed",
            }]
            s["tool_call_count"] = 1
            return s

        steps = [
            step(0, "git status"),
            step(1, "FOO=1 npm test"),
            step(2, "sudo pytest -q"),
            step(3, "git status"),
            step(4, "python3 tools/run_eval.py"),
            step(5, "python -m pytest"),
            _opencode_step(6, agent="build", session_id="ses_root", tool="Read"),
        ]
        fig = build_tool_chart(steps)
        self.assertEqual(len(fig.data), 1)
        labels = list(fig.data[0].y)
        counts = {label: fig.data[0].x[i] for i, label in enumerate(labels)}
        self.assertEqual(counts["git"], 2)
        self.assertEqual(counts["npm"], 1)
        self.assertEqual(counts["pytest"], 2)  # sudo pytest + python -m pytest
        self.assertEqual(counts["run_eval.py"], 1)
        self.assertEqual(counts["Read"], 1)
        self.assertNotIn("Bash", labels)
        self.assertNotIn("python3", labels)
        self.assertNotIn("python", labels)

    def test_workflow_badges_use_timeline_labels(self):
        steps = [
            _opencode_step(0, agent="plan", session_id="ses_root"),
            _opencode_step(1, agent="explore", session_id="ses_ex1"),
        ]
        html = render_workflow_html(steps)
        self.assertIn("plan", html)
        self.assertIn("explore", html)
        color_map, labels, agent_id_of = bind_timeline_agents(steps)
        self.assertGreater(len(color_map), 1)
        ids = {agent_id_of(s) for s in steps}
        self.assertEqual(len(ids), 2)
        self.assertTrue(all("::" in aid for aid in ids))
        self.assertIn("plan", labels.values())
        self.assertIn("explore", labels.values())

    def test_color_map_keys_match_tool_chart_lookup(self):
        steps = [
            _opencode_step(0, agent="plan", session_id="ses_root", tool="Read"),
            _opencode_step(1, agent="explore", session_id="ses_ex1", tool="Grep"),
        ]
        color_map, _labels, agent_id_of = bind_timeline_agents(steps)
        for step in steps:
            aid = agent_id_of(step)
            self.assertIn(aid, color_map)

    def test_dsh_tagged_subagents_label_main_and_sub(self):
        steps = [
            {
                "index": 0,
                "role": "assistant",
                "agent": "standard",
                "session_id": "session-parent",
                "is_sub_agent": False,
                "tokens": {"total": 1},
                "tool_call_count": 0,
            },
            {
                "index": 1,
                "role": "assistant",
                "agent": "standard",
                "session_id": "child-aaa-bbb",
                "is_sub_agent": True,
                "tokens": {"total": 1},
                "tool_call_count": 0,
            },
        ]
        _color_map, labels, _agent_id_of = bind_timeline_agents(steps)
        values = {label for aid, label in labels.items() if aid}
        self.assertIn("main", values)
        self.assertTrue(any(label.startswith("sub ") for label in values))
        self.assertFalse(any("standard" in label for label in values))


class SkillAgentChartTests(unittest.TestCase):
    def test_empty_when_no_skill_calls(self):
        steps = [
            _opencode_step(0, agent="plan", session_id="ses_root", tool="Read"),
        ]
        fig = build_skill_agent_chart(steps)
        self.assertEqual(len(fig.data), 0)

    def test_sankey_links_agents_to_skills(self):
        plan = _opencode_step(0, agent="plan", session_id="ses_root", tool="Skill")
        explore = _opencode_step(1, agent="explore", session_id="ses_ex1", tool="Skill")
        plan["tool_calls"][0]["input"] = {"skill": "create-hook"}
        explore["tool_calls"][0]["input"] = {"skill": "canvas"}
        explore["tool_calls"].append(
            {"tool_name": "Skill", "status": "completed", "input": {"skill": "create-hook"}}
        )
        fig = build_skill_agent_chart([plan, explore])
        self.assertEqual(len(fig.data), 1)
        sankey = fig.data[0]
        self.assertEqual(sankey.type, "sankey")
        labels = list(sankey.node.label)
        self.assertTrue(any(lab.startswith("plan") for lab in labels))
        self.assertTrue(any(lab.startswith("explore") for lab in labels))
        self.assertTrue(any("create-hook" in lab for lab in labels))
        self.assertTrue(any("canvas" in lab for lab in labels))
        self.assertEqual(len(sankey.link.value), 3)
        self.assertEqual(sorted(int(v) for v in sankey.link.value), [1, 1, 1])

        node_of = {lab.split(" (")[0]: i for i, lab in enumerate(labels)}
        pairs = {
            (labels[int(s)].split(" (")[0], labels[int(t)].split(" (")[0])
            for s, t in zip(sankey.link.source, sankey.link.target, strict=True)
        }
        self.assertIn(("plan", "create-hook"), pairs)
        self.assertIn(("explore", "canvas"), pairs)
        self.assertIn(("explore", "create-hook"), pairs)
        self.assertEqual(node_of["plan"], 0)
        self.assertGreater(node_of["create-hook"], node_of["plan"])

    def test_aggregates_repeat_calls(self):
        steps = [
            _opencode_step(0, agent="plan", session_id="ses_root", tool="Skill"),
            _opencode_step(1, agent="plan", session_id="ses_root", tool="Skill"),
        ]
        for step in steps:
            step["tool_calls"][0]["input"] = {"skill": "create-rule"}
        fig = build_skill_agent_chart(steps)
        sankey = fig.data[0]
        self.assertEqual([int(v) for v in sankey.link.value], [2])
        labels = list(sankey.node.label)
        self.assertTrue(any(lab.startswith("create-rule (2)") for lab in labels))
        self.assertTrue(any(lab.endswith("(2)") and "create-rule" not in lab for lab in labels))


class ScorecardDisplayTests(unittest.TestCase):
    def test_peak_zero_is_numeric_not_dash(self):
        html = build_run_group_scorecard_html(
            {
                "rows": [
                    {
                        "run_id": "a",
                        "label": "A",
                        "error": None,
                        "finished": True,
                        "steps": 1,
                        "wall_clock_s": 1,
                        "wall_clock_fmt": "1s",
                        "tokens": 1,
                        "tool_calls": 1,
                        "tool_success_pct": 100,
                        "peak_occupancy": 0,
                        "peak_pct": None,
                        "compactions": 1,
                        "format": "test",
                    },
                    {
                        "run_id": "b",
                        "label": "B",
                        "error": None,
                        "finished": True,
                        "steps": 2,
                        "wall_clock_s": 2,
                        "wall_clock_fmt": "2s",
                        "tokens": 2,
                        "tool_calls": 2,
                        "tool_success_pct": 100,
                        "peak_occupancy": 100,
                        "peak_pct": 50,
                        "compactions": 2,
                        "format": "test",
                    },
                ],
                "behavior": None,
                "ok": True,
                "error": None,
            }
        )
        self.assertIn("100 (50%)", html)
        self.assertNotIn("—", html)

    def test_empty_action_lists_are_identical(self):
        behavior = build_behavioral_comparison(
            [
                {"run_id": "a", "label": "A", "actions": []},
                {"run_id": "b", "label": "B", "actions": []},
            ]
        )
        self.assertEqual(behavior["similarity"]["a"]["b"], 1.0)
        self.assertEqual(behavior["similarity"]["a"]["a"], 1.0)
        html = build_run_group_scorecard_html(
            {
                "rows": [
                    {
                        "run_id": "a",
                        "label": "A",
                        "error": None,
                        "finished": True,
                        "steps": 1,
                        "wall_clock_s": 1,
                        "wall_clock_fmt": "1s",
                        "tokens": 1,
                        "tool_calls": 0,
                        "tool_success_pct": 100,
                        "peak_occupancy": 1,
                        "peak_pct": None,
                        "compactions": 0,
                        "format": "test",
                    },
                    {
                        "run_id": "b",
                        "label": "B",
                        "error": None,
                        "finished": True,
                        "steps": 1,
                        "wall_clock_s": 1,
                        "wall_clock_fmt": "1s",
                        "tokens": 1,
                        "tool_calls": 0,
                        "tool_success_pct": 100,
                        "peak_occupancy": 1,
                        "peak_pct": None,
                        "compactions": 0,
                        "format": "test",
                    },
                ],
                "behavior": behavior,
                "ok": True,
                "error": None,
            }
        )
        self.assertIn("No non-file actions", html)

    def test_namespaced_tool_keeps_prefix_when_truncated(self):
        long_tool = "org.example/" + ("plugin/" * 12) + "list_issues"
        self.assertGreater(len(long_tool), 64)
        behavior = build_behavioral_comparison(
            [
                {
                    "run_id": "a",
                    "label": "A",
                    "actions": [_action("SEARCH", "x")],
                    "steps": _steps_with_tools([(long_tool, {})]),
                },
                {
                    "run_id": "b",
                    "label": "B",
                    "actions": [_action("SEARCH", "x")],
                    "steps": _steps_with_tools([(long_tool, {})]),
                },
            ]
        )
        tools = {r["key"]: r for r in behavior["tool_matrix"]}
        self.assertIn(long_tool, tools)
        short = tools[long_tool]["short"]
        self.assertTrue(short.endswith("list_issues"))
        self.assertIn("plugin/", short)
        self.assertNotEqual(short, "list_issues")


class CoverageMatrixExpandTests(unittest.TestCase):
    def test_keeps_all_actions_and_files_with_show_all_toggle(self):
        from trajviz.insight.run_group import _ACTION_MATRIX_PREVIEW

        n_files = 62
        n_actions = _ACTION_MATRIX_PREVIEW + 7
        runs = [
            {
                "run_id": "a",
                "label": "A",
                "actions": (
                    [_action("FILE_READ", f"src/f{i}.py", i) for i in range(n_files)]
                    + [_action("SEARCH", f"query-{i}", n_files + i) for i in range(n_actions)]
                ),
            },
            {
                "run_id": "b",
                "label": "B",
                "actions": (
                    [_action("FILE_READ", f"src/f{i}.py", i) for i in range(n_files)]
                    + [_action("SEARCH", f"query-{i}", n_files + i) for i in range(n_actions)]
                ),
            },
        ]
        behavior = build_behavioral_comparison(runs)
        self.assertEqual(len(behavior["file_matrix"]), n_files)
        self.assertEqual(behavior["file_matrix_total"], n_files)
        self.assertEqual(len(behavior["action_matrix"]), n_actions)
        self.assertEqual(behavior["action_matrix_total"], n_actions)
        html = build_run_group_behavior_html({"behavior": behavior, "ok": True})
        self.assertIn(f"Show all {n_actions} actions", html)
        self.assertIn("src/", html)
        self.assertIn(f"{n_files} files", html)
        self.assertIn(f"f{n_files - 1}.py", html)
        self.assertIn(f"query-{n_actions - 1}", html)
        self.assertIn("rg-tree-dir", html)
        self.assertNotIn("rg-expand-files", html)
        self.assertIn("rg-expand-actions", html)
        self.assertEqual(html.count("rg-matrix-tail"), n_actions - _ACTION_MATRIX_PREVIEW)
        self.assertNotIn("Showing ", html)

    def test_small_matrices_have_no_toggle(self):
        behavior = build_behavioral_comparison(
            [
                {
                    "run_id": "a",
                    "label": "A",
                    "actions": [_action("FILE_READ", "a.py", 1), _action("SEARCH", "todo", 2)],
                },
                {
                    "run_id": "b",
                    "label": "B",
                    "actions": [_action("FILE_READ", "a.py", 1)],
                },
            ]
        )
        html = build_run_group_behavior_html({"behavior": behavior, "ok": True})
        self.assertNotIn("rg-matrix-toggle", html)
        self.assertNotIn("Show all", html)
        self.assertIn("a.py", html)
        self.assertIn("todo", html)


class FileCoverageFolderTests(unittest.TestCase):
    def test_groups_shared_folder_and_keeps_root_files_flat(self):
        from trajviz.insight.run_group import _render_file_matrix_html

        runs = [
            {
                "run_id": "a",
                "label": "Claude",
                "actions": [
                    _action("FILE_READ", "src/a.py", 1),
                    _action("FILE_READ", "src/b.py", 2),
                    _action("FILE_READ", "src/nested/c.py", 3),
                    _action("FILE_READ", "README.md", 4),
                    _action("FILE_WRITE", "src/a.py", 5),
                ],
            },
            {
                "run_id": "b",
                "label": "Codex",
                "actions": [
                    _action("FILE_READ", "src/a.py", 1),
                    _action("FILE_READ", "other.py", 2),
                ],
            },
        ]
        behavior = build_behavioral_comparison(runs)
        html = _render_file_matrix_html(behavior)
        self.assertIn("rg-file-summary", html)
        self.assertIn("src/", html)
        self.assertIn("files", html)
        self.assertIn("rg-mix-bar", html)
        self.assertIn("README.md", html)
        self.assertIn("other.py", html)
        self.assertIn("only in Codex", html)
        self.assertIn("Read in both", html)
        self.assertIn("src/a.py", html)
        self.assertIn("rg-tree-both", html)
        self.assertIn("rg-badge-shared", html)
        self.assertIn("rg-tree-unique", html)
        self.assertIn("rg-run-0", html)
        self.assertIn("rg-run-1", html)
        self.assertIn("Claude only", html)
        self.assertIn("Codex only", html)
        self.assertIn("rg-run-swatch", html)

    def test_single_file_folder_stays_a_row(self):
        from trajviz.insight.run_group import _render_file_matrix_html

        behavior = build_behavioral_comparison(
            [
                {
                    "run_id": "a",
                    "label": "A",
                    "actions": [_action("FILE_READ", "src/main.py", 1)],
                },
                {
                    "run_id": "b",
                    "label": "B",
                    "actions": [_action("FILE_READ", "src/main.py", 1)],
                },
            ]
        )
        html = _render_file_matrix_html(behavior)
        self.assertIn("src/main.py", html)
        self.assertIn("rg-tree-both", html)
        self.assertNotIn("rg-tree-dir", html)

    def test_home_absolutes_keep_project_under_work(self):
        from trajviz.insight.run_group import _file_tree_from_rows

        root = "/home/kevin/Work/TrajectoryVisualizer"
        runs = [
            {
                "run_id": "a",
                "label": "A",
                "actions": [
                    _action("FILE_READ", f"{root}/src/a.py", 1),
                    _action("FILE_READ", f"{root}/src/b.py", 2),
                    _action("FILE_READ", f"{root}/tests/test_a.py", 3),
                    _action("FILE_READ", f"{root}/README.md", 4),
                    _action("FILE_READ", "/home/kevin/.cursor/projects/foo/bar.py", 5),
                ],
            },
            {
                "run_id": "b",
                "label": "B",
                "actions": [
                    _action("FILE_READ", f"{root}/src/a.py", 1),
                    _action("FILE_READ", f"{root}/tests/test_b.py", 2),
                    _action("FILE_READ", "src/rel.py", 3),
                ],
            },
        ]
        behavior = build_behavioral_comparison(runs)
        tree = _file_tree_from_rows(behavior["file_matrix"])
        self.assertIn("Work", tree["dirs"])
        work = tree["dirs"]["Work"]
        self.assertIn("TrajectoryVisualizer", work["dirs"])
        proj = work["dirs"]["TrajectoryVisualizer"]
        self.assertIn("src", proj["dirs"])
        self.assertIn("tests", proj["dirs"])
        self.assertNotIn("src", tree["dirs"])
        self.assertNotIn("home", tree["dirs"])
        self.assertIn(".cursor", tree["dirs"])
        src_names = {row.get("name") for row in proj["dirs"]["src"]["files"]}
        self.assertIn("rel.py", src_names)

    def test_wsl_unc_merges_with_posix_and_lists_both_reads(self):
        from trajviz.converge.canonical import _normalize_target
        from trajviz.insight.run_group import _render_file_matrix_html

        unc = r"\\wsl.localhost\Ubuntu-26.04\home\kevin\Work\telephony_call_manager\src\app.py"
        posix = "/home/kevin/Work/telephony_call_manager/src/app.py"
        self.assertEqual(_normalize_target(unc), posix)
        runs = [
            {
                "run_id": "a",
                "label": "Claude",
                "actions": [
                    _action("FILE_READ", unc, 1),
                    _action("FILE_READ", r"\\wsl.localhost\Ubuntu-26.04\home\kevin\Work\telephony_call_manager\src\only_a.py", 2),
                ],
            },
            {
                "run_id": "b",
                "label": "Codex",
                "actions": [
                    _action("FILE_READ", posix, 1),
                    _action("FILE_READ", "/home/kevin/Work/telephony_call_manager/tests/test_app.py", 2),
                ],
            },
        ]
        behavior = build_behavioral_comparison(runs)
        paths = {r["path"] for r in behavior["file_matrix"]}
        self.assertIn(posix, paths)
        self.assertEqual(len([p for p in paths if p.endswith("/app.py")]), 1)
        app = next(r for r in behavior["file_matrix"] if r["path"] == posix)
        self.assertEqual(app["cells"]["a"]["read"], 1)
        self.assertEqual(app["cells"]["b"]["read"], 1)
        html = _render_file_matrix_html(behavior)
        self.assertIn("telephony_call_manager", html)
        self.assertIn("Read in both", html)
        self.assertIn("Work/telephony_call_manager/src/app.py", html)
        self.assertNotIn("wsl.localhost", html)
        self.assertIn("Work/", html)
        self.assertIn("src/", html)
        self.assertIn("tests/", html)

    def test_project_stays_nested_under_work(self):
        from trajviz.insight.run_group import _file_tree_from_rows

        root = "/home/kevin/Work/telephony_call_manager"
        runs = [
            {
                "run_id": "a",
                "label": "A",
                "actions": [
                    _action("FILE_READ", f"{root}/src/a.py", 1),
                    _action("FILE_READ", f"{root}/src/b.py", 2),
                    _action("FILE_READ", f"{root}/tests/test_a.py", 3),
                    _action("FILE_READ", "/home/kevin/Work/notes.md", 4),
                ],
            },
            {
                "run_id": "b",
                "label": "B",
                "actions": [
                    _action("FILE_READ", f"{root}/src/a.py", 1),
                ],
            },
        ]
        behavior = build_behavioral_comparison(runs)
        tree = _file_tree_from_rows(behavior["file_matrix"])
        self.assertIn("Work", tree["dirs"])
        work = tree["dirs"]["Work"]
        self.assertIn("telephony_call_manager", work["dirs"])
        self.assertIn("src", work["dirs"]["telephony_call_manager"]["dirs"])
        self.assertNotIn("src", tree["dirs"])
        names = {row.get("name") for row in work["files"]}
        self.assertIn("notes.md", names)


if __name__ == "__main__":
    unittest.main()
