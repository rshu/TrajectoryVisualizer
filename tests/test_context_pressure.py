"""Context-window pressure: occupancy, compaction detection, and chart builders."""

import unittest

from trajviz.insight.charts import build_context_pressure_chart
from trajviz.insight.context_usage import (
    DEFAULT_CONTEXT_WINDOW_LIMIT,
    PRESSURE_ALL_AGENTS,
    PRESSURE_MAIN_AGENT,
    coerce_window_limit,
    context_pressure_series,
    detect_compaction_events,
    detect_premature_compactions,
    infer_context_window_limit,
    pressure_agent_choices,
    resolve_context_window_limit,
    step_context_occupancy,
)
from trajviz.insight.parser import parse_steps


def _tokens(*, total, inp, out=0, reasoning=0, cache_read=0):
    return {
        "total": total,
        "input": inp,
        "output": out,
        "reasoning": reasoning,
        "cache_read": cache_read,
    }


def _step(
    idx,
    *,
    role="assistant",
    agent="",
    is_sub_agent=False,
    session_id="",
    tokens=None,
    parts=None,
    tool_calls=None,
    summary=False,
    is_compaction_checkpoint=False,
    message_type="",
    model_id="",
    session_title="",
):
    return {
        "index": idx,
        "role": role,
        "agent": agent,
        "is_sub_agent": is_sub_agent,
        "session_id": session_id,
        "tokens": tokens or _tokens(total=0, inp=0),
        "parts": parts or [],
        "tool_calls": tool_calls or [],
        "summary": summary,
        "is_compaction_checkpoint": is_compaction_checkpoint,
        "message_type": message_type,
        "model_id": model_id,
        "session_title": session_title,
    }


class OccupancySchemaTests(unittest.TestCase):
    def test_opencode_input_plus_cache(self):
        # total = input + output + reasoning + cache_read (input is already fresh)
        step = _step(1, tokens=_tokens(total=100, inp=50, out=20, reasoning=10, cache_read=20))
        occ = step_context_occupancy(step)
        self.assertEqual(occ["fresh"], 50)
        self.assertEqual(occ["cache_read"], 20)
        self.assertEqual(occ["occupancy"], 70)

    def test_claude_cache_exclusive_input(self):
        # total = input + output + reasoning (input includes cache)
        step = _step(1, tokens=_tokens(total=100, inp=70, out=20, reasoning=10, cache_read=20))
        occ = step_context_occupancy(step)
        self.assertEqual(occ["fresh"], 50)
        self.assertEqual(occ["cache_read"], 20)
        self.assertEqual(occ["occupancy"], 70)


class AgentGroupingTests(unittest.TestCase):
    def test_main_vs_opencode_subagent_vs_codearts_session(self):
        steps = [
            _step(0, role="user", is_sub_agent=False),
            _step(1, agent="build", is_sub_agent=False,
                  tokens=_tokens(total=80, inp=60, out=20, cache_read=0)),
            _step(2, agent="explore (subagent)", is_sub_agent=True, session_id="ses_child",
                  tokens=_tokens(total=40, inp=30, out=10, cache_read=0)),
            _step(3, agent="Agent", is_sub_agent=True, session_id="ses_ca",
                  tokens=_tokens(total=50, inp=40, out=10, cache_read=0)),
        ]
        series = context_pressure_series(steps)
        ids = [a["agent_id"] for a in series["agents"]]
        self.assertEqual(ids, ["", "ses_child", "ses_ca"])

        main = context_pressure_series(steps, agent_key=PRESSURE_MAIN_AGENT)
        self.assertEqual(len(main["agents"]), 1)
        self.assertEqual(main["agents"][0]["agent_id"], "")
        self.assertEqual([p["occupancy"] for p in main["agents"][0]["points"]], [60])

        child = context_pressure_series(steps, agent_key="ses_child")
        self.assertEqual(len(child["agents"]), 1)
        self.assertEqual(child["agents"][0]["points"][0]["occupancy"], 30)

    def test_dropdown_choices_include_all_and_main(self):
        steps = [
            _step(0, is_sub_agent=False, tokens=_tokens(total=10, inp=10)),
            _step(1, agent="explore (subagent)", is_sub_agent=True, session_id="ses_x",
                  tokens=_tokens(total=10, inp=10)),
        ]
        choices = pressure_agent_choices(steps)
        values = [value for _, value in choices]
        self.assertEqual(values[0], PRESSURE_ALL_AGENTS)
        self.assertIn(PRESSURE_MAIN_AGENT, values)
        self.assertIn("ses_x", values)
        labels = [label for label, _ in choices]
        self.assertIn("main", labels)
        self.assertTrue(any("explore" in lab or lab.startswith("sub ") for lab in labels))

    def test_dsh_shared_preset_labels_main_and_sub(self):
        steps = [
            _step(0, agent="standard", session_id="session-parent",
                  is_sub_agent=False, tokens=_tokens(total=10, inp=10)),
            _step(1, agent="standard", session_id="child-aaa-bbb",
                  is_sub_agent=True, tokens=_tokens(total=10, inp=10)),
        ]
        series = context_pressure_series(steps)
        labels = [a["label"] for a in series["agents"]]
        self.assertIn("main", labels)
        self.assertTrue(any(lab.startswith("sub ") for lab in labels))
        self.assertFalse(any("standard" in lab for lab in labels))


class CompactionDetectionTests(unittest.TestCase):
    def test_compaction_part(self):
        steps = [
            _step(0, role="user", parts=[{"type": "compaction", "summary": "prior work"}]),
            _step(1, tokens=_tokens(total=20, inp=20)),
        ]
        events = detect_compaction_events(steps)
        kinds = [e["kind"] for e in events]
        self.assertIn("compaction_part", kinds)
        part = next(e for e in events if e["kind"] == "compaction_part")
        self.assertEqual(part["occupancy_after"], 20)

    def test_compaction_part_after_is_next_occupancy_not_zero(self):
        steps = [
            _step(0, session_id="s", tokens=_tokens(total=100, inp=100)),
            _step(1, role="user", session_id="s",
                  parts=[{"type": "compaction", "summary": "prior work"}]),
            _step(2, session_id="s", tokens=_tokens(total=20, inp=20)),
        ]
        part = next(
            e for e in detect_compaction_events(steps) if e["kind"] == "compaction_part"
        )
        self.assertEqual(part["occupancy_before"], 100)
        self.assertEqual(part["occupancy_after"], 20)
        self.assertEqual(part["dropped"], 80)

    def test_summary_flag(self):
        steps = [
            _step(0, tokens=_tokens(total=100, inp=100)),
            _step(1, summary=True, tokens=_tokens(total=30, inp=30)),
        ]
        events = detect_compaction_events(steps)
        self.assertTrue(any(e["kind"] == "summary" for e in events))
        # Explicit summary suppresses a duplicate occupancy_drop on the same step
        self.assertFalse(any(e["kind"] == "occupancy_drop" and e["step"] == 1 for e in events))

    def test_v2_compaction_checkpoint(self):
        steps = [
            _step(0, tokens=_tokens(total=80, inp=80)),
            _step(1, role="compaction", is_compaction_checkpoint=True, message_type="compaction"),
            _step(2, tokens=_tokens(total=25, inp=25)),
        ]
        events = detect_compaction_events(steps)
        self.assertTrue(any(e["kind"] == "compaction_message" for e in events))
        series = context_pressure_series(steps)
        # Checkpoint is not an occupancy point; splice a vertical cliff there.
        occ = [(p["step"], p["occupancy"]) for p in series["agents"][0]["points"]]
        self.assertIn((0, 80), occ)
        self.assertIn((1, 80), occ)
        self.assertIn((1, 25), occ)
        self.assertIn((2, 25), occ)

    def test_tool_time_compacted(self):
        steps = [
            _step(0, tokens=_tokens(total=50, inp=50), tool_calls=[
                {"tool_name": "read", "time_compacted": 1_700_000_000_000},
            ]),
        ]
        kinds = [e["kind"] for e in detect_compaction_events(steps)]
        self.assertEqual(kinds, ["tool_prune"])

    def test_prune_uses_compacted_time_not_the_original_tool_step(self):
        """OpenCode stamps time.compacted when outputs are pruned, later."""
        steps = [
            _step(0, session_id="s", tokens=_tokens(total=8_000, inp=8_000),
                  tool_calls=[{"tool_name": "read", "time_compacted": 5_000}]),
            _step(1, session_id="s", tokens=_tokens(total=8_500, inp=8_500)),
            _step(2, session_id="s", tokens=_tokens(total=3_000, inp=3_000)),
        ]
        steps[0]["time_created_ms"] = 1_000
        steps[1]["time_created_ms"] = 2_000
        steps[2]["time_created_ms"] = 6_000
        prunes = [e for e in detect_compaction_events(steps) if e["kind"] == "tool_prune"]
        self.assertEqual(len(prunes), 1)
        self.assertEqual(prunes[0]["step"], 2)
        self.assertEqual(prunes[0]["occupancy_after"], 3_000)

    def test_one_prune_wave_from_many_compacted_tools(self):
        steps = [
            _step(0, session_id="s", tokens=_tokens(total=8_000, inp=8_000),
                  tool_calls=[
                      {"tool_name": "read", "time_compacted": 5_000},
                      {"tool_name": "bash", "time_compacted": 5_010},
                  ]),
            _step(1, session_id="s", tokens=_tokens(total=9_000, inp=9_000),
                  tool_calls=[{"tool_name": "grep", "time_compacted": 5_020}]),
            _step(2, session_id="s", tokens=_tokens(total=3_000, inp=3_000)),
        ]
        steps[0]["time_created_ms"] = 1_000
        steps[1]["time_created_ms"] = 2_000
        steps[2]["time_created_ms"] = 6_000
        prunes = [e for e in detect_compaction_events(steps) if e["kind"] == "tool_prune"]
        self.assertEqual(len(prunes), 1)
        self.assertEqual(prunes[0]["step"], 2)

    def test_prune_after_last_step_with_no_drop_is_filtered(self):
        """A prune whose timestamp falls after the last logged step produces
        dropped=0 (we cannot observe its effect).  This is noise, not a
        compaction event."""
        steps = [
            _step(0, session_id="s", tokens=_tokens(total=8_000, inp=8_000),
                  tool_calls=[{"tool_name": "read", "time_compacted": 9_000}]),
            _step(1, session_id="s", tokens=_tokens(total=8_500, inp=8_500)),
            _step(2, session_id="s", tokens=_tokens(total=8_700, inp=8_700)),
        ]
        steps[0]["time_created_ms"] = 1_000
        steps[1]["time_created_ms"] = 2_000
        steps[2]["time_created_ms"] = 3_000
        prunes = [e for e in detect_compaction_events(steps) if e["kind"] == "tool_prune"]
        self.assertEqual(len(prunes), 0)
        sid = "ses"
        steps = [
            _step(0, session_id=sid, tokens=_tokens(total=10_000, inp=10_000)),
            _step(1, role="user", session_id=sid,
                  parts=[{"type": "compaction", "summary": "prior"}]),
            _step(2, session_id=sid, agent="compaction", summary=True,
                  tokens=_tokens(total=2_000, inp=2_000)),
            _step(3, session_id=sid, tokens=_tokens(total=1_300, inp=1_300)),
            _step(4, session_id=sid, tokens=_tokens(total=1_500, inp=1_500)),
        ]
        kinds = [e["kind"] for e in detect_compaction_events(steps)]
        self.assertNotIn("occupancy_drop", kinds)
        self.assertIn("compaction_part", kinds)

    def test_occupancy_drop_same_agent(self):
        steps = [
            _step(0, tokens=_tokens(total=10_000, inp=10_000, cache_read=8_000)),
            _step(1, tokens=_tokens(total=2_000, inp=2_000, cache_read=1_000)),
        ]
        events = detect_compaction_events(steps)
        drops = [e for e in events if e["kind"] == "occupancy_drop"]
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0]["step"], 1)
        self.assertEqual(drops[0]["occupancy_before"], 10_000)
        self.assertEqual(drops[0]["occupancy_after"], 2_000)
        self.assertEqual(drops[0]["dropped"], 8_000)

    def test_one_step_occupancy_dip_that_recovers_is_not_compaction(self):
        steps = [
            _step(0, tokens=_tokens(total=10_000, inp=10_000, cache_read=8_000)),
            _step(1, tokens=_tokens(total=6_000, inp=6_000, cache_read=4_000)),
            _step(2, tokens=_tokens(total=9_500, inp=9_500, cache_read=7_500)),
        ]
        events = detect_compaction_events(steps)
        self.assertFalse(any(e["kind"] == "occupancy_drop" for e in events))

    def test_no_occupancy_drop_without_cache_read(self):
        """OpenCode without cache reports per-turn fresh input, not cumulative
        context.  Input naturally swings between turns (large tool output then
        a short reply), so occupancy_drop must not fire."""
        steps = [
            _step(0, tokens=_tokens(total=9949, inp=9870)),
            _step(1, tokens=_tokens(total=256, inp=229)),
            _step(2, tokens=_tokens(total=3767, inp=3796)),
            _step(3, tokens=_tokens(total=288, inp=183)),
            _step(4, tokens=_tokens(total=12944, inp=13001)),
            _step(5, tokens=_tokens(total=241, inp=262)),
        ]
        events = detect_compaction_events(steps)
        self.assertFalse(any(e["kind"] == "occupancy_drop" for e in events))

    def test_context_window_snapshot_drop_is_compaction_not_estimated_swings(self):
        """Cursor copies context_window onto assistants; estimated input still
        swings on earlier steps and must not spawn fake compaction markers.

        A ~30% snapshot drop (268k → 189k) is below the billed 0.7 ratio but
        is still summarization.
        """
        def _snap(idx, window, *, role="assistant", inp=10):
            tokens = _tokens(total=inp, inp=inp)
            if window:
                tokens["context_window"] = window
            return _step(idx, role=role, session_id="s", tokens=tokens)

        steps = [
            _snap(0, None, inp=5000),
            _snap(1, None, inp=10),
            _snap(2, 268_224),
            _snap(3, 268_224),
            _snap(4, 188_604, role="user", inp=40),
            _snap(5, 188_604),
            _snap(6, 188_604),
        ]
        events = detect_compaction_events(steps)
        drops = [e for e in events if e["kind"] == "occupancy_drop"]
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0]["step"], 4)
        self.assertEqual(drops[0]["occupancy_before"], 268_224)
        self.assertEqual(drops[0]["occupancy_after"], 188_604)
        self.assertEqual(drops[0]["dropped"], 268_224 - 188_604)

    def test_no_false_drop_when_agents_interleave(self):
        steps = [
            _step(0, is_sub_agent=False, tokens=_tokens(total=10_000, inp=10_000)),
            _step(1, agent="explore (subagent)", is_sub_agent=True, session_id="ses_child",
                  tokens=_tokens(total=1_000, inp=1_000)),
            _step(2, is_sub_agent=False, tokens=_tokens(total=11_000, inp=11_000)),
            _step(3, agent="explore (subagent)", is_sub_agent=True, session_id="ses_child",
                  tokens=_tokens(total=1_200, inp=1_200)),
        ]
        events = detect_compaction_events(steps)
        self.assertFalse(any(e["kind"] == "occupancy_drop" for e in events))

    def test_interleaved_opencode_sessions_without_issubagent_are_separate_windows(self):
        """Consolidated OpenCode logs often omit isSubAgent; session_id is the window."""
        steps = [
            _step(0, agent="plan", session_id="ses_plan",
                  tokens=_tokens(total=10_000, inp=10_000)),
            _step(1, agent="explore", session_id="ses_a",
                  tokens=_tokens(total=1_000, inp=1_000)),
            _step(2, agent="explore", session_id="ses_b",
                  tokens=_tokens(total=2_000, inp=2_000)),
            _step(3, agent="plan", session_id="ses_plan",
                  tokens=_tokens(total=11_000, inp=11_000)),
            _step(4, agent="explore", session_id="ses_a",
                  tokens=_tokens(total=1_200, inp=1_200)),
        ]
        events = detect_compaction_events(steps)
        self.assertFalse(any(e["kind"] == "occupancy_drop" for e in events))
        series = context_pressure_series(steps)
        ids = [a["agent_id"] for a in series["agents"]]
        self.assertEqual(ids, ["ses_plan", "ses_a", "ses_b"])
        labels = [a["label"] for a in series["agents"]]
        self.assertEqual(labels[0], "plan")
        self.assertTrue(any("explore" in lab for lab in labels))

    def test_compress_step_name(self):
        steps = [
            _step(0, parts=[
                {"type": "step_start", "name": "Compact conversation"},
            ], tokens=_tokens(total=40, inp=40)),
        ]
        kinds = [e["kind"] for e in detect_compaction_events(steps)]
        self.assertIn("compress_step", kinds)


class WindowLimitTests(unittest.TestCase):
    def test_codearts_metadata_wins(self) -> None:
        steps = [_step(0, model_id="claude-sonnet-4", tokens=_tokens(total=10, inp=10))]
        limit = infer_context_window_limit(steps, raw={"metadata": {"context_tokens": 32_000}})
        self.assertEqual(limit, 32_000)

    def test_cursor_context_snapshot_max_tokens(self) -> None:
        steps = [_step(0, tokens=_tokens(total=10, inp=10))]
        limit = infer_context_window_limit(
            steps,
            raw={"metadata": {"context_snapshot": {"max_tokens": 300_000}}},
        )
        self.assertEqual(limit, 300_000)

    def test_claude_prefix_table(self):
        steps = [_step(0, model_id="claude-sonnet-4-5", tokens=_tokens(total=10, inp=10))]
        self.assertEqual(infer_context_window_limit(steps), 200_000)

    def test_unknown_model_falls_back_to_default_window(self):
        steps = [_step(0, model_id="mystery-model", tokens=_tokens(total=10, inp=10))]
        self.assertIsNone(infer_context_window_limit(steps))
        self.assertEqual(resolve_context_window_limit(steps), DEFAULT_CONTEXT_WINDOW_LIMIT)
        series = context_pressure_series(steps)
        self.assertEqual(series["window_limit"], DEFAULT_CONTEXT_WINDOW_LIMIT)

    def test_override_beats_inferred_and_default(self):
        steps = [_step(0, model_id="claude-sonnet-4", tokens=_tokens(total=10, inp=10))]
        self.assertEqual(resolve_context_window_limit(steps, override=256_000), 256_000)
        self.assertEqual(coerce_window_limit("256k"), 256_000)
        series = context_pressure_series(steps, window_limit=100_000)
        self.assertEqual(series["window_limit"], 100_000)


class ParserRoundTripTests(unittest.TestCase):
    def test_compaction_part_and_time_compacted_survive_parse_steps(self):
        raw = {
            "messages": [
                {
                    "info": {"role": "user", "id": "u1", "sessionID": "ses"},
                    "parts": [
                        {"type": "compaction", "summary": "earlier work", "reason": "auto"},
                    ],
                },
                {
                    "info": {
                        "role": "assistant",
                        "id": "a1",
                        "sessionID": "ses",
                        "summary": True,
                        "tokens": {
                            "total": 40, "input": 30, "output": 10,
                            "reasoning": 0, "cache": {"read": 0, "write": 0},
                        },
                    },
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "read",
                            "callID": "c1",
                            "state": {
                                "status": "completed",
                                "input": {},
                                "output": "ok",
                                "time": {"start": 1, "end": 2, "compacted": 99},
                            },
                        },
                    ],
                },
                {
                    "type": "compaction",
                    "info": {"id": "k1", "sessionID": "ses", "reason": "overflow"},
                    "parts": [],
                },
            ],
        }
        steps = parse_steps(raw)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["parts"][0]["type"], "compaction")
        self.assertEqual(steps[0]["parts"][0]["summary"], "earlier work")
        self.assertTrue(steps[1]["summary"])
        self.assertEqual(steps[1]["tool_calls"][0]["time_compacted"], 99)
        self.assertTrue(steps[2]["is_compaction_checkpoint"])
        self.assertEqual(steps[2]["role"], "compaction")
        kinds = {e["kind"] for e in detect_compaction_events(steps)}
        self.assertIn("compaction_part", kinds)
        self.assertIn("tool_prune", kinds)
        self.assertIn("compaction_message", kinds)


class ChartBuilderTests(unittest.TestCase):
    def test_empty_steps(self):
        fig = build_context_pressure_chart([])
        self.assertTrue(fig.data == () or len(fig.data) == 0)
        self.assertTrue(any("No context occupancy" in (a.text or "") for a in fig.layout.annotations))

    def test_all_vs_single_agent(self):
        steps = [
            _step(0, is_sub_agent=False, tokens=_tokens(total=100, inp=100)),
            _step(1, agent="explore (subagent)", is_sub_agent=True, session_id="ses_child",
                  tokens=_tokens(total=40, inp=40)),
            _step(2, is_sub_agent=False, tokens=_tokens(total=20, inp=20)),
        ]
        overlay = build_context_pressure_chart(steps, agent_key=PRESSURE_ALL_AGENTS)
        names = [t.name for t in overlay.data]
        self.assertIn("main", names)
        self.assertTrue(any(n != "Compaction" and n != "main" for n in names))

        single = build_context_pressure_chart(steps, agent_key=PRESSURE_MAIN_AGENT)
        single_names = [t.name for t in single.data]
        self.assertIn("Occupancy", single_names)
        self.assertIn("Fresh input", single_names)
        self.assertIn("Cache read", single_names)

    def test_single_agent_occupancy_cliffs_at_compaction(self):
        steps = [
            _step(0, agent="explore", session_id="ses_a",
                  tokens=_tokens(total=10_000, inp=10_000)),
            _step(1, role="user", agent="explore", session_id="ses_a",
                  parts=[{"type": "compaction", "summary": "prior work"}]),
            _step(2, agent="explore", session_id="ses_a", summary=True,
                  tokens=_tokens(total=2_000, inp=2_000)),
        ]
        series = context_pressure_series(steps, agent_key="ses_a")
        ys = [p["occupancy"] for p in series["agents"][0]["points"]]
        self.assertTrue(any(a == 10_000 and b == 2_000 for a, b in zip(ys, ys[1:], strict=False)))

        fig = build_context_pressure_chart(steps, agent_key="ses_a")
        occ = next(t for t in fig.data if t.name == "Occupancy")
        occ_ys = list(occ.y)
        self.assertTrue(
            any(a == 10_000 and b == 2_000 for a, b in zip(occ_ys, occ_ys[1:], strict=False))
        )
        self.assertNotEqual(occ.line.color, "#3b82f6")

    def test_overlay_uses_distinct_colors_and_per_agent_compaction(self):
        steps = [
            _step(0, agent="plan", session_id="ses_plan",
                  tokens=_tokens(total=10_000, inp=10_000, cache_read=8_000)),
            _step(1, agent="explore", session_id="ses_a",
                  tokens=_tokens(total=4_000, inp=4_000, cache_read=3_000)),
            _step(2, agent="plan", session_id="ses_plan",
                  tokens=_tokens(total=2_000, inp=2_000, cache_read=1_000)),
            _step(3, agent="plan", session_id="ses_plan",
                  tokens=_tokens(total=2_100, inp=2_100, cache_read=1_100)),
            _step(4, role="user", agent="explore", session_id="ses_a",
                  parts=[{"type": "compaction", "summary": "prior work"}]),
            _step(5, agent="explore", session_id="ses_a", summary=True,
                  tokens=_tokens(total=800, inp=800, cache_read=600)),
        ]
        fig = build_context_pressure_chart(steps, agent_key=PRESSURE_ALL_AGENTS)
        occupancy_colors = [
            t.line.color for t in fig.data
            if t.name in ("plan", "explore") and t.line is not None
        ]
        self.assertEqual(len(occupancy_colors), 2)
        self.assertEqual(len(set(occupancy_colors)), 2)
        compact_names = [t.name for t in fig.data if t.name and "compact" in t.name]
        self.assertTrue(any("explore" in n for n in compact_names))
        self.assertTrue(any("plan" in n for n in compact_names))
        # Full-height vlines would cut through every series.
        for shape in fig.layout.shapes or []:
            self.assertEqual(shape.type, "line")
            self.assertEqual(shape.y0, shape.y1)


class PrematureCompactionTests(unittest.TestCase):
    @staticmethod
    def _steps(before, after):
        return [
            _step(0, session_id="s", model_id="gpt-4o", tokens=_tokens(total=before, inp=before)),
            _step(1, role="compaction", session_id="s", is_compaction_checkpoint=True,
                  message_type="compaction"),
            _step(2, session_id="s", model_id="gpt-4o", tokens=_tokens(total=after, inp=after)),
        ]

    def test_healthy_compaction_not_flagged(self):
        # 100k of the 128k window, then a real cliff to 20k.
        flagged = detect_premature_compactions(self._steps(100_000, 20_000))
        self.assertEqual(flagged, [])

    def test_low_occupancy_compaction_flagged(self):
        flagged = detect_premature_compactions(self._steps(20_000, 8_000))
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["reason"], "low_occupancy")
        self.assertFalse(flagged[0]["grew"])
        self.assertEqual(flagged[0]["occupancy_before"], 20_000)
        self.assertEqual(flagged[0]["occupancy_after"], 8_000)
        self.assertAlmostEqual(flagged[0]["before_pct"], 15.6, places=1)
        self.assertEqual(flagged[0]["window_limit"], 128_000)

    def test_window_growing_compaction_flagged(self):
        flagged = detect_premature_compactions(self._steps(10_000, 12_000))
        self.assertEqual(len(flagged), 1)
        self.assertTrue(flagged[0]["grew"])
        self.assertEqual(flagged[0]["reason"], "window_grew")

    def test_override_limit_suppresses_premature_flag(self):
        # Same low-occupancy compaction, but with the real 30k window known.
        steps = self._steps(20_000, 8_000)
        self.assertEqual(detect_premature_compactions(steps, window_limit="30k"), [])

    def test_inferred_occupancy_drop_is_not_considered(self):
        # A cliff with no explicit compaction record stays out of the detector;
        # it is only heuristic pressure-series material.
        steps = [
            _step(0, session_id="s", tokens=_tokens(total=100_000, inp=100_000, cache_read=90_000)),
            _step(1, session_id="s", tokens=_tokens(total=20_000, inp=20_000, cache_read=15_000)),
            _step(2, session_id="s", tokens=_tokens(total=21_000, inp=21_000, cache_read=16_000)),
        ]
        self.assertEqual(detect_premature_compactions(steps), [])


if __name__ == "__main__":
    unittest.main()
