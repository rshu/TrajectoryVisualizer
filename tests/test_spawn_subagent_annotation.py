"""OpenCode Task-tool spawn metadata → is_sub_agent annotation."""

from __future__ import annotations

import unittest
from pathlib import Path

from trajviz.insight.charts import build_run_group_agent_timeline
from trajviz.insight.loaders import load_trajectory
from trajviz.insight.parser import parse_steps, spawned_child_session_id
from trajviz.insight.patterns import extract_subagent_sessions

FIXTURE = Path(__file__).resolve().parents[1] / "scripts" / "exploration_subagent.json"


class SpawnSubagentAnnotationTests(unittest.TestCase):
    def test_spawned_child_session_id_guards(self):
        self.assertEqual(
            spawned_child_session_id({"sessionId": "child"}, caller_session_id="parent"),
            "child",
        )
        self.assertEqual(
            spawned_child_session_id({"sessionId": "parent"}, caller_session_id="parent"),
            "",
        )
        self.assertEqual(
            spawned_child_session_id(
                {"sessionID": "root"},
                caller_session_id="parent",
                root_session_id="root",
            ),
            "",
        )
        self.assertEqual(spawned_child_session_id(None), "")

    @unittest.skipUnless(FIXTURE.is_file(), "exploration_subagent.json missing")
    def test_exploration_fixture_marks_explore_sessions(self):
        raw = load_trajectory(str(FIXTURE))
        steps = parse_steps(raw)
        root = (raw.get("info") or {}).get("id", "")
        self.assertTrue(root)

        build_steps = [s for s in steps if s.get("session_id") == root]
        explore_steps = [s for s in steps if s.get("agent") == "explore"]
        self.assertTrue(build_steps)
        self.assertTrue(explore_steps)
        self.assertTrue(all(not s.get("is_sub_agent") for s in build_steps))
        self.assertTrue(all(s.get("is_sub_agent") for s in explore_steps))
        self.assertTrue(all(s.get("parent_session_id") == root for s in explore_steps))

        sessions = extract_subagent_sessions(steps, raw.get("messages") or [])
        self.assertEqual(len(sessions), 2)
        self.assertEqual({s["spawn_step"] for s in sessions}, {1, 16})

        fig = build_run_group_agent_timeline([{"label": "exploration", "steps": steps}])
        legend = {t.name for t in fig.data if t.showlegend}
        self.assertIn("build", legend)
        self.assertNotIn("compaction", legend)
        self.assertEqual(len([n for n in legend if n.startswith("explore")]), 2)


if __name__ == "__main__":
    unittest.main()
