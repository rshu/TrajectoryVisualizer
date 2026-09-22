"""The LLM surfaces must never send trajectory content without a user action.

TrajViz is offline-by-default analytics over traces that routinely contain
source code, absolute local paths, shell commands and customer data. Any
outbound LLM call therefore needs an explicit, declinable user action. These
tests read the real Gradio dependency graph of the built dashboard rather than
source text, so a future rewiring cannot slip past them.
"""

import unittest

import gradio as gr

from trajviz.insight.insight import build_ui


def _deps_for(demo, fn_name):
    """Dependencies in the built UI whose handler is *fn_name*."""
    out = []
    for bf in demo.fns.values():
        fn = getattr(bf, "fn", None)
        if getattr(fn, "__name__", None) == fn_name:
            out.append(bf)
    return out


class IssuesJudgeOptInTests(unittest.TestCase):
    """The Overview Issues LLM judge is opt-in (regression: auto-fired on load)."""

    @classmethod
    def setUpClass(cls):
        cls.demo = build_ui()

    def test_judge_is_bound_exactly_once(self):
        self.assertEqual(len(_deps_for(self.demo, "on_suggest_fixes")), 1)

    def test_judge_is_never_chained_to_a_load_event(self):
        # `.then(...)` on the load events is what made every opened file ship
        # trajectory excerpts to a third-party endpoint with no consent.
        for bf in _deps_for(self.demo, "on_suggest_fixes"):
            events = [event for _block_id, event in bf.targets]
            self.assertNotIn("then", events)
            self.assertEqual(set(events), {"click"})

    def test_judge_trigger_is_a_button_that_names_the_egress(self):
        blocks = self.demo.blocks
        for bf in _deps_for(self.demo, "on_suggest_fixes"):
            for block_id, _event in bf.targets:
                trigger = blocks[block_id]
                self.assertIsInstance(trigger, gr.Button)
                label = (trigger.value or "").lower()
                self.assertIn("llm", label)
                self.assertIn("sends", label)


class SidebarAnalysisOptInTests(unittest.TestCase):
    """The AI analysis sidebar only calls out when the user opened the panel."""

    def test_load_with_closed_sidebar_does_not_call_the_provider(self):
        from trajviz.insight import assistant
        from trajviz.insight.ui import sidebar

        calls = []

        def _boom(*a, **k):
            calls.append(1)
            raise AssertionError("provider must not be called on load")

        original = assistant.analyze_loaded_trajectory
        sidebar.analyze_loaded_trajectory = _boom
        try:
            demo = build_ui()
            handlers = _deps_for(demo, "on_trajectory_loaded")
            self.assertTrue(handlers, "sidebar load handler not found")
            steps = [{"index": 0, "role": "user", "parts": [], "tool_calls": []}]
            for bf in handlers:
                # sidebar_open=False is the default: panel starts collapsed.
                bf.fn(steps, "", False)
        finally:
            sidebar.analyze_loaded_trajectory = original
            assistant.analyze_loaded_trajectory = original
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
