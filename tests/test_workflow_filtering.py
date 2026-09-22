import inspect
import json
import shutil
import subprocess
import unittest


def _chip_state_source():
    """Extract the pure chip state machine JS embedded in the Workflow tab."""
    from trajviz.insight.ui import workflow_tab

    source = inspect.getsource(workflow_tab)
    begin = source.index("__WF_CHIP_STATE_BEGIN__")
    end = source.index("__WF_CHIP_STATE_END__")
    block = source[begin:end]
    block = block[block.index("window.__wfComputeChipState"):]
    return block[:block.rindex("};") + 2]


def _run_chip_state(scenarios):
    """Execute __wfComputeChipState in Node for each (state, action) pair."""
    script = (
        "var window = {};\n"
        + _chip_state_source()
        + "\nvar scenarios = " + json.dumps(scenarios) + ";\n"
        + "console.log(JSON.stringify(scenarios.map(function(s) {\n"
        + "    return window.__wfComputeChipState(s.state, s.action);\n"
        + "})));"
    )
    proc = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node failed: {proc.stderr}")
    return json.loads(proc.stdout)


def _chip_state(roles=None, features=None, agents=None):
    state = {
        "roles": {"Assistant": True, "User": True},
        "features": {"All": True, "Tool Calls": False, "Errors": False, "Reasoning": False},
        "agents": {},
    }
    if roles:
        state["roles"].update(roles)
    if features:
        state["features"].update(features)
    if agents is not None:
        state["agents"] = dict(agents)
    return state


def _step(
    index,
    *,
    role="assistant",
    agent="",
    tool=False,
    error=False,
    reasoning=False,
    text="",
    tool_name="Read",
    tool_input=None,
):
    tool_calls = []
    if tool:
        tool_calls.append({
            "tool_name": tool_name,
            "input": tool_input or {},
        })
    return {
        "index": index,
        "role": role,
        "agent": agent,
        "is_sub_agent": bool(agent),
        "session_id": "",
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "error_count": int(error),
        "has_reasoning": reasoning,
        "text_preview": text,
        "duration": None,
        "parts": [],
        "tokens": {
            "total": 0,
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        },
    }


class WorkflowFilteringTests(unittest.TestCase):
    def setUp(self):
        self.steps = [
            _step(0, role="user", text="build the dashboard"),
            _step(1, tool=True, text="inspect files"),
            _step(2, agent="sub-agent", text="plain assistant response"),
            _step(3, agent="sub-agent", tool=True, error=True, text="command failed"),
            _step(4, agent="sub-agent", reasoning=True, text="consider options"),
        ]

    def test_roles_are_combined_with_or(self):
        from trajviz.insight.presenters import filter_workflow_steps

        self.assertEqual(
            filter_workflow_steps(self.steps, ["Assistant", "User", "All"]),
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            filter_workflow_steps(self.steps, ["Assistant", "All"]),
            [1, 2, 3, 4],
        )
        self.assertEqual(
            filter_workflow_steps(self.steps, ["User", "All"]),
            [0],
        )

    def test_errors_feature_includes_provider_abort(self):
        from trajviz.insight.presenters import filter_workflow_steps

        steps = [
            _step(0, role="assistant", text="ok"),
            _step(1, role="assistant", text="api down"),
        ]
        steps[1]["finish"] = "error"
        self.assertEqual(
            filter_workflow_steps(steps, ["Assistant", "Errors"]),
            [1],
        )

    def test_features_are_combined_with_or_inside_selected_roles(self):
        from trajviz.insight.presenters import filter_workflow_steps

        self.assertEqual(
            filter_workflow_steps(
                self.steps,
                ["Assistant", "Tool Calls", "Reasoning"],
            ),
            [1, 3, 4],
        )

    def test_role_and_feature_groups_are_combined_with_and(self):
        from trajviz.insight.presenters import filter_workflow_steps

        self.assertEqual(
            filter_workflow_steps(self.steps, ["Assistant", "Tool Calls"]),
            [1, 3],
        )
        self.assertEqual(
            filter_workflow_steps(self.steps, ["User", "Tool Calls"]),
            [],
        )

    def test_all_or_an_omitted_feature_selection_has_no_feature_predicate(self):
        from trajviz.insight.presenters import filter_workflow_steps

        expected = [1, 2, 3, 4]
        self.assertEqual(
            filter_workflow_steps(self.steps, ["Assistant", "All"]),
            expected,
        )
        self.assertEqual(
            filter_workflow_steps(self.steps, ["Assistant"]),
            expected,
        )

    def test_no_selected_role_is_empty(self):
        from trajviz.insight.presenters import filter_workflow_steps

        self.assertEqual(
            filter_workflow_steps(self.steps, ["All", "Tool Calls"]),
            [],
        )

    def test_subagent_handoff_is_not_a_user_filter_match(self):
        from trajviz.insight.presenters import filter_workflow_steps

        spawn = _step(5, role="user", agent="explore", text="Explore the repo")
        spawn["parts"] = [{"type": "text", "text": "Explore the repo"}]
        steps = [*self.steps, spawn]

        self.assertEqual(
            filter_workflow_steps(steps, ["User", "All"]),
            [0],
        )
        self.assertIn(5, filter_workflow_steps(steps, ["Assistant", "All"]))

    def test_agent_tokens_filter_by_timeline_agent_id(self):
        from trajviz.insight.presenters import filter_workflow_steps
        from trajviz.insight.rendering import MAIN_AGENT_FILTER, agent_filter_token

        self.assertEqual(
            filter_workflow_steps(
                self.steps,
                ["Assistant", "All", agent_filter_token("sub-agent")],
            ),
            [2, 3, 4],
        )
        self.assertEqual(
            filter_workflow_steps(
                self.steps,
                ["Assistant", "All", MAIN_AGENT_FILTER],
            ),
            [1],
        )
        self.assertEqual(
            filter_workflow_steps(
                self.steps,
                ["Assistant", "All", "agent:All"],
            ),
            [1, 2, 3, 4],
        )

    def test_agent_and_feature_groups_are_combined_with_and(self):
        from trajviz.insight.presenters import filter_workflow_steps
        from trajviz.insight.rendering import agent_filter_token

        self.assertEqual(
            filter_workflow_steps(
                self.steps,
                ["Assistant", "Tool Calls", agent_filter_token("sub-agent")],
            ),
            [3],
        )

    def test_keyword_is_anded_with_role_and_features(self):
        from trajviz.insight.presenters import filter_workflow_steps

        self.steps[3]["tool_calls"][0] = {
            "tool_name": "Bash",
            "input": {"command": "pytest workflow"},
        }
        filters = ["Assistant", "Tool Calls"]

        self.assertEqual(filter_workflow_steps(self.steps, filters, "pytest"), [3])
        self.assertEqual(filter_workflow_steps(self.steps, filters, "inspect"), [1])

    def test_filtered_outputs_keep_cards_count_and_toc_in_sync(self):
        from trajviz.insight.presenters import build_filtered_workflow_outputs

        workflow, count, toc = build_filtered_workflow_outputs(
            self.steps,
            "Assistant,Errors",
            "",
        )

        self.assertIn("wf-card-3", workflow)
        self.assertNotIn("wf-card-1", workflow)
        self.assertIn("Showing 1 of 5 steps", count)
        self.assertIn("data-step-idx='3'", toc)
        self.assertNotIn("data-step-idx='1'", toc)

    def test_filter_chips_render_two_explicit_groups_without_agents(self):
        from trajviz.insight.rendering import render_filter_chips

        chips = render_filter_chips()

        self.assertIn("data-filter-group-container='role'", chips)
        self.assertIn("data-filter-group-container='feature'", chips)
        self.assertNotIn("data-filter-group-container='agent'", chips)
        self.assertIn("data-filter='All'", chips)
        self.assertIn("data-wf-action='reset-filters'", chips)
        self.assertIn("select at least one", chips)
        self.assertIn("match any selected", chips)
        self.assertNotIn("agent:", chips)
        self.assertNotIn("Clear all", chips)
        self.assertNotIn("onclick=", chips)

    def test_filter_chips_render_agent_group_for_multi_agent_steps(self):
        from trajviz.insight.rendering import (
            AGENT_ALL_FILTER,
            MAIN_AGENT_FILTER,
            agent_filter_token,
            render_filter_chips,
            workflow_agent_chip_options,
        )

        options = workflow_agent_chip_options(self.steps)
        self.assertGreater(len(options), 1)
        chips = render_filter_chips(agent_options=options)

        self.assertIn("data-filter-group-container='agent'", chips)
        self.assertIn(f"data-filter='{AGENT_ALL_FILTER}'", chips)
        self.assertIn(f"data-filter='{MAIN_AGENT_FILTER}'", chips)
        self.assertIn(f"data-filter='{agent_filter_token('sub-agent')}'", chips)
        self.assertIn("Agent: All", chips)

    def test_default_chips_select_both_roles_and_all_only(self):
        from trajviz.insight.rendering import render_filter_chips

        chips = render_filter_chips()

        for name in ("Assistant", "User", "All"):
            start = chips.index(f"data-filter='{name}'")
            button_start = chips.rfind("<button", 0, start)
            self.assertIn("chip-active", chips[button_start:start])
        for name in ("Tool Calls", "Errors", "Reasoning"):
            start = chips.index(f"data-filter='{name}'")
            button_start = chips.rfind("<button", 0, start)
            self.assertNotIn("chip-active", chips[button_start:start])

    def test_chip_handler_routes_clicks_through_the_pure_state_machine(self):
        # Thin DOM-glue assertions; the state machine itself is executed
        # behaviorally in WorkflowChipStateMachineTests below.
        from trajviz.insight.ui import workflow_tab

        source = inspect.getsource(workflow_tab)

        self.assertIn("window.__wfComputeChipState(", source)
        self.assertIn("window.__wfReadChipState(bar)", source)
        self.assertIn("window.__wfApplyChipState(bar, next)", source)
        self.assertIn("data-wf-action=\"reset-filters\"", source)
        self.assertIn("window.__updateWorkflowFilterQuery(root)", source)
        self.assertIn("window.__syncWorkflowFilters(bar)", source)

    def test_chip_handler_updates_the_real_hidden_input_and_all_outputs(self):
        from trajviz.insight.ui import workflow_tab

        source = inspect.getsource(workflow_tab)

        hidden_filter_source = source[source.index("wf_filter_hidden = gr.Textbox("):]
        hidden_filter_source = hidden_filter_source[:hidden_filter_source.index("wf_count_html")]
        self.assertIn("visible=True", hidden_filter_source)
        self.assertIn("hiddenEl.tagName === 'TEXTAREA'", source)
        self.assertIn("descriptor.set.call(hiddenEl, active.join(','))", source)
        self.assertIn("new InputEvent('input'", source)
        self.assertIn("new Event('change'", source)
        self.assertIn("wf_filter_hidden.change(", source)
        self.assertIn("outputs=[workflow_html, wf_count_html, toc_html]", source)

    def test_hidden_filter_bridge_is_visually_hidden_with_css(self):
        from pathlib import Path

        styles = Path("trajviz/insight/styles.py").read_text()

        start = styles.index("#wf-filter-hidden")
        bridge_css = styles[start:styles.index(".filter-summary {", start)]
        self.assertIn("display: none !important", bridge_css)

    def test_filter_rerender_preserves_collapsed_toc(self):
        from trajviz.insight.presenters import build_filtered_workflow_outputs

        collapsed_toc = "<nav class='wf-toc-sidebar toc-hidden' id='wf-toc-sidebar'></nav>"
        _, _, toc = build_filtered_workflow_outputs(
            self.steps, "Assistant,All", "", collapsed_toc,
        )
        self.assertIn("toc-hidden", toc)

        open_toc = "<nav class='wf-toc-sidebar' id='wf-toc-sidebar'></nav>"
        _, _, toc = build_filtered_workflow_outputs(
            self.steps, "Assistant,All", "", open_toc,
        )
        self.assertNotIn("toc-hidden", toc)

    def test_filter_callbacks_read_and_write_the_toc(self):
        from trajviz.insight.ui import workflow_tab

        source = inspect.getsource(workflow_tab)

        self.assertIn(
            "inputs=[state_steps, wf_filter_hidden, wf_search, toc_html]", source,
        )
        self.assertIn("outputs=[workflow_html, wf_count_html, toc_html]", source)

    def test_hidden_selection_watcher_reacts_to_rerenders_not_page_load(self):
        # The message must come from a DOM watcher (re-renders never re-run
        # js_on_load in Gradio 6), must never fire while the detail panel
        # still shows its placeholder, and must restore the step's detail
        # when its card reappears. A stale URL hash is dropped, not
        # explained away as "hidden by the current filters".
        from trajviz.insight.ui import workflow_tab

        source = inspect.getsource(workflow_tab)

        self.assertIn("new MutationObserver(", source)
        self.assertIn("data-wf-hidden-msg", source)
        self.assertIn("data-wf-detail-placeholder", source)
        self.assertIn("Selected step is hidden by the current filters", source)
        watcher = source[source.index("__wfHiddenStepObserverAttached"):]
        watcher = watcher[:watcher.index("MutationObserver")]
        self.assertIn("selectCard(card, { pushHistory: false })", watcher)
        deep_link = source[source.index("Deep link: on load"):source.index("Hidden-selection watcher")]
        self.assertNotIn("hidden by the current filters", deep_link)
        self.assertIn("waitForDeepLink", deep_link)
        self.assertIn("tvGotoWorkflowStep", deep_link)
        self.assertIn("restore: true", deep_link)
        self.assertIn("history.pushState(", source)
        self.assertIn("pushHistory", source)
        self.assertIn("tvWorkflowStepUrl", source)


@unittest.skipUnless(shutil.which("node"), "requires Node.js to execute the chip state machine")
class WorkflowChipStateMachineTests(unittest.TestCase):
    """Execute the embedded __wfComputeChipState source against real scenarios."""

    def test_selecting_a_feature_makes_it_exclusive_with_all(self):
        (result,) = _run_chip_state([{
            "state": _chip_state(),
            "action": {"type": "toggle", "group": "feature", "name": "Errors"},
        }])

        self.assertFalse(result["rejected"])
        self.assertFalse(result["features"]["All"])
        self.assertTrue(result["features"]["Errors"])
        self.assertEqual(result["roles"], {"Assistant": True, "User": True})

    def test_deselecting_the_last_feature_auto_restores_all(self):
        (result,) = _run_chip_state([{
            "state": _chip_state(features={"All": False, "Errors": True}),
            "action": {"type": "toggle", "group": "feature", "name": "Errors"},
        }])

        self.assertTrue(result["features"]["All"])
        self.assertFalse(result["features"]["Errors"])

    def test_deselecting_one_of_two_features_keeps_the_other_without_all(self):
        (result,) = _run_chip_state([{
            "state": _chip_state(features={"All": False, "Errors": True, "Reasoning": True}),
            "action": {"type": "toggle", "group": "feature", "name": "Errors"},
        }])

        self.assertFalse(result["features"]["All"])
        self.assertFalse(result["features"]["Errors"])
        self.assertTrue(result["features"]["Reasoning"])

    def test_selecting_all_clears_every_specific_feature(self):
        (result,) = _run_chip_state([{
            "state": _chip_state(features={"All": False, "Errors": True, "Reasoning": True}),
            "action": {"type": "toggle", "group": "feature", "name": "All"},
        }])

        self.assertEqual(
            result["features"],
            {"All": True, "Tool Calls": False, "Errors": False, "Reasoning": False},
        )

    def test_the_last_active_role_cannot_be_deselected(self):
        (result,) = _run_chip_state([{
            "state": _chip_state(roles={"Assistant": True, "User": False}),
            "action": {"type": "toggle", "group": "role", "name": "Assistant"},
        }])

        self.assertTrue(result["rejected"])
        self.assertTrue(result["roles"]["Assistant"])
        self.assertFalse(result["roles"]["User"])

    def test_role_toggles_do_not_touch_features(self):
        results = _run_chip_state([
            {
                "state": _chip_state(features={"All": False, "Tool Calls": True}),
                "action": {"type": "toggle", "group": "role", "name": "User"},
            },
            {
                "state": _chip_state(roles={"Assistant": False, "User": True}),
                "action": {"type": "toggle", "group": "role", "name": "Assistant"},
            },
        ])

        self.assertFalse(results[0]["roles"]["User"])
        self.assertEqual(
            results[0]["features"],
            {"All": False, "Tool Calls": True, "Errors": False, "Reasoning": False},
        )
        self.assertTrue(results[1]["roles"]["Assistant"])
        self.assertTrue(results[1]["roles"]["User"])

    def test_reset_restores_all_roles_and_the_all_feature(self):
        (result,) = _run_chip_state([{
            "state": _chip_state(
                roles={"Assistant": False, "User": True},
                features={"All": False, "Tool Calls": True, "Errors": True},
            ),
            "action": {"type": "reset"},
        }])

        self.assertEqual(result["roles"], {"Assistant": True, "User": True})
        self.assertEqual(
            result["features"],
            {"All": True, "Tool Calls": False, "Errors": False, "Reasoning": False},
        )
        self.assertEqual(result["agents"], {})

    def test_agent_all_is_exclusive_like_features(self):
        agents = {
            "agent:All": True,
            "agent:__main__": False,
            "agent:sub-agent": False,
        }
        (selected,) = _run_chip_state([{
            "state": _chip_state(agents=agents),
            "action": {"type": "toggle", "group": "agent", "name": "agent:sub-agent"},
        }])
        self.assertFalse(selected["rejected"])
        self.assertFalse(selected["agents"]["agent:All"])
        self.assertTrue(selected["agents"]["agent:sub-agent"])
        self.assertFalse(selected["agents"]["agent:__main__"])
        self.assertTrue(selected["features"]["All"])

        (restored,) = _run_chip_state([{
            "state": _chip_state(agents={
                "agent:All": False,
                "agent:__main__": False,
                "agent:sub-agent": True,
            }),
            "action": {"type": "toggle", "group": "agent", "name": "agent:sub-agent"},
        }])
        self.assertTrue(restored["agents"]["agent:All"])
        self.assertFalse(restored["agents"]["agent:sub-agent"])

    def test_reset_restores_agent_all(self):
        (result,) = _run_chip_state([{
            "state": _chip_state(
                features={"All": False, "Errors": True},
                agents={
                    "agent:All": False,
                    "agent:__main__": False,
                    "agent:sub-agent": True,
                },
            ),
            "action": {"type": "reset"},
        }])
        self.assertTrue(result["features"]["All"])
        self.assertEqual(
            result["agents"],
            {
                "agent:All": True,
                "agent:__main__": False,
                "agent:sub-agent": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
