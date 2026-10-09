"""What the dashboard must clear, contain, or fire exactly once on a load.

Three properties live here, all of them about the wiring rather than the
numbers:

* **reset** — a tab that is outside the load packer (`ui.load.LOAD_UNITS`) keeps
  whatever it last rendered unless something clears it explicitly. Attribution
  does that; Comparison did not, so a scorecard captioned with run A's filename
  could sit beside run B's Overview metrics.
* **containment** — Gradio answers a raising callback with HTTP 500 and an empty
  error body when `show_error` is off (it is), so the browser shows nothing and
  every output keeps its stale value. The callbacks with no containment at any
  layer must carry `shared.safe_callback`; the ones already contained one layer
  down must NOT, or the inner error surface is replaced by a toast.
* **one gesture, one load** — choosing a file and clicking Load must run the
  analytics fan-out once, not twice.

Plus pins (passing before and after the change that recorded them) for the
deliberately light-only palette and for the JS block that
`tests/test_workflow_detail_ui.py` slices out of `build_ui`'s source.
"""

from __future__ import annotations

import dataclasses
import inspect
import unittest

import gradio as gr

from trajviz.insight import insight as insight_mod
from trajviz.insight.ui import comparison_tab, overview_tab, shared, upload

_TRAJECTORY_FILE_LABEL = "Trajectory (.json / .jsonl / .zip)"
_LOAD_BUTTON_LABEL = "Load Trajectory"

_APP = None
_CFG = None


def _app():
    """Build the real Blocks once — construction is the expensive part."""
    global _APP, _CFG
    if _APP is None:
        _APP = insight_mod.build_ui()
        _CFG = _APP.get_config_file()
    return _APP, _CFG


def _fns(app) -> list:
    fns = getattr(app, "fns", None) or app.default_config.fns
    return list(fns.values())


def _component_id(cfg, comp_type: str, prop: str, value: str) -> int:
    for component in cfg["components"]:
        if component["type"] == comp_type and (component.get("props") or {}).get(prop) == value:
            return component["id"]
    raise AssertionError(f"no {comp_type} component with {prop}={value!r}")


def _deps_targeting(cfg, component_id: int, event: str) -> list[dict]:
    return [
        dep
        for dep in cfg["dependencies"]
        if any(cid == component_id and name == event for cid, name in dep.get("targets", []))
    ]


class ComparisonResetOnLoadTests(unittest.TestCase):
    """U1: Comparison is outside the load packer, so load must clear it."""

    def test_reset_restores_the_layout_placeholders(self):
        status, scorecard, timeline, behavior, report, phase_count, phase_duration = (
            comparison_tab._reset_comparison()
        )
        self.assertEqual(status, comparison_tab._CMP_STATUS_PLACEHOLDER)
        self.assertEqual(scorecard, comparison_tab._RG_SCORECARD_PLACEHOLDER)
        self.assertFalse(timeline["visible"])
        self.assertEqual(behavior, "")
        self.assertEqual(report, "")
        # Empty figures rather than None, matching what the run callbacks return.
        self.assertEqual(phase_count.data, ())
        self.assertEqual(phase_duration.data, ())

    def test_loading_a_trajectory_clears_the_comparison_panels(self):
        app, cfg = _app()
        load_btn = _component_id(cfg, "button", "value", _LOAD_BUTTON_LABEL)
        resets = [bf for bf in _fns(app) if bf.fn is comparison_tab._reset_comparison]
        self.assertEqual(len(resets), 1, "Comparison reset is not wired exactly once")
        self.assertEqual(list(resets[0].targets), [(load_btn, "click")])
        self.assertEqual(len(resets[0].outputs), 7)


class SingleLoadPathTests(unittest.TestCase):
    """U4: one gesture must not run the whole analytics fan-out twice."""

    def test_choosing_a_file_has_no_second_backend_load(self):
        app, cfg = _app()
        file_id = _component_id(cfg, "file", "label", _TRAJECTORY_FILE_LABEL)
        change_deps = _deps_targeting(cfg, file_id, "change")
        self.assertEqual(len(change_deps), 1, "the Trajectory file input still has extra listeners")
        self.assertTrue(change_deps[0].get("js"), "auto-load should re-click Load in the browser")
        self.assertFalse(change_deps[0].get("backend_fn"), "auto-load still runs a second Python load")

    def test_the_export_writer_is_registered_once(self):
        app, _ = _app()
        exports = [bf for bf in _fns(app) if bf.fn is upload.prepare_html_export]
        self.assertEqual(len(exports), 1, "one load gesture writes the HTML report twice")

    def test_the_load_resets_are_registered_once_each(self):
        app, cfg = _app()
        load_btn = _component_id(cfg, "button", "value", _LOAD_BUTTON_LABEL)
        by_name: dict[str, list] = {}
        for bf in _fns(app):
            name = getattr(bf.fn, "__name__", "")
            if name in ("_reset_labels", "_clear_attribution"):
                by_name.setdefault(name, []).append(bf)
        for name in ("_reset_labels", "_clear_attribution"):
            with self.subTest(reset=name):
                bound = by_name.get(name, [])
                self.assertEqual(len(bound), 1, f"{name} is bound to more than one load trigger")
                self.assertEqual(list(bound[0].targets), [(load_btn, "click")])


class SafeCallbackTests(unittest.TestCase):
    """U3: an unguarded callback fails as a silent 500 with stale outputs."""

    def test_wrapper_reports_the_type_and_hides_the_message(self):
        @shared.safe_callback("Context utilization")
        def boom():
            raise RuntimeError("/Users/someone/runs/claude_code/astropy__astropy-12907.json")

        with self.assertRaises(gr.Error) as ctx:
            boom()
        message = ctx.exception.message
        self.assertIn("Context utilization", message)
        self.assertIn("RuntimeError", message)
        self.assertNotIn("astropy__astropy-12907", message)

    def test_wrapper_passes_a_user_facing_error_through_unchanged(self):
        @shared.safe_callback("Outer")
        def already_reported():
            raise gr.Error("Inner already said it.")

        with self.assertRaises(gr.Error) as ctx:
            already_reported()
        self.assertEqual(ctx.exception.message, "Inner already said it.")

    def test_wrapper_keeps_the_callback_callable_as_itself(self):
        @shared.safe_callback("Echo")
        def echo(a, b=2):
            return a + b

        self.assertEqual(echo(1), 3)
        self.assertEqual(echo.__name__, "echo")
        self.assertTrue(echo.__tv_safe__)

    def test_the_unguarded_callbacks_are_contained(self):
        app, _ = _app()
        marked = {
            getattr(bf.fn, "__name__", ""): getattr(bf.fn, "__tv_safe__", False)
            for bf in _fns(app)
            if bf.fn is not None
        }
        for name in (
            "_rebuild_utilization",
            "on_agent_change",
            "do_filter_workflow",
            "on_toc_toggle",
            "on_run_group_scorecard",
            "on_diagnose",
        ):
            with self.subTest(callback=name):
                self.assertIn(name, marked, f"{name} is no longer a registered callback")
                self.assertTrue(marked[name], f"{name} has no error containment")

    def test_callbacks_with_their_own_error_surface_are_left_alone(self):
        """Wrapping these would replace a rendered banner with a toast."""
        app, _ = _app()
        for bf in _fns(app):
            name = getattr(bf.fn, "__name__", "")
            if name in ("_do_load", "do_load_labels", "on_analysis_ask", "on_run_comparison"):
                with self.subTest(callback=name):
                    self.assertFalse(getattr(bf.fn, "__tv_safe__", False))


class OverviewSectionMappingTests(unittest.TestCase):
    """U7/U8: the nav names and the Column outputs are zipped by position."""

    @staticmethod
    def _layout():
        with gr.Blocks():
            kpi_html = gr.HTML("")
            with gr.Tabs():
                refs = overview_tab.layout(kpi_html)
            upload_refs = upload.layout()
            shared_state = shared.SharedState(
                state_steps=gr.State([]),
                state_dark=gr.State(False),
                state_raw=gr.State({}),
                state_analysis_brief=gr.State(""),
            )
        return refs, shared_state, upload_refs

    def test_the_context_section_is_named_for_what_it_holds(self):
        self.assertIn(("Context Utilization", "context_section"), overview_tab.OVERVIEW_SECTIONS)
        self.assertEqual(overview_tab.OVERVIEW_SECTION_NAMES, [n for n, _ in overview_tab.OVERVIEW_SECTIONS])
        refs, _, _ = self._layout()
        self.assertTrue(hasattr(refs, "context_section"))
        self.assertFalse(hasattr(refs, "efficiency_section"))

    def test_two_names_mapped_onto_one_column_is_rejected(self):
        refs, shared_state, upload_refs = self._layout()
        with gr.Blocks():
            tampered = dataclasses.replace(refs, context_section=refs.tools_section)
            with self.assertRaises(ValueError):
                overview_tab.bind(tampered, shared_state, upload_refs)


class LightOnlyPinTests(unittest.TestCase):
    """U6 pin: `state_dark` is write-once, so every chart renders light."""

    def test_only_the_page_load_handler_writes_state_dark(self):
        app, cfg = _app()
        page_load = [dep for dep in cfg["dependencies"] if any(name == "load" for _, name in dep.get("targets", []))]
        self.assertEqual(len(page_load), 1)
        dark_ids = list(page_load[0]["outputs"])
        self.assertEqual(len(dark_ids), 1, "the page-load handler no longer writes exactly state_dark")
        writers = [dep for dep in cfg["dependencies"] if dark_ids[0] in dep.get("outputs", [])]
        self.assertEqual(len(writers), 1, "something other than page load now writes state_dark")


class LaunchPresentationTests(unittest.TestCase):
    """U9: `css`/`head` are launch() parameters in Gradio 6, not Blocks ones."""

    def test_launch_presentation_carries_the_stylesheet_and_the_head_tag(self):
        from trajviz.insight.styles import APP_CSS

        presentation = insight_mod.LAUNCH_PRESENTATION
        self.assertIs(presentation["css"], APP_CSS)
        self.assertIn("color-scheme", presentation["head"])


class BindJumpsContractTests(unittest.TestCase):
    """U10 pin: the markers are a test contract, enforced here without Node.

    `tests/test_workflow_detail_ui.py::_bind_jumps_source` slices
    `inspect.getsource(build_ui)` between the markers and runs the slice under
    Node — but it is skipped where Node is absent, so a reflow of the block
    would show up as a skip rather than a failure. These assertions hold
    everywhere.
    """

    def test_the_slice_still_extracts_exactly_one_balanced_function(self):
        src = inspect.getsource(insight_mod.build_ui)
        self.assertEqual(src.count("/* __TV_BIND_JUMPS_BEGIN__ */"), 1)
        self.assertEqual(src.count("/* __TV_BIND_JUMPS_END__ */"), 1)
        block = src.split("/* __TV_BIND_JUMPS_BEGIN__ */")[1].split("/* __TV_BIND_JUMPS_END__ */")[0]
        self.assertEqual(block.count("window.tvBindChartWorkflowJumps"), 1)
        # The slice is executed as-is, and taken up to its LAST `};`: exactly one
        # top-level declaration, nothing after it.
        self.assertTrue(block.strip().startswith("window.tvBindChartWorkflowJumps = function"))
        self.assertTrue(block.rstrip().endswith("};"), "the block must end with the function it declares")

    def test_the_contract_is_recorded_next_to_the_markers(self):
        src = inspect.getsource(insight_mod.build_ui)
        preamble = src.split("/* __TV_BIND_JUMPS_BEGIN__ */")[0].rstrip().rsplit("/*", 1)[-1]
        self.assertIn("tests/test_workflow_detail_ui.py", preamble)
        self.assertIn("markers", preamble)


if __name__ == "__main__":
    unittest.main()
