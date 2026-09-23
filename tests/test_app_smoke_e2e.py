"""End-to-end smoke test: the dashboard actually launches, serves, and loads.

The rest of ``tests/`` exercises builders and handlers in isolation, so a
dashboard that composes fine but never reaches a browser would still be green.
This file drives the composed app the way a browser does — ``build_ui()`` ->
``launch()`` -> HTTP -> the Gradio config the frontend renders from — plus one
real load through ``do_load``, the function the Load button is bound to.

Hermetic: binds a free loopback port, disables Gradio's telemetry before the
Blocks are constructed, and uses only the vendored fixture trajectory.
"""

import inspect
import json
import os
import re
import socket
import unittest
import urllib.request
from pathlib import Path

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures" / "corpus" / "data" / "trajectory" / "claude_code"
    / "astropy__astropy-13033.json"
)

# Every tab the composer builds. A tab that stops reaching the browser takes its
# whole feature with it and raises nothing, so the set is asserted explicitly.
EXPECTED_TABS = {"Overview", "Patterns", "Attribution", "Comparison", "Workflow", "Raw Data"}


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - loopback only
        return response.status, response.read().decode("utf-8", "replace")


def _tab_labels(components):
    return {
        comp.get("props", {}).get("label")
        for comp in components
        if comp.get("type") == "tabitem"
    }


class ServedDashboardTests(unittest.TestCase):
    """The composed Blocks must actually start a server and ship every tab."""

    @classmethod
    def setUpClass(cls):
        # Read by Blocks.__init__, so it must be set before build_ui(): keeps the
        # version-check thread and launch telemetry off the network.
        cls._analytics = os.environ.get("GRADIO_ANALYTICS_ENABLED")
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        from trajviz.insight.insight import build_ui

        cls.app = build_ui()
        cls.port = _free_port()
        cls.app.launch(
            prevent_thread_lock=True,
            server_name="127.0.0.1",
            server_port=cls.port,
            share=False,
            quiet=True,
            inbrowser=False,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        try:
            cls.app.close()
        finally:
            if cls._analytics is None:
                os.environ.pop("GRADIO_ANALYTICS_ENABLED", None)
            else:
                os.environ["GRADIO_ANALYTICS_ENABLED"] = cls._analytics

    def test_root_page_is_served(self):
        """A dashboard that 500s on the index is broken for every user."""
        status, page = _get(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn("TrajViz", page)

    def test_every_tab_the_composer_builds_reaches_the_browser(self):
        """Tabs are built in one place and consumed in another; keep them equal.

        Only the load-backed tabs are covered by the packer's key check, so a
        dropped Attribution or Comparison tab would raise nothing at all.
        """
        status, body = _get(self.base + "/config")
        self.assertEqual(status, 200)
        served = _tab_labels(json.loads(body).get("components", []))
        self.assertEqual(served, EXPECTED_TABS)

        built = {
            block.label
            for block in self.app.blocks.values()
            if getattr(block, "get_block_name", lambda: "")() == "tabitem"
        }
        self.assertEqual(served, built)

    def test_cross_tab_jumps_use_the_gradio_6_tab_selector(self):
        """``.tab-nav`` does not exist in Gradio 6: it would be a silent no-op.

        Every chart click and diagnostics chip reaches the Workflow tab through
        a label lookup over ``button[role=tab]``. If a ``.tab-nav`` selector ever
        comes back, the jumps stop working with no error anywhere.
        """
        _status, body = _get(self.base + "/config")
        self.assertIn("button[role=tab]", body)
        self.assertNotIn("tab-nav", body)

    def test_the_workflow_jump_has_exactly_one_emitter(self):
        """Two divergent definitions of the jump would fight over the same name."""
        _status, body = _get(self.base + "/config")
        self.assertEqual(body.count("window.tvGotoWorkflowStep = function"), 1)
        self.assertEqual(body.count("window.tvClickMainTab = function"), 1)


class ChartJumpWiringTests(unittest.TestCase):
    """The chart ids the jump binder looks up must be ids the UI really renders."""

    def test_every_chart_the_binder_targets_exists_in_the_ui(self):
        """``getElementById`` on a renamed chart returns null and binds nothing.

        The binder's id list lives in a JS string and the ids live in the tab
        layouts, so nothing but this check couples them.
        """
        from trajviz.insight import insight as insight_mod

        source = inspect.getsource(insight_mod.build_ui)
        block = source.split("__TV_BIND_JUMPS_BEGIN__")[1].split("__TV_BIND_JUMPS_END__")[0]
        targeted = re.findall(r"'([a-z0-9-]+-chart)'", block)
        self.assertTrue(targeted, "jump binder no longer lists any chart ids")

        app = insight_mod.build_ui()
        rendered = {
            getattr(block_obj, "elem_id", None)
            for block_obj in app.blocks.values()
        }
        for chart_id in targeted:
            self.assertIn(chart_id, rendered, f"{chart_id} is bound for jumps but never rendered")


@unittest.skipUnless(_FIXTURE.is_file(), "vendored fixture trajectory not present")
class LoadButtonTests(unittest.TestCase):
    """One real load through the exact function the Load button is bound to."""

    def _load(self, dark):
        from trajviz.insight.ui.load import do_load, load_slot_keys

        packed = do_load(str(_FIXTURE), dark, "")
        self.assertEqual(frozenset(packed), load_slot_keys())
        return packed

    def test_a_real_trajectory_fills_every_slot_without_an_error_banner(self):
        """``do_load`` swallows exceptions into a banner, so a crash looks green.

        Asserting on the banner is the only way a test notices that the load
        path blew up rather than rendering the trajectory.
        """
        for dark in (False, True):
            with self.subTest(dark=dark):
                packed = self._load(dark)
                banner = packed["summary_banner"]
                value = banner.get("value") if isinstance(banner, dict) else banner
                self.assertNotIn("Error loading", str(value or ""))
                self.assertNotIn("Unrecognized", str(value or ""))
                self.assertTrue(packed["state_steps"], "load produced no steps")
                self.assertTrue(packed["workflow_html"].strip())
                self.assertTrue(packed["raw_json"].strip())

    def test_a_missing_file_reports_an_error_instead_of_raising(self):
        """The Load button must survive a bad path: the UI has no other recovery."""
        from trajviz.insight.ui.load import do_load, load_slot_keys

        packed = do_load(str(_FIXTURE.parent / "does-not-exist.json"), False, "")
        self.assertEqual(frozenset(packed), load_slot_keys())
        banner = packed["summary_banner"]
        value = banner.get("value") if isinstance(banner, dict) else banner
        self.assertTrue(str(value or "").strip(), "a failed load must say so")
        self.assertEqual(packed["state_steps"], [])


if __name__ == "__main__":
    unittest.main()
