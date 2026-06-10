"""Regression: do_load must not crash on valid-JSON-but-wrong-shape input.

A top-level list/scalar (a common heterogeneous-export shape) used to raise an
uncaught AttributeError inside detect_format, surfacing as a generic Gradio error
with nothing logged. do_load now wraps the load and returns a friendly banner.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def _find_do_load(app):
    fns = app.fns.values() if hasattr(app.fns, "values") else app.fns
    for f in fns:
        fn = getattr(f, "fn", None)
        if fn is not None and getattr(fn, "__name__", "") == "do_load":
            return fn
    return None


class DoLoadRobustnessTests(unittest.TestCase):
    def test_malformed_trajectory_returns_banner_not_crash(self):
        from trajectory_visualizer.insight.insight import build_ui

        app = build_ui()
        do_load = _find_do_load(app)
        self.assertIsNotNone(do_load, "do_load callback not found in built UI")

        tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        try:
            json.dump([1, 2, 3], tf)  # valid JSON, wrong shape (top-level list)
            tf.close()
            result = do_load(tf.name, False, "ccsession")  # must not raise
        finally:
            os.unlink(tf.name)

        self.assertIsInstance(result, tuple)
        # The summary-banner slot should carry a load-failure message. The
        # loader now rejects a wrong-shape top-level value with a precise
        # "_error" ("Expected a JSON object …") handled by do_load's _error
        # branch; if it ever crashes instead, the generic banner is shown.
        # Either is acceptable here — the point is "no crash, shows an error".
        self.assertTrue(
            any(
                isinstance(x, str)
                and ("Could not load this trajectory" in x
                     or "Expected a JSON object" in x
                     or "top level" in x)
                for x in result
            ),
            "expected a load-failure banner in do_load output",
        )


if __name__ == "__main__":
    unittest.main()
