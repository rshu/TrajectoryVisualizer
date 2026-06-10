"""Regression: error panels must not leak tracebacks / paths / raw exception text."""

import unittest

from trajectory_visualizer.converge.app import _error_panel


class ErrorPanelLeakageTests(unittest.TestCase):
    def test_panel_is_generic_and_path_free(self):
        secret = "/Users/someone/.ssh/id_rsa"
        out = _error_panel("Comparison", FileNotFoundError(f"{secret} not found"))
        self.assertNotIn(secret, out)
        self.assertNotIn(".ssh", out)
        self.assertNotIn("Traceback", out)
        self.assertNotIn("<pre", out)
        self.assertIn("Comparison failed", out)

    def test_context_is_html_escaped(self):
        out = _error_panel("<img src=x onerror=alert(1)>", ValueError("x"))
        self.assertNotIn("<img", out)
        self.assertIn("&lt;img", out)


if __name__ == "__main__":
    unittest.main()
