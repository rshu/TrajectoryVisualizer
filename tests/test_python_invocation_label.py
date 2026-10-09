"""A Python script read from stdin is charted as the interpreter, never as ``-``.

``python -`` reads its program from stdin; the heredoc form
``python3 - <<'EOF' ... EOF`` is how agents most often run inline scripts. The
label helper treats ``-c`` and a bare interpreter as "no stable script identity,
keep the interpreter name", but it read the lone ``-`` as a script path and
charted a tool literally named ``-``, which on Codex runs became one of the most
frequent "tools" in the frequency and duration charts.
"""

from __future__ import annotations

import unittest

from trajviz.insight.shell_cmd import tool_chart_name


def _bash(command: str) -> dict:
    return {"tool_name": "bash", "input": {"command": command}}


class PythonStdinLabelTests(unittest.TestCase):
    def test_stdin_scripts_keep_the_interpreter_name(self):
        cases = {
            "python -": "python",
            "python3 - <<'EOF'\nprint(1)\nEOF": "python3",
            "python -u -": "python",
            "bash -c 'python -'": "python",
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(tool_chart_name(_bash(command)), expected)

    def test_existing_labels_are_unchanged(self):
        cases = {
            "python -c 'print(1)'": "python",
            "python -m pytest -q": "pytest",
            "python scripts/run.py --flag": "run.py",
            "python": "python",
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(tool_chart_name(_bash(command)), expected)


if __name__ == "__main__":
    unittest.main()
