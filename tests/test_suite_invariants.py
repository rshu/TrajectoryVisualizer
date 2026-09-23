"""Invariants about the test suite itself.

The suite grew 199 -> 602 tests in the PR #13 fork sync and is now edited by
several people at once, mostly by appending to existing files. That makes three
specific failure modes likely, and all three are *invisible*: the suite stays
green while quietly testing less than it appears to.

  * **Shadowed tests.** Two ``def test_x`` in one class is legal Python — the
    second binding silently replaces the first, so the earlier test is deleted
    without a diff marker, a collection error, or a change in the pass count.
  * **Files that collect nothing.** A module whose tests were renamed out of the
    ``test_*`` convention (or a file added with only helpers) reports no failure
    at all; its subject stops being covered while the file still looks tested.
  * **Unconditionally disabled tests.** A bare ``@unittest.skip`` / ``skip`` /
    ``xfail`` marker turns a failing test into a green run forever. Conditional
    skips (``skipUnless``/``skipIf``, which guard optional fixtures and the
    DECAF integration) are legitimate and are deliberately allowed.

These are checked by parsing the test sources rather than importing them, so
the check itself has no import side effects and no dependence on which optional
backends are installed.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent

# Markers that disable a test outright, as opposed to guarding it on a
# condition the environment can satisfy.
_UNCONDITIONAL_SKIP_ATTRS = {"skip", "xfail"}


def _test_modules() -> list[Path]:
    return sorted(p for p in _TESTS_DIR.glob("test_*.py") if p.is_file())


def _decorator_attr_path(node: ast.expr) -> str:
    """Dotted name of a decorator, ignoring any call arguments."""
    if isinstance(node, ast.Call):
        node = node.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class SuiteInvariantTests(unittest.TestCase):
    """Meta-tests: the suite must not silently shrink."""

    def setUp(self) -> None:
        self.modules = _test_modules()
        self.assertTrue(self.modules, f"no test modules found under {_TESTS_DIR}")

    def test_no_test_is_shadowed_by_a_later_definition_of_the_same_name(self):
        """Re-defining a test name deletes the earlier test with no signal.

        The pass count does not drop (the name still runs once), nothing errors,
        and review diffs show only an addition — so the lost assertions are
        found, if ever, by a bug reaching production.
        """
        collisions: list[str] = []

        def scan(body: list[ast.stmt], scope: str, path: Path) -> None:
            seen: dict[str, int] = {}
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name in seen:
                        collisions.append(
                            f"{path.name}:{node.lineno} {scope}.{node.name} "
                            f"redefines the one at line {seen[node.name]}"
                        )
                    seen[node.name] = node.lineno
                if isinstance(node, ast.ClassDef):
                    scan(node.body, f"{scope}.{node.name}", path)

        for path in self.modules:
            scan(ast.parse(path.read_text(encoding="utf-8")).body, path.stem, path)

        self.assertEqual(collisions, [], "shadowed test/class definitions: " + "; ".join(collisions))

    def test_every_test_module_defines_at_least_one_test(self):
        """A module that collects nothing is a hole that reports as healthy.

        pytest says nothing about a file with zero ``test_*`` callables, so the
        subject it was written for silently stops being exercised.
        """
        empty: list[str] = []
        for path in self.modules:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            has_test = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test")
                for node in ast.walk(tree)
            )
            if not has_test:
                empty.append(path.name)
        self.assertEqual(empty, [], f"test modules that collect no tests: {empty}")

    def test_no_test_is_disabled_unconditionally(self):
        """``@unittest.skip`` / ``@pytest.mark.skip`` / ``xfail`` never expires.

        Conditional guards (``skipUnless``/``skipIf``) are fine — they run as
        soon as the environment provides the fixture. A bare marker makes the
        test permanently green, which is indistinguishable from passing.
        """
        disabled: list[str] = []
        for path in self.modules:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                for dec in node.decorator_list:
                    dotted = _decorator_attr_path(dec)
                    tail = dotted.rsplit(".", 1)[-1]
                    if tail in _UNCONDITIONAL_SKIP_ATTRS and (
                        dotted.startswith("unittest.") or "mark." in dotted or dotted == tail
                    ):
                        disabled.append(f"{path.name}:{node.lineno} {node.name} @{dotted}")
        self.assertEqual(
            disabled, [],
            "tests disabled unconditionally (use skipIf/skipUnless, or delete them): "
            + "; ".join(disabled),
        )


if __name__ == "__main__":
    unittest.main()
