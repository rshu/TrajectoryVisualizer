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

# Every ``@unittest.expectedFailure`` in the suite, as ``file.py::name``. Each
# entry is a defect the code is known to break and the test documents; see
# ``test_the_expected_failure_inventory_is_the_pinned_one``.
_PINNED_EXPECTED_FAILURES: list[str] = []


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

    def test_the_expected_failure_inventory_is_the_pinned_one(self):
        """``@unittest.expectedFailure`` is legitimate here, but must stay counted.

        The repo uses it as a *defect specification*: a test that states a
        contract the code currently breaks, with the defect's file:line in its
        docstring, and the decorator comes off in the same commit as the fix. An
        unexpected pass then fails the run, which is the point.

        The sibling check above cannot see it — ``expectedFailure`` is neither
        ``skip`` nor ``xfail`` — so without this the inventory could grow one
        decorator at a time and the suite would stay green while testing less.
        It is pinned rather than forbidden: adding one is allowed, and updating
        this list is the deliberate, reviewed act that records the decision.

        Empty as of the gap-remediation batch: the six that existed (four
        never-raise ingest violations, a stale cache-ratio assertion, and a live
        report-injection path) were fixed rather than re-pinned.
        """
        expected: list[str] = []
        for path in self.modules:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                for dec in node.decorator_list:
                    if _decorator_attr_path(dec).rsplit(".", 1)[-1] == "expectedFailure":
                        expected.append(f"{path.name}::{node.name}")

        self.assertEqual(
            sorted(expected), _PINNED_EXPECTED_FAILURES,
            "the @unittest.expectedFailure inventory changed. Fixing one? Remove it "
            "from _PINNED_EXPECTED_FAILURES. Adding one? Append it there, and put the "
            "defect's file:line plus the intended remedy in the test's docstring.",
        )


class EnvironmentGatedCoverageTests(unittest.TestCase):
    """Guards for the two ways coverage can shrink without a failing test."""

    # Test-function inventory of every module holding environment-gated tests.
    # Those tests only run where the optional DECAF integration is importable,
    # so whatever skips is invisible in a default run — which is exactly how a
    # skipped block quietly becomes an empty one. Pinning the count means the
    # modules cannot be deleted, renamed out of the convention, or un-gated
    # without someone editing this list on purpose. Not every test counted here
    # skips; the inventory is the guard, not the skip set.
    #
    # Measured by blocking the ``awe`` import and diffing the run: 37 tests go
    # from passing to skipped — 14 in test_attribution, 20 of the 21 in
    # test_attribution_live, 1 of the 3 in test_attribution_ui (its
    # reference-data check goes through DECAF's config), 1 of the 8 in
    # test_dsh_source_identity, and 1 in test_concurrency_isolation, which is
    # not pinned here because its other tests do not depend on DECAF.
    _GATED_MODULES = {
        "test_attribution.py": 14,
        "test_attribution_live.py": 21,
        "test_attribution_ui.py": 3,
        "test_dsh_source_identity.py": 8,
    }

    def test_the_decaf_gated_test_inventory_is_the_pinned_one(self):
        actual: dict[str, int] = {}
        for name in self._GATED_MODULES:
            path = _TESTS_DIR / name
            self.assertTrue(path.exists(), f"{name} is gone; update _GATED_MODULES")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            actual[name] = sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test")
            )
        self.assertEqual(
            actual, self._GATED_MODULES,
            "the environment-gated test inventory changed. These skip wherever the "
            "optional integration is absent, so a loss here is invisible in a normal "
            "run — update the pin deliberately.",
        )

    def test_no_package_directory_is_bytecode_only(self):
        """A directory with ``__pycache__`` but no source is a deleted module.

        Python cannot import from a bare ``__pycache__``, so such a directory is
        inert — but it reads as live code to anyone browsing the tree, and it is
        exactly what a half-finished deletion leaves behind.
        """
        pkg_root = _TESTS_DIR.parent / "trajviz"
        stale = [
            str(d.relative_to(pkg_root.parent))
            for d in pkg_root.rglob("*")
            if d.is_dir() and d.name != "__pycache__"
            and (d / "__pycache__").is_dir()
            and not any(d.glob("*.py"))
        ]
        self.assertEqual(stale, [], f"bytecode-only package directories: {stale}")


if __name__ == "__main__":
    unittest.main()
