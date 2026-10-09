"""Import-surface guards for the post-PR-#13 package split.

PR #13 split ``insight.py``/``charts.py`` into the ``ui/``, ``presenters/``,
``formats/`` and ``charts/`` packages.  Most of the new modules are reached only
from inside a Gradio callback (``ui/attribution_tab.py`` imports
``trajviz.insight.attribution`` inside ``on_diagnose``, ``ui/overview_tab.py``
imports the judge inside its handler, ...), so a stale or circular import in one
of them cannot fail the test suite — it fails the first time a user clicks.  A
refactor of this size is exactly the change that leaves such a module behind.

Two invariants are pinned here:

1. *Every* module under ``trajviz/`` and ``scripts/`` imports cleanly, whether
   or not any other test happens to reach it.
2. Importing them is inert.  Importing a library must not ingest configuration
   from the process launch directory (``llm_config.load_env_files`` documents
   this explicitly: "Call from the dashboard entry / ``build_ui``, not at import
   time"), must not open a network connection, and must not write files next to
   the caller's CWD.  A module that reads ``.env`` at import time would silently
   arm the LLM egress paths for anything that merely imports TrajViz — including
   ``pytest`` collection.

Two further invariants guard the *inputs* to that surface — where imports resolve
from, and what the package claims to need:

3. The suite's ``sys.path`` and ``rootdir`` come from a declaration in this
   repository's ``pyproject.toml``, not from a parent directory and not from an
   install-mode accident.  TrajViz may be checked out inside a larger tree that
   has its own ``[tool.pytest.ini_options]``, and ``scripts`` is an implicit
   namespace package that neighbouring checkouts may also provide.
4. Every declared runtime dependency is either imported by shipped code or
   carries a comment saying why it is declared anyway.  An unimported,
   unexplained requirement is indistinguishable from a leftover.

All tests are hermetic: no real data, no network, no dependence on the user's home
directory.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import tomllib
import traceback
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys written into the throwaway launch-directory ``.env`` the probe runs in.
# ``AWE_DECAF_PATH`` is deliberately among them: honouring it at import time
# would prepend a launch-directory-controlled path to ``sys.path``.
_DOTENV_KEYS = {
    "ANALYZE_API_KEY": "sk-import-surface-sentinel",
    "ANALYZE_BASE_URL": "https://import-surface.invalid/v1",
    "ANALYZE_MODEL": "import-surface-model",
    "LABEL_API_KEY": "sk-import-surface-label",
    "TRAJVIZ_IMPORT_SURFACE_SENTINEL": "leaked",
    "AWE_DECAF_PATH": "",  # filled in with a path inside the temp launch dir
}

# No module may write the process environment at import. The DECAF bridge reads
# its settings once at import and configures DECAF through its attributes, not
# through environment variables, so there is nothing to allow here.

_PROBE = '''
"""Import every trajviz/scripts module and report what the imports did."""
import json
import os
import pathlib
import socket
import sys
import traceback

repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo))
cwd = pathlib.Path.cwd()
files_before = sorted(p.name for p in cwd.iterdir())
path_before = list(sys.path)
env_before = dict(os.environ)
network = []


def _blocked(*args, **kwargs):
    network.append(repr(args)[:160])
    raise OSError("network access blocked by the import-surface probe")


socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

modules = []
for py in sorted(repo.glob("trajviz/**/*.py")) + sorted(repo.glob("scripts/*.py")):
    parts = list(py.relative_to(repo).parts)
    parts = parts[:-1] if parts[-1] == "__init__.py" else parts[:-1] + [parts[-1][:-3]]
    if parts:
        modules.append(".".join(parts))

failures = {}
for name in modules:
    try:
        __import__(name)
    except BaseException:
        failures[name] = traceback.format_exc()[-800:]

print(json.dumps({
    "modules": modules,
    "failures": failures,
    "env_added": {k: v for k, v in os.environ.items() if k not in env_before},
    "env_changed": sorted(k for k, v in os.environ.items() if k in env_before and env_before[k] != v),
    "new_files": [p.name for p in cwd.iterdir() if p.name not in files_before],
    "new_sys_path": [p for p in sys.path if p not in path_before],
    "network": network,
}))
'''


def _discover_modules() -> list[str]:
    """Dotted names for every module under ``trajviz/`` and ``scripts/``."""
    names: list[str] = []
    for py in sorted(_REPO_ROOT.glob("trajviz/**/*.py")) + sorted(_REPO_ROOT.glob("scripts/*.py")):
        parts = list(py.relative_to(_REPO_ROOT).parts)
        parts = parts[:-1] if parts[-1] == "__init__.py" else parts[:-1] + [parts[-1][:-3]]
        if parts:
            names.append(".".join(parts))
    return names


class ModuleImportTests(unittest.TestCase):
    """Every shipped module must import, including the lazily-imported ones."""

    def test_discovery_finds_the_whole_package(self):
        """A broken walk would make the import test vacuously green."""
        names = _discover_modules()
        self.assertGreater(len(names), 60, f"only found {len(names)} modules — the package walk is broken")
        for expected in (
            "trajviz.insight.insight",
            "trajviz.insight.attribution",      # imported inside on_diagnose only
            "trajviz.insight.issue_judge",      # imported inside the Issues handler only
            "trajviz.insight.charts.label_charts",
            "trajviz.insight.formats.dsh",
            "trajviz.converge.cli",
            "scripts.cursor_consolidator",
        ):
            self.assertIn(expected, names)

    def test_every_module_imports(self):
        """A module only reachable from a Gradio callback must still import.

        Nothing else in the suite executes ``converge/cli.py``,
        ``charts/label_charts.py`` or ``ui/attribution_tab.py``'s lazy imports,
        so without this a NameError or a bad relative import in any of them ships
        green and raises on the user's first click.
        """
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        failures: dict[str, str] = {}
        for name in _discover_modules():
            try:
                __import__(name)
            except Exception:
                failures[name] = traceback.format_exc(limit=6)
        self.assertEqual(
            failures, {},
            "modules failed to import:\n" + "\n".join(f"--- {k}\n{v}" for k, v in failures.items()),
        )


class ImportSideEffectTests(unittest.TestCase):
    """Importing TrajViz must be inert: no config ingest, no egress, no writes."""

    @classmethod
    def setUpClass(cls):
        cls._launch = tempfile.TemporaryDirectory(prefix="trajviz-launch-")
        cls._probe_dir = tempfile.TemporaryDirectory(prefix="trajviz-probe-")
        launch = Path(cls._launch.name)
        dotenv = dict(_DOTENV_KEYS)
        dotenv["AWE_DECAF_PATH"] = str(launch / "fake-decaf")
        cls.dotenv = dotenv
        (launch / ".env").write_text(
            "".join(f"{k}={v}\n" for k, v in dotenv.items()), encoding="utf-8",
        )
        probe = Path(cls._probe_dir.name) / "import_probe.py"
        probe.write_text(_PROBE, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(probe), str(_REPO_ROOT)],
            cwd=str(launch),
            capture_output=True,
            text=True,
            timeout=600,
            # Keep the child out of the parent's pytest/coverage plumbing and stop
            # it writing __pycache__ into the launch directory.
            env={"PATH": "/usr/bin:/bin", "HOME": str(launch), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        cls.proc = proc
        cls.report = json.loads(proc.stdout.splitlines()[-1]) if proc.returncode == 0 and proc.stdout else None

    @classmethod
    def tearDownClass(cls):
        cls._launch.cleanup()
        cls._probe_dir.cleanup()

    def setUp(self):
        if self.report is None:
            self.fail(f"import probe failed (rc={self.proc.returncode}):\n{self.proc.stderr[-2000:]}")

    def test_probe_imported_the_whole_package(self):
        """Guards the other assertions: they mean nothing if nothing imported."""
        self.assertGreater(len(self.report["modules"]), 60)
        self.assertEqual(self.report["failures"], {}, "modules failed to import in a clean interpreter")

    def test_import_does_not_read_the_launch_directory_dotenv(self):
        """Import time must not ingest ``.env`` from wherever the process was started.

        ``llm_config.load_env_files`` is fill-only and belongs to ``build_ui``.
        If any module started calling it (or its own dotenv reader) at import,
        merely importing TrajViz — or collecting this test suite — would arm
        ``ANALYZE_*``/``LABEL_*`` and point ``AWE_DECAF_PATH`` at a directory
        chosen by whoever owns the launch directory.
        """
        leaked = sorted(k for k in self.dotenv if k in self.report["env_added"])
        self.assertEqual(leaked, [], f"import read the launch directory's .env: {leaked}")
        self.assertEqual(self.report["env_changed"], [], "import overwrote existing environment variables")
        fake_decaf = self.dotenv["AWE_DECAF_PATH"]
        self.assertNotIn(
            fake_decaf, self.report["new_sys_path"],
            "import put a launch-directory-controlled path on sys.path",
        )

    def test_import_writes_no_environment_variables(self):
        """Importing TrajViz must not change the process environment.

        A module writing process-global environment state at import is an
        invisible coupling between "imported TrajViz" and "changed the process".
        """
        added = sorted(self.report["env_added"])
        self.assertEqual(added, [], f"import set environment variables: {added}")

    def test_import_makes_no_network_call(self):
        """No module may phone home (telemetry, model probe, CDN check) on import."""
        self.assertEqual(self.report["network"], [], "import attempted a network connection")

    def test_import_writes_nothing_into_the_launch_directory(self):
        """Importing a library must not drop caches, logs or exports next to the CWD."""
        self.assertEqual(self.report["new_files"], [], "import created files in the process CWD")


def _declared_dependencies(pyproject: Path) -> list[str]:
    return list(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"])


def _requirement_name(spec: str) -> str:
    """``gradio_client>=2.5.0`` -> ``gradio_client`` (also drops extras/markers)."""
    return re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0].strip()


def _undocumented_dependencies(pyproject: Path, sources: list[Path]) -> list[str]:
    """Declared runtime dependencies that are neither imported nor commented.

    Takes the file as an argument so the check can be run against any revision of
    ``pyproject.toml``, not only the working tree's.
    """
    text = "\n".join(p.read_text(encoding="utf-8") for p in sources)
    lines = pyproject.read_text(encoding="utf-8").splitlines()
    offenders = []
    for spec in _declared_dependencies(pyproject):
        module = _requirement_name(spec).replace("-", "_")
        if re.search(rf"^\s*(?:import\s+{re.escape(module)}\b|from\s+{re.escape(module)}[\s.])", text, re.M):
            continue
        # Not imported: the declaration needs a reason, on the line(s) above it.
        idx = next((i for i, line in enumerate(lines) if line.strip().strip(",").strip('"') == spec), None)
        annotated = idx is not None and idx > 0 and lines[idx - 1].lstrip().startswith("#")
        if not annotated:
            offenders.append(spec)
    return offenders


class DeclaredDependencyTests(unittest.TestCase):
    """What the package says it needs must match what it uses — and requirements.txt."""

    PYPROJECT = _REPO_ROOT / "pyproject.toml"
    REQUIREMENTS = _REPO_ROOT / "requirements.txt"

    def _sources(self) -> list[Path]:
        return sorted(_REPO_ROOT.glob("trajviz/**/*.py")) + sorted(_REPO_ROOT.glob("scripts/*.py"))

    def test_requirements_txt_mirrors_project_dependencies(self):
        """requirements.txt claims to mirror ``[project.dependencies]``; hold it to that.

        Two hand-maintained lists of the same thing drift silently — and
        ``requirements.txt`` is the install path for anyone not using uv, so a
        drift there is a missing dependency at someone else's runtime, not here.
        """
        declared = {spec.replace(" ", "") for spec in _declared_dependencies(self.PYPROJECT)}
        pinned = set()
        for raw in self.REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            spec = raw.split("#", 1)[0].strip().replace(" ", "")
            if spec:
                pinned.add(spec)
        self.assertEqual(
            pinned, declared,
            "requirements.txt and [project.dependencies] disagree:\n"
            f"  only in requirements.txt: {sorted(pinned - declared)}\n"
            f"  only in pyproject.toml:   {sorted(declared - pinned)}",
        )

    def test_every_declared_dependency_is_imported_or_annotated(self):
        """An unimported requirement must say why it is there.

        ``pandas`` is the precedent for a legitimate one: nothing needs its API,
        but ``charts/_layout.py`` pre-imports it to dodge a plotly circular
        import under Gradio's async threads, and says so. Without this test the
        next dependency that stops being used just stays, and nobody can tell it
        apart from one that is load-bearing for a reason off the import graph.
        """
        offenders = _undocumented_dependencies(self.PYPROJECT, self._sources())
        self.assertEqual(
            offenders, [],
            "declared, never imported, and unexplained: "
            f"{offenders} — import it, drop it, or comment the line above it",
        )


def test_pytest_rootdir_and_path_come_from_this_repository(pytestconfig):
    """Function-style because it needs pytest's resolved config (``pytestconfig``).

    Asserts the three consequences of this repo owning its own
    ``[tool.pytest.ini_options]``: rootdir is here, nothing is inherited from a
    parent directory's pytest config, and the ``scripts`` import path is declared rather
    than supplied by an editable-install ``.pth``, by ``python -m``'s CWD
    insertion, or by a ``PYTHONPATH`` env var in CI.
    """
    assert Path(pytestconfig.rootpath) == _REPO_ROOT, (
        f"rootdir is {pytestconfig.rootpath}, not {_REPO_ROOT} — pytest walked up out of this repository"
    )
    assert Path(pytestconfig.inipath or "") == _REPO_ROOT / "pyproject.toml"
    # Read the table out of whichever file pytest actually resolved. `asyncio_mode`
    # comes from a parent directory (pytest-asyncio is not installed here), so finding it means
    # this suite is running on someone else's pytest configuration — which pytest
    # only reports as a PytestConfigWarning about an unknown option.
    ini_table = tomllib.loads(Path(pytestconfig.inipath).read_text(encoding="utf-8"))
    ini_table = ini_table.get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert "asyncio_mode" not in ini_table, f"inherited pytest config from {pytestconfig.inipath}"
    # getini resolves `pythonpath` entries against rootdir, so "." arrives as a Path.
    assert _REPO_ROOT in pytestconfig.getini("pythonpath"), 'pyproject.toml must declare pythonpath = ["."]'


def test_the_scripts_modules_resolve_inside_this_repository():
    """``scripts`` is a namespace package that sibling trees also define.

    This is not hypothetical: ``trajviz.insight.attribution`` puts the DECAF
    checkout's parent on ``sys.path`` to import ``awe``, and if that checkout also
    has a ``scripts/`` directory then ``scripts.__path__`` really does contain it
    ahead of ours.
    No module name collides today, so resolution is still correct — which is
    exactly why this needs a test rather than a comment: the first colliding
    filename would redirect ``from scripts import X`` into another project with no
    error at all.  Assert on resolution, not on ``__path__``, because the portion
    list depends on whether attribution has been imported yet.
    """
    import importlib

    for name in ("_common", "cursor_consolidator", "opencode_consolidator", "step_labeler", "step_labeler_v2"):
        module = importlib.import_module(f"scripts.{name}")
        resolved = Path(module.__file__).resolve()
        assert resolved == _REPO_ROOT / "scripts" / f"{name}.py", f"scripts.{name} resolved to {resolved}"


if __name__ == "__main__":
    unittest.main()
