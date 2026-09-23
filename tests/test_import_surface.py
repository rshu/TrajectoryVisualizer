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

Both tests are hermetic: no corpus, no network, no dependence on the user's home
directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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

# The DECAF bridge captures its corpus root and judge-model namespace once at
# import (``trajviz/insight/attribution.py`` lines 82 and 86, both
# ``os.environ.setdefault``).  That is the only sanctioned import-time
# environment write; anything outside this prefix is a new side effect.
_ALLOWED_ENV_PREFIX = "AWE_"

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

    def test_import_time_environment_writes_stay_in_the_decaf_namespace(self):
        """Only the DECAF bridge may set environment defaults at import.

        ``attribution.py`` captures ``AWE_ARGUS_ROOT``/``AWE_JUDGE_MODEL`` once at
        import by design (tests/conftest.py depends on that ordering).  Any other
        module writing process-global environment state at import is a new,
        invisible coupling between "imported TrajViz" and "changed the process".
        """
        stray = sorted(k for k in self.report["env_added"] if not k.startswith(_ALLOWED_ENV_PREFIX))
        self.assertEqual(stray, [], f"import set unexpected environment variables: {stray}")

    def test_import_makes_no_network_call(self):
        """No module may phone home (telemetry, model probe, CDN check) on import."""
        self.assertEqual(self.report["network"], [], "import attempted a network connection")

    def test_import_writes_nothing_into_the_launch_directory(self):
        """Importing a library must not drop caches, logs or exports next to the CWD."""
        self.assertEqual(self.report["new_files"], [], "import created files in the process CWD")


if __name__ == "__main__":
    unittest.main()
