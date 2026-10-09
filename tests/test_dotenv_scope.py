"""The dashboard's ``.env`` may configure the analysis LLM, and nothing else.

``load_env_files`` runs when the dashboard starts and reads ``.env`` from the
launch directory. It used to accept every key, so launching TrajViz inside an
untrusted folder (an unpacked trajectory dataset, say) let that folder's
``.env`` set ``AWE_DECAF_PATH`` — which decides what code the Attribution tab
imports — or any other variable the process reads. The documented purpose
(README, ``.env.example``) is the ``ANALYZE_*`` / ``LABEL_*`` LLM settings, so
that is all it loads now.

A redirected ``ANALYZE_BASE_URL`` is still possible by design, so the sidebar
status names the endpoint host the trajectory would be sent to. And the
"repo-root" ``.env`` is read only from a source checkout: in an installed
wheel, ``parents[2]`` of the module is ``site-packages``.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trajviz.insight import llm_config
from trajviz.insight.llm_config import config_status_html, load_env_files


class DotenvScopeTests(unittest.TestCase):
    def _load(self, text: str, *, repo_root: Path | None = None) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text(text, encoding="utf-8")
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                with patch.object(llm_config, "_REPO_ROOT", repo_root or Path(tmp) / "no-checkout"), \
                        patch.dict(os.environ, {}, clear=True):
                    load_env_files()
                    return dict(os.environ)
            finally:
                os.chdir(previous)

    def test_only_analysis_and_labeler_keys_are_loaded(self):
        env = self._load(
            "ANALYZE_MODEL=m\nLABEL_API_KEY=k\nAWE_DECAF_PATH=/tmp/evil\n"
            "TRAJVIZ_REFERENCE_ROOT=/tmp/x\nPYTHONPATH=/tmp/p\nTRAJVIZ_DSH_EXPORT_ROOT=/\n"
        )
        self.assertEqual(env.get("ANALYZE_MODEL"), "m")
        self.assertEqual(env.get("LABEL_API_KEY"), "k")
        for key in ("AWE_DECAF_PATH", "TRAJVIZ_REFERENCE_ROOT", "PYTHONPATH", "TRAJVIZ_DSH_EXPORT_ROOT"):
            self.assertNotIn(key, env)

    def test_repo_root_env_is_read_only_from_a_source_checkout(self):
        with tempfile.TemporaryDirectory() as fake_site_packages:
            Path(fake_site_packages, ".env").write_text("ANALYZE_MODEL=from-site-packages\n", encoding="utf-8")
            env = self._load("", repo_root=Path(fake_site_packages))
            self.assertNotIn("ANALYZE_MODEL", env)

    def test_status_names_the_endpoint_host(self):
        env = {"ANALYZE_BASE_URL": "https://llm.example.invalid/v1", "ANALYZE_API_KEY": "sk-x",
               "ANALYZE_MODEL": "m"}
        with patch.dict(os.environ, env, clear=True):
            html_out = config_status_html(loaded_steps=3)
        self.assertIn("llm.example.invalid", html_out)
        self.assertNotIn("sk-x", html_out)


if __name__ == "__main__":
    unittest.main()
