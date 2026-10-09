"""Pytest configuration for TrajViz.

Points DECAF failure-attribution at the vendored fixture (tests/fixtures/reference +
tests/fixtures/decaf_cache) so the integration tests run **hermetically** — no
dependency on a developer's local reference data or gitignored DECAF caches,
so they also run in CI. Locates DECAF via AWE_DECAF_PATH (default: a sibling
../DECAF checkout).

`TRAJVIZ_REQUIRE_ATTRIBUTION=1` (set in CI) turns a missing integration
environment into a hard failure instead of a silent skip — so the suite can never
be green merely because the attribution tests were skipped.
"""
import os
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_FIX = _TESTS / "fixtures"
_REFERENCE = _FIX / "reference"
_CACHE = _FIX / "decaf_cache" / "cache"

# 1. Reference root -> the vendored fixture, BEFORE attribution is imported
#    (attribution.py captures its default root once, at import).
if (_REFERENCE / "data" / "requirements").is_dir():
    os.environ["TRAJVIZ_REFERENCE_ROOT"] = str(_REFERENCE)
# FORCE (not setdefault) the judge-model namespace: the vendored verdicts live
# under deepseek-v4-pro, and an inherited shell AWE_JUDGE_MODEL would point the
# cache lookups at an empty namespace, silently changing what the golden tests
# exercise. Hermeticity beats the developer's environment here.
os.environ["AWE_JUDGE_MODEL"] = "deepseek-v4-pro"


def _decaf_root():
    env = os.environ.get("AWE_DECAF_PATH")
    for c in ([Path(env)] if env else []) + [_TESTS.parents[1] / "DECAF"]:
        if (c / "awe" / "__init__.py").is_file():
            return c
    return None


_DECAF = _decaf_root()
if _DECAF and str(_DECAF) not in sys.path:
    sys.path.insert(0, str(_DECAF))

# 2. Redirect DECAF's judge/arbiter caches to the vendored fixture caches, so the
#    arbiter-dependent tests need no developer-local (gitignored) caches — and
#    redirect the normtraj/trajsig DISK caches to a throwaway tmp dir so
#    fixture runs never write into (or thrash) DECAF's own caches.
_DECAF_OK = False
try:
    import atexit
    import shutil
    import tempfile
    from awe import config as _cfg
    import awe.arbiter as _arb
    import awe.adapters as _ad
    import awe.trajsignals as _ts
    # DECAF resolves its own repo-local INPUTS under config.PROJECT_DATA at call
    # time: the per-test grader records (pertest/) and the resolved-outcome
    # overrides (overrides/). Left pointing at DECAF's own data/, a fixture
    # run silently reads real records for whatever (agent, instance) it is handed,
    # so the golden cases stop being hermetic — measured: the deductive golden case
    # gains a ('code_editing', 'regression_introduced') fault that is derived from
    # a record living outside this fixture tree. Point PROJECT_DATA inside the
    # fixture tree, where cache/ already lives and pertest/ + overrides/ are
    # absent; both inputs degrade to "no record", which is what hermetic means here.
    _cfg.PROJECT_DATA = _FIX / "decaf_cache"
    if (_CACHE / "judge").is_dir():
        _cfg.JUDGE_CACHE_DIR = _CACHE / "judge"
        _arb.ARBITER_CACHE_DIR = _CACHE / "arbiter"
    _TMP_CACHE = Path(tempfile.mkdtemp(prefix="trajviz-decaf-cache-"))
    # The redirected caches are per-run throwaways: remove them at interpreter
    # exit so repeated suite runs do not leak tmp dirs (ignore_errors keeps
    # teardown from ever failing the run).
    atexit.register(shutil.rmtree, _TMP_CACHE, ignore_errors=True)
    _ad._NORMTRAJ_CACHE = _TMP_CACHE / "normtraj"
    _ts._CACHE = _TMP_CACHE / "trajsig"
    _DECAF_OK = True
except Exception:
    _DECAF_OK = False


def pytest_collection_finish(session):
    # No silent green: if CI demands the integration tests, the environment must
    # be ready (DECAF importable + reference fixture present) or the run fails here.
    if os.environ.get("TRAJVIZ_REQUIRE_ATTRIBUTION"):
        assert _DECAF_OK, ("TRAJVIZ_REQUIRE_ATTRIBUTION set but DECAF (awe) is not "
                           "importable — set AWE_DECAF_PATH to a DECAF checkout")
        assert (_REFERENCE / "data" / "requirements"
                / "astropy__astropy-13033.json").is_file(), \
            "TRAJVIZ_REQUIRE_ATTRIBUTION set but the reference fixture is missing"
