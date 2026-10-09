"""Phase 2 tests: the Attribution tab renderer + UI wiring.

- render:       build_attribution_html produces the scorecard + evidence panels
                for a real diagnosis, and a clean notice when degraded.
- construction: build_ui() constructs the whole Blocks (incl. the Attribution
                tab + its callback wiring) without error — catches component /
                callback mismatches.
"""
from dataclasses import asdict

import pytest

from trajviz.insight import attribution
from trajviz.insight.rendering import build_attribution_html

GOLD_AGENT = "claude_code"
GOLD_INST = "astropy__astropy-13033"


def _reference_present() -> bool:
    if not attribution.DECAF_AVAILABLE:
        return False
    from awe import config
    return config.requirements_path(GOLD_INST).is_file()


def test_render_degraded_shows_reason():
    out = build_attribution_html(
        {"available": False, "reason": "no gold reference for foo__bar-1"})
    assert "no gold reference for foo__bar-1" in out
    # no scorecard / fault panels when unavailable
    assert "score-dim-grid" not in out


@pytest.mark.skipif(not _reference_present(), reason="reference data not present")
def test_render_full_attribution():
    res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST)
    assert res.available
    out = build_attribution_html(asdict(res))
    # banner
    assert "Primary cause" in out
    assert "Code Editing" in out and "incorrect_patch" in out
    # scorecard grid with all seven capabilities present as cards
    assert "score-dim-grid" in out
    assert "Tool Use" in out and "Requirement Understanding" in out
    # evidence chain rendered as collapsible panels with a strength tier
    assert "judge-panel" in out
    assert "Deductive" in out


def test_build_ui_constructs_with_attribution_tab():
    # Constructing the Blocks exercises the Attribution tab declaration + the
    # attr_run_btn.click wiring; a mismatch (bad component ref, arity) raises here.
    from trajviz.insight.insight import build_ui
    app = build_ui()
    assert app is not None
