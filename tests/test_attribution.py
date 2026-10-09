"""Tests for the DECAF attribution seam (trajviz/insight/attribution.py).

Run hermetically against the vendored fixture (tests/fixtures/reference +
tests/fixtures/decaf_cache — wired by conftest.py), so they need no developer
reference data and run in CI. Golden expectations are PINNED LITERALS (not read from
DECAF's dossier artifacts), so a drift in DECAF's diagnosis fails loudly here.

  * golden       — diagnose() reproduces the pinned v0.33 diagnosis for a
                   deductive case and an arbiter-refuted case.
  * identity     — a displayed file that is not byte-identical to the canonical
                   run's trajectory is refused (no provenance mixing, no silent
                   fallback); a byte-identical copy (Gradio upload) passes.
  * offline      — works with no OPENROUTER_API_KEY (reads cached verdicts).
  * degradation  — missing gold / ids degrade cleanly; never raise or fabricate.
  * root-switch  — configure() clears gold AND outcome caches (load_outcomes).
  * semantics    — blamed iff weight>0; unassessed judge caps are n/a, not clean;
                   a card's tier belongs to its displayed top_error.
  * parity       — DECAF's adapter on the raw file == DECAF's load_normalized.

Skipped only when DECAF itself is absent (the standalone CI job); the
integration job sets TRAJVIZ_REQUIRE_ATTRIBUTION=1 so absence fails hard.
"""
import json
import shutil

import pytest

from trajviz.insight import attribution

GOLD_AGENT = "claude_code"
GOLD_INST = "astropy__astropy-13033"            # deductive-only case
ARB_AGENT, ARB_INST = "claude_code", "django__django-11477"  # arbiter-refuted case

# ---- pinned v0.34 golden expectations (do not read from DECAF artifacts) ----
GOLD_EXPECT = {
    "blame_status": "primary",
    "primary": {"capability": "code_editing", "error_type": "incorrect_patch"},
    "fault_set": {("code_editing", "incorrect_patch"),
                  ("code_verification", "test_expectation_rewritten"),
                  ("self_repair_loop", "repeated_ineffective_attempt")},
}
ARB_EXPECT = {"blame_status": "refuted_unattributed", "primary": None}

pytestmark = pytest.mark.skipif(
    not attribution.DECAF_AVAILABLE, reason="DECAF (awe) not importable")


def _traj_path(agent, inst):
    from awe import config
    return str(config.TRAJECTORY_DIR / agent / f"{inst}.json")


def _reference_present() -> bool:
    if not attribution.DECAF_AVAILABLE:
        return False
    from awe import config
    return config.requirements_path(GOLD_INST).is_file()


requires_reference = pytest.mark.skipif(
    not _reference_present(), reason="reference fixture not present")


# --------------------------------------------------------------- golden
@requires_reference
def test_golden_deductive_case_matches_pinned_expectations():
    res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                               source_path=_traj_path(GOLD_AGENT, GOLD_INST),
                               fmt="ccsession")
    assert res.available is True
    assert res.mode == "reference"
    assert res.blame_status == GOLD_EXPECT["blame_status"]
    assert res.primary == GOLD_EXPECT["primary"]
    assert {(f["capability"], f["error_type"]) for f in res.faults} == \
        GOLD_EXPECT["fault_set"]
    for f in res.faults:
        chain = f.get("evidence_chain") or {}
        assert chain.get("strength") in {"deductive", "associational", "model_inferred"}
        assert isinstance(chain.get("links"), list) and chain["links"]


@requires_reference
def test_golden_arbiter_refuted_case_is_not_blamed():
    """Pins the vendored model namespace (deepseek-v4-pro caches) + refuted-not-blamed
    semantics: under a wrong judge-model namespace the arbiter would be inert and
    this case would show a blamed primary."""
    res = attribution.diagnose(agent=ARB_AGENT, instance_id=ARB_INST,
                               source_path=_traj_path(ARB_AGENT, ARB_INST),
                               fmt="ccsession")
    assert res.available
    assert res.notes == []               # vendored verdicts are provenance-stamped
    assert res.blame_status == ARB_EXPECT["blame_status"]
    assert res.primary is None
    assert not any(s.blamed for s in res.scorecard)
    # judge verdict is vendored for this case -> judge assessment is PER-CAP:
    # each judge capability is assessed only when ITS opportunity opened (DECAF
    # can judge requirement_understanding while task_planning is out of scope)
    assert res.used_judge is True
    by_cap = {s.capability: s for s in res.scorecard}
    from awe.detect import detect
    opp = detect(ARB_INST, ARB_AGENT)["opportunities"]
    for cap in ("requirement_understanding", "task_planning"):
        assert by_cap[cap].assessed == bool(opp.get(cap))
        assert by_cap[cap].blamed is False
    # the arbiter verdict is surfaced for the UI to render
    assert (res.arbiter or {}).get("applied") == "refuted_unattributed"


@requires_reference
def test_unverifiable_llm_verdicts_are_disabled(tmp_path, monkeypatch):
    """A cached judge/arbiter verdict whose trajectory_sha256 does not match the
    current canonical trajectory must NOT shape the diagnosis (a stale verdict
    could otherwise refute a different trajectory citing nonexistent steps)."""
    import awe.arbiter as _arb
    from awe import config as _cfg

    jdst = tmp_path / "judge"
    adst = tmp_path / "arbiter"
    shutil.copytree(_cfg.JUDGE_CACHE_DIR, jdst)
    shutil.copytree(_arb.ARBITER_CACHE_DIR, adst)
    for p in list(jdst.rglob("*.json")) + list(adst.rglob("*.json")):
        rec = json.loads(p.read_text())
        rec["trajectory_sha256"] = "0" * 64
        p.write_text(json.dumps(rec))
    monkeypatch.setattr(_cfg, "JUDGE_CACHE_DIR", jdst)
    monkeypatch.setattr(_arb, "ARBITER_CACHE_DIR", adst)

    res = attribution.diagnose(agent=ARB_AGENT, instance_id=ARB_INST)
    assert res.available
    # LLM layers disabled -> the rule-elected omission primary STANDS (no
    # arbiter refutation), and the mismatch is noted for the UI
    assert res.blame_status == "primary"
    assert res.notes and any("provenance" in n for n in res.notes)
    assert res.used_judge is False


@requires_reference
def test_changed_gold_invalidates_llm_verdicts(tmp_path):
    """Verdict provenance covers ALL prompt inputs: a byte-identical trajectory
    with a CHANGED task/gold must not reuse the old judge/arbiter verdicts."""
    # clone the reference fixture, then mutate ONLY the requirements (task) file
    root = tmp_path / "reference"
    shutil.copytree(attribution._DEFAULT_ROOT / "data", root / "data")
    rp = root / "data" / "requirements" / f"{ARB_INST}.json"
    req = json.loads(rp.read_text())
    req["problem_statement"] = "A COMPLETELY DIFFERENT TASK STATEMENT."
    rp.write_text(json.dumps(req))

    res = attribution.diagnose(agent=ARB_AGENT, instance_id=ARB_INST,
                               reference_root=root)
    assert res.available
    # old verdicts unverifiable for the changed task -> LLM layers disabled,
    # the rule-elected primary stands, and the mismatch is noted
    assert res.notes and any("provenance" in n for n in res.notes)
    assert res.blame_status == "primary"
    assert res.used_judge is False


@requires_reference
def test_toctou_mutated_canonical_file_is_refused(tmp_path):
    """Content identity, not path identity: if the reference file changes AFTER the
    UI captured the displayed bytes' hash, diagnosis must refuse (the verdict
    would describe different bytes than the ones shown)."""
    root = tmp_path / "reference"
    shutil.copytree(attribution._DEFAULT_ROOT / "data", root / "data")
    tpath = root / "data" / "trajectory" / "claude_code" / f"{GOLD_INST}.json"
    import hashlib
    loaded_sha = hashlib.sha256(tpath.read_bytes()).hexdigest()  # captured at "load"
    # the canonical file mutates after load
    raw = json.loads(tpath.read_text())
    raw["_mutated_after_load"] = True
    tpath.write_text(json.dumps(raw))

    res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                               expected_sha=loaded_sha, reference_root=root)
    assert res.available is False
    assert "changed after it was loaded" in (res.reason or "") or \
           "does not match the canonical" in (res.reason or "")


def test_path_traversal_instance_id_is_rejected():
    """instance_id must never reach filesystem path construction: traversal is
    rejected up front with a uniform message (no file-existence oracle)."""
    for evil in ("../../../../etc/passwd", "..%2Fetc", "a/../b", "x/y", ".hidden"):
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=evil)
        assert res.available is False
        assert "invalid instance id" in (res.reason or ""), evil
        assert res.faults == []


# --------------------------------------------------------------- identity
@requires_reference
def test_byte_identical_copy_passes_identity_check(tmp_path):
    # a Gradio upload is a verbatim copy at a temp path with a hash parent dir
    copy = tmp_path / "upload.json"
    shutil.copyfile(_traj_path(GOLD_AGENT, GOLD_INST), copy)
    res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                               source_path=str(copy), fmt="ccsession")
    assert res.available is True
    assert res.blame_status == GOLD_EXPECT["blame_status"]


@requires_reference
def test_mismatched_displayed_trajectory_is_refused(tmp_path):
    """A displayed file that differs from the canonical run must DEGRADE (its
    patch/outcome belong to a different execution) — never silently diagnose the
    reference copy, never fabricate."""
    doctored = tmp_path / "doctored.json"
    with open(_traj_path(GOLD_AGENT, GOLD_INST)) as f:
        raw = json.load(f)
    raw["_tampered"] = True
    doctored.write_text(json.dumps(raw))
    res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                               source_path=str(doctored), fmt="ccsession")
    assert res.available is False
    assert "does not match the canonical" in (res.reason or "")
    assert res.faults == [] and res.primary is None


# --------------------------------------------------------------- offline
@requires_reference
def test_diagnose_offline_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    res = attribution.diagnose(agent=ARB_AGENT, instance_id=ARB_INST)
    assert res.available is True
    assert res.blame_status == ARB_EXPECT["blame_status"]


# --------------------------------------------------------------- degradation
def test_gold_free_upload_degrades_cleanly():
    res = attribution.diagnose(agent=GOLD_AGENT,
                               instance_id="nonexistent__instance-99999")
    assert res.available is False
    assert res.mode == "gold_free"
    reason = (res.reason or "").lower()
    assert "gold" in reason or "reference" in reason
    assert res.faults == [] and res.primary is None


def test_missing_identifiers_degrade_cleanly():
    res = attribution.diagnose(agent=None, instance_id=None)
    assert res.available is False
    assert res.faults == []


# ----------------------------------------------------------------- root switch
@requires_reference
def test_configure_clears_outcome_and_gold_caches(tmp_path):
    """Switching reference roots must not serve the previous root's outcomes: the
    lru_cache lives on load_outcomes (resolved() is uncached) — clearing the
    wrong symbol left stale outcomes active."""
    from awe.outcomes import resolved
    orig_root = attribution._DEFAULT_ROOT
    try:
        assert resolved(GOLD_AGENT, GOLD_INST) is False  # warm the cache
        empty = tmp_path / "empty_reference"
        (empty / "data" / "requirements").mkdir(parents=True)
        attribution.configure(empty)
        # same (agent, instance) key, new root with NO eval file -> must be None,
        # not the cached False from the previous root
        assert resolved(GOLD_AGENT, GOLD_INST) is None
    finally:
        attribution.configure(orig_root)
        assert resolved(GOLD_AGENT, GOLD_INST) is False


# --------------------------------------------------------------- semantics
@requires_reference
def test_scorecard_semantics_and_tier_alignment():
    res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST)
    assert res.available
    by_cap = {s.capability: s for s in res.scorecard}
    assert set(by_cap) == set(attribution.CAPABILITIES)
    # primary capability is blamed with weight > 0
    p = by_cap[res.primary["capability"]]
    assert p.blamed and p.weight > 0
    # deductive core always assessed; never blamed-but-unassessed
    assert by_cap["code_localization"].assessed and by_cap["code_editing"].assessed
    for s in res.scorecard:
        assert not (s.blamed and not s.assessed)
    # no judge verdict is vendored for THIS case -> judge caps are n/a, and
    # used_judge is False (judge availability, not emitted blame)
    assert res.used_judge is False
    assert by_cap["requirement_understanding"].assessed is False
    # tier belongs to the displayed top_error's fault
    tier_of = {(f["capability"], f["error_type"]): (f.get("evidence_chain") or {}).get("strength")
               for f in res.faults}
    for s in res.scorecard:
        if s.blamed and s.top_error:
            assert s.tier == tier_of[(s.capability, s.top_error)]


# --------------------------------------------------------------- parity
@requires_reference
def test_adapter_parity_claude_code():
    """DECAF's adapter on the raw trajectory file == DECAF's own load_normalized."""
    from awe import config
    from awe.adapters import adapt_claude_code, load_normalized

    raw = json.loads((config.TRAJECTORY_DIR / GOLD_AGENT / f"{GOLD_INST}.json").read_text())
    direct = adapt_claude_code(raw, GOLD_AGENT, GOLD_INST)
    disk = load_normalized(GOLD_AGENT, GOLD_INST)
    assert disk is not None
    assert len(direct.steps) == len(disk.steps)
    for a, b in zip(direct.steps, disk.steps, strict=False):
        assert (a.actor, a.type, a.canonical_action, tuple(a.files or []),
                a.test_run, a.tool_error) == \
               (b.actor, b.type, b.canonical_action, tuple(b.files or []),
                b.test_run, b.tool_error)


