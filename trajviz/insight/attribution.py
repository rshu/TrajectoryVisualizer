"""DECAF failure-attribution seam for TrajViz.

The single, one-directional bridge between TrajViz and DECAF's ``awe`` package:
this module imports ``awe``; ``awe`` never imports ``trajviz``. Every DECAF call
the UI makes goes through :func:`diagnose`, which returns a plain
:class:`AttributionResult` (no Gradio types) so it is unit-testable headless.

Integrity model (hardened over three review rounds):

* **Gold-grounded.** DECAF's blame needs the run's reference patch + test
  outcome; without them we degrade honestly (never fabricate).
* **Canonical identity.** A displayed file must be byte-identical to the file
  DECAF itself parses for (agent, instance) — resolved by DECAF's own
  ``canonical_trajectory_path`` (codex prefers ``.jsonl``) so verification and
  diagnosis can never read different sources.
* **Cache provenance.** DECAF's normtraj/trajsig caches are content-keyed
  (v0.34). Judge/arbiter verdicts are accepted only when their stamped
  ``trajectory_sha256`` matches the canonical trajectory; unverifiable verdicts
  disable the LLM layers for that diagnosis (noted in the result) rather than
  shape it.
* **Isolation.** ``diagnose`` always configures an explicit reference root (the
  immutable import-time default unless the caller passes one) under a module
  lock, so no request inherits another session's root and concurrent callers
  cannot interleave configuration and diagnosis.
* **Input validation.** ``agent`` must be an agent DECAF knows and
  ``instance_id`` a plain filename stem — path traversal cannot reach the
  filesystem.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Causal capability order (mirrors DECAF's UPSTREAM_ORDER).
CAPABILITIES: tuple[str, ...] = (
    "requirement_understanding", "task_planning", "code_localization",
    "code_editing", "code_verification", "self_repair_loop", "tool_use",
)
_JUDGE_CAPS = frozenset({"requirement_understanding", "task_planning"})
# Always assessed on a failed case (the deductive core); the rest are conditional
# on an opportunity in the trajectory / a judged verdict.
_ALWAYS_ASSESSED = frozenset({"code_localization", "code_editing"})

# A safe instance id: a single filename stem (no separators, no leading dot).
_VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# RLock: diagnose() holds it end-to-end, and the public configure() acquires the
# SAME lock so no caller can rewrite DECAF's process-global root/caches while a
# diagnosis is in flight (reentrancy lets diagnose call configure internally).
_LOCK = threading.RLock()


# --------------------------------------------------------------------------- #
# DECAF discovery + import (stdlib-only, not a pip package)
# --------------------------------------------------------------------------- #
def _decaf_root() -> Path | None:
    candidates = []
    env = os.environ.get("AWE_DECAF_PATH")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[3] / "DECAF")
    for c in candidates:
        if (c / "awe" / "__init__.py").is_file():
            return c
    return None


_DECAF_ROOT = _decaf_root()
if _DECAF_ROOT is not None and str(_DECAF_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECAF_ROOT))

def _reference_root() -> Path:
    """Where the gold reference data lives, resolved by LAYOUT, not by name.

    ``TRAJVIZ_REFERENCE_ROOT`` wins. Otherwise look for a sibling
    of this checkout that actually has the layout attribution needs —
    ``data/requirements/`` and ``data/trajectory/`` — and take it only if exactly
    one sibling qualifies. Discovering it by content rather than by a hard-coded
    directory name keeps TrajViz independent of whatever the reference set is
    called locally, and keeps an ambiguous tree from being silently picked.

    With no match, the neighbouring directory is returned unchanged; attribution
    then degrades with a message naming the root it searched rather than guessing.
    """
    env = os.environ.get("TRAJVIZ_REFERENCE_ROOT")
    if env:
        return Path(env)

    neighbourhood = Path(__file__).resolve().parents[3]
    try:
        matches = [
            d for d in neighbourhood.iterdir()
            if d.is_dir()
            and (d / "data" / "requirements").is_dir()
            and (d / "data" / "trajectory").is_dir()
        ]
    except OSError:
        return neighbourhood
    return matches[0] if len(matches) == 1 else neighbourhood


# Captured ONCE at import and immutable thereafter. Every diagnose() call
# explicitly configures either this default or the caller's root — never
# "whatever the previous caller left behind".
_DEFAULT_ROOT = _reference_root()
# Read before awe is imported: an explicit AWE_JUDGE_MODEL always wins.
_JUDGE_MODEL_EXPLICIT = bool(os.environ.get("AWE_JUDGE_MODEL"))


def _select_judge_model(judge_cache_dir: Path, schema: int) -> str | None:
    """The model whose cached verdicts DECAF's current schema can verify.

    DECAF partitions judge verdicts by model slug and stamps each with the
    evidence schema it was produced under; :func:`_verdict_verifies` refuses any
    other schema. A model named by hand therefore goes stale on the next
    move to another model or schema bump — every verdict in its namespace is refused and
    both LLM layers silently switch off. Choose by content instead: among the
    namespaces holding verdicts stamped with *schema*, the one with the most.
    The model id comes from the records, because the directory slug is lossy
    (``/`` becomes ``__``). ``None`` when no namespace qualifies.
    """
    counts: dict[str, int] = {}
    try:
        namespaces = sorted(d for d in Path(judge_cache_dir).iterdir() if d.is_dir())
    except OSError:
        return None
    for ns in namespaces:
        for record in ns.glob("*/*.json"):
            try:
                doc = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            model = doc.get("model") if isinstance(doc, dict) else None
            if isinstance(model, str) and model and doc.get("evidence_schema") == schema:
                counts[model] = counts.get(model, 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda m: (-counts[m], m))


DECAF_AVAILABLE = False
_IMPORT_ERROR: str | None = None
try:
    if _DECAF_ROOT is None:
        raise ImportError(
            "DECAF (awe) not found — set AWE_DECAF_PATH to a DECAF checkout")
    from awe import config as _cfg  # noqa: E402
    DECAF_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

if DECAF_AVAILABLE and not _JUDGE_MODEL_EXPLICIT:
    # DECAF reads config.JUDGE_MODEL at call time (judge, arbiter and
    # model_slug), so setting it after import takes effect everywhere. Without a
    # match DECAF keeps its own default and the layers degrade with a note.
    from awe.trajectory import SCHEMA_VERSION as _SCHEMA_VERSION  # noqa: E402

    _detected_model = _select_judge_model(_cfg.JUDGE_CACHE_DIR, _SCHEMA_VERSION)
    if _detected_model:
        _cfg.JUDGE_MODEL = _detected_model


def configure(reference_root: str | os.PathLike) -> None:
    """Point DECAF at a reference root and clear its root-dependent memory caches.

    Acquires the SAME module lock as diagnose(), so an external caller cannot
    rewrite DECAF's process-global root/caches while a diagnosis is in flight.
    diagnose() itself re-configures per request (explicit root or the immutable
    default) — configuration is never inherited across callers.
    The outcome lru_cache lives on load_outcomes (resolved() is an uncached
    wrapper); clearing must fail loudly on DECAF API drift, not except-pass.
    """
    if not DECAF_AVAILABLE:
        return
    with _LOCK:
        _configure_unlocked(reference_root)


def _configure_unlocked(reference_root: str | os.PathLike) -> None:
    # DECAF resolves every input path from these attributes at call time, so
    # setting them is the whole configuration; the process environment is left
    # alone.
    root = Path(reference_root).resolve()
    _cfg.DATA = root / "data"
    _cfg.REQUIREMENTS_DIR = _cfg.DATA / "requirements"
    _cfg.PATCH_DIR = _cfg.DATA / "patch"
    _cfg.TRAJECTORY_DIR = _cfg.DATA / "trajectory"
    _cfg.LABELS_DIR = _cfg.DATA / "labels"
    from awe.gold import build_gold_reference
    from awe.outcomes import load_outcomes
    build_gold_reference.cache_clear()
    load_outcomes.cache_clear()


if DECAF_AVAILABLE:
    # DECAF's own default is its checkout; point it at the reference root from
    # import on, so a caller that reaches DECAF before any diagnose() reads the
    # same data diagnose() will. Every diagnose() re-configures explicitly.
    _configure_unlocked(_DEFAULT_ROOT)


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
@dataclass
class CapabilityScore:
    capability: str
    assessed: bool                # was this capability in scope for this case?
    blamed: bool                  # NON-ZERO blame — refuted faults are not blamed
    weight: float
    tier: str | None              # deductive | associational | model_inferred
    top_error: str | None


@dataclass
class AttributionResult:
    available: bool
    reason: str | None = None
    mode: str = "gold_free"       # reference | gold_free
    agent: str | None = None
    instance_id: str | None = None
    blame_status: str | None = None
    primary: dict | None = None
    scorecard: list[CapabilityScore] = field(default_factory=list)
    faults: list[dict] = field(default_factory=list)
    arbiter: dict | None = None
    used_judge: bool = False      # judge verdict AVAILABLE for this case
    task: dict | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sha256(p: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _verdict_verifies(rec: dict, *, canon_sha: str, req_sha: str | None,
                      prompt_version: str, schema: int) -> bool:
    """A cached LLM verdict is trusted only when its FULL prompt provenance is
    verifiable: the trajectory it judged, the gold reference whose task
    statement its prompt embedded, the prompt version, and the evidence schema
    all match the current inputs. Legacy/unstamped records fail (cannot verify)."""
    return (rec.get("trajectory_sha256") == canon_sha
            and rec.get("requirements_sha256") == req_sha
            and rec.get("prompt_version") == prompt_version
            and rec.get("evidence_schema") == schema)


def _llm_layer_policy(agent: str, instance_id: str, canon_sha: str):
    """Decide whether cached judge/arbiter verdicts may shape this diagnosis.

    Validates the complete prompt fingerprint (trajectory + requirements +
    prompt_version + evidence_schema; the arbiter's claim is already part of
    its cache key). Any unverifiable record disables the corresponding layer —
    a stale verdict must never judge a different trajectory OR a different task.

    Returns (use_judge, adjudicate, notes): use_judge False disables the judge;
    adjudicate None disables the arbiter; _AUTO keeps DECAF's default.
    """
    import hashlib
    import json as _json
    import awe.arbiter as _arb
    import awe.judge as _judge
    from awe.adapters import requirements_source_sha256
    from awe.dossier import _AUTO
    from awe.trajectory import SCHEMA_VERSION

    notes: list[str] = []
    req_sha = requirements_source_sha256(instance_id)
    j_prompt = hashlib.sha256(_judge.SYSTEM_PROMPT.encode()).hexdigest()[:12]
    a_prompt = hashlib.sha256(_arb.SYSTEM_PROMPT.encode()).hexdigest()[:12]

    use_judge: bool | None = None      # DECAF auto
    jrec = _judge.verdict_cached(agent, instance_id)
    if jrec is not None and not _verdict_verifies(
            jrec, canon_sha=canon_sha, req_sha=req_sha,
            prompt_version=j_prompt, schema=SCHEMA_VERSION):
        use_judge = False
        notes.append("judge verdict ignored: its recorded provenance "
                     "(trajectory/task/prompt/schema) does not match the current "
                     "inputs (unverifiable)")

    adjudicate = _AUTO
    adir = _arb.ARBITER_CACHE_DIR / _cfg.model_slug() / agent
    if adir.is_dir():
        # Match ONLY this instance's verdict files: exact legacy "{id}.json" or
        # claim-keyed "{id}__{cap}__{et}.json". A bare "{id}*" prefix glob would
        # also match a DIFFERENT instance whose id extends this one (e.g.
        # proj-1147 matching proj-11477's files) and wrongly disable the arbiter.
        candidates = (list(adir.glob(f"{instance_id}.json"))
                      + list(adir.glob(f"{instance_id}__*.json")))
        for f in candidates:
            try:
                rec = _json.loads(f.read_text())
            except (OSError, ValueError):
                # Unreadable or malformed candidate file — skip it; provenance
                # verification below decides whether the arbiter stays enabled.
                continue
            if not _verdict_verifies(rec, canon_sha=canon_sha, req_sha=req_sha,
                                     prompt_version=a_prompt,
                                     schema=SCHEMA_VERSION):
                adjudicate = None
                notes.append("arbiter verdict ignored: its recorded provenance "
                             "(trajectory/task/prompt/schema) does not match "
                             "the current inputs (unverifiable)")
                break
    return use_judge, adjudicate, notes


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #
def diagnose(*, agent: str | None, instance_id: str | None,
             source_path: str | os.PathLike | None = None, fmt: str | None = None,
             expected_sha: str | None = None, merged_sources: int = 0,
             reference_root: str | os.PathLike | None = None) -> AttributionResult:
    """Diagnose the failure of the *displayed* trajectory. Never raises; never
    fabricates; degrades with an explicit reason. See the module docstring for
    the integrity model.

    ``expected_sha`` — the sha256 captured when the UI LOADED the trajectory
    (the immutable identity of the displayed content). Diagnosis requires the
    canonical file's CURRENT bytes to equal it, so a real export mutated between
    load and diagnosis is refused rather than diagnosed while the UI still
    shows the old state.

    ``merged_sources`` — how many ADDITIONAL files the loader merged into the
    displayed trajectory (``_source_merged_count``; non-zero only for a DSH
    export tree). Anything above zero is unverifiable here by construction, so
    it is refused with that reason rather than compared.
    """
    if not DECAF_AVAILABLE:
        return AttributionResult(False, reason=f"DECAF unavailable ({_IMPORT_ERROR})")
    with _LOCK:
        return _diagnose_locked(agent=agent, instance_id=instance_id,
                                source_path=source_path, fmt=fmt,
                                expected_sha=expected_sha,
                                merged_sources=merged_sources,
                                reference_root=reference_root)


def _diagnose_locked(*, agent, instance_id, source_path, fmt, expected_sha,
                     reference_root, merged_sources=0):
    # per-call, explicit configuration — never inherited from a previous caller
    _configure_unlocked(reference_root if reference_root else _DEFAULT_ROOT)

    if not (agent and instance_id):
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason="capability attribution needs (agent, instance_id); an arbitrary "
                   "trajectory upload has no reference patch to ground blame")
    # Validate BEFORE any filesystem access: agent from DECAF's known agents,
    # instance_id a single safe filename stem (no traversal, no separators).
    from awe import config
    if agent not in config.AGENTS:
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason=f"could not determine the agent for '{instance_id}' "
                   f"(got '{agent}'). Set the agent override to one of: "
                   f"{', '.join(config.AGENTS)}")
    if not _VALID_ID.fullmatch(instance_id) or ".." in instance_id:
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason="invalid instance id (must be a plain instance name)")

    if not config.requirements_path(instance_id).is_file():
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason=f"no gold reference (data/requirements/{instance_id}.json) under "
                   f"{config.DATA.parent} — attribution needs the reference patch + "
                   f"test outcome; showing trajectory-only signals instead")

    # Canonical identity: resolve via DECAF's OWN source resolution so we verify
    # the very file diagnosis will parse (codex prefers .jsonl).
    from awe.adapters import canonical_trajectory_path
    canon = canonical_trajectory_path(agent, instance_id)
    if canon is None:
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason=f"no real trajectory for {agent}/{instance_id} under "
                   f"{config.DATA.parent} — cannot verify the displayed trajectory "
                   f"belongs to this run")
    if merged_sources:
        # A merged export (DSH parent log + N sub-agent logs) has no single-file
        # identity to compare: `canonical_trajectory_path` resolves exactly ONE
        # file per (agent, instance) and the LLM-layer provenance is keyed on
        # that file's sha, so the composite the loader stamped cannot match by
        # construction. Say that, rather than let the sha comparison below blame
        # a mismatch or (worse) certify a tree whose children were never hashed.
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason=f"the displayed trajectory is a merged multi-file export "
                   f"(parent log + {merged_sources} sub-agent log"
                   f"{'s' if merged_sources != 1 else ''}), while the canonical "
                   f"source for {agent}/{instance_id} is a single file — its "
                   f"content cannot be certified as this run, so no "
                   f"gold-grounded verdict is issued")
    canon_sha = _sha256(canon)
    # Identity is CONTENT identity, never path identity: the displayed bytes'
    # hash (captured at load) — or, failing that, the current source file's
    # bytes — must equal the canonical file's CURRENT bytes. A path match alone
    # says nothing if the file changed after the UI parsed it (TOCTOU).
    displayed_sha = expected_sha or (
        _sha256(Path(source_path)) if source_path is not None else None)
    if source_path is not None and displayed_sha is None:
        # A displayed source whose identity cannot be established (unreadable
        # at load AND at diagnose time) must refuse, not silently skip the gate.
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason="cannot establish the displayed trajectory's content identity "
                   "(source unreadable) — refusing rather than diagnosing "
                   "unverified bytes")
    if displayed_sha is not None and displayed_sha != canon_sha:
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason=f"the displayed trajectory does not match the canonical "
                   f"{agent}/{instance_id} run's current content — either it "
                   f"belongs to a different execution, or the real export "
                   f"changed after it was loaded (reload to re-sync). A "
                   f"gold-grounded verdict would otherwise describe different "
                   f"bytes than the ones shown")

    # LLM layers only when their cached verdicts verifiably belong to THIS
    # trajectory content.
    use_judge, adjudicate, notes = _llm_layer_policy(agent, instance_id, canon_sha)

    from awe.dossier import case_record
    from awe.detect import detect
    try:
        rec = case_record(agent, instance_id, use_judge=use_judge,
                          adjudicate=adjudicate)
        d = detect(instance_id, agent, use_judge=use_judge)
    except Exception as exc:  # never surface a raw traceback to the UI
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason=f"diagnosis failed for {agent}/{instance_id}: {type(exc).__name__}: {exc}")
    if rec is None:
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason="this run resolved (or its outcome is unknown); DECAF attributes "
                   "failures only")
    # Close the mid-diagnosis mutation window: case_record/detect re-read the
    # reference data from disk AFTER the identity gate above, and an external
    # writer (e.g. a git pull in that data) could swap the file in between — the
    # result would describe bytes the gate never approved and the UI never
    # displayed. Recompute the canonical hash after ALL reads and refuse on any
    # change (the in-process lock cannot serialize other processes).
    if _sha256(canon) != canon_sha:
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason=f"the real trajectory for {agent}/{instance_id} changed "
                   f"while the diagnosis was running — reload and retry")
    return _shape(rec, d.get("opportunities", {}), notes)


def _shape(rec: dict, opportunities: dict, notes: list[str]) -> AttributionResult:
    """case_record dict + detection opportunities -> AttributionResult.

    Blame is NON-ZERO only (arbiter-refuted faults carry weight 0). Assessment
    is per capability: the deductive core always; every other capability —
    including EACH judge capability separately — only when its own opportunity
    opened (DECAF can judge requirement_understanding while task_planning is out
    of scope for the same case)."""
    faults = rec.get("faults", [])
    primary = next(({"capability": f["capability"], "error_type": f["error_type"]}
                    for f in faults if f.get("is_primary")), None)

    agg: dict[str, dict] = {}
    for f in faults:
        cap = f["capability"]
        w = float(f.get("blame_weight") or 0.0)
        a = agg.setdefault(cap, {"weight": 0.0, "errors": {}, "tier_of": {}})
        a["weight"] += w
        if w > 0:
            et = f["error_type"]
            a["errors"][et] = a["errors"].get(et, 0.0) + w
            a["tier_of"][et] = (f.get("evidence_chain") or {}).get("strength")

    scorecard = []
    for cap in CAPABILITIES:
        a = agg.get(cap)
        weight = round(a["weight"], 3) if a else 0.0
        blamed = weight > 0
        assessed = blamed or cap in _ALWAYS_ASSESSED or bool(opportunities.get(cap))
        top = (max(a["errors"], key=a["errors"].get) if (a and a["errors"]) else None)
        tier = a["tier_of"].get(top) if (a and top) else None
        scorecard.append(CapabilityScore(cap, assessed, blamed, weight, tier, top))

    used_judge = any(bool(opportunities.get(c)) for c in _JUDGE_CAPS) or \
        any(f["capability"] in _JUDGE_CAPS for f in faults)

    return AttributionResult(
        available=True, mode="reference",
        agent=rec.get("agent"), instance_id=rec.get("instance_id"),
        blame_status=rec.get("blame_status"), primary=primary,
        scorecard=scorecard, faults=faults, arbiter=rec.get("arbiter"),
        used_judge=used_judge, task=rec.get("task"), notes=notes,
    )
