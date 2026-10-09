"""Content identity of DeepSeek-Harness loads, and the zip reader's contract.

A DSH export is a *tree*: the parent ``session.jsonl`` plus every
``subagents/<id>/session.jsonl`` it merges. The zip branch already hashes the
whole archive (``tests/test_attribution_live.py::ZipSourceIdentityTests``), but
the directory / bare-``session.jsonl`` branch hashed only the parent file while
the converter merged the siblings — so two trees whose child logs differed
carried the SAME ``_source_sha256`` and attribution would certify either of
them as the other's run. These tests pin the composite digest that closes that
hole, including the two properties that make it usable as an identity (stable
when the tree is relocated, unchanged when nothing was merged).

The last two groups pin the zip reader: the duplicate-child tie-break must not
depend on ``namelist()`` order, and an unreadable member (encrypted, or a bad
CRC) must degrade to an ``_error`` dict rather than raise out of
``load_trajectory`` — the never-raise ingest contract.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from trajviz.insight import attribution
from trajviz.insight.formats.dsh import _zip_dsh_members
from trajviz.insight.loaders import load_trajectory
from trajviz.insight.parser import parse_steps

GOLD_AGENT = "claude_code"
GOLD_INST = "astropy__astropy-13033"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference_present() -> bool:
    if not attribution.DECAF_AVAILABLE:
        return False
    from awe import config

    return config.requirements_path(GOLD_INST).is_file()


requires_decaf = unittest.skipUnless(attribution.DECAF_AVAILABLE, "DECAF (awe) not importable")
requires_reference = unittest.skipUnless(_reference_present(), "reference fixture not present")


def _dsh_events(child_text: str) -> tuple[list[dict], list[dict]]:
    """A two-message parent log and one sub-agent log that answers it."""

    def evt(kind, seq, time, data):
        return {"type": kind, "seq": seq, "time": time, "data": data}

    def session(**extra):
        base = {
            "type": "session",
            "version": 0,
            "id": "session-parent",
            "createdAt": 1_787_623_647_203,
            "cwd": "/p",
            "delegationDepth": 0,
            "agentPreset": "standard",
        }
        base.update(extra)
        return base

    def user(seq, time, text, mid):
        return evt(
            "user/message",
            seq,
            time,
            {"content": [{"type": "text", "text": text}], "source": {"kind": "user"}, "role": "user", "id": mid},
        )

    def assistant(seq, time, text, mid):
        return evt(
            "assistant/message",
            seq,
            time,
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "source": {"kind": "model", "provider": "deepseek-official", "model": "deepseek-v4-pro"},
                    "id": mid,
                },
                "usage": {"inputTokens": 10, "outputTokens": 3, "cacheReadTokens": 0, "reasoningTokens": 0},
            },
        )

    parent = [
        session(),
        user(1, 1_787_623_647_300, "do it", "u1"),
        assistant(2, 1_787_623_647_500, "parent done", "a1"),
    ]
    child = [
        session(
            id="child-1",
            origin="subagent",
            parentSession="session-parent",
            seedLength=1,
            createdAt=1_787_623_648_050,
        ),
        user(2, 1_787_623_648_200, "explore", "cu"),
        assistant(3, 1_787_623_648_400, child_text, "ca"),
    ]
    return parent, child


def _jsonl(events: list[dict]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


def _build_tree(tmp: Path, name: str, child_text: str | None) -> Path:
    """An unzipped DSH export dir; ``child_text`` of None means no ``subagents/``."""
    export = tmp / name / "export"
    export.mkdir(parents=True)
    parent, child = _dsh_events(child_text or "")
    (export / "session.jsonl").write_text(_jsonl(parent))
    if child_text is not None:
        child_dir = export / "subagents" / "child-1"
        child_dir.mkdir(parents=True)
        (child_dir / "session.jsonl").write_text(_jsonl(child))
    return export


# ------------------------------------------------------- directory identity
class DirectorySourceIdentityTests(unittest.TestCase):
    """The non-zip branch must hash everything it merged, not just the parent."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-identity-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_changed_subagent_log_changes_the_identity(self):
        """Two trees with byte-identical parents and different children are two
        different runs: identical ``_source_sha256`` would let attribution
        certify one as the other, which is the whole point of the gate."""
        first = _build_tree(self.tmp, "first", "CHILD-ALPHA")
        second = _build_tree(self.tmp, "second", "CHILD-BETA-DIFFERENT-TEXT")
        self.assertEqual(_sha(first / "session.jsonl"), _sha(second / "session.jsonl"))

        # Both entry points reach the same merge: the directory, and the parent
        # log the UI gets after resolve_dsh_session_path() rewrites it.
        for entry in ("", "session.jsonl"):
            raw_first = load_trajectory(str(first / entry) if entry else str(first))
            raw_second = load_trajectory(str(second / entry) if entry else str(second))
            self.assertNotIn("_error", raw_first)
            self.assertNotIn("_error", raw_second)
            self.assertEqual(raw_first["metadata"]["sub_agent_count"], 1)
            self.assertNotEqual(
                parse_steps(raw_first),
                parse_steps(raw_second),
                "the child log must reach the analysed trajectory",
            )
            self.assertNotEqual(
                raw_first["_source_sha256"],
                raw_second["_source_sha256"],
                f"a changed subagent log left the identity unchanged (loaded via {entry or 'directory'})",
            )

    def test_merged_load_reports_the_parent_sha_and_the_merge_count(self):
        """The composite is not the parent's sha, so the plain one stays visible
        (debugging a refusal needs to tell 'merged tree' from 'wrong file')."""
        export = _build_tree(self.tmp, "counted", "CHILD-ALPHA")
        raw = load_trajectory(str(export))
        self.assertEqual(raw["_source_file_sha256"], _sha(export / "session.jsonl"))
        self.assertEqual(raw["_source_merged_count"], 1)
        self.assertNotEqual(raw["_source_sha256"], raw["_source_file_sha256"])

    def test_a_tree_with_no_subagents_keeps_the_plain_file_sha(self):
        """Backward compatibility, and the reason existing identities cannot move:
        with nothing merged the identity is still sha256 of the file's bytes."""
        export = _build_tree(self.tmp, "lonely", None)
        raw = load_trajectory(str(export))
        self.assertNotIn("_error", raw)
        self.assertEqual(raw["metadata"]["sub_agent_count"], 0)
        self.assertEqual(raw["_source_sha256"], _sha(export / "session.jsonl"))
        self.assertNotIn("_source_merged_count", raw)
        self.assertNotIn("_source_file_sha256", raw)

    def test_the_identity_survives_relocating_the_export(self):
        """Keyed on content and session ids, never on where the tree sits: a
        re-download to another directory is the same run."""
        export = _build_tree(self.tmp, "home", "CHILD-ALPHA")
        before = load_trajectory(str(export))["_source_sha256"]
        moved = self.tmp / "elsewhere" / "deeper" / "export"
        moved.parent.mkdir(parents=True)
        shutil.copytree(export, moved)
        self.assertEqual(load_trajectory(str(moved))["_source_sha256"], before)


# ---------------------------------------------------- directory attribution
@requires_decaf
@requires_reference
class DirectoryAttributionTests(unittest.TestCase):
    """A merged DSH tree driven through ``diagnose`` — the zip path had the only
    identity test, so this branch could certify a merged tree unnoticed."""

    def setUp(self):
        attribution.configure(attribution._DEFAULT_ROOT)
        self.addCleanup(attribution.configure, attribution._DEFAULT_ROOT)
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-attr-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_merged_export_is_refused_instead_of_certified(self):
        """DECAF's canonical source for (agent, instance) is ONE file, so a
        trajectory assembled from several cannot be certified as that run even
        when the parent log's bytes match it exactly."""
        root = self.tmp / "reference"
        shutil.copytree(attribution._DEFAULT_ROOT / "data", root / "data")
        canon = root / "data" / "trajectory" / GOLD_AGENT / f"{GOLD_INST}.json"
        self.assertTrue(canon.is_file(), canon)

        reasons = set()
        for name, child_text in (("alpha", "CHILD-ALPHA"), ("beta", "CHILD-BETA-DIFFERENT-TEXT")):
            export = _build_tree(self.tmp, name, child_text)
            # The canonical file IS the parent log: the only thing left that can
            # tell the two trees apart is whether the merge is accounted for.
            shutil.copyfile(export / "session.jsonl", canon)
            raw = load_trajectory(str(export))
            self.assertNotIn("_error", raw)
            res = attribution.diagnose(
                agent=GOLD_AGENT,
                instance_id=GOLD_INST,
                source_path=raw["_source_path"],
                expected_sha=raw["_source_sha256"],
                merged_sources=raw.get("_source_merged_count") or 0,
                reference_root=root,
            )
            self.assertFalse(res.available, f"{name}: a merged export was certified: {res.reason}")
            self.assertEqual(res.faults, [], f"{name}: blame was attributed to a merged export")
            self.assertIn("merged", res.reason.lower(), res.reason)
            reasons.add(res.reason)
        self.assertEqual(len(reasons), 1, "both trees must be refused for the same stated reason")


# ------------------------------------------------------------ zip tie-break
class ZipDuplicateChildTests(unittest.TestCase):
    """``_zip_dsh_members`` picks ONE member per child id; the choice must be a
    property of the paths, not of the order the archive happens to list them."""

    def test_equal_depth_duplicates_keep_the_shorter_path(self):
        names = [
            "session.jsonl",
            "aaaaaa/subagents/c1/session.jsonl",
            "b/subagents/c1/session.jsonl",
        ]
        for ordering in (names, list(reversed(names))):
            parent, children = _zip_dsh_members(ordering)
            self.assertEqual(parent, "session.jsonl")
            self.assertEqual(
                children,
                [("c1", "b/subagents/c1/session.jsonl")],
                f"the chosen member depends on namelist() order (input: {ordering})",
            )


# ------------------------------------------------------- zip read robustness
class ZipMemberReadContractTests(unittest.TestCase):
    """Ingest never raises: an unreadable member is an ``_error`` dict (parent)
    or a skipped child, never an exception out of ``load_trajectory``."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-zipread-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _load(self, path: Path) -> dict:
        try:
            return load_trajectory(str(path))
        except Exception as exc:  # noqa: BLE001 - the contract is "no exception of any type"
            self.fail(f"load_trajectory({path.name!r}) raised {type(exc).__name__}: {exc}")

    def test_an_encrypted_parent_member_is_a_clean_error(self):
        """``zipfile`` raises a bare ``RuntimeError`` for an encrypted member."""
        parent, _ = _dsh_events("CHILD-ALPHA")
        plain = self.tmp / "plain.zip"
        with zipfile.ZipFile(plain, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr("session.jsonl", _jsonl(parent))
        data = bytearray(plain.read_bytes())
        # Set the "encrypted" general-purpose flag in the local (+6) and
        # central-directory (+8) headers without actually encrypting anything.
        for magic, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
            i = 0
            while True:
                i = data.find(magic, i)
                if i < 0:
                    break
                data[i + offset] |= 0x01
                i += 4
        encrypted = self.tmp / "encrypted.zip"
        encrypted.write_bytes(bytes(data))
        self.assertIn("_error", self._load(encrypted))

    def test_a_corrupted_child_member_is_skipped_not_fatal(self):
        """A damaged CHILD raises ``BadZipFile`` from the same read: the parent's
        steps are intact and displayable, so losing them is the wrong failure."""
        parent, child = _dsh_events("CHILD-ALPHA")
        path = self.tmp / "crc.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("session.jsonl", _jsonl(parent))
            archive.writestr("subagents/child-1/session.jsonl", _jsonl(child))
        data = bytearray(path.read_bytes())
        # Stored (undeflated) members appear verbatim, so flipping a byte of the
        # child's payload breaks its CRC and nothing else.
        idx = data.find(b"CHILD-ALPHA")
        self.assertGreater(idx, 0)
        data[idx] = ord("X")
        path.write_bytes(bytes(data))

        result = self._load(path)
        self.assertNotIn("_error", result)
        self.assertEqual(result["metadata"]["sub_agent_count"], 0, "the damaged child must be dropped")
        self.assertEqual(len(parse_steps(result)), 2, "the parent's own steps must survive")


if __name__ == "__main__":
    unittest.main()
