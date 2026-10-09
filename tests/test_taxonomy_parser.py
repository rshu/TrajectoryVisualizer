"""Parser-level guards for ``step_labeler.load_taxonomy``.

``tests/test_scripts_fixes.py::TaxonomyVersionTests`` pins what the *shipped*
TAXONOMY_REFERENCE.md parses to today (C15).  This file pins the *parser* against
inputs the shipped file does not yet contain, because the taxonomy is a prose
document that anyone may edit: it is the single source for the LLM system prompt,
for ``_build_valid_sets``' validation of model output, and (through
``step_labeler_v2``) for every label sidecar it writes.  A parser
that attributes prose to a phase does not fail loudly — it silently widens the
valid-action set, so an invalid model label passes validation and lands in a
sidecar as if it were taxonomy.

Both labelers share this one function (``step_labeler_v2`` calls
``v1.load_taxonomy``), so these tests cover v1 and v2 at once.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import step_labeler

# The pinned inventory of the shipped taxonomy: 6 phases, 40 actions.  Kept here
# as per-phase counts rather than a total so that a bullet moving between phases
# is caught too — a phase/action distribution over a set of runs is only comparable
# across runs if this mapping is stable.
SHIPPED_PHASE_COUNTS = {
    "understand": 9,
    "plan": 5,
    "implement": 8,
    "debug": 4,
    "validate": 9,
    "report": 5,
}

TAXONOMY = Path(step_labeler.__file__).resolve().parent / "TAXONOMY_REFERENCE.md"


def _write_taxonomy(tmpdir: str, body: str) -> str:
    path = Path(tmpdir) / "TAXONOMY_REFERENCE.md"
    path.write_text(body, encoding="utf-8")
    return str(path)


class PhaseScopeTests(unittest.TestCase):
    """A phase section must end where its heading's scope ends."""

    def test_a_notes_bullet_is_not_parsed_as_a_report_action(self) -> None:
        """``## Notes`` closes the last ``### <phase>`` section.

        ``## Notes`` is the last heading in the shipped file and sits *after*
        ``### report``, so a parser that only ever opens phases attributes every
        later bullet to ``report``.  The shipped Notes bullets happen to be
        prose that misses the ``- `name`:`` shape by a single backtick; one
        future Notes bullet written in the file's own dominant style would make
        a documentation term an LLM-assignable action.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_taxonomy(
                tmp,
                "# Phase and Action Taxonomy Reference (v1)\n"
                "\n"
                "## Hierarchy\n"
                "\n"
                "### report\n"
                "- `final_reporting`: final completion summary.\n"
                "\n"
                "## Notes\n"
                "- `reserved_user`: not a taxonomy action, only documentation.\n",
            )
            mapping, version = step_labeler.load_taxonomy(path)

        self.assertEqual(version, "v1")
        self.assertEqual(mapping, {"report": ["final_reporting"]})
        _, valid_actions, action_to_phase = step_labeler._build_valid_sets(mapping)
        self.assertNotIn("reserved_user", valid_actions)
        self.assertNotIn("reserved_user", action_to_phase)

    def test_a_level_one_heading_also_closes_the_phase(self) -> None:
        """Appendices live behind ``# ``; nothing after one belongs to a phase."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_taxonomy(
                tmp,
                "# Taxonomy (v1)\n"
                "\n"
                "### validate\n"
                "- `run_tests`: execute the test suite.\n"
                "\n"
                "# Appendix\n"
                "- `deprecated_action`: removed in v1, kept for historical notes.\n",
            )
            mapping, _ = step_labeler.load_taxonomy(path)

        self.assertEqual(mapping, {"validate": ["run_tests"]})

    def test_a_sub_subsection_stays_inside_its_phase(self) -> None:
        """``#### `` is deliberately *not* a phase boundary.

        The ``### ``/``#### `` carve-out in ``load_taxonomy`` encodes the decision
        that a sub-subsection refines the phase it sits under, so its bullets are
        actions of that phase.  Closing the phase on any ``#`` would silently
        drop them.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_taxonomy(
                tmp,
                "# Taxonomy (v1)\n"
                "\n"
                "### debug\n"
                "- `reproduce`: reproduce the failure.\n"
                "\n"
                "#### Rarely used\n"
                "- `bisect`: bisect to the offending change.\n",
            )
            mapping, _ = step_labeler.load_taxonomy(path)

        self.assertEqual(mapping, {"debug": ["reproduce", "bisect"]})


class ShippedInventoryTests(unittest.TestCase):
    """The shipped taxonomy parses to the pinned inventory of phases and actions."""

    def test_the_shipped_taxonomy_parses_to_the_pinned_inventory(self) -> None:
        """Passes before *and* after the phase-boundary fix.

        That is the point: it is the evidence that tightening the parser
        re-labels nothing.  Every reported phase/action distribution rests on
        these exact 6 phases and 40 actions.
        """
        mapping, version = step_labeler.load_taxonomy(str(TAXONOMY))
        self.assertEqual(version, "v1")
        self.assertEqual({k: len(v) for k, v in mapping.items()}, SHIPPED_PHASE_COUNTS)
        _, valid_actions, _ = step_labeler._build_valid_sets(mapping)
        self.assertEqual(len(valid_actions), sum(SHIPPED_PHASE_COUNTS.values()))

    def test_no_action_is_declared_twice(self) -> None:
        """``action_to_phase`` is a dict: a duplicated action silently reassigns."""
        mapping, _ = step_labeler.load_taxonomy(str(TAXONOMY))
        flat = [a for actions in mapping.values() for a in actions]
        self.assertEqual(len(flat), len(set(flat)), f"duplicate actions: {sorted(flat)}")


if __name__ == "__main__":
    unittest.main()
