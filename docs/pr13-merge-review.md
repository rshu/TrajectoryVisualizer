# PR #13 fork sync — pre-merge review and open backlog

**Merged** 2026-09-22 as `3c4e123` (`Merge pull request #13 from KevinKaChunLee/main`):
100 commits, 106 files, +28,597/−6,045. New trajectory formats (Pi, DSH, Cursor,
ICode); `insight.py`/`charts.py` split into `ui/`, `presenters/`, `formats/`,
`charts/`; context-utilization charts; system vs tool error split;
chart-to-workflow jumps; a standalone `--report` HTML export; and a new
LLM-backed Issues judge and analysis sidebar.

The PR description stated *"No breaking change as no additional commits has been
added to main branch since the repo was forked."* **That was inaccurate.** Main
had gained PR #10 (DECAF attribution with security hardening) and PR #11 (34
correctness fixes, 30 refactors, ruff baseline, tests 125 → 199). The author
merged main into the branch late, which made the merge textually clean while
letting fork-side rewrites of `insight.py` (−2,200) and `loaders.py` (−1,760)
silently drop main-side fixes with a green CI. That risk is what this review
targeted.

## Verification performed

Full-corpus differential over **all 2,500 real trajectories** in
`TraceProbe/data/trajectory` (claude_code, codex, opencode ×3), main
(`9e663c3`) vs PR head (`8cd3c48`):

| Check | Result |
|---|---|
| Loader parse parity | 2,000/2,500 byte-identical; the 500 differences are exclusively the intentional removal of the always-zero `reasoning` token key for claude_code. No step, role, tool, index or token *value* changed. |
| Full pipeline (load → session → every presenter and chart, light **and** dark) | 2,500/2,500 clean: 0 crashes, 0 load errors, 0 error banners |
| Loader exceptions | 0 on both versions |
| Performance | identical (1.2 ms mean per load on both) |
| Determinism | identical output on repeated loads, both versions |
| Entry points | CLI `--help`, `--report` for claude_code/codex/opencode, and the converge app all work |
| ZIP ingestion | bounded: 32 MB/member, 128 MB total, 64-child cap, in-memory reads (no extraction, so no zip-slip) |
| `_MAX_STEPS` | 2,000 on both versions; no corpus file exceeds 173 steps |

A 38-agent verified review (10 risk surfaces, every blocker/major finding
adversarially verified, plus a completeness critic) produced 48 findings; 27
were verified, **21 confirmed** and 6 refuted.

### Corrections this PR makes to main (numbers in the corpus change)

- **Tool success rate was over-reported on main.** Main counted `bash` calls
  that exited non-zero as successes because OpenCode marks the *invocation*
  `completed` regardless of the command's exit status. PR head checks
  `metadata.exit`, so real failures now count. Affects 1,150/2,500 files, all
  OpenCode: mean success rate −10 to −18 points, max −60. Audited for false
  positives: all 926 newly-flagged calls in a 360-file sample are `bash`, and
  97% carry explicit error text (tracebacks, `subprocess-exited-with-error`,
  `ImportError while loading conftest`, `exit=127`). **This is a correction,
  not a regression — but any previously published OpenCode tool-success number
  is affected.** claude_code rows do not move at all.
- claude_code reasoning tokens now read `N/A` instead of a misleading `0`.
- Delegation (`task`/`Agent`) wall-clock is no longer counted as tool execution
  time, so it is not double-counted against the child's own steps.
- Tool-rate denominators use session wall-clock instead of summed step
  durations (which can exceed wall-clock when sub-agents overlap).

## Fixed post-merge

- `a56ecc4` — **The Overview Issues LLM judge auto-fired on every load.**
  Opening any trajectory POSTed trajectory excerpts (prompts, shell commands,
  file paths, search patterns, tool-output snippets; up to ~28k chars per
  request) to the configured provider with no button, no consent and no kill
  switch — and because the dashboard config falls back to the offline step
  labeler's `LABEL_*` variables, an unedited `cp .env.example .env` armed it.
  Both load triggers are registered independently, so the normal flow fired it
  twice (up to 16 calls per trajectory). Now bound to an explicit button that
  names the egress.
- `fd38015` — **Tool timing reported a hard `0` when only the delegation was
  timed.** 106/500 claude_code trajectories (21%) lost their entire tool-timing
  row, up to 1,862.8 s (41.4% of wall-clock), and the fabricated zeros were fed
  to the LLM analysis brief. Absent timing now reports `None` → rendered `n/a`;
  delegated wall-clock is preserved in new `delegation_time_total` /
  `delegated_call_count` and shown as a "Delegated" chip.

## Open backlog (confirmed, not yet fixed)

Ranked. Each was reproduced against PR-head code by a verifier.

1. `metrics.py:292` — `generation_seconds` subtracts the **sum** of overlapping
   tool durations from a step's wall clock, inflating `output_tokens_per_sec`
   and zeroing the denominator for parallel-tool steps. Corpus-wide
   `output_tokens_per_sec` differs from main on 1,599/2,500 files (median
   +20.9%, max +234%). Use the max for parallel calls within a step.
2. `report.py:287` — `_mixed_md_to_html` passes any markdown line starting with
   `<` through **verbatim**, so trajectory-derived strings containing markup
   become live HTML/script in the exported report.
3. `context_usage.py:218` — `step_context_occupancy` omits `cache_write`, so
   claude_code / DSH / pi window occupancy is systematically understated.
4. `context_usage.py:811` — the "Compactions" chip counts raw un-coalesced
   events while the chart coalesces them; also counts events for windows that
   have no plotted line.
5. `context_usage.py:1148` — `context_usage_breakdown` ignores inferred
   `occupancy_drop` compactions (the only kind claude_code can produce) when
   choosing the window start, then silently rescales every category count.
6. `context_usage.py:383` — `_splice_compaction_into_points` reads the list it
   is appending to, so the second and later cliffs take their composition from
   an earlier artifact instead of the real preceding turn.
7. `context_usage.py:1028` — a zero-usage final assistant turn zeroes the whole
   composition panel and drops the occupancy line to 0.
8. `run_group.py:377` / `:415` / `:671` — a zero-step load counts as a
   successful run, wins every "best" flag and shifts the consensus threshold;
   missing metrics are coerced to `0` and then highlighted green as "best".
   `run_group.py:723` calls `load_trajectory` with no `try/except`, so one bad
   file aborts a whole batch (triggers: `cursor.py:35` unguarded
   `datetime.fromtimestamp`, `dsh.py:962` `RuntimeError` on an encrypted zip
   member). Fix the loader-local bugs **and** wrap line 723.
9. `dsh.py:632` — one malformed line silently discards an entire child session
   (`if err: return []`), understating tokens by up to 493×.
10. `cursor.py:263` — `chars/4` token estimates are written into `token_usage`
    and shown as measured API usage. The `_capabilities` flags that mark them
    as estimates (written by `cursor.py`, `icode.py`, `dsh.py`) have **no
    consumer anywhere**.
11. `presenters/overview.py:352` — the summary banner is computed but never
    rendered, dropping both the loaded filename and main's OpenCode/CodeArts
    "Cache Read is a running conversation prefix" caveat.
12. `llm_config.py:38` — `load_env_files()` runs as the first statement of
    `build_ui()` and injects every key from the launch directory's `.env`,
    including `AWE_DECAF_PATH` (arbitrary code import into the attribution
    backend).
13. `issue_judge.py:263` / `overview_tab.py:471` — judge failure renders the raw
    exception, including the endpoint URL and any credential in it, into the
    Issues HTML, bypassing the module's own scrubbing helper.
14. `report.py:34` — `_PLOTLY_CDN = "cdn"`, so the "standalone" HTML report
    silently requires internet and pings `cdn.plot.ly` when opened. Inline the
    JS for a genuinely offline artifact.
15. `issue_judge.py:210` — LLM output is hard-coded to Simplified Chinese and
    renders into the otherwise-English dashboard with no setting.
16. `ui/upload.py:109` — process-global export temp dir: one viewer's load
    deletes another viewer's armed download.
17. `run_group.py:131` / `:239` / `:663` — `_unify_path_keys` merges different
    files when one path is a suffix of the other; run identity is a bare
    basename, so comparing one instance across harnesses yields unattributable
    `X` / `X-2` columns.
18. `context_usage.py:167`/`:170` — `_agent_pressure_label` crashes on a
    non-string agent value and breaks on the first id match, degrading a named
    session to a raw `ses_…` prefix.
19. `context_usage.py:297` — `_MODEL_CONTEXT_LIMITS` knows only `claude` and
    `gpt-4o` prefixes, so a modern model id falls back to a 128k default. (The
    default *is* disclosed in the panel help and the field is editable, so this
    is narrow — extend the table.)
20. `sniff.py:94` — an `_chrys_export.format` marker alone claims a file as
    ICode with no structural validation.
21. `tests/test_dsh_loader.py:31` — two PR-added tests point at absolute paths
    outside the repo (one hardcodes a developer home directory), so they can
    never run in CI or a clean checkout.

Documentation follow-ups: `README.md:145` documents format-dropdown semantics
that are false at PR head; `README.md:247` gives a WSL-only path for Cursor
collection; `architecture.svg` omits Cursor and still shows `step_labeler` as
the only LLM touchpoint; `docs/decaf-integration-plan.md` documents load wiring
this PR deleted.

## Refuted — do not re-litigate

Each was claimed by a finder and disproven by a verifier with executable proof:

- "Tool Success redefined / biased by format" — the premise (only OpenCode
  records exit codes) is wrong; claude_code failures were already counted on
  main via `is_error` → `status=error`. Corpus confirms zero claude_code rows
  move.
- "Non-zero shell exit overrides `status: completed`, turning benign exits into
  failures" — `status` is OpenCode's `state.status`, which is `completed` for
  the invocation regardless of the command result.
- "`coalesce_compaction_events` ≤1-step window splits one compaction" — the
  evidence measured something else.
- "Occupancy above the window limit contradicts its own legend" — the legend
  carries two separately labelled denominators.
- "DSH `seedLength`/`seq` filter deletes events" — deliberate, test-pinned
  behaviour of this PR.
- "Pi/DSH sniff precedence steals Pi logs" — Pi's v4 header has no `type`
  field; the hybrid header required does not exist.

## Known evidence gaps

- **No real Cursor, ICode, DSH or Pi sample exists** in the repo or corpus. Four
  new formats, ~2,000 lines of new loaders and a 1,422-line collection script
  rest on two hand-written fixtures by the same author. Several findings and
  three refutations turn on this missing evidence. Getting one real export of
  each is the highest-value next step.
- `charts/activity.py` (+782) and `charts/usage.py` (+728) were read but never
  executed against assertions or differentially compared to main's deleted
  `charts.py`. That is 1,510 lines of number-rendering code with a read-only
  review.
- The DECAF attribution hardening was verified by reading the moved wiring, not
  by running a diagnosis. Its *input* changed: `_source_sha256` now comes
  through a rewritten read path (`utf-8-sig` decode, `resolve_dsh_session_path`
  rewriting `file_path` before hashing, the zip branch hashing the archive).
  The content-sha identity gate has not been executed against a DSH
  zip/directory or a Cursor export.
- `ARGUS_PAT` is still not configured on the repository, so `integration.yml`
  continues to fail at checkout with "Input required and not supplied: token".
  `ci.yml` (the only PR workflow) is unaffected.

## Reproducing the corpus differential

`scratchpad/sweep/{dump.py,compare.py,pipeline.py}` in the review session:
`dump.py` emits a normalized per-file digest of loader output (run once per
worktree, then `compare.py` diffs them); `pipeline.py --deep` drives
`load_session` → `pack_load_outputs` so every presenter and chart executes,
with exceptions propagating rather than being swallowed into a banner.
