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
| Determinism | identical output on repeated loads, both versions — but this check ran **in-process**, so it could not see string-hash variation; the cross-process check later found backlog item 5 |
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
- `ff1c7e8` — **Parallel tool calls were double-counted and collapsed
  generation windows credited tokens to zero seconds.**
  `non_spawn_tool_seconds` now counts the interval **union** of
  `time_start`/`time_end` windows instead of their sum, and steps whose
  generation window collapses to `<= 0` are excluded from the output tok/s
  aggregation (the coverage figure already discloses the omission). Corrected
  `tool_time_total` on 661 OpenCode files.
- `a868bdf` + follow-up — **Converge produced a confident all-zeros comparison
  for a trajectory it could not read.** See the post-merge test campaign below.
- `1faaaf2` — **Delegation wall-clock was summed across parallel spawns and the
  "Delegated" chip was unreachable for formats that also time ordinary tool
  calls.** `delegation_time_total` now uses the existing per-step
  `spawn_wait_seconds` (max, not sum — the same quantity
  `step_duration_excluding_spawn` subtracts), and the chip is emitted whenever
  delegation was timed. Three parallel 60s spawns in a 70s step reported 180s;
  now 60s. OpenCode's 456s of sub-agent wall-clock was invisible on 239/2,500
  trajectories; now shown alongside the tool-timing chips.

## Post-merge test campaign (2026-09-23)

A 66-agent campaign (6.5M tokens, 3,198 tool calls) tested twelve surfaces —
coverage mapping, chart execution, live DECAF attribution, live-app E2E,
new-format synthesis, ingest fuzzing, render injection, concurrency/global
state, test-suite quality, review of the post-merge fixes, corpus-scale numeric
regression, and CLI/report/offline — with every claimed defect adversarially
verified, plus a completeness critic that mutation-tested the result.

**Merge review verdict: no behavioural regressions.** Pre-merge main's original
199-test suite was executed against current production code. 28 fail, and every
one is mechanical: 24 are source-introspection or import tests broken by the
`insight.py` → `ui/`+`presenters/` split, 2 are the intentional new-format
additions to `FORMAT_LABELS`, 1 is a deliberate banner rename ("output tok/s" →
"gen tok/s"), and 1 is `analytics.tool_time_share`, an internal value that
reaches no rendered surface (verified: it appears only in the LLM brief, where
it correctly reads `n/a`).

**Suite:** 602 → 818 tests (11 new files, ~203 tests), 0 red, ruff clean.
**Coverage:** 74.2% → 78.2% combined (+4.0pp); 520 statements and 308 branch
destinations newly covered. Gains are concentrated where there was nothing:
`presenters/patterns.py` +49.1pp, `ui/attribution_tab.py` +45.5pp,
`charts/label_charts.py` +32.6pp, `formats/claude_code.py` +19.3pp.

**Mutation testing (critic, 14 mutants): 12 killed, 2 survived.** The single
best piece of evidence that these are real tests: reverting the PR's headline
numeric correction (`metrics.py:608`, the `metadata.exit` failure branch)
survived all 615 committed tests but is killed by 4 of the new
`test_format_synthesis.py` tests. Both survivors were fixed — see below.

**Known-wrong numbers still displayed** (backlog items 1, 3, 4 below): avg cache
% can render 25,386.2% with a green health verdict, Input can render
-3,175,801, and Codex peak context pressure is ~2x too high on 500/500 files.
The dashboard is sound as an exploratory tool; **numbers destined for a paper
should come from a build with those items fixed.**

### Fixed during the campaign

- `build_comparison_report` (converge) guarded only on `_error`, but
  `load_trajectory` returns an unrecognised JSON **object** unchanged with no
  `_error` at all. A file like `{"hello":"world"}` therefore sailed through,
  yielded 0 steps, and reproduced the full 25% batch-aggregate distortion that
  `a868bdf` was written to eliminate — recorded as `success`, exit 0, with no
  trace anywhere in the output. The guard now also refuses a side that parsed
  no steps, naming the detected format.
- `tests/test_render_injection.py`'s presenter sweep could not fail for the
  agent-name sink: the 30-char payload is truncated to 20 chars by
  `metrics.py:40` before reaching `_kpi_breakdown_html`, so deleting
  `html.escape` at `presenters/overview.py:144` survived the entire suite. A
  16-char payload (`<svg onload=x()>`) survives that truncation intact and the
  Overview KPI card is on the default-visible tab. Added a short-payload case;
  verified it goes red under exactly that mutation.
- `tests/test_attribution_live.py::_RootRestoringTest` restored DECAF's
  process-global root on cleanup but never established it on entry, so
  pollution from `test_attribution.py` leaked in. 21 new tests had been added
  into that blast radius. Now configured in `setUp` as well.

## Open backlog (confirmed, not yet fixed)

Ranked. Each was reproduced against current code by an adversarial verifier.
The campaign added ~32 distinct defects (37 confirmed findings, 5 pairs
double-counted across surfaces); the highest-value ones are folded in below.

1. `metrics.py:382` — `cache_ratio = cache_read / tok_total` is unbounded, and
   `:575` averages it into `avg_cache_ratio`. Providers that report `total`
   excluding cache reads make this exceed 100%: a real corpus file renders
   **avg cache % = 25,386.2%** with a **green** "strong" health verdict at
   `:846`. Clamp it, and make the verdict refuse out-of-range input.
2. `report.py:287` — `_mixed_md_to_html` passes any markdown line starting with
   `<` through **verbatim**, so trajectory-derived strings containing markup
   become live HTML in the exported report. (A sweep of all 2,500 generated
   reports found zero non-plotly `<script>` in `<body>`, so real corpus content
   does not currently reach the sink — but nothing prevents it.)
3. `metrics.py:571`/`:579` — provider token fields are never clamped, so
   `input_tokens` can be **negative** (-3,175,801 on real data; 370 assistant
   steps across the corpus) and `output_input_ratio` is computed from it.
4. `context_usage.py:288-297` — `_MODEL_CONTEXT_LIMITS` knows only `claude` and
   `gpt-4o` prefixes. No Codex step carries a `model_id` at all, so all 500
   Codex trajectories fall back to the 128k default while their exports declare
   `model_context_window = 258400` — every Codex peak-pressure percentage is
   **2.02x too high**, flipping the verdict colour on 163/500. (This is the
   escalation of former item #19: the fix is to read the declared window, not
   just extend the table.)
5. `patterns.py:513-518` — `compute_plan_metrics` iterates a `set[str]`, so
   plan item order follows per-process string-hash randomisation.
   `charts/activity.py:364` sorts with a non-total key and a stable sort, so
   the **Plan Progress Timeline's row order changes between runs on identical
   input** (116/502 trajectories with ≥2 plan items), and `rendering.py:1361`
   slices `stalled[:2]`, so *which* stalled items the Overview names also
   varies. A figure in a paper is not reproducible from the same file. Fix:
   `dict.fromkeys(...)` for insertion order plus a total sort key.
6. `charts/usage.py:148` and `:698` — `cache_write` is dropped from the Token
   Usage and Agent Token charts (computed at `:131`, passed only on the
   `opencode`/`codearts` branch; `compute_agent_summary` never accumulates it
   at all), so the stacked bars do not sum to the reported total. `:144` also
   draws reasoning tokens **twice** on OpenCode.
7. `formats/claude_code.py:23` — `total = inp + out + cache_read + cache_write`
   double-counts on every one of the 500 claude_code trajectories (median ~4%
   inflation of reported total tokens).
8. `run_group.py:377`/`:415`/`:671` — a zero-step load counts as a successful
   run, wins every "best" flag and shifts the consensus threshold; missing
   metrics are coerced to `0` and highlighted green as "best". `:723` calls
   `load_trajectory` with no `try/except`, so one bad file aborts a batch
   (triggers: `cursor.py:35` unguarded `datetime.fromtimestamp`, `dsh.py:962`
   `RuntimeError` on an encrypted zip member). Fix the loader-local bugs **and**
   wrap `:723`. This is the same class as the converge blocker fixed above.
9. `ui/comparison_tab.py` — a stale Run Group scorecard is never cleared on a
   new load, so the panel can show another trajectory's numbers. The file is
   50% covered and carries a confirmed defect with no regression test.
10. `formats/icode.py:405`/`:190` — per-call `_chrys_timing` is never read, so
    every real ICode session loses all tool timing; `:120` leaves real ICode
    tool names uncanonicalised (`_ICODE_TOOL_NAMES` lacks `read_file`,
    `write_file`, `edit_file`, `zsh`).
11. `formats/dsh.py:235` — `component_total = inp + out + reasoning +
    cache_read + cache_write` double-counts; `:279` derives tool failure only
    from `data.error`, missing the item-level flag (3/342 real DSH tool results
    carry it); `:632` silently discards an entire child session on one
    malformed line.
12. `context_usage.py:218` — occupancy omits `cache_write`, understating window
    usage for claude_code/DSH/pi.
13. `context_usage.py:811`, `:1148`, `:383`, `:1028` — the compaction chip
    counts raw un-coalesced events while the chart coalesces them; inferred
    `occupancy_drop` compactions are ignored when choosing the window start;
    the splice reads its own appended artifacts for the second and later
    cliffs; a zero-usage final turn zeroes the whole composition panel.
14. `cursor.py:263` — `chars/4` token estimates are written into `token_usage`
    and shown as measured API usage. The `_capabilities` flags that mark them
    as estimates (written by `cursor.py`, `icode.py`, `dsh.py`) have **no
    consumer anywhere**.
15. `report.py:151` — `write_report_file` opens the destination with no guard,
    and `:176-179` derives the report name from the basename stem, so two
    trajectories sharing a task stem silently overwrite each other's report.
16. `charts/swimlanes.py:274` — `build_tool_outcome_timeline` is the last
    consumer classifying tool outcomes with its own inline predicate instead of
    the shared `tool_failure.tool_call_failed`, so the timeline disagrees with
    every other surface about which calls failed.
17. `charts/usage.py:490` — the Tool Call Duration chart plots the **sum** of
    per-call durations while the "Tool time" chip directly above it reports the
    union, so the two contradict each other on parallel calls.
18. `shell_cmd.py:510` — `return base or interpreter` emits `-` as a command
    label for certain shell forms.
19. `presenters/workflow.py:44`/`:105` — role filtering drops steps whose label
    set does not intersect the filter, including steps that should match.
20. `presenters/overview.py:352` — the summary banner is computed but never
    rendered, dropping both the loaded filename and main's OpenCode/CodeArts
    "Cache Read is a running conversation prefix" caveat.
21. `llm_config.py:38` — `load_env_files()` runs as the first statement of
    `build_ui()` and injects every key from the launch directory's `.env`,
    including `AWE_DECAF_PATH` (arbitrary code import into the attribution
    backend).
22. `issue_judge.py:263` / `overview_tab.py:471` — judge failure renders the raw
    exception, including the endpoint URL and any credential in it, into the
    Issues HTML, bypassing the module's own scrubbing helper.
23. `report.py:34` — `_PLOTLY_CDN = "cdn"`, so the "standalone" report pings
    `cdn.plot.ly` when opened. (Note: the inline plotly bundle also contains
    one `cdn.plot.ly` occurrence — its `topojsonURL` default — so a naive
    string count does not distinguish the two.)
24. `issue_judge.py:210` — LLM output is hard-coded to Simplified Chinese and
    renders into the otherwise-English dashboard with no setting.
25. `ui/upload.py:109-117` — process-global export scratch dir: one viewer's
    load can delete another viewer's armed download.
26. `run_group.py:131`/`:239`/`:663` — `_unify_path_keys` merges different files
    when one path is a suffix of the other; run identity is a bare basename, so
    comparing one instance across harnesses yields unattributable `X` / `X-2`
    columns. `:765-790` `_best_worst_flags` ranks incomparable columns.
27. `formats/parse.py:21` — `splitlines(keepends=True)` also splits on U+2028,
    U+2029 and U+0085, so a rollout containing those characters fails to load.
    Bare `NaN`/`Infinity` JSON literals pass with no `_error` (no
    `parse_constant`) and crash downstream at `metrics.py:569`.
28. `converge/cli.py:237` — advertises `prog="trajectory-converge"`, a console
    script `pyproject.toml` does not define, while `python -m trajviz.converge`
    launches the Gradio app instead. The CLI is only reachable as
    `python -m trajviz.converge.cli` and is 3.5% covered.
29. `batch.build_batch_report` — `per_task` drops `compared_format` and
    `confidence`, so batch output is strictly less informative than the
    pairwise report about whether a comparison was trustworthy.
30. `sniff.py:94` — an `_chrys_export.format` marker alone claims a file as
    ICode with no structural validation.
31. `tests/test_workflow_detail_ui.py:66`/`:71`, `tests/test_workflow_filtering.py`
    — three tests read `trajviz/insight/styles.py` through a cwd-relative path
    and fail whenever pytest is launched from outside the repo root.
32. `tests/test_dsh_loader.py:885`, `tests/test_spawn_subagent_annotation.py:36`
    — the two remaining skips point at absolute paths outside the repo (one a
    developer home directory), so they can never run in CI.

Superseded: the pre-merge backlog's item #1 (`generation_seconds` summing
overlapping tool durations, `output_tokens_per_sec` differing from pre-merge
main on 1,599/2,500 files, median +20.9%, max +234%) was fixed by `ff1c7e8`.

Also still open from the pre-merge review:

33. `context_usage.py:167`/`:170` — `_agent_pressure_label` crashes on a
    non-string agent value and breaks on the first id match, degrading a named
    session to a raw `ses_…` prefix.

Documentation follow-ups: `README.md:145` documents format-dropdown semantics
that are false at PR head; `README.md:247` gives a WSL-only path for Cursor
collection; `architecture.svg` omits Cursor and still shows `step_labeler` as
the only LLM touchpoint; `docs/decaf-integration-plan.md` documents load wiring
this PR deleted. `trajviz/converge/app.py:16` passes `css`/`theme` to the
`Blocks` constructor, which Gradio 6 warns about and will eventually remove —
the CSS *is* still applied today (verified by launching and inspecting the
served page), and the sibling Insight app already passes it to `launch()`.

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
