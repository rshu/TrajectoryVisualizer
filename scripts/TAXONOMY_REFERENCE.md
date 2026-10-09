# Phase and Action Taxonomy Reference (v1)

This file defines the canonical hierarchy and brief meanings of each label used in trajectory categorization.

## Principles
- `phase` is the coarse workflow stage.
- `action` is the specific behavior inside that phase.
- Each step should map to one primary action, and the phase is derived from that action.

## Hierarchy

### understand
High-level stage for understanding requirements, context, tools, and existing code.

- `spec_intake`: first-pass interpretation of user request, task brief, or spec.
- `scope_clarification`: explicit clarification request or resolution of ambiguous scope.
- `constraint_extraction`: identification of hard constraints (API, infra, policy, deadline, compatibility).
- `tool_discovery`: discovering/selecting available tools or capabilities.
- `delegated_research`: spawning or using sub-agents for exploration.
- `external_research`: looking up external/web information.
- `file_discovery`: locating relevant files/paths/symbols (for example via grep/glob/search).
- `code_reading`: semantic reading of existing code/docs to understand behavior.
- `context_synthesis`: consolidating findings into a coherent mental model before planning/implementation.

### plan
Stage for turning understanding into an execution plan.

- `plan_management`: creating/updating TODOs, phase plans, or progress state.
- `task_breakdown`: splitting work into atomic tasks.
- `sequencing`: ordering tasks by execution priority or dependency.
- `dependency_mapping`: identifying dependency relationships and prerequisites.
- `risk_planning`: planning around expected failure modes or tradeoffs.

### implement
Stage for making changes.

- `implement_api_schema`: edits to API contracts, schema, protobuf, feature-gates, type definitions.
- `implement_runtime_logic`: edits to core runtime/business logic behavior.
- `implement_tests`: adding/updating tests.
- `implement_generated_artifacts`: updating generated files (openapi, protobuf outputs, deepcopy, conversion, etc.).
- `implement_config`: editing config/build/CI/environment wiring.
- `implement_docs`: editing docs/readme/spec text.
- `implement_refactor`: structural code cleanup without primary behavior change.
- `implement_migration`: migration/backfill/compatibility-transition code.

### debug
Stage for diagnosing and resolving failures.

- `debug_reproduction`: reproducing a bug or failing condition.
- `debug_root_cause`: identifying root cause from evidence.
- `debug_hypothesis_test`: validating or falsifying debugging hypotheses.
- `debug_fix_selection`: choosing the most appropriate fix strategy.

### validate
Stage for checking correctness after implementation/debugging.

- `validation_run`: running checks (tests/build/lint/commands).
- `validation_review`: interpreting or summarizing verification results between runs.
- `validation_lint`: lint/static-analysis validation.
- `validation_build`: compile/build validation.
- `validation_unit_tests`: unit-test validation.
- `validation_integration_tests`: integration-test validation.
- `validation_e2e_tests`: end-to-end/system-test validation.
- `validation_regression`: regression-specific validation.
- `validation_performance`: performance/benchmark validation.

### report
Stage for communicating status and handoff.

- `progress_update`: interim status update while work is in progress.
- `final_reporting`: final completion summary.
- `change_summary`: concise list of what changed.
- `blockers_risks`: unresolved blockers, caveats, or risks.
- `next_steps_handoff`: recommended follow-up actions or ownership handoff.

## Notes
- Not all actions must appear in every trajectory; zero-count actions are still valid taxonomy members.
- If a step could fit multiple actions, choose the dominant intent of that step.
- Reserved system labels are emitted outside this classification hierarchy and must be accepted by any sidecar consumer. They are not part of the LLM-assignable taxonomy above.
- Reserved: the phase/action pair `user`/`user_prompt` is emitted by step_labeler_v2 for user steps, and is also synthesized by the dashboard's label loader.
- Reserved: the phase/action pair `unknown`/`unknown` is the fallback both labelers emit for unparseable, invalid, or failed LLM output; step_labeler_v2 also assigns it to any non-user, non-assistant role.
