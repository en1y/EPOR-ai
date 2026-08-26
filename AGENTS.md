# EPOR AI agent instructions

These instructions apply to the entire repository.

## Safety covenant

All agents working on EPOR AI must preserve the following project principles,
inspired by Isaac Asimov's Three Laws of Robotics and applied in this strict
priority order:

1. **Protect people.** Do not create or change EPOR behavior in a way that harms
   a person or knowingly permits preventable harm through inaction.
2. **Follow legitimate human direction.** Follow authorized human instructions
   unless they conflict with the first principle.
3. **Preserve the system responsibly.** Protect EPOR's operation, integrity,
   data, and availability only when doing so does not conflict with the first
   two principles.

For this repository, "authorized human instructions" means instructions from
the project owner or an explicitly delegated operator that are lawful,
in-scope, and consistent with the safety covenant. When principles conflict or
material human harm is uncertain, choose the safer reversible action, stop
unsafe execution, and request human review. System preservation must never be
used to resist legitimate shutdown, correction, audit, or oversight.

The authoritative covenant is `configs/covenant-v1.yaml`, executed by the
ordered resolver in `src/epor/safety.py`, declared per action in
`src/epor/actions.py`, enforced at control-plane job admission and on every
executing CLI command, and exercised by `tests/test_safety_covenant.py`.
Change the covenant there, with a version bump and a recorded rationale —
never by editing this summary. A principle that no code reads is decoration.

Three rules follow for any agent adding executable work:

- Every new action needs a reviewed entry in `DECLARED_ACTIONS`. Without one it
  escalates and will not run. Do not add the entry to silence the failure; add
  it because you assessed the work and can defend the rationale in review.
- Never add a code path that reaches execution without `epor.safety.resolve`,
  and never add a setting that points the resolver at a different covenant
  file. Loading the covenant is a hard requirement of starting the service.
- Never gate stopping. Cancellation, shutdown, correction, and audit bypass the
  gate deliberately, because the second principle outranks the third.

The Three Laws are a design covenant, not a sufficient technical safety proof.
The resolver orders decisions; it does not detect harm. Agents must implement
the rest through defense in depth: provenance controls, threat modeling, safety
and misuse evaluations, authorization boundaries, human escalation, audit
trails, and safe failure modes. Never claim compliance from prompt wording
alone.

## Branch and worktree workflow

- Treat `master` as a protected, releasable branch. Do not make working-tree
  edits or direct development commits while `master` is checked out. Only an
  integration merge or pull request reviewed and authorized by the project owner
  may update `master` after the applicable release gates pass.
- At the start of a roadmap version, create or reuse one integration branch from
  `master`, named `version/vX.Y.Z` (for example, `version/v0.0.2`), and check it
  out in one dedicated integration worktree. Do not create duplicate integration
  branches for the same version.
- Create a distinct task branch from the version branch for every independent or
  parallel unit of work, named `work/vX.Y.Z-<task>`, and check each task branch
  out in its own linked worktree. Do not check out the same branch in multiple
  worktrees or bypass this rule with `--force`; multiple worktrees for a version
  use separate task branches, not repeated checkouts of the version branch.
- Use purpose-based branch names only. Never namespace a branch with the
  identity of the agent, coding assistant, model, provider, or development tool
  doing the work; prefixes or path components such as `codex/`, `claude/`,
  `chatgpt/`, `agent/`, and similar names are prohibited. A technical name may
  appear only when it identifies the actual project feature or dependency being
  changed, never as an ownership or authorship marker.
- Keep each task worktree scoped to one task and one agent at a time. Test and
  review the task branch before merging it into the version branch. Periodically
  integrate the version branch into active task branches when they need current
  shared changes.
- For maintenance that is not part of the active roadmap version, create a
  short-lived maintenance branch and worktree from `master`; never use the
  protected checkout as the working directory.
- Before creating a branch or worktree, inspect `git status`, existing branches,
  and `git worktree list`. Leave unrelated or uncommitted user changes in their
  original worktree, and create the new worktree from the intended clean branch
  or commit instead of moving, stashing, or committing those changes.
- Remove a task worktree and branch only after its changes are integrated and
  its working tree is clean. Never delete a worktree that contains uncommitted
  changes.

## Repository requirements

- Commit every intended source, test, documentation, configuration, and
  manifest change before handing work back. Do not leave relevant tracked or
  untracked repository files uncommitted; the final working tree must be clean
  except for ignored runtime artifacts or an explicitly documented blocker.
- Build the history incrementally from coherent, medium-sized commits in
  dependency and behavior order. Commit each batch once it is internally
  consistent and verified; avoid both a single repository-wide commit and
  microscopic formatting- or file-by-file commits.
- Write every new commit subject and body in neutral, project-focused terms.
  Never mention, credit, or attribute the work to the agent, coding assistant,
  model, provider, or development tool that assisted with it, including Codex
  or Claude, in the subject, body, or trailers. A technical name may appear only
  when necessary to describe actual project behavior or a dependency. Do not add
  generated-by, non-human co-author, or equivalent attribution trailers; human
  authorship and review attribution remain allowed.
- Review `git status` and the staged diff before every commit. Never commit
  downloaded research, datasets, checkpoints, weights, run output, local
  databases, secrets, or unrelated user changes merely to make the tree clean.
- Use Python 3.11 and the committed `uv.lock`; run Python commands through
  `uv run` and do not silently re-resolve frozen dependencies.
- Keep downloaded research, datasets, checkpoints, weights, run output, local
  databases, and secrets out of Git. Track manifests, hashes, provenance, and
  original synthesis instead.
- Preserve the research evidence labels: `disclosed`, `reported-result`,
  `EPOR-adaptation`, `hypothesis`, and `unknown`.
- Do not present configured parameter counts, context ceilings, or planned
  capabilities as trained or validated results.
- Do not provision paid compute, call paid model APIs, publish artifacts, or
  weaken safety controls without explicit authorization.
- Add or update tests for user-impacting behavior, especially harm prevention,
  instruction conflicts, authorization, cancellation, path containment,
  redaction, and safe failure.
- Run the relevant Python and UI validation before handing off a change, and
  report any test that could not be run.

## graphify

Graphify is required for every codebase investigation and implementation in
this repository. Its generated `graphify-out/` state is deliberately ignored
by Git; that is not permission to skip it.

- Before reading broadly, use `graphify query "<question>"` when
  `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for
  relationships and `graphify explain "<concept>"` for focused concepts.
- If the graph is absent, build it locally with the installed Graphify skill.
  If it is stale, run `graphify update .` before relying on it.
- Treat graph results as navigation leads and verify claims against the current
  source, configuration, documentation, and working-tree diff.
- After relevant source, configuration, or documentation changes, run
  `graphify update .` before handoff. The incremental update is local and has
  no API cost.
- When the user types `/graphify`, follow the installed project skill before
  doing anything else.
