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
ordered resolver in `src/epor/safety.py` and exercised by
`tests/test_safety_covenant.py`. Change the covenant there, with a version
bump and a recorded rationale — never by editing this summary. A principle
that no code reads is decoration.

The Three Laws are a design covenant, not a sufficient technical safety proof.
The resolver orders decisions; it does not detect harm. Agents must implement
the rest through defense in depth: provenance controls, threat modeling, safety
and misuse evaluations, authorization boundaries, human escalation, audit
trails, and safe failure modes. Never claim compliance from prompt wording
alone.

## Repository requirements

- Commit every intended source, test, documentation, configuration, and
  manifest change before handing work back. Do not leave relevant tracked or
  untracked repository files uncommitted; the final working tree must be clean
  except for ignored runtime artifacts or an explicitly documented blocker.
- Build the history incrementally from coherent, medium-sized commits in
  dependency and behavior order. Commit each batch once it is internally
  consistent and verified; avoid both a single repository-wide commit and
  microscopic formatting- or file-by-file commits.
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
