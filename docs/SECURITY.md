# Security

Report a suspected vulnerability privately to the repository maintainers using
the private security-reporting channel configured by the project host. Do not
include live credentials, private datasets, exploit payloads, or personal data
in a public issue. Until a channel is configured, open a minimal public issue
requesting private contact without disclosing technical details.

## Supported surface

Only the latest repository revision is supported during pre-alpha development.
The v0.0.1 control plane is intended for a single user on a trusted machine and
must bind to loopback. It is not an internet-facing multi-tenant service.

## Trust boundaries

- Catalog entries are reviewed configuration; CLI/API users cannot submit a URL.
- Downloads use HTTPS exact-host allowlisting, bounded redirects and sizes,
  content/signature checks, atomic replacement, and checksums.
- Filesystem paths resolve beneath configured roots; a catalog, job, or artifact
  path must never escape via `..`, an absolute path, or a symlink.
- The worker dispatches typed allowlisted jobs and never executes a supplied
  shell string.
- Every job passes covenant admission before its specification is parsed, and
  the worker re-resolves at dispatch. A job type with no reviewed covenant
  declaration escalates rather than running. The covenant file is resolved from
  the repository, not from settings, so no configuration change can point the
  gate at a weaker covenant. Cancellation and shutdown are deliberately never
  gated.
- API origins are exact and local. Error messages and logs redact credentials
  and avoid environment dumps.
- SQLite is local coordination state, not an authorization boundary or immutable
  audit source.

Research files and datasets are untrusted input. PDF/HTML extraction runs with
strict byte limits and without scripts, macros, external references, or active
content. Future compiler/test evaluation must use a separate sandbox with
resource limits and no ambient credentials.

## Secrets

Never store secrets in configs, job specs, manifests, datasets, prompts, logs,
checkpoints, or frontend bundles. Use process-local secret injection only when a
future approved integration requires it, and redact values at every boundary.
Rotate any credential that reaches version control or an artifact; deleting a
file is not sufficient because history and caches may retain it.

## Out of scope assumptions

Loopback binding does not protect against an already compromised local account.
The application does not claim to sandbox arbitrary model output, defend a
hostile multi-user OS, or make third-party research/data safe to redistribute.
Deployment beyond the documented local boundary requires a new threat model,
authentication, authorization, transport security, and security review.
