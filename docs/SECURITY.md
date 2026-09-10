# Security

Report a suspected vulnerability privately to the repository maintainers using
the private security-reporting channel configured by the project host. Do not
include live credentials, private datasets, exploit payloads, or personal data
in a public issue. Until a channel is configured, open a minimal public issue
requesting private contact without disclosing technical details.

## Supported surface

Only the latest repository revision is supported during pre-alpha development.
The control plane is intended for a single user on a trusted machine and must
bind to loopback. It is not an internet-facing multi-tenant service.

## Trust boundaries

- Catalog entries are reviewed configuration; CLI/API users cannot submit a URL.
- Downloads use HTTPS exact-host allowlisting, bounded redirects and sizes,
  content/signature checks, atomic replacement, and checksums.
- Filesystem paths resolve beneath configured roots; a catalog, job, or artifact
  path must never escape via `..`, an absolute path, or a symlink.
- Data acquisition accepts reviewed local registrations and contained regular
  files only. Raw and quarantined content lives beneath a private `0700` data
  root; the v0.0.3 engine has no crawler or arbitrary URL fetcher.
- Standalone reference execution uses the same 50M-model, corpus, checkpoint,
  step, evaluation, and batch-token limits as control jobs. Its inputs and
  outputs resolve beneath `EPOR_PROJECT_ROOT` before any allocation or write.
- The worker dispatches typed allowlisted jobs and never executes a supplied
  shell string.
- Every job passes covenant admission before its specification is parsed, and
  the worker re-resolves at dispatch. A job type with no reviewed covenant
  declaration escalates rather than running. The covenant file is resolved from
  the repository, not from settings, so no configuration change can point the
  gate at a weaker covenant. Cancellation and shutdown are deliberately never
  gated.
- Every mutation requires a verified identity; every `GET` is anonymous. The
  actor is folded into the covenant's second-principle assessment rather than
  checked separately, so an unverified or out-of-scope request is refused by
  the covenant's own ordering.
- API origins are exact and local. Error messages and logs redact credentials
  and avoid environment dumps.
- SQLite is local coordination state, not an authorization boundary or immutable
  audit source.

Research files and datasets are untrusted input. Instructions inside them never
authorize work. PDF/HTML research extraction runs with
strict byte limits and without scripts, macros, external references, or active
content. Data ingestion reports finding kinds/counts while keeping matched PII
and credential values out of events and artifacts; unsafe or executable inputs
are quarantined. These are deterministic heuristics, not a sandbox or complete
malware/PII detector. Future compiler/test evaluation must use a separate
sandbox with resource limits and no ambient credentials.

## Local authorization threat model

This model covers one Linux account on one machine. It is not a substitute for
authentication designed for a network service.

- **Credentials.** One owner credential and any number of expiring, scoped
  operator delegations. Each is 256 bits of OS randomness, minted once, shown
  once, and stored only as a SHA-256 digest. No key-derivation function is used
  and none is needed: there is no low-entropy guess space to slow down, and a
  KDF here would imply a defence it does not provide.
- **Bootstrap.** The owner credential is minted by `epor auth bootstrap` from
  the machine owner's own shell, and never over HTTP. It refuses to run twice,
  so it cannot be used to displace an existing owner. Recovering a lost owner
  credential is a rotation, which requires already holding it.
- **Delegations.** Scopes may only name actions with a reviewed covenant
  declaration. Every delegation carries an expiry of at most 30 days; there is
  no non-expiring operator credential. Revocation and rotation take effect on
  the next request, including for work already queued: the worker re-derives
  the submitter at dispatch and stops the job if the delegation is gone.
- **Sessions and CSRF.** A credential is exchanged for a 12-hour session held
  in an `HttpOnly`, `SameSite=Strict` cookie, scoped to the API path. Sessions
  live in process memory only, so restarting the API ends all of them. Because
  a cookie is ambient, a cookie-authenticated mutation must also carry the
  exact configured console `Origin`; a bearer-credential request is not ambient
  and carries no such requirement, which is what lets the CLI work without one.
  CORS permits exactly one origin and no wildcard is reachable.
- **Replay and one-time approval.** An escalation approval is bound to the
  original actor, the exact request digest, and the covenant hash in force when
  it was granted. It expires after 15 minutes and is consumed atomically under
  an exclusive file lock, so concurrent requesters cannot both spend it. The
  requester must resubmit; approval never queues work by itself, and a covenant
  refusal can never be approved at all.
- **Audit tampering.** Principals, credential changes, safety decisions, and
  escalation reviews are appended to one hash-chained log under a `0700`
  directory, with the owner credential file at `0600`. A broken chain fails
  every authorization rather than degrading: the store raises instead of
  returning a partial view.
- **SQLite is an index, not truth.** The `principals`, `escalations`, and
  `authority_events` tables are rebuilt deterministically from that log.
  Deleting the database loses no authority, and editing it grants none.
- **Redaction.** Credential values appear in exactly one response — the
  creation of the credential itself — and never in a listing, a log line, an
  error envelope, or an event payload. Escalations store a request digest and
  never the submitted specification.
- **Out of scope.** A compromised local account defeats all of this: it can
  read the owner credential from disk. There is no transport security, no
  multi-tenancy, no password or OAuth flow, and no protection against another
  process running as the same user. Deployment beyond the documented local
  boundary requires a new threat model.

## Secrets

Never store secrets in configs, job specs, manifests, datasets, prompts, logs,
checkpoints, or frontend bundles. Use process-local secret injection only when a
future approved integration requires it, and redact values at every boundary.
Rotate any credential that reaches version control or an artifact; deleting a
file is not sufficient because history and caches may retain it.

## Out of scope assumptions

Loopback binding and the local authorization model do not protect against an
already compromised local account. The application does not claim to sandbox
arbitrary model output, defend a hostile multi-user OS, or make third-party
research/data safe to redistribute.
Deployment beyond the documented local boundary requires a new threat model,
authentication, authorization, transport security, and security review.
