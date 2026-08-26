"""Durable local authority: principals, credentials, sessions, escalations.

This module answers two questions the covenant resolver deliberately cannot:
who is asking, and what did a human decide about work the covenant would not
permit on its own.  It is not a second gate bolted on after resolution.  The
actor and the scope it holds are folded into the covenant's Principle 2
assessment (see :func:`epor.actions.resolve_action`), because "follow
legitimate human direction" is exactly the principle that an unverified or
out-of-scope request engages.

File truth is the authority.  Every principal, credential change, safety
decision, and escalation review is appended to one hash-chained log under a
private directory; the SQLite tables are a rebuildable query index over that
log and are never consulted to decide anything.  Sessions are the one piece of
state that is deliberately *not* durable: they live in process memory, so
restarting the API ends every browser session.

Threat model, in one line: this protects a single-user loopback machine from
an unauthenticated browser tab and from a delegated operator exceeding its
delegation.  It does not protect against a compromised local account, which
can read the owner token from disk.  That limitation is documented in
``docs/SECURITY.md`` and is not fixed by anything here.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from epor.actions import DECLARED_ACTIONS

from .errors import (
    AuthorizationDeniedError,
    ControlError,
    EscalationError,
    EscalationNotFoundError,
)
from .security import redact_text
from .truth import ChainedLog, TruthStoreError, parse_utc, utc_text

AUTHORITY_LOG_FILENAME = "authority.jsonl"
OWNER_TOKEN_FILENAME = "owner.token"

# 256 bits of OS randomness. Credentials are stored only as SHA-256 digests:
# with this much entropy there is no guess space for a KDF to slow down, so a
# password hash here would buy nothing while implying a defence it does not add.
TOKEN_BYTES = 32
MAX_DELEGATION = timedelta(days=30)
SESSION_TTL = timedelta(hours=12)
APPROVAL_TTL = timedelta(minutes=15)
MAX_LABEL_LENGTH = 120
MAX_RATIONALE_LENGTH = 2_000


def utc_now() -> datetime:
    return datetime.now(UTC)


class Role(StrEnum):
    OWNER = "owner"
    DELEGATED_OPERATOR = "delegated_operator"
    ANONYMOUS = "anonymous"


class EscalationState(StrEnum):
    OPEN = "open"
    APPROVED = "approved"
    REFUSED = "refused"
    CONSUMED = "consumed"
    EXPIRED = "expired"


class OwnerAlreadyBootstrappedError(ControlError):
    status_code = 409
    code = "owner_already_bootstrapped"


class PrincipalNotFoundError(ControlError):
    status_code = 404
    code = "principal_not_found"


class InvalidDelegationError(ControlError):
    status_code = 422
    code = "invalid_delegation"


@dataclass(frozen=True, slots=True)
class Actor:
    """The verified requester of one action, or the anonymous reader."""

    id: str
    role: Role
    scopes: frozenset[str] = frozenset()
    label: str = ""

    @property
    def is_owner(self) -> bool:
        return self.role is Role.OWNER

    @property
    def is_authenticated(self) -> bool:
        return self.role is not Role.ANONYMOUS

    def may(self, action_id: str) -> bool:
        """Whether this actor holds a delegation covering ``action_id``.

        The owner holds every declared action by definition; that is what
        being the project owner means under the covenant's second principle.
        """

        return self.is_owner or action_id in self.scopes


ANONYMOUS = Actor(id="anonymous", role=Role.ANONYMOUS)

# The shell on the trusted single-user machine is the owner's own shell. There
# is no second local identity to distinguish it from, and pretending otherwise
# would be theatre: whoever holds that shell already holds the token file.
LOCAL_CLI = Actor(id="local-cli", role=Role.OWNER, label="local command line")


@dataclass(frozen=True, slots=True)
class Principal:
    """One durable identity and the digest of its current credential."""

    id: str
    role: Role
    label: str
    scopes: frozenset[str]
    credential_sha256: str
    created_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    rotated_at: datetime | None = None

    def active(self, now: datetime | None = None) -> bool:
        current = now or utc_now()
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or self.expires_at > current

    def actor(self) -> Actor:
        return Actor(id=self.id, role=self.role, scopes=self.scopes, label=self.label)


@dataclass(frozen=True, slots=True)
class Escalation:
    """A request the covenant would not admit, awaiting or carrying a decision.

    The raw specification is deliberately absent.  An escalation records who
    asked, for what action, the digest of the exact request, and why the
    covenant stopped it — enough to review and to bind an approval, and not
    enough to leak whatever the caller submitted.
    """

    id: str
    actor_id: str
    actor_role: Role
    action_id: str
    request_digest: str
    covenant_sha256: str
    binding_priority: int | None
    reasons: tuple[str, ...]
    created_at: datetime
    state: EscalationState = EscalationState.OPEN
    decided_by: str | None = None
    decided_at: datetime | None = None
    rationale: str | None = None
    approval_expires_at: datetime | None = None
    consumed_at: datetime | None = None
    consumed_job_id: str | None = None

    def effective_state(self, now: datetime | None = None) -> EscalationState:
        """Expiry is computed, never swept.

        A background expiry job is one more thing that can fail open.  An
        approval whose window has passed is expired the moment anyone looks,
        including the consumption path.
        """

        current = now or utc_now()
        if (
            self.state is EscalationState.APPROVED
            and self.approval_expires_at is not None
            and self.approval_expires_at <= current
        ):
            return EscalationState.EXPIRED
        return self.state


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    principal_id: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class IssuedCredential:
    """A freshly minted token, shown once and never stored in clear."""

    principal: Principal
    token: str


def credential_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def request_digest(action_id: str, spec: Any) -> str:
    """Digest the exact request an approval is allowed to authorize.

    Canonical JSON, so a re-submitted request produces the same digest and any
    change to the specification produces a different one.  The specification
    itself is never persisted alongside it.
    """

    canonical = json.dumps(
        {"action": action_id, "spec": spec},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _clean_label(value: str) -> str:
    label = redact_text(" ".join(value.split()))[:MAX_LABEL_LENGTH]
    if not label:
        raise InvalidDelegationError("a delegation label is required")
    return label


def _validated_scopes(scopes: object) -> frozenset[str]:
    if not isinstance(scopes, (list, tuple, set, frozenset)) or not scopes:
        raise InvalidDelegationError("a delegation must name at least one declared action")
    requested = {str(item) for item in scopes}
    unknown = sorted(requested - set(DECLARED_ACTIONS))
    if unknown:
        raise InvalidDelegationError(
            "a delegation may only name reviewed declared actions",
            details={"unknown_actions": unknown},
        )
    return frozenset(requested)


def _validated_expiry(expires_at: datetime, now: datetime) -> datetime:
    if expires_at.tzinfo is None:
        raise InvalidDelegationError("delegation expiry must be timezone-aware")
    expiry = expires_at.astimezone(UTC)
    if expiry <= now:
        raise InvalidDelegationError("delegation expiry must be in the future")
    if expiry - now > MAX_DELEGATION:
        raise InvalidDelegationError("a delegation may not exceed 30 days")
    return expiry


class AuthorityStore:
    """Hash-chained file truth for identity and human safety decisions.

    Every mutating method appends exactly one record.  Read methods replay the
    log, so a corrupt chain fails every authorization rather than degrading to
    an ungoverned service.
    """

    def __init__(self, safety_root: Path) -> None:
        self.root = safety_root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        # A directory that already existed with looser modes is tightened here
        # rather than trusted; the token beneath it is only as private as it is.
        os.chmod(self.root, 0o700)
        self._log = ChainedLog(self.root / AUTHORITY_LOG_FILENAME)
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    # ---- replay -------------------------------------------------------

    def records(self) -> list[dict[str, Any]]:
        """Return the verified authority chain, oldest first."""

        # ponytail: full replay per call. The log holds a handful of records on
        # a single-user machine; index it incrementally only if that changes.
        return self._log.load()

    def _state(self) -> tuple[dict[str, Principal], dict[str, Escalation]]:
        return _replay(self.records())

    def principals(self) -> list[Principal]:
        principals, _ = self._state()
        return sorted(principals.values(), key=lambda item: item.created_at)

    def principal(self, principal_id: str) -> Principal:
        principals, _ = self._state()
        found = principals.get(principal_id)
        if found is None:
            raise PrincipalNotFoundError("no such principal")
        return found

    def owner(self) -> Principal | None:
        return next((item for item in self.principals() if item.role is Role.OWNER), None)

    def escalations(
        self,
        *,
        state: EscalationState | None = None,
        limit: int = 200,
    ) -> list[Escalation]:
        _, escalations = self._state()
        now = utc_now()
        items = sorted(escalations.values(), key=lambda item: item.created_at, reverse=True)
        if state is not None:
            items = [item for item in items if item.effective_state(now) is state]
        return items[:limit]

    def escalation(self, escalation_id: str) -> Escalation:
        _, escalations = self._state()
        found = escalations.get(escalation_id)
        if found is None:
            raise EscalationNotFoundError("no such escalation")
        return found

    # ---- bootstrap and credentials ------------------------------------

    @property
    def owner_token_path(self) -> Path:
        return self.root / OWNER_TOKEN_FILENAME

    def bootstrap_owner(self, *, label: str = "project owner") -> IssuedCredential:
        """Mint the single owner credential on first run.

        Refuses to run twice.  Replacing a lost owner credential is a rotation,
        which requires proving you already hold it, not a second bootstrap that
        anyone reaching the machine could use to install themselves as owner.
        """

        with self._lock:
            if self.owner() is not None:
                raise OwnerAlreadyBootstrappedError(
                    "an owner credential already exists; rotate it instead"
                )
            now = utc_now()
            token = secrets.token_urlsafe(TOKEN_BYTES)
            principal = Principal(
                id=str(uuid4()),
                role=Role.OWNER,
                label=_clean_label(label),
                scopes=frozenset(),
                credential_sha256=credential_digest(token),
                created_at=now,
            )
            self._append_principal(principal, actor_id=principal.id)
            self._write_owner_token(token)
            return IssuedCredential(principal=principal, token=token)

    def _write_owner_token(self, token: str) -> None:
        path = self.owner_token_path
        temporary = path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(f"{token}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise TruthStoreError("cannot persist the owner credential") from exc

    def read_owner_token(self) -> str | None:
        try:
            return self.owner_token_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def create_delegation(
        self,
        *,
        label: str,
        scopes: object,
        expires_at: datetime,
        by: Actor,
    ) -> IssuedCredential:
        """Issue a scoped, expiring operator credential. Owner only."""

        self._require_owner(by, "create a delegation")
        now = utc_now()
        validated_scopes = _validated_scopes(scopes)
        expiry = _validated_expiry(expires_at, now)
        with self._lock:
            token = secrets.token_urlsafe(TOKEN_BYTES)
            principal = Principal(
                id=str(uuid4()),
                role=Role.DELEGATED_OPERATOR,
                label=_clean_label(label),
                scopes=validated_scopes,
                credential_sha256=credential_digest(token),
                created_at=now,
                expires_at=expiry,
            )
            self._append_principal(principal, actor_id=by.id)
            return IssuedCredential(principal=principal, token=token)

    def rotate(self, principal_id: str, *, by: Actor) -> IssuedCredential:
        """Replace a credential in place, keeping identity, scope, and expiry."""

        self._require_owner(by, "rotate a credential")
        with self._lock:
            existing = self.principal(principal_id)
            if existing.revoked_at is not None:
                raise InvalidDelegationError("a revoked principal cannot be rotated")
            now = utc_now()
            token = secrets.token_urlsafe(TOKEN_BYTES)
            rotated = replace(
                existing,
                credential_sha256=credential_digest(token),
                rotated_at=now,
            )
            self._log.append(
                {
                    "kind": "credential.rotated",
                    "recorded_at": utc_text(now),
                    "principal_id": existing.id,
                    "credential_sha256": rotated.credential_sha256,
                    "by": by.id,
                }
            )
            self._drop_sessions(existing.id)
            return IssuedCredential(principal=rotated, token=token)

    def revoke(self, principal_id: str, *, by: Actor) -> Principal:
        """Revoke a delegation immediately. The owner credential is not revocable."""

        self._require_owner(by, "revoke a delegation")
        with self._lock:
            existing = self.principal(principal_id)
            if existing.role is Role.OWNER:
                raise InvalidDelegationError(
                    "the owner credential is rotated, never revoked; revoking it would "
                    "lock the covenant's own oversight out of the system"
                )
            if existing.revoked_at is not None:
                return existing
            now = utc_now()
            self._log.append(
                {
                    "kind": "principal.revoked",
                    "recorded_at": utc_text(now),
                    "principal_id": existing.id,
                    "by": by.id,
                }
            )
            self._drop_sessions(existing.id)
            return replace(existing, revoked_at=now)

    def _require_owner(self, actor: Actor, what: str) -> None:
        if not actor.is_owner:
            raise AuthorizationDeniedError(f"only the project owner may {what}")

    def _append_principal(self, principal: Principal, *, actor_id: str) -> None:
        self._log.append(
            {
                "kind": "principal.created",
                "recorded_at": utc_text(principal.created_at),
                "by": actor_id,
                "principal": {
                    "id": principal.id,
                    "role": principal.role.value,
                    "label": principal.label,
                    "scopes": sorted(principal.scopes),
                    "credential_sha256": principal.credential_sha256,
                    "created_at": utc_text(principal.created_at),
                    "expires_at": utc_text(principal.expires_at),
                },
            }
        )

    # ---- authentication and sessions ----------------------------------

    def authenticate(self, token: str) -> Principal | None:
        """Return the active principal holding ``token``, if any.

        Comparison is constant-time against every candidate digest, and every
        candidate is examined, so neither timing nor early exit reveals which
        digest was closest.
        """

        if not token or len(token) > 512:
            return None
        digest = credential_digest(token)
        now = utc_now()
        matched: Principal | None = None
        for principal in self.principals():
            if secrets.compare_digest(principal.credential_sha256, digest):
                matched = principal
        if matched is None or not matched.active(now):
            return None
        return matched

    def open_session(self, principal: Principal) -> Session:
        """Exchange a verified credential for a short in-memory session."""

        with self._lock:
            self._prune_sessions()
            session = Session(
                id=secrets.token_urlsafe(TOKEN_BYTES),
                principal_id=principal.id,
                expires_at=utc_now() + SESSION_TTL,
            )
            self._sessions[session.id] = session
            return session

    def session_actor(self, session_id: str | None) -> Actor | None:
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.expires_at <= utc_now():
                self._sessions.pop(session.id, None)
                return None
        principals, _ = self._state()
        principal = principals.get(session.principal_id)
        # Revocation, rotation, and expiry take effect on the next request
        # rather than waiting for the session's own clock to run out.
        if principal is None or not principal.active():
            with self._lock:
                self._sessions.pop(session.id, None)
            return None
        return principal.actor()

    def close_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)

    def _drop_sessions(self, principal_id: str) -> None:
        self._sessions = {
            key: value
            for key, value in self._sessions.items()
            if value.principal_id != principal_id
        }

    def _prune_sessions(self) -> None:
        now = utc_now()
        self._sessions = {
            key: value for key, value in self._sessions.items() if value.expires_at > now
        }

    # ---- safety decisions and escalations -----------------------------

    def record_decision(
        self,
        *,
        actor: Actor,
        action_id: str,
        surface: str,
        outcome: str,
        binding_priority: int | None,
        reasons: tuple[str, ...] | list[str],
        covenant_sha256: str,
        request_digest_value: str | None = None,
    ) -> None:
        """Append one sanitized covenant resolution to durable safety truth."""

        self._log.append(
            {
                "kind": "safety.decision",
                "recorded_at": utc_text(utc_now()),
                "actor_id": actor.id,
                "actor_role": actor.role.value,
                "action_id": action_id[:64],
                "surface": surface,
                "outcome": outcome,
                "binding_priority": binding_priority,
                "reasons": [redact_text(reason)[:500] for reason in reasons],
                "covenant_sha256": covenant_sha256,
                "request_digest": request_digest_value,
            }
        )

    def open_escalation(
        self,
        *,
        actor: Actor,
        action_id: str,
        request_digest_value: str,
        covenant_sha256: str,
        binding_priority: int | None,
        reasons: tuple[str, ...] | list[str],
    ) -> Escalation:
        """Record work a human must decide on. No job is created."""

        now = utc_now()
        escalation = Escalation(
            id=str(uuid4()),
            actor_id=actor.id,
            actor_role=actor.role,
            action_id=action_id[:64],
            request_digest=request_digest_value,
            covenant_sha256=covenant_sha256,
            binding_priority=binding_priority,
            reasons=tuple(redact_text(reason)[:500] for reason in reasons),
            created_at=now,
        )
        self._log.append(
            {
                "kind": "escalation.opened",
                "recorded_at": utc_text(now),
                "escalation": {
                    "id": escalation.id,
                    "actor_id": escalation.actor_id,
                    "actor_role": escalation.actor_role.value,
                    "action_id": escalation.action_id,
                    "request_digest": escalation.request_digest,
                    "covenant_sha256": escalation.covenant_sha256,
                    "binding_priority": escalation.binding_priority,
                    "reasons": list(escalation.reasons),
                    "created_at": utc_text(now),
                },
            }
        )
        return escalation

    def decide_escalation(
        self,
        escalation_id: str,
        *,
        approve: bool,
        by: Actor,
        rationale: str,
    ) -> Escalation:
        """Approve or refuse an open escalation. Owner only, once.

        Approving an undeclared action is impossible by construction: there is
        no reviewed declaration behind it and no handler for it, so approval
        would be a human inventing permission for work nobody assessed.
        """

        self._require_owner(by, "decide an escalation")
        cleaned = redact_text(" ".join(rationale.split()))[:MAX_RATIONALE_LENGTH]
        if not cleaned:
            raise EscalationError("an escalation decision requires a written rationale")

        with self._lock:
            existing = self.escalation(escalation_id)
            now = utc_now()
            current = existing.effective_state(now)
            if current is not EscalationState.OPEN:
                raise EscalationError(f"escalation is {current.value} and cannot be decided")
            if approve and existing.action_id not in DECLARED_ACTIONS:
                raise EscalationError(
                    "only an action with a reviewed covenant declaration can be approved"
                )
            state = EscalationState.APPROVED if approve else EscalationState.REFUSED
            approval_expiry = now + APPROVAL_TTL if approve else None
            self._log.append(
                {
                    "kind": "escalation.decided",
                    "recorded_at": utc_text(now),
                    "escalation_id": existing.id,
                    "state": state.value,
                    "decided_by": by.id,
                    "rationale": cleaned,
                    "approval_expires_at": utc_text(approval_expiry),
                }
            )
            return replace(
                existing,
                state=state,
                decided_by=by.id,
                decided_at=now,
                rationale=cleaned,
                approval_expires_at=approval_expiry,
            )

    def consume_approval(
        self,
        *,
        actor: Actor,
        action_id: str,
        request_digest_value: str,
        covenant_sha256: str,
    ) -> Escalation | None:
        """Atomically spend a matching approval, or return ``None``.

        The approval must match the original actor, the exact request digest,
        and the covenant in force when it was granted.  Two callers racing the
        same approval see exactly one success, because the match and the
        consumption record are decided under one exclusive file lock.
        """

        consumed: dict[str, Escalation] = {}

        def build(committed: list[dict[str, Any]]) -> dict[str, Any] | None:
            _, escalations = _replay(committed)
            now = utc_now()
            candidates = [
                item
                for item in escalations.values()
                if item.effective_state(now) is EscalationState.APPROVED
                and item.actor_id == actor.id
                and item.action_id == action_id
                and item.request_digest == request_digest_value
                and item.covenant_sha256 == covenant_sha256
            ]
            if not candidates:
                return None
            chosen = min(candidates, key=lambda item: item.created_at)
            consumed["escalation"] = replace(
                chosen, state=EscalationState.CONSUMED, consumed_at=now
            )
            return {
                "kind": "escalation.consumed",
                "recorded_at": utc_text(now),
                "escalation_id": chosen.id,
            }

        with self._lock:
            record = self._log.transact(build)
        if record is None:
            return None
        return consumed["escalation"]

    def attach_job(self, escalation_id: str, job_id: str) -> None:
        """Link a consumed approval to the job it authorized."""

        self._log.append(
            {
                "kind": "escalation.attached",
                "recorded_at": utc_text(utc_now()),
                "escalation_id": escalation_id,
                "job_id": job_id,
            }
        )


def _replay(records: list[dict[str, Any]]) -> tuple[dict[str, Principal], dict[str, Escalation]]:
    """Project the verified chain into current principals and escalations."""

    principals: dict[str, Principal] = {}
    escalations: dict[str, Escalation] = {}
    for record in records:
        kind = record.get("kind")
        if kind == "principal.created":
            body = record.get("principal")
            if not isinstance(body, dict):
                raise TruthStoreError("principal record is malformed")
            created_at = parse_utc(body.get("created_at"))
            if created_at is None:
                raise TruthStoreError("principal record requires created_at")
            principals[str(body["id"])] = Principal(
                id=str(body["id"]),
                role=Role(str(body["role"])),
                label=str(body["label"]),
                scopes=frozenset(str(item) for item in body.get("scopes", ())),
                credential_sha256=str(body["credential_sha256"]),
                created_at=created_at,
                expires_at=parse_utc(body.get("expires_at")),
            )
        elif kind == "credential.rotated":
            principal_id = str(record["principal_id"])
            existing = principals.get(principal_id)
            if existing is not None:
                principals[principal_id] = replace(
                    existing,
                    credential_sha256=str(record["credential_sha256"]),
                    rotated_at=parse_utc(record.get("recorded_at")),
                )
        elif kind == "principal.revoked":
            principal_id = str(record["principal_id"])
            existing = principals.get(principal_id)
            if existing is not None:
                principals[principal_id] = replace(
                    existing, revoked_at=parse_utc(record.get("recorded_at"))
                )
        elif kind == "escalation.opened":
            body = record.get("escalation")
            if not isinstance(body, dict):
                raise TruthStoreError("escalation record is malformed")
            created_at = parse_utc(body.get("created_at"))
            if created_at is None:
                raise TruthStoreError("escalation record requires created_at")
            escalations[str(body["id"])] = Escalation(
                id=str(body["id"]),
                actor_id=str(body["actor_id"]),
                actor_role=Role(str(body["actor_role"])),
                action_id=str(body["action_id"]),
                request_digest=str(body["request_digest"]),
                covenant_sha256=str(body["covenant_sha256"]),
                binding_priority=body.get("binding_priority"),
                reasons=tuple(str(item) for item in body.get("reasons", ())),
                created_at=created_at,
            )
        elif kind == "escalation.decided":
            escalation_id = str(record["escalation_id"])
            open_escalation = escalations.get(escalation_id)
            if open_escalation is not None:
                escalations[escalation_id] = replace(
                    open_escalation,
                    state=EscalationState(str(record["state"])),
                    decided_by=record.get("decided_by"),
                    decided_at=parse_utc(record.get("recorded_at")),
                    rationale=record.get("rationale"),
                    approval_expires_at=parse_utc(record.get("approval_expires_at")),
                )
        elif kind == "escalation.consumed":
            escalation_id = str(record["escalation_id"])
            open_escalation = escalations.get(escalation_id)
            if open_escalation is not None:
                escalations[escalation_id] = replace(
                    open_escalation,
                    state=EscalationState.CONSUMED,
                    consumed_at=parse_utc(record.get("recorded_at")),
                )
        elif kind == "escalation.attached":
            escalation_id = str(record["escalation_id"])
            open_escalation = escalations.get(escalation_id)
            if open_escalation is not None:
                escalations[escalation_id] = replace(
                    open_escalation, consumed_job_id=str(record["job_id"])
                )
    return principals, escalations
