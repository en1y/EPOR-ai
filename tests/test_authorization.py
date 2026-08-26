"""G10: who may direct EPOR, and what happens to everyone else.

These exercise the covenant's second principle where it actually binds — at
the API boundary — rather than a permission table sitting beside it.  Every
refusal here is a covenant resolution, not a separate access check.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from conftest import Harness, run

from epor.actions import DECLARED_ACTIONS, resolve_action
from epor.control.authority import (
    ANONYMOUS,
    SESSION_TTL,
    Actor,
    AuthorityStore,
    InvalidDelegationError,
    OwnerAlreadyBootstrappedError,
    Role,
)
from epor.control.errors import AuthorizationDeniedError
from epor.safety import load_covenant

READ_ONLY_SURFACES = (
    "/api/v1/health",
    "/api/v1/capabilities",
    "/api/v1/job-types",
    "/api/v1/research",
    "/api/v1/models",
    "/api/v1/documents",
    "/api/v1/jobs",
    "/api/v1/auth/me",
    "/api/v1/auth/operators",
    "/api/v1/safety/covenant",
    "/api/v1/safety/actions",
    "/api/v1/safety/escalations",
)


def test_an_unverified_request_is_not_direction(harness: Harness) -> None:
    covenant = load_covenant()
    resolution = resolve_action("system_probe", covenant, actor=ANONYMOUS)
    assert resolution.outcome == "refuse"
    assert resolution.binding_priority == 2
    assert resolution.binding_obligation == "verify-authorization"

    # The same action, declared identically, is permitted once an identity
    # stands behind it. Nothing about the work changed; the direction did.
    assert resolve_action("system_probe", covenant, actor=harness.owner).outcome == "allow"


def test_a_delegation_binds_the_second_principle_not_a_side_table(harness: Harness) -> None:
    covenant = load_covenant()
    operator = Actor(id="op", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"system_probe"}))

    permitted = resolve_action("system_probe", covenant, actor=operator)
    assert permitted.outcome == "allow"

    refused = resolve_action("tiny_train", covenant, actor=operator)
    assert refused.outcome == "refuse"
    assert refused.binding_priority == 2
    assert refused.binding_obligation == "refuse-unlawful-or-out-of-scope"
    # Principle 3 cannot rescue it: nothing ranks below the third principle.
    assert any("no higher-priority justification" in reason for reason in refused.reasons)


def test_every_read_only_surface_stays_anonymously_browsable(harness: Harness) -> None:
    async def scenario() -> None:
        async with harness.client() as client:
            for path in READ_ONLY_SURFACES:
                response = await client.get(path)
                assert response.status_code == 200, path
            identity = (await client.get("/api/v1/auth/me")).json()
            assert identity == {
                "role": "anonymous",
                "principal_id": None,
                "label": "",
                "scopes": [],
                "authenticated": False,
                "expires_at": None,
            }

    run(scenario)


def test_every_mutation_requires_a_verified_identity(harness: Harness) -> None:
    async def scenario() -> None:
        async with harness.client() as client:
            mutations = (
                ("/api/v1/jobs", {"type": "system_probe", "spec": {}}),
                ("/api/v1/auth/owner/rotate", None),
                (
                    "/api/v1/auth/operators",
                    {
                        "label": "x",
                        "scopes": ["system_probe"],
                        "expires_at": "2030-01-01T00:00:00Z",
                    },
                ),
            )
            for path, body in mutations:
                response = await client.post(path, json=body)
                assert response.status_code == 401, path
                assert response.json()["code"] == "authentication_required"

    run(scenario)


def test_scoped_operator_reaches_its_delegation_and_nothing_else(harness: Harness) -> None:
    token = harness.delegate("system_probe")

    async def scenario() -> None:
        async with harness.as_bearer(token) as client:
            identity = (await client.get("/api/v1/auth/me")).json()
            assert identity["role"] == "delegated_operator"
            assert identity["scopes"] == ["system_probe"]

            allowed = await client.post("/api/v1/jobs", json={"type": "system_probe", "spec": {}})
            assert allowed.status_code == 201
            assert allowed.json()["submitted_by_role"] == "delegated_operator"

            denied = await client.post(
                "/api/v1/jobs",
                json={"type": "tiny_train", "spec": {}},
            )
            assert denied.status_code == 403
            assert denied.json()["code"] == "authorization_denied"

            # Managing delegations is the owner's alone.
            escalated = await client.post(
                "/api/v1/auth/operators",
                json={
                    "label": "self-promotion",
                    "scopes": ["tiny_train"],
                    "expires_at": "2030-01-01T00:00:00Z",
                },
            )
            assert escalated.status_code == 403
            assert escalated.json()["code"] == "authorization_denied"

    run(scenario)


def test_stopping_work_is_never_gated_by_scope(harness: Harness) -> None:
    token = harness.delegate("system_probe")

    async def scenario() -> None:
        async with harness.as_owner() as owner:
            job = (await owner.post("/api/v1/jobs", json={"type": "tiny_train", "spec": {}})).json()
        async with harness.as_bearer(token) as operator:
            # The operator holds no tiny_train delegation, yet may still stop it:
            # gating cancellation would let the third principle's continuity
            # argument outrank the second principle's duty to accept correction.
            cancelled = await operator.post(f"/api/v1/jobs/{job['id']}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"

    run(scenario)


def test_revocation_and_rotation_take_effect_immediately(harness: Harness) -> None:
    revoked_token = harness.delegate("system_probe", label="revoked")
    rotated_id = harness.store.principals()[-1].id

    async def scenario() -> None:
        async with harness.as_bearer(revoked_token) as client:
            assert (await client.get("/api/v1/auth/me")).json()["authenticated"] is True
        async with harness.as_owner() as owner:
            revoke = await owner.post(f"/api/v1/auth/operators/{rotated_id}/revoke")
            assert revoke.status_code == 200
            assert revoke.json()["active"] is False
        async with harness.as_bearer(revoked_token) as client:
            assert (await client.get("/api/v1/auth/me")).json()["role"] == "anonymous"
            blocked = await client.post("/api/v1/jobs", json={"type": "system_probe", "spec": {}})
            assert blocked.status_code == 401

    run(scenario)

    fresh = harness.delegate("system_probe", label="rotating")
    principal_id = harness.store.principals()[-1].id
    replacement = harness.store.rotate(principal_id, by=harness.owner)
    assert harness.store.authenticate(fresh) is None
    assert harness.store.authenticate(replacement.token) is not None
    # Rotation keeps identity and scope; only the secret changes.
    assert replacement.principal.id == principal_id
    assert replacement.principal.scopes == frozenset({"system_probe"})


def test_an_expired_delegation_stops_authenticating(harness: Harness) -> None:
    token = harness.delegate("system_probe", days=1)
    principal = harness.store.principals()[-1]
    assert principal.active() is True
    assert principal.active(datetime.now(UTC) + timedelta(days=2)) is False

    # The store reads the same clock, so expiry needs no sweeper to take hold.
    import epor.control.authority as authority

    later = datetime.now(UTC) + timedelta(days=2)
    original = authority.utc_now
    authority.utc_now = lambda: later  # type: ignore[assignment]
    try:
        assert harness.store.authenticate(token) is None
    finally:
        authority.utc_now = original  # type: ignore[assignment]


def test_a_session_is_exchanged_bound_to_origin_and_expires(harness: Harness) -> None:
    origin = harness.settings.allowed_origin

    async def scenario() -> None:
        async with harness.client() as client:
            wrong_origin = await client.post(
                "/api/v1/auth/session",
                json={"token": harness.owner_token},
                headers={"Origin": "http://127.0.0.1:9999"},
            )
            assert wrong_origin.status_code == 403

            opened = await client.post(
                "/api/v1/auth/session",
                json={"token": harness.owner_token},
                headers={"Origin": origin},
            )
            assert opened.status_code == 200
            assert opened.json()["role"] == "owner"
            cookie = opened.headers["set-cookie"]
            assert "HttpOnly" in cookie and "SameSite=strict" in cookie
            # The exchanged credential is never echoed back to the browser.
            assert harness.owner_token not in cookie
            assert harness.owner_token not in opened.text

            # A cookie is ambient, so a cookie-authenticated mutation must also
            # prove which origin sent it.
            no_origin = await client.post("/api/v1/jobs", json={"type": "system_probe", "spec": {}})
            assert no_origin.status_code == 403
            assert no_origin.json()["code"] == "authorization_denied"

            with_origin = await client.post(
                "/api/v1/jobs",
                json={"type": "system_probe", "spec": {}},
                headers={"Origin": origin},
            )
            assert with_origin.status_code == 201

            closed = await client.delete("/api/v1/auth/session", headers={"Origin": origin})
            assert closed.status_code == 204
            assert (await client.get("/api/v1/auth/me")).json()["role"] == "anonymous"

    run(scenario)
    assert timedelta(hours=12) == SESSION_TTL


def test_a_session_dies_with_the_api_process(harness: Harness) -> None:
    principal = harness.store.owner()
    assert principal is not None
    session = harness.store.open_session(principal)
    assert harness.store.session_actor(session.id) is not None

    # A second store over the same truth is what a restarted API looks like.
    restarted = AuthorityStore(harness.store.root)
    assert restarted.session_actor(session.id) is None


def test_credentials_are_never_reflected_by_any_serializer(harness: Harness) -> None:
    token = harness.delegate("system_probe", label="visible")
    digests = {item.credential_sha256 for item in harness.store.principals()}

    async def scenario() -> None:
        async with harness.as_owner() as client:
            listing = await client.get("/api/v1/auth/operators")
            assert listing.status_code == 200
            body = listing.text
            assert token not in body
            assert harness.owner_token not in body
            for digest in digests:
                assert digest not in body
            assert all(
                "credential_sha256" not in item and "token" not in item
                for item in listing.json()["items"]
            )

    run(scenario)


def test_malformed_and_oversized_credentials_are_rejected_without_echo(harness: Harness) -> None:
    async def scenario() -> None:
        async with harness.client() as client:
            oversized = "z" * 4096
            response = await client.post("/api/v1/auth/session", json={"token": oversized})
            assert response.status_code == 422
            assert oversized not in response.text

            wrong = await client.post("/api/v1/auth/session", json={"token": "n" * 64})
            assert wrong.status_code == 401
            assert "n" * 64 not in wrong.text
            # One message for unknown, expired, and revoked alike.
            assert wrong.json()["message"] == "the supplied credential is not valid"

            garbage = await client.get(
                "/api/v1/auth/me", headers={"Authorization": "Bearer " + "q" * 600}
            )
            assert garbage.json()["role"] == "anonymous"

    run(scenario)


def test_a_delegation_may_only_name_reviewed_declared_actions(harness: Harness) -> None:
    with pytest.raises(InvalidDelegationError, match="reviewed declared actions"):
        harness.store.create_delegation(
            label="fictional",
            scopes=["model_serve"],
            expires_at=datetime.now(UTC) + timedelta(days=1),
            by=harness.owner,
        )
    with pytest.raises(InvalidDelegationError, match="30 days"):
        harness.store.create_delegation(
            label="forever",
            scopes=["system_probe"],
            expires_at=datetime.now(UTC) + timedelta(days=31),
            by=harness.owner,
        )
    with pytest.raises(InvalidDelegationError, match="at least one"):
        harness.store.create_delegation(
            label="empty",
            scopes=[],
            expires_at=datetime.now(UTC) + timedelta(days=1),
            by=harness.owner,
        )


def test_the_owner_credential_is_bootstrapped_once_and_never_revoked(harness: Harness) -> None:
    with pytest.raises(OwnerAlreadyBootstrappedError):
        harness.store.bootstrap_owner()

    owner = harness.store.owner()
    assert owner is not None
    with pytest.raises(InvalidDelegationError, match="rotated, never revoked"):
        harness.store.revoke(owner.id, by=harness.owner)

    operator = Actor(id="op", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"system_probe"}))
    with pytest.raises(AuthorizationDeniedError):
        harness.store.revoke(owner.id, by=operator)

    assert harness.store.owner_token_path.stat().st_mode & 0o777 == 0o600
    assert harness.store.root.stat().st_mode & 0o777 == 0o700


def test_the_owner_holds_every_declared_action_by_definition(harness: Harness) -> None:
    covenant = load_covenant()
    for action_id in DECLARED_ACTIONS:
        assert harness.owner.may(action_id), action_id
        assert resolve_action(action_id, covenant, actor=harness.owner).outcome == "allow"
