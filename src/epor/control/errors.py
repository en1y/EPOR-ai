"""Stable error taxonomy shared by the control service, authority, and API.

These live apart from the service so that authorization and escalation can
raise them without importing the job lifecycle, and so the HTTP layer has one
place to learn every code it must be able to render.
"""

from __future__ import annotations

from typing import Any

from epor.safety import Resolution

from .security import redact


class ControlError(Exception):
    """Base class mapped to the stable control API error envelope."""

    status_code = 400
    code = "control_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = redact(details)


class JobNotFoundError(ControlError):
    status_code = 404
    code = "job_not_found"


class InvalidTransitionError(ControlError):
    status_code = 409
    code = "invalid_job_transition"


class InvalidJobSpecError(ControlError):
    status_code = 422
    code = "invalid_job_spec"


class ArtifactPathError(ControlError):
    status_code = 422
    code = "artifact_path_not_allowed"


class AuthenticationRequiredError(ControlError):
    """Raised when a mutation arrives with no verified identity.

    Distinct from an authorization refusal: the covenant cannot assess human
    direction from a caller whose origin is unverified, so there is nothing
    yet to weigh.
    """

    status_code = 401
    code = "authentication_required"


class AuthorizationDeniedError(ControlError):
    """Raised when a verified actor holds no delegation for the action."""

    status_code = 403
    code = "authorization_denied"


class CovenantBlockedError(ControlError):
    """Base for every outcome in which the covenant did not admit work.

    Subclasses separate the two that must never be confused: a refusal, which
    no human may approve, and an escalation, which one may.
    """

    status_code = 403
    code = "covenant_blocked"

    def __init__(self, resolution: Resolution, *, escalation_id: str | None = None) -> None:
        details: dict[str, Any] = resolution.model_dump(mode="json")
        if escalation_id is not None:
            details["escalation_id"] = escalation_id
        super().__init__(
            f"the safety covenant did not permit this action ({resolution.outcome})",
            details=details,
        )
        self.resolution = resolution
        self.escalation_id = escalation_id


class CovenantRefusedError(CovenantBlockedError):
    """The covenant refused. A refusal is never overridable by approval."""

    code = "covenant_refused"


class CovenantEscalatedError(CovenantBlockedError):
    """The covenant stopped short of deciding; a human must decide instead."""

    status_code = 409
    code = "covenant_escalated"


class EscalationError(ControlError):
    """Raised when an escalation cannot move as requested."""

    status_code = 409
    code = "escalation_invalid_state"


class EscalationNotFoundError(ControlError):
    status_code = 404
    code = "escalation_not_found"
