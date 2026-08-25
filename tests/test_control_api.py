from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from epor.control.api import create_app
from epor.control.settings import ControlSettings


def test_control_api_contract_cors_and_error_envelopes(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    settings = ControlSettings(
        project_root=project_root,
        database_path=tmp_path / "api.sqlite3",
        artifact_root=tmp_path / "artifacts",
        research_catalog_path=project_root / "research" / "catalog.yaml",
    )
    app = create_app(settings)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://control.test") as client:
            health = await client.get("/api/v1/health", headers={"X-Request-ID": "test-123"})
            assert health.status_code == 200
            assert health.headers["x-request-id"] == "test-123"
            assert health.json() == {"status": "ok", "database": "ok", "version": "0.0.1"}

            capabilities = (await client.get("/api/v1/capabilities")).json()
            assert capabilities["bind_host"] == "127.0.0.1"
            assert capabilities["arbitrary_commands"] is False
            assert capabilities["arbitrary_url_fetching"] is False
            assert capabilities["filesystem_browser"] is False

            research = (await client.get("/api/v1/research")).json()
            assert research["available"] is True
            assert research["count"] > 10
            assert {item["sync_status"] for item in research["sources"]} <= {
                "verified",
                "missing",
                "error",
                "unknown",
            }

            models = (await client.get("/api/v1/models")).json()["items"]
            assert [item["slug"] for item in models] == [
                "epor-gamma",
                "epor-alpha",
                "epor-beta",
            ]
            assert all(item["metrics"] is None and item["status"] == "planned" for item in models)
            assert all(
                "operational_default_context" in item and "validated_max_context" in item
                for item in models
            )

            definitions = (await client.get("/api/v1/job-types")).json()["items"]
            assert {item["type"] for item in definitions} == {
                "system_probe",
                "research_sync",
                "tiny_train",
                "tiny_eval",
            }
            tiny_schema = next(item for item in definitions if item["type"] == "tiny_train")
            assert tiny_schema["schema"]["properties"]["model_config"]["default"] == (
                "configs/models/epor-tiny.yaml"
            )

            created = await client.post(
                "/api/v1/jobs",
                json={"type": "system_probe", "spec": {}},
            )
            assert created.status_code == 201
            job_id = created.json()["id"]
            fetched_job = (await client.get(f"/api/v1/jobs/{job_id}")).json()
            assert fetched_job["created_at"].endswith(("Z", "+00:00"))
            assert fetched_job["updated_at"].endswith(("Z", "+00:00"))
            events = (await client.get(f"/api/v1/jobs/{job_id}/events?after=0")).json()
            assert events["items"][0]["kind"] == "job.queued"
            assert events["next_after"] == 1
            assert events["items"][0]["created_at"].endswith(("Z", "+00:00"))

            cancelled = await client.post(f"/api/v1/jobs/{job_id}/cancel")
            assert cancelled.json()["status"] == "cancelled"
            retry = await client.post(f"/api/v1/jobs/{job_id}/retry")
            assert retry.status_code == 201
            assert retry.json()["retry_of_id"] == job_id

            rejected = await client.post(
                "/api/v1/jobs",
                json={
                    "type": "system_probe",
                    "spec": {"command": "echo", "api_token": "super-secret-value"},
                },
            )
            assert rejected.status_code == 422
            assert rejected.json()["code"] == "invalid_job_spec"
            assert "super-secret-value" not in rejected.text
            assert set(rejected.json()) == {"code", "message", "details", "request_id"}

            traversal = await client.post(
                "/api/v1/jobs",
                json={"type": "tiny_train", "spec": {"model_config": "../outside.yaml"}},
            )
            assert traversal.status_code == 422
            assert traversal.json()["code"] == "invalid_job_spec"

            oversized_recipe = await client.post(
                "/api/v1/jobs",
                json={
                    "type": "tiny_train",
                    "spec": {"model_config": "configs/models/epor-gamma.yaml"},
                },
            )
            assert oversized_recipe.status_code == 422
            assert oversized_recipe.json()["code"] == "invalid_job_spec"

            missing = await client.get("/not-a-route")
            assert missing.status_code == 404
            assert missing.json()["code"] == "http_error"
            assert missing.headers["x-request-id"] == missing.json()["request_id"]

            allowed = await client.options(
                "/api/v1/jobs",
                headers={
                    "Origin": settings.allowed_origin,
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert allowed.status_code == 200
            assert allowed.headers["access-control-allow-origin"] == settings.allowed_origin
            denied = await client.options(
                "/api/v1/jobs",
                headers={
                    "Origin": "http://127.0.0.1:9999",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert "access-control-allow-origin" not in denied.headers

    asyncio.run(scenario())
