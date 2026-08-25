from __future__ import annotations

from pathlib import Path

from alembic import command
from sqlalchemy import inspect, text

from epor.control.api import create_app
from epor.control.database import (
    create_sqlite_engine,
    initialize_database,
    migration_config,
)
from epor.control.models import Base
from epor.control.settings import ControlSettings


def test_default_bootstrap_is_alembic_current_and_has_no_schema_drift(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    settings = ControlSettings(
        project_root=project_root,
        database_path=tmp_path / "bootstrap.sqlite3",
        artifact_root=tmp_path / "artifacts",
        research_catalog_path=project_root / "research" / "catalog.yaml",
    )
    app = create_app(settings)
    engine = app.state.engine

    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0001_control_plane"
        )
        assert {"jobs", "job_events", "job_artifacts", "alembic_version"} <= set(
            inspect(connection).get_table_names()
        )
        assert connection.scalar(text("PRAGMA journal_mode")) == "wal"

    # Re-running normal bootstrap follows Alembic and is an idempotent upgrade,
    # rather than attempting to create existing tables outside its lineage.
    initialize_database(engine)
    config = migration_config(engine)
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.current(config, check_heads=True)
        command.upgrade(config, "head")
        command.check(config)


def test_pre_alembic_create_all_schema_is_safely_adopted(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-create-all.sqlite3"
    engine = create_sqlite_engine(database_path)
    Base.metadata.create_all(engine)
    with engine.connect() as connection:
        assert "alembic_version" not in inspect(connection).get_table_names()

    initialize_database(engine)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0001_control_plane"
        )
    initialize_database(engine)
