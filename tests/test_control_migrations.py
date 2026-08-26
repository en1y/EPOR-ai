from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import Engine, inspect, text

from epor.control.api import create_app
from epor.control.database import (
    create_sqlite_engine,
    initialize_database,
    migration_config,
)
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
            "0002_safety_identity"
        )
        assert {
            "jobs",
            "job_events",
            "job_artifacts",
            "principals",
            "escalations",
            "authority_events",
            "alembic_version",
        } <= set(inspect(connection).get_table_names())
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


def _create_legacy_schema(engine: Engine) -> None:
    """Build the exact schema a pre-Alembic v0.0.1 database had.

    Running revision 0001 and then dropping the version table reproduces that
    state exactly. Using the current ``Base.metadata`` would not: it already
    carries the columns later revisions add, so the fixture would silently
    stop testing the migration it exists to test.
    """

    config = migration_config(engine)
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "0001_control_plane")
        connection.commit()
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE alembic_version"))


def test_pre_alembic_create_all_schema_is_safely_adopted(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-create-all.sqlite3"
    engine = create_sqlite_engine(database_path)
    # A genuine v0.0.1 database has only the three original tables and no
    # Alembic lineage. Later revisions' tables are legitimately absent, and
    # their absence must read as work still to do, not as schema drift.
    _create_legacy_schema(engine)
    with engine.connect() as connection:
        table_names = set(inspect(connection).get_table_names())
        assert "alembic_version" not in table_names
        assert "principals" not in table_names

    initialize_database(engine)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0002_safety_identity"
        )
        assert {"principals", "escalations", "authority_events"} <= set(
            inspect(connection).get_table_names()
        )
    initialize_database(engine)


def test_a_drifted_unversioned_v001_schema_is_still_rejected(tmp_path: Path) -> None:
    # Filtering later revisions' tables out of the drift check must not blind it
    # to real drift in the three tables revision 0001 actually owns.
    engine = create_sqlite_engine(tmp_path / "drifted.sqlite3")
    _create_legacy_schema(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE jobs DROP COLUMN worker_id"))

    with pytest.raises(RuntimeError, match=r"differs from the v0\.0\.1 migration"):
        initialize_database(engine)
