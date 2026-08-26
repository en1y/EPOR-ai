"""SQLite engine setup with v0.0.1 durability and concurrency settings."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, MetaData, create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker


def create_sqlite_engine(database_path: Path) -> Engine:
    path = database_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite+pysqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30.0},
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection: sqlite3.Connection, _record: object) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def initialize_database(engine: Engine) -> None:
    """Upgrade the rebuildable index through the bundled Alembic lineage."""

    config = migration_config(engine)
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        table_names = set(inspect(connection).get_table_names())
        control_tables = {"jobs", "job_events", "job_artifacts"}
        if control_tables & table_names and "alembic_version" not in table_names:
            if not control_tables <= table_names:
                raise RuntimeError("unversioned control schema is incomplete")
            migration_context = MigrationContext.configure(
                connection,
                opts={"compare_type": True, "render_as_batch": True},
            )
            # Compare against a scratch database built by revision 0001, not
            # against current metadata. Current metadata carries every column
            # later revisions add, so a legitimate v0.0.1 database would read
            # as drift and a genuinely drifted one could slip through.
            differences = compare_metadata(
                migration_context, _revision_metadata("0001_control_plane")
            )
            if differences:
                raise RuntimeError("unversioned control schema differs from the v0.0.1 migration")
            # Safe only after an empty autogenerate diff proves the pre-Alembic
            # create_all schema is exactly revision 0001. The upgrade below then
            # applies every later revision normally.
            command.stamp(config, "0001_control_plane")
        command.upgrade(config, "head")
        connection.commit()


def _revision_metadata(revision: str) -> MetaData:
    """Reflect the schema one Alembic revision produces, in a scratch database."""

    with tempfile.TemporaryDirectory(prefix="epor-schema-") as directory:
        engine = create_engine(f"sqlite+pysqlite:///{Path(directory) / 'reference.sqlite3'}")
        try:
            config = migration_config(engine)
            with engine.connect() as connection:
                config.attributes["connection"] = connection
                command.upgrade(config, revision)
                connection.commit()
            metadata = MetaData()
            metadata.reflect(bind=engine)
        finally:
            engine.dispose()
    metadata.remove(metadata.tables["alembic_version"])
    return metadata


def migration_config(engine: Engine) -> Config:
    """Build an Alembic config bound to an existing control-plane engine."""

    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "migrations" / "alembic.ini"
    script_location = project_root / "migrations"
    if not config_path.is_file() or not script_location.is_dir():
        raise RuntimeError("bundled control-plane migrations are unavailable")
    config = Config(str(config_path))
    config.set_main_option("script_location", str(script_location))
    config.set_main_option(
        "sqlalchemy.url",
        engine.url.render_as_string(hide_password=False),
    )
    return config
