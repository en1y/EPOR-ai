"""SQLite engine setup with v0.0.1 durability and concurrency settings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


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
            differences = compare_metadata(migration_context, Base.metadata)
            if differences:
                raise RuntimeError("unversioned control schema differs from the v0.0.1 migration")
            # This is safe only after an empty autogenerate diff proves the
            # pre-Alembic create_all schema is exactly revision 0001.
            command.stamp(config, "0001_control_plane")
        command.upgrade(config, "head")
        connection.commit()


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
