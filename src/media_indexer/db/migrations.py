"""Database migration system.

REQ-022: Database schema migration support for forward compatibility.
REQ-024: Extensible migration system for schema evolution.
REQ-087: Forward migration of existing databases to accommodate schema changes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pony.orm import db_session

logger = logging.getLogger(__name__)

# Current schema version - increment when adding new migrations
CURRENT_SCHEMA_VERSION = 3


class Migration(ABC):
    """Base class for database migrations.

    REQ-024: Abstract base for all database migrations.
    """

    def __init__(self, version: int, description: str) -> None:
        """Initialize migration.

        Parameters
        ----------
        version : int
            Schema version this migration applies to.
        description : str
            Human-readable description of the migration.
        """
        self.version = version
        self.description = description

    @abstractmethod
    def upgrade(self, connection: Any) -> None:
        """Apply migration to database.

        Parameters
        ----------
        connection : Any
            Database connection object (SQLite connection).
        """
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Migration(version={self.version}, description='{self.description}')"


class Migration001_AddFaceAttributes(Migration):
    """Migration 001: Add attributes column to Face table.

    REQ-081: Adds support for age and emotion attributes in Face model.
    """

    def __init__(self) -> None:
        """Initialize migration."""
        super().__init__(
            version=1,
            description="Add attributes column to Face table for REQ-081",
        )

    def upgrade(self, connection: Any) -> None:
        """Add attributes column to Face table.

        Parameters
        ----------
        connection : Any
            SQLite database connection.
        """
        cursor = connection.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(Face)")
        columns = [row[1] for row in cursor.fetchall()]

        if "attributes" not in columns:
            logger.info(
                "REQ-024: Applying migration 001: Adding attributes column to Face table"
            )
            cursor.execute(
                "ALTER TABLE Face ADD COLUMN attributes TEXT"
            )
            connection.commit()
            logger.info("REQ-024: Migration 001 completed successfully")
        else:
            logger.debug("REQ-024: Migration 001 already applied (attributes column exists)")


class Migration002_AddDetectedAtToFaces(Migration):
    """Migration 002: Add detected_at column to Face table if missing.

    Ensures all Face records have timestamp support.
    """

    def __init__(self) -> None:
        """Initialize migration."""
        super().__init__(
            version=2,
            description="Add detected_at column to Face table if missing",
        )

    def upgrade(self, connection: Any) -> None:
        """Add detected_at column to Face table if missing.

        Parameters
        ----------
        connection : Any
            SQLite database connection.
        """
        cursor = connection.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(Face)")
        columns = [row[1] for row in cursor.fetchall()]

        if "detected_at" not in columns:
            logger.info(
                "REQ-024: Applying migration 002: Adding detected_at column to Face table"
            )
            cursor.execute(
                "ALTER TABLE Face ADD COLUMN detected_at REAL"
            )
            connection.commit()
            logger.info("REQ-024: Migration 002 completed successfully")
        else:
            logger.debug("REQ-024: Migration 002 already applied (detected_at column exists)")


# Registry of all migrations - add new migrations here
MIGRATIONS: list[Migration] = [
    Migration001_AddFaceAttributes(),
    Migration002_AddDetectedAtToFaces(),
]


def get_schema_version(connection: Any) -> int:
    """Get current schema version from database.

    Parameters
    ----------
    connection : Any
        SQLite database connection.

    Returns
    -------
    int
        Current schema version, or 0 if version table doesn't exist.
    """
    cursor = connection.cursor()

    # Check if schema_version table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    if not cursor.fetchone():
        return 0

    # Get version
    cursor.execute("SELECT version FROM schema_version LIMIT 1")
    result = cursor.fetchone()
    if result:
        return int(result[0])

    return 0


def set_schema_version(connection: Any, version: int) -> None:
    """Set schema version in database.

    Parameters
    ----------
    connection : Any
        SQLite database connection.
    version : int
        Schema version to set.
    """
    cursor = connection.cursor()

    # Create schema_version table if it doesn't exist
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)"
    )

    # Insert or update version
    cursor.execute("DELETE FROM schema_version")
    cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    connection.commit()


def run_migrations(connection: Any) -> None:
    """Run all pending migrations.

    REQ-024: Apply forward migrations to bring database schema up to date.
    REQ-087: Automatically detect schema version differences and apply migrations.

    Parameters
    ----------
    connection : Any
        SQLite database connection.

    Raises
    ------
    RuntimeError
        If migration fails or schema version is ahead of current code.
    """
    current_version = get_schema_version(connection)

    if current_version > CURRENT_SCHEMA_VERSION:
        error_msg = (
            f"REQ-024: Database schema version ({current_version}) is ahead of "
            f"code version ({CURRENT_SCHEMA_VERSION}). Please update the code."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if current_version == CURRENT_SCHEMA_VERSION:
        logger.debug(f"REQ-024: Database schema is up to date (version {current_version})")
        return

    logger.info(
        f"REQ-024: Current schema version: {current_version}, "
        f"target version: {CURRENT_SCHEMA_VERSION}"
    )

    # Apply migrations in order
    for migration in MIGRATIONS:
        if migration.version > current_version:
            logger.info(
                f"REQ-024: Applying migration {migration.version}: {migration.description}"
            )
            try:
                migration.upgrade(connection)
                set_schema_version(connection, migration.version)
                logger.info(
                    f"REQ-024: Migration {migration.version} completed, "
                    f"schema version updated to {migration.version}"
                )
            except Exception as e:
                error_msg = (
                    f"REQ-024: Migration {migration.version} failed: {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    # Set final version
    set_schema_version(connection, CURRENT_SCHEMA_VERSION)
    logger.info(
        f"REQ-024: All migrations completed. Schema version: {CURRENT_SCHEMA_VERSION}"
    )

