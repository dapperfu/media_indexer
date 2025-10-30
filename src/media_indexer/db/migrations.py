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


class Migration003_AddRelationalEXIF(Migration):
    """Migration 003: Add relational EXIF tables and migrate existing data.

    REQ-024: Creates EXIFTag and EXIFTagValue tables and migrates existing
    JSON EXIF data to relational format.
    """

    def __init__(self) -> None:
        """Initialize migration."""
        super().__init__(
            version=3,
            description="Add relational EXIF tables (EXIFTag, EXIFTagValue) and migrate existing data",
        )

    def upgrade(self, connection: Any) -> None:
        """Create relational EXIF tables and migrate data.

        Parameters
        ----------
        connection : Any
            SQLite database connection.
        """
        cursor = connection.cursor()

        # Check if EXIFTag table already exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='EXIFTag'"
        )
        if cursor.fetchone():
            logger.debug("REQ-024: Migration 003 already applied (EXIFTag table exists)")
            return

        logger.info(
            "REQ-024: Applying migration 003: Creating relational EXIF tables"
        )

        # Create EXIFTag table
        cursor.execute(
            """
            CREATE TABLE EXIFTag (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                "group" TEXT,
                description TEXT,
                tag_type TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX idx_exiftag_name ON EXIFTag(name)")
        cursor.execute("CREATE INDEX idx_exiftag_group ON EXIFTag(\"group\")")

        # Create EXIFTagValue table
        cursor.execute(
            """
            CREATE TABLE EXIFTagValue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image INTEGER NOT NULL,
                tag INTEGER NOT NULL,
                value_text TEXT,
                value_numeric REAL,
                value_json TEXT,
                extracted_at REAL,
                FOREIGN KEY (image) REFERENCES Image(id) ON DELETE CASCADE,
                FOREIGN KEY (tag) REFERENCES EXIFTag(id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute("CREATE INDEX idx_exiftagvalue_image ON EXIFTagValue(image)")
        cursor.execute("CREATE INDEX idx_exiftagvalue_tag ON EXIFTagValue(tag)")
        cursor.execute("CREATE INDEX idx_exiftagvalue_text ON EXIFTagValue(value_text)")
        cursor.execute("CREATE INDEX idx_exiftagvalue_numeric ON EXIFTagValue(value_numeric)")

        connection.commit()

        # Migrate existing EXIF data from JSON blob to relational format
        logger.info("REQ-024: Migrating existing EXIF data to relational format")
        cursor.execute("SELECT id, image, data FROM EXIFData WHERE data IS NOT NULL")
        exif_records = cursor.fetchall()

        migrated_count = 0
        import json
        import time

        for exif_id, image_id, exif_json_str in exif_records:
            try:
                if isinstance(exif_json_str, str):
                    exif_dict = json.loads(exif_json_str)
                else:
                    exif_dict = exif_json_str

                if not isinstance(exif_dict, dict):
                    continue

                extracted_at = time.time()

                for tag_name, tag_value in exif_dict.items():
                    if tag_value is None:
                        continue

                    # Infer group
                    group = self._infer_exif_group(tag_name)

                    # Get or create tag
                    cursor.execute(
                        "SELECT id FROM EXIFTag WHERE name = ?",
                        (tag_name,),
                    )
                    tag_row = cursor.fetchone()
                    if tag_row:
                        tag_id = tag_row[0]
                    else:
                        cursor.execute(
                            "INSERT INTO EXIFTag (name, \"group\") VALUES (?, ?)",
                            (tag_name, group),
                        )
                        tag_id = cursor.lastrowid

                    # Convert value
                    value_text = str(tag_value)
                    value_numeric: float | None = None
                    value_json: str | None = None

                    if isinstance(tag_value, (int, float)):
                        value_numeric = float(tag_value)
                    elif isinstance(tag_value, str):
                        try:
                            if "." in tag_value:
                                value_numeric = float(tag_value)
                            else:
                                value_numeric = float(int(tag_value))
                        except (ValueError, TypeError):
                            pass

                    if isinstance(tag_value, (list, dict)):
                        value_json = json.dumps(tag_value)
                        value_text = value_json

                    # Insert tag value
                    cursor.execute(
                        """
                        INSERT INTO EXIFTagValue
                        (image, tag, value_text, value_numeric, value_json, extracted_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (image_id, tag_id, value_text, value_numeric, value_json, extracted_at),
                    )

                migrated_count += 1
            except Exception as e:
                logger.warning(
                    f"REQ-024: Failed to migrate EXIF data for image {image_id}: {e}"
                )
                continue

        connection.commit()
        logger.info(
            f"REQ-024: Migration 003 completed - migrated {migrated_count} EXIF records"
        )

    @staticmethod
    def _infer_exif_group(tag_name: str) -> str | None:
        """Infer EXIF group from tag name."""
        tag_upper = tag_name.upper()

        if tag_upper.startswith("GPS") or "LATITUDE" in tag_upper or "LONGITUDE" in tag_upper:
            return "GPS"

        ifd0_tags = [
            "IMAGE",
            "MAKE",
            "MODEL",
            "ORIENTATION",
            "XRESOLUTION",
            "YRESOLUTION",
            "RESOLUTIONUNIT",
            "SOFTWARE",
            "DATETIME",
            "ARTIST",
            "COPYRIGHT",
        ]
        if any(tag in tag_upper for tag in ifd0_tags):
            return "IFD0"

        exif_tags = [
            "EXPOSURE",
            "FOCAL",
            "APERTURE",
            "ISO",
            "SHUTTER",
            "WHITEBALANCE",
            "FLASH",
            "METERING",
            "FOCUS",
        ]
        if any(tag in tag_upper for tag in exif_tags):
            return "EXIF"

        if "INTEROP" in tag_upper:
            return "Interop"

        return None


# Registry of all migrations - add new migrations here
MIGRATIONS: list[Migration] = [
    Migration001_AddFaceAttributes(),
    Migration002_AddDetectedAtToFaces(),
    Migration003_AddRelationalEXIF(),
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

