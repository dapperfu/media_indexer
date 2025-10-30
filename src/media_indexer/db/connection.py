"""Database connection manager.

REQ-022: Use PonyORM for database operations.
REQ-023: Database connection management.
"""

import logging
from pathlib import Path

from pony.orm import Database

logger = logging.getLogger(__name__)

# Global database instance
db: Database | None = None


class DatabaseConnection:
    """Database connection manager for PonyORM.

    REQ-022: Manage PonyORM database connection.
    REQ-024: Handle database initialization and connection.
    """

    def __init__(self, database_path: Path) -> None:
        """Initialize database connection.

        Args:
            database_path: Path to SQLite database file.
        """
        # REQ-066: Resolve to absolute path for consistent storage location
        self.database_path = Path(database_path).resolve()
        logger.info(f"REQ-022: Initializing database at {self.database_path}")

    def connect(self) -> Database:
        """Connect to the database and create tables.

        REQ-022: Connect to PonyORM database.
        REQ-024: Create database schema.

        Returns:
            Database instance.

        Raises:
            RuntimeError: If database connection fails.
        """
        global db

        try:
            # Create database directory if it doesn't exist
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # REQ-022: Create PonyORM database
            db = Database()

            # Import models to register them with the database
            # This must be done after creating Database instance but before binding
            from media_indexer.db.exif import EXIFData  # noqa: F401
            from media_indexer.db.face import Face  # noqa: F401
            from media_indexer.db.image import Image  # noqa: F401
            from media_indexer.db.object import Object  # noqa: F401
            from media_indexer.db.pose import Pose  # noqa: F401

            # REQ-022: Bind to SQLite database
            db.bind(
                provider="sqlite",
                filename=str(self.database_path),
                create_db=True,
            )

            # REQ-028: Configure SQLite for concurrent access
            @db.on_connect(provider="sqlite")
            def setup_sqlite(db_instance, connection):  # type: ignore[no-untyped-def]
                """Configure SQLite for optimal concurrent access.

                REQ-028: Enable WAL mode and optimize SQLite settings for concurrent access.
                """
                # Enable WAL mode for better concurrency
                connection.execute("PRAGMA journal_mode=WAL;")
                # Use NORMAL sync mode (safer than OFF, faster than FULL)
                connection.execute("PRAGMA synchronous=NORMAL;")
                # Allocate cache (~1 GB) - optimized for systems with large RAM
                connection.execute("PRAGMA cache_size=-262144;")
                # Use memory for temporary storage
                connection.execute("PRAGMA temp_store=MEMORY;")
                # Set busy timeout (retry if locked, 5 seconds)
                connection.execute("PRAGMA busy_timeout=5000;")

            # REQ-024: Generate database schema (creates tables if they don't exist)
            db.generate_mapping(create_tables=True)
            logger.info("REQ-024: Database schema generated")

            # REQ-024, REQ-087: Run migrations to update existing schema
            # This must be done after generate_mapping to ensure tables exist
            import sqlite3

            # Get raw SQLite connection for migrations
            raw_conn = sqlite3.connect(str(self.database_path))
            try:
                from media_indexer.db.migrations import run_migrations

                run_migrations(raw_conn)
            finally:
                raw_conn.close()

            # REQ-066: Verify tables were created by making a test connection
            from pony.orm import db_session

            from media_indexer.db.image import Image as ImageEntity

            with db_session:
                try:
                    # Try to count images to verify the Image table exists
                    count = ImageEntity.select().count()
                    logger.info(f"REQ-066: Database verified - Image table exists with {count} records")
                except Exception as e:
                    error_msg = f"REQ-066: Database verification failed - tables not created: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

            return db

        except Exception as e:
            error_msg = f"REQ-022: Failed to connect to database: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def close(self) -> None:
        """Close database connection.

        REQ-022: Close PonyORM database connection.
        """
        global db
        if db is not None:
            db.disconnect()
            logger.info("REQ-022: Database connection closed")
            db = None


def get_db() -> Database:
    """Get database instance.

    REQ-022: Get PonyORM database instance.

    Returns:
        Database instance.

    Raises:
        RuntimeError: If database is not initialized.
    """
    global db
    if db is None:
        raise RuntimeError("REQ-022: Database not initialized. Call DatabaseConnection.connect() first.")
    return db


def is_connected() -> bool:
    """Check if database is connected.

    REQ-022: Check database connection status.

    Returns:
        True if database is connected, False otherwise.
    """
    global db
    return db is not None
