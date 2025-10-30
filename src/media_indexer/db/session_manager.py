"""
Database Session Manager

REQ-022: Context manager for database sessions.
REQ-023: Simplified database connection management.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any

from pony.orm import db_session as pony_db_session

from media_indexer.db.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class DatabaseSession:
    """
    Context manager for database sessions.

    REQ-022: Provides simplified database connection and session management
    with automatic cleanup.

    Attributes:
        db_connection: Database connection instance.
        _session_context: PonyORM session context manager.
    """

    def __init__(self, database_path: Path) -> None:
        """
        Initialize database session.

        REQ-022: Create database connection.

        Args:
            database_path: Path to SQLite database file.
        """
        self.database_path: Path = Path(database_path).resolve()
        self.db_connection: DatabaseConnection | None = None
        self._session_context: Any | None = None

    def __enter__(self) -> Any:
        """
        Enter context manager.

        REQ-022: Connect to database and return session.

        Returns:
            Database session context manager.
        """
        self.db_connection = DatabaseConnection(self.database_path)
        self.db_connection.connect()
        self._session_context = pony_db_session().__enter__()
        return self._session_context

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit context manager.

        REQ-022: Close session and database connection.
        """
        if self._session_context is not None:
            try:
                self._session_context.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.error(f"REQ-022: Error closing session: {e}")

        if self.db_connection is not None:
            try:
                self.db_connection.close()
            except Exception as e:
                logger.error(f"REQ-022: Error closing database: {e}")
