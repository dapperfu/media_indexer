"""
Statistics Tracking

REQ-012: Track and display processing statistics.
REQ-010: All code components directly linked to requirements.
"""

import logging
from datetime import datetime
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class StatisticsTracker:
    """
    Thread-safe statistics tracker.

    REQ-012: Tracks processing statistics with thread-safe updates.
    """

    def __init__(self) -> None:
        """
        Initialize statistics tracker.

        REQ-012: Create statistics dictionary and lock.
        """
        self.stats: dict[str, Any] = {
            "total_images": 0,
            "processed_images": 0,
            "skipped_images": 0,
            "error_images": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
        }
        self.stats_lock: Lock = Lock()

    def update_increment(self, key: str) -> None:
        """
        Thread-safe stats update.

        Args:
            key: Stat key to increment.
        """
        with self.stats_lock:
            self.stats[key] += 1

    def set(self, key: str, value: Any) -> None:
        """
        Set a statistics value.

        Args:
            key: Stat key to set.
            value: Value to set.
        """
        with self.stats_lock:
            self.stats[key] = value

    def get(self, key: str) -> Any:
        """
        Get a statistics value.

        Args:
            key: Stat key to get.

        Returns:
            Statistics value.
        """
        return self.stats.get(key)

    def print_statistics(self) -> None:
        """
        Print processing statistics.

        REQ-012: Display processing statistics to log.
        """
        logger.info("REQ-012: Processing complete")
        logger.info(f"  Total images: {self.stats['total_images']}")
        logger.info(f"  Processed: {self.stats['processed_images']}")
        logger.info(f"  Skipped: {self.stats['skipped_images']}")
        logger.info(f"  Errors: {self.stats['error_images']}")
        logger.info(f"  Start time: {self.stats['start_time']}")
        logger.info(f"  End time: {self.stats['end_time']}")

    def finalize(self) -> None:
        """
        Finalize statistics.

        REQ-012: Set end time and return statistics dictionary.
        """
        self.set("end_time", datetime.now().isoformat())

    def get_stats(self) -> dict[str, Any]:
        """
        Get all statistics.

        Returns:
            Statistics dictionary.
        """
        return self.stats.copy()

