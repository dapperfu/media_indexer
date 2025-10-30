"""
Cancellation Manager for Graceful Shutdown

REQ-015: Handle user interruption gracefully with quick shutdown.
REQ-010: All code components directly linked to requirements.
"""

import logging
import signal
import threading
from typing import Any

logger = logging.getLogger(__name__)


class CancellationManager:
    """
    Thread-safe cancellation manager for graceful shutdown.

    REQ-015: Provides centralized cancellation flag and signal handling
    for quick shutdown of processing operations.

    Attributes:
        _flag: Thread-safe event flag for cancellation state.
        _original_handler: Original signal handler to restore on cleanup.
    """

    def __init__(self) -> None:
        """
        Initialize cancellation manager.

        REQ-015: Create thread-safe cancellation flag.
        """
        self._flag: threading.Event = threading.Event()
        self._original_handler: Any = None

    def setup_signal_handler(self) -> None:
        """
        Register signal handler for SIGINT.

        REQ-015: Set up signal handler to catch user interrupts.
        """
        self._original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        logger.debug("REQ-015: Signal handler registered")

    def _signal_handler(self, _signum: int, _frame: Any) -> None:
        """Signal handler for SIGINT (KeyboardInterrupt).

        REQ-015: Set shutdown flag when user interrupts processing.
        """
        logger.warning("REQ-015: Processing interrupted by user")
        self._flag.set()

    def reset(self) -> None:
        """
        Reset cancellation flag.

        REQ-015: Clear cancellation state for new processing batch.
        """
        self._flag.clear()
        logger.debug("REQ-015: Cancellation flag reset")

    def is_cancelled(self) -> bool:
        """
        Check if cancellation has been requested.

        REQ-015: Thread-safe check for cancellation state.

        Returns:
            True if cancellation requested, False otherwise.
        """
        return self._flag.is_set()

    def cancel(self) -> None:
        """
        Set cancellation flag.

        REQ-015: Manually trigger cancellation.
        """
        self._flag.set()
        logger.debug("REQ-015: Cancellation requested")

    def cleanup(self) -> None:
        """
        Restore original signal handler.

        REQ-015: Clean up signal handler on completion.
        """
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
            self._original_handler = None
            logger.debug("REQ-015: Signal handler restored")
