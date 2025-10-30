"""
Parallel Executor with Cancellation Support

REQ-020: Parallel processing with thread-based I/O operations.
REQ-015: Handle user interruption gracefully with quick shutdown.
REQ-010: All code components directly linked to requirements.
"""

import logging
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ParallelExecutor(Generic[T, R]):
    """
    Context manager for parallel execution with cancellation support.

    REQ-020: Provides thread-based parallel execution with configurable workers.
    REQ-015: Supports graceful cancellation and quick shutdown.

    Attributes:
        workers: Number of parallel workers.
        cancellation_manager: Cancellation manager for shutdown handling.
        progress_bar: Optional progress bar for tracking.
        executor: Thread pool executor instance.
    """

    def __init__(
        self,
        workers: int,
        cancellation_manager: Any,  # CancellationManager (avoid circular import)
        progress_bar: Any | None = None,
    ) -> None:
        """
        Initialize parallel executor.

        REQ-020: Configure worker count and cancellation handling.

        Args:
            workers: Number of parallel workers.
            cancellation_manager: Cancellation manager instance.
            progress_bar: Optional progress bar for updates.
        """
        self.workers: int = workers
        self.cancellation_manager: Any = cancellation_manager
        self.progress_bar: Any | None = progress_bar
        self.executor: ThreadPoolExecutor | None = None

    def execute(
        self,
        items: list[T],
        func: Callable[[T], tuple[bool, R]],
    ) -> list[tuple[T, bool, R]]:
        """
        Execute function on items in parallel.

        REQ-020: Process items in parallel with thread pool.
        REQ-015: Check for cancellation and handle gracefully.

        Args:
            items: List of items to process.
            func: Function to execute on each item (returns success, result).

        Returns:
            List of tuples (item, success, result) for each processed item.
        """
        if not items:
            return []

        results: list[tuple[T, bool, R]] = []

        if self.workers == 1:
            # Sequential processing for single worker
            for item in items:
                if self.cancellation_manager.is_cancelled():
                    logger.warning("REQ-015: Processing interrupted by user")
                    break

                try:
                    success, result = func(item)
                    results.append((item, success, result))
                except Exception as e:
                    logger.error(f"REQ-015: Error processing {item}: {e}")
                    results.append((item, False, None))  # type: ignore[assignment]
                finally:
                    if self.progress_bar:
                        self.progress_bar.update(1)
        else:
            # REQ-020: Parallel processing with thread pool
            try:
                self.executor = ThreadPoolExecutor(max_workers=self.workers)
                futures = {
                    self.executor.submit(func, item): item for item in items
                }

                for future in as_completed(futures):
                    # REQ-015: Check for cancellation
                    if self.cancellation_manager.is_cancelled():
                        logger.warning("REQ-015: Processing interrupted by user")
                        # Cancel remaining futures
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break

                    item = futures[future]
                    try:
                        success, result = future.result()
                        results.append((item, success, result))
                    except CancelledError:
                        # REQ-015: Future was cancelled, skip
                        logger.debug(f"REQ-015: Processing cancelled for {item}")
                        results.append((item, False, None))  # type: ignore[assignment]
                    except Exception as e:
                        logger.error(f"REQ-015: Error processing {item}: {e}")
                        results.append((item, False, None))  # type: ignore[assignment]
                    finally:
                        if self.progress_bar:
                            self.progress_bar.update(1)
            finally:
                # REQ-015: Shutdown executor quickly when interrupted
                if self.executor:
                    if self.cancellation_manager.is_cancelled():
                        self.executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        self.executor.shutdown(wait=True)

        return results

    def __enter__(self) -> "ParallelExecutor[T, R]":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit context manager.

        REQ-015: Ensure executor is properly shutdown.
        """
        if self.executor:
            if self.cancellation_manager.is_cancelled():
                self.executor.shutdown(wait=False, cancel_futures=True)
            else:
                self.executor.shutdown(wait=True)

