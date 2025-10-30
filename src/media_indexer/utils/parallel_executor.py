"""Parallel execution helpers with cooperative cancellation support.

REQ-020: Parallel processing with thread-based I/O operations.
REQ-015: Handle user interruption gracefully with quick shutdown.
REQ-010: All code components directly linked to requirements.
"""

import logging
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R_co = TypeVar("R_co")


@dataclass(slots=True)
class ParallelResult(Generic[T, R_co]):
    """Container describing the outcome of processing a single item.

    Parameters
    ----------
    item : T
        Work item that was processed.
    success : bool
        Indicator specifying whether ``func`` completed without raising.
    value : R_co | None
        Value returned by ``func`` when successful.
    error : Exception | None
        Captured exception when execution fails.

    Notes
    -----
    REQ-070
        Provides a structured payload for downstream error handling while
        preserving type information required by the project's mypy policy
        (REQ-057).
    """

    item: T
    success: bool
    value: R_co | None
    error: Exception | None = None


class ParallelExecutor(Generic[T, R_co]):
    """Coordinate callable execution across a worker pool.

    Parameters
    ----------
    workers : int
        Number of worker threads to allocate.
    cancellation_manager : Any
        Object exposing ``is_cancelled`` used to coordinate graceful
        shutdown (REQ-015).
    progress_bar : Any, optional
        Optional UI element providing an ``update`` method.

    Notes
    -----
    REQ-070
        Exposes structured results instead of raw tuples so callers can
        distinguish success, payload, and exceptions without sentinel
        values.
    """

    def __init__(
        self,
        workers: int,
        cancellation_manager: Any,  # CancellationManager (avoid circular import)
        progress_bar: Any | None = None,
    ) -> None:
        self.workers: int = workers
        self.cancellation_manager: Any = cancellation_manager
        self.progress_bar: Any | None = progress_bar
        self.executor: ThreadPoolExecutor | None = None

    def execute(
        self,
        items: list[T],
        func: Callable[[T], tuple[bool, R_co | None]],
    ) -> list[ParallelResult[T, R_co]]:
        """Execute ``func`` across ``items`` with cooperative cancellation.

        Parameters
        ----------
        items : list[T]
            Items scheduled for execution.
        func : Callable[[T], tuple[bool, R_co | None]]
            Callable returning a success flag and optional payload for each
            processed item.

        Returns
        -------
        list[ParallelResult[T, R_co]]
            Structured execution results preserving original items and any
            captured exceptions.

        Notes
        -----
        REQ-070
            Ensures type-safe propagation of worker outcomes and failures
            back to the caller.
        """
        if not items:
            return []

        results: list[ParallelResult[T, R_co]] = []

        if self.workers == 1:
            # Sequential processing for single worker
            for item in items:
                if self.cancellation_manager.is_cancelled():
                    logger.warning("REQ-015: Processing interrupted by user")
                    break

                try:
                    success, result = func(item)
                    results.append(ParallelResult(item=item, success=success, value=result))
                except Exception as exc:  # noqa: BLE001
                    logger.error("REQ-015: Error processing %s: %s", item, exc)
                    results.append(ParallelResult(item=item, success=False, value=None, error=exc))
                finally:
                    if self.progress_bar:
                        self.progress_bar.update(1)
        else:
            # REQ-020: Parallel processing with thread pool
            try:
                self.executor = ThreadPoolExecutor(max_workers=self.workers)
                futures = {self.executor.submit(func, item): item for item in items}

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
                        results.append(ParallelResult(item=item, success=success, value=result))
                    except CancelledError as exc:
                        # REQ-015: Future was cancelled, skip
                        logger.debug("REQ-015: Processing cancelled for %s", item)
                        results.append(ParallelResult(item=item, success=False, value=None, error=exc))
                    except Exception as exc:  # noqa: BLE001
                        logger.error("REQ-015: Error processing %s: %s", item, exc)
                        results.append(ParallelResult(item=item, success=False, value=None, error=exc))
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

    def __enter__(self) -> "ParallelExecutor[T, R_co]":
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
