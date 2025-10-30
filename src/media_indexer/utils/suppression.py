"""
Output Suppression Utilities

REQ-016: Suppress verbose output from libraries (OpenCV, ONNX Runtime, YOLO).
REQ-010: All code components directly linked to requirements.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

logger = logging.getLogger(__name__)


def setup_suppression() -> None:
    """Configure library log levels for clean CLI output.

    Notes
    -----
    REQ-016
        Suppresses ONNX Runtime and OpenCV verbosity so that the log
        levels controlled by CLI flags remain authoritative.
    """
    # REQ-016: Suppress ONNX Runtime verbose output before importing insightface
    os.environ.setdefault("ORT_LOG_LEVEL", "3")  # 3 = ERROR level (suppress INFO/VERBOSE)

    # REQ-016: Suppress OpenCV warnings when cv2 is imported
    try:
        import cv2

        # Suppress all OpenCV warnings except errors
        # OpenCV uses numeric log levels: 0=SILENT, 1=FATAL, 2=ERROR, 3=WARN, 4=INFO, 5=DEBUG, 6=VERBOSE
        # Set to ERROR level (2) to suppress WARN and below
        try:
            cv2.setLogLevel(2)  # ERROR level
        except (TypeError, AttributeError):
            # Fallback: try with string constant if available
            if hasattr(cv2, "LOG_LEVEL_ERROR"):
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
            elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except (ImportError, AttributeError):
        # OpenCV logging API not available or cv2 not installed
        pass


@contextmanager
def suppress_output() -> Iterator[None]:
    """Temporarily redirect both stdout and stderr to in-memory buffers.

    Yields
    ------
    None
        Control is yielded back to the caller while output is suppressed.

    Notes
    -----
    REQ-016
        Ensures verbose third-party libraries remain silent within the
        managed scope.
    REQ-069
        Guarantees stdout and stderr are redirected together without
        leaking descriptors or mutating global handlers beyond the
        context lifetime.
    """
    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(StringIO()))
        stack.enter_context(redirect_stderr(StringIO()))
        yield


def suppress_stderr_context() -> Any:
    """Construct a context manager that silences only stderr output.

    Returns
    -------
    Context manager
        A redirect context that captures stderr into an in-memory buffer.

    Notes
    -----
    REQ-016
        Useful when stdout must remain visible while suppressing noisy
        native-library warnings.
    REQ-069
        Leverages the same redirection guarantees as :func:`suppress_output`
        while targeting only stderr.
    """
    return redirect_stderr(StringIO())
