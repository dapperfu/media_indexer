"""
Output Suppression Utilities

REQ-016: Suppress verbose output from libraries (OpenCV, ONNX Runtime, YOLO).
REQ-010: All code components directly linked to requirements.
"""

import logging
import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

logger = logging.getLogger(__name__)


def setup_suppression() -> None:
    """
    Setup global output suppression for verbose libraries.

    REQ-016: Suppress ONNX Runtime and OpenCV verbose output.

    This function should be called early in the application startup,
    before importing libraries that may produce verbose output.
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
            if hasattr(cv2, 'LOG_LEVEL_ERROR'):
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
            elif hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except (ImportError, AttributeError):
        # OpenCV logging API not available or cv2 not installed
        pass


def suppress_output() -> Any:
    """
    Context manager to suppress stdout and stderr.

    REQ-016: Suppress output from libraries during initialization or processing.

    Returns:
        Context manager that suppresses stdout and stderr.
    """
    return redirect_stdout(StringIO()) and redirect_stderr(StringIO())


def suppress_stderr_context() -> Any:
    """
    Context manager to suppress stderr only.

    REQ-016: Suppress stderr from libraries (e.g., CR2 corruption messages).

    Returns:
        Context manager that suppresses stderr.
    """
    return redirect_stderr(StringIO())

