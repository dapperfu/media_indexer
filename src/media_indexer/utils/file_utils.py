"""
File Utilities

REQ-018, REQ-040: File extension and path utilities.
REQ-010: All code components directly linked to requirements.
"""

from pathlib import Path
from typing import Set


def get_image_extensions() -> Set[str]:
    """
    Get set of supported image file extensions.

    REQ-018: Support common image formats including RAW files.

    Returns:
        Set of image file extensions (lowercase, with leading dot).
    """
    return {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".raw", ".cr2", ".nef", ".arw"}


def get_raw_extensions() -> Set[str]:
    """
    Get set of RAW image file extensions.

    REQ-040: Support RAW image formats.

    Returns:
        Set of RAW file extensions (lowercase, with leading dot).
    """
    return {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef", ".raw"}


def is_image_file(file_path: Path) -> bool:
    """
    Check if file is a supported image format.

    REQ-018: Check if file is a supported image format.

    Args:
        file_path: Path to the file.

    Returns:
        True if file is a supported image format.
    """
    return file_path.suffix.lower() in get_image_extensions()


def is_raw_file(file_path: Path) -> bool:
    """
    Check if image file is a RAW format.

    REQ-040: Check if file is a RAW format.

    Args:
        file_path: Path to image file.

    Returns:
        True if file is a RAW format.
    """
    return file_path.suffix.lower() in get_raw_extensions()

