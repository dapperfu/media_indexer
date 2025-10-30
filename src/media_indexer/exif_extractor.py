"""
EXIF Extractor Module

REQ-003: EXIF parsing using fast-exif-rs-py for optimal performance.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any

import fast_exif_rs_py

logger = logging.getLogger(__name__)


class EXIFExtractor:
    """
    EXIF Extractor using fast-exif-rs-py.

    REQ-003: Extract EXIF data using the fast-exif-rs-py library.
    """

    def __init__(self) -> None:
        """
        Initialize EXIF extractor.

        Raises:
            RuntimeError: If fast-exif-rs-py is not available (REQ-021).
        """
        logger.debug("REQ-003: EXIF extractor initialized with fast-exif-rs-py")

    def extract_from_path(self, image_path: Path) -> dict[str, Any]:
        """
        Extract EXIF data from an image file.

        REQ-003: Extract EXIF metadata from image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict containing EXIF data.

        Raises:
            RuntimeError: If extraction fails (REQ-021).
        """
        logger.debug(f"REQ-003: Extracting EXIF from {image_path}")
        # REQ-003: Use fast-exif-rs-py for extraction
        reader = fast_exif_rs_py.PyFastExifReader()
        exif_data: dict[str, Any] = reader.read_file(str(image_path))

        logger.debug(f"REQ-003: Successfully extracted EXIF from {image_path}")
        return exif_data

    def extract_from_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Extract EXIF data from image bytes.

        REQ-003: Extract EXIF metadata from image bytes.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Dict containing EXIF data.

        Raises:
            RuntimeError: If extraction fails (REQ-021).
        """
        logger.debug("REQ-003: Extracting EXIF from bytes")
        # REQ-003: Use fast-exif-rs-py for extraction
        reader = fast_exif_rs_py.PyFastExifReader()
        exif_data: dict[str, Any] = reader.read_bytes(image_bytes)

        logger.debug("REQ-003: Successfully extracted EXIF from bytes")
        return exif_data


def get_exif_extractor() -> EXIFExtractor:
    """
    Factory function to get EXIF extractor instance.

    REQ-003: Factory function for EXIF extraction.

    Returns:
        EXIFExtractor: Configured EXIF extractor.
    """
    return EXIFExtractor()
