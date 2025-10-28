"""
EXIF Extractor Module

REQ-003: EXIF parsing using fast-exif-rs-py for optimal performance.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any, Dict

try:
    import fast_exif_rs
except ImportError:
    # Fallback for development/testing without the library
    fast_exif_rs = None  # type: ignore[assignment]

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
            ImportError: If fast-exif-rs-py is not available.
        """
        if fast_exif_rs is None:
            raise ImportError(
                "REQ-003: fast-exif-rs-py is not available. Please install it."
            )
        logger.debug("REQ-003: EXIF extractor initialized with fast-exif-rs-py")

    def extract_from_path(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract EXIF data from an image file.

        REQ-003: Extract EXIF metadata from image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict containing EXIF data.

        Raises:
            IOError: If image cannot be read.
            ValueError: If EXIF data cannot be parsed.
        """
        try:
            logger.debug(f"REQ-003: Extracting EXIF from {image_path}")
            # REQ-003: Use fast-exif-rs-py for extraction
            exif_data: Dict[str, Any] = fast_exif_rs.get_exif(
                str(image_path)
            )  # type: ignore[arg-type]

            logger.debug(f"REQ-003: Successfully extracted EXIF from {image_path}")
            return exif_data

        except Exception as e:
            logger.warning(
                f"REQ-003: Failed to extract EXIF from {image_path}: {e}"
            )
            return {}

    def extract_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract EXIF data from image bytes.

        REQ-003: Extract EXIF metadata from image bytes.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Dict containing EXIF data.

        Raises:
            ValueError: If EXIF data cannot be parsed.
        """
        try:
            logger.debug("REQ-003: Extracting EXIF from bytes")
            # REQ-003: Use fast-exif-rs-py for extraction
            exif_data: Dict[str, Any] = fast_exif_rs.get_exif_bytes(
                image_bytes
            )  # type: ignore[arg-type]

            logger.debug("REQ-003: Successfully extracted EXIF from bytes")
            return exif_data

        except Exception as e:
            logger.warning(f"REQ-003: Failed to extract EXIF from bytes: {e}")
            return {}


def get_exif_extractor() -> EXIFExtractor:
    """
    Factory function to get EXIF extractor instance.

    REQ-003: Factory function for EXIF extraction.

    Returns:
        EXIFExtractor: Configured EXIF extractor.
    """
    return EXIFExtractor()


