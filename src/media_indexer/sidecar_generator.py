"""
Sidecar Generator Module

REQ-004: Sidecar file generation in binary format using image-sidecar-rust.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any, Dict

try:
    import image_sidecar_rust
except ImportError:
    image_sidecar_rust = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class SidecarGenerator:
    """
    Sidecar file generator using image-sidecar-rust.

    REQ-004: Generate sidecar files in binary format.
    """

    def __init__(self) -> None:
        """
        Initialize sidecar generator.

        Raises:
            ImportError: If image-sidecar-rust is not available.
        """
        if image_sidecar_rust is None:
            raise ImportError(
                "REQ-004: image-sidecar-rust is not available. Please install it."
            )
        logger.debug("REQ-004: Sidecar generator initialized with image-sidecar-rust")

    def generate_sidecar(
        self, image_path: Path, metadata: Dict[str, Any]
    ) -> Path:
        """
        Generate sidecar file for an image.

        REQ-004: Generate binary sidecar file for image.

        Args:
            image_path: Path to the image file.
            metadata: Metadata to store in sidecar (faces, objects, poses, EXIF).

        Returns:
            Path to the generated sidecar file.

        Raises:
            IOError: If sidecar file cannot be written.
        """
        try:
            logger.debug(f"REQ-004: Generating sidecar for {image_path}")

            # REQ-004: Determine sidecar filename
            sidecar_path: Path = Path(f"{image_path}.sidecar")

            # REQ-004: Use image-sidecar-rust to generate binary sidecar
            # Note: Actual API depends on image-sidecar-rust interface
            image_sidecar_rust.write_sidecar(
                str(sidecar_path), metadata  # type: ignore[misc, arg-type]
            )

            logger.debug(f"REQ-004: Generated sidecar at {sidecar_path}")
            return sidecar_path

        except Exception as e:
            error_msg = f"REQ-004: Failed to generate sidecar for {image_path}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

    def read_sidecar(self, sidecar_path: Path) -> Dict[str, Any]:
        """
        Read metadata from a sidecar file.

        REQ-004: Read binary sidecar file.

        Args:
            sidecar_path: Path to the sidecar file.

        Returns:
            Dict containing metadata from sidecar.

        Raises:
            IOError: If sidecar file cannot be read.
        """
        try:
            logger.debug(f"REQ-004: Reading sidecar from {sidecar_path}")
            # REQ-004: Use image-sidecar-rust to read binary sidecar
            metadata: Dict[str, Any] = image_sidecar_rust.read_sidecar(
                str(sidecar_path)  # type: ignore[misc, arg-type]
            )

            logger.debug(f"REQ-004: Read sidecar from {sidecar_path}")
            return metadata

        except Exception as e:
            logger.warning(f"REQ-004: Failed to read sidecar {sidecar_path}: {e}")
            return {}


def get_sidecar_generator() -> SidecarGenerator:
    """
    Factory function to get sidecar generator instance.

    REQ-004: Factory function for sidecar generation.

    Returns:
        SidecarGenerator: Configured sidecar generator.
    """
    return SidecarGenerator()

