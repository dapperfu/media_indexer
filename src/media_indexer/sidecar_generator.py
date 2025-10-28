"""
Sidecar Generator Module

REQ-004: Sidecar file generation using image-sidecar-rust library.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any

import image_sidecar_rust

logger = logging.getLogger(__name__)


class SidecarGenerator:
    """
    Sidecar file generator using image-sidecar-rust library.

    REQ-004: Generate sidecar files containing extracted metadata.
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize sidecar generator.

        Args:
            output_dir: Directory where sidecar files should be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"REQ-004: Sidecar generator initialized with output_dir={output_dir}")

    def generate_sidecar(
        self, image_path: Path, metadata: dict[str, Any]
    ) -> Path:
        """
        Generate sidecar file for an image.

        REQ-004: Generate sidecar file for image containing metadata.

        Args:
            image_path: Path to the image file.
            metadata: Metadata to store in sidecar (faces, objects, poses, EXIF).

        Returns:
            Path to the generated sidecar file.

        Raises:
            IOError: If sidecar file cannot be written.
        """
        logger.debug(f"REQ-004: Generating sidecar for {image_path}")

        # REQ-004: Use image-sidecar-rust to handle sidecar file creation
        # The library determines the sidecar filename based on format
        sidecar_path: Path = image_sidecar_rust.write_sidecar(  # type: ignore[misc, arg-type]
            str(image_path), metadata, str(self.output_dir)
        )

        logger.debug(f"REQ-004: Generated sidecar at {sidecar_path}")
        return sidecar_path

    def read_sidecar(self, sidecar_path: Path) -> dict[str, Any]:
        """
        Read metadata from a sidecar file.

        REQ-004: Read sidecar file containing metadata.

        Args:
            sidecar_path: Path to the sidecar file.

        Returns:
            Dict containing metadata from sidecar.

        Raises:
            IOError: If sidecar file cannot be read.
        """
        logger.debug(f"REQ-004: Reading sidecar from {sidecar_path}")
        # REQ-004: Read sidecar file using image-sidecar-rust
        metadata: dict[str, Any] = image_sidecar_rust.read_sidecar(  # type: ignore[misc, arg-type]
            str(sidecar_path)
        )

        logger.debug(f"REQ-004: Read sidecar from {sidecar_path}")
        return metadata


def get_sidecar_generator(output_dir: Path) -> SidecarGenerator:
    """
    Factory function to get sidecar generator instance.

    REQ-004: Factory function for sidecar generation.

    Args:
        output_dir: Directory where sidecar files should be saved.

    Returns:
        SidecarGenerator: Configured sidecar generator.
    """
    return SidecarGenerator(output_dir)

