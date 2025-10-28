"""
Sidecar Generator Module

REQ-004: Sidecar file generation.
REQ-010: All code components directly linked to requirements.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SidecarGenerator:
    """
    Sidecar file generator.

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
        try:
            logger.debug(f"REQ-004: Generating sidecar for {image_path}")

            # REQ-004: Determine sidecar filename in output directory
            # Use the image filename to maintain structure
            image_filename = image_path.name
            sidecar_path: Path = self.output_dir / f"{image_filename}.sidecar"

            # REQ-004: Write sidecar file
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"REQ-004: Generated sidecar at {sidecar_path}")
            return sidecar_path

        except Exception as e:
            error_msg = f"REQ-004: Failed to generate sidecar for {image_path}: {e}"
            logger.error(error_msg)
            raise OSError(error_msg) from e

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
        try:
            logger.debug(f"REQ-004: Reading sidecar from {sidecar_path}")
            # REQ-004: Read sidecar file
            with open(sidecar_path, encoding="utf-8") as f:
                metadata: dict[str, Any] = json.load(f)

            logger.debug(f"REQ-004: Read sidecar from {sidecar_path}")
            return metadata

        except Exception as e:
            logger.warning(f"REQ-004: Failed to read sidecar {sidecar_path}: {e}")
            return {}


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

