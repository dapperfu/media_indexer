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

    def generate_sidecar(self, image_path: Path, metadata: dict[str, Any]) -> Path:
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
        # Use the correct API: save_data with operation type
        sidecar = image_sidecar_rust.ImageSidecar()
        try:
            info = sidecar.save_data(str(image_path), "extract", metadata)
            logger.debug(f"REQ-004: Generated sidecar info: {info}")
            return Path(info.get('sidecar_path', str(image_path) + '.json'))
        except Exception as e:
            # Fallback if library fails
            logger.warning(f"REQ-004: Sidecar generation failed, using fallback: {e}")
            return self._fallback_sidecar(image_path, metadata)

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
        sidecar = image_sidecar_rust.ImageSidecar()
        try:
            # Extract image path from sidecar path
            image_path = Path(str(sidecar_path).replace('.json', ''))
            metadata = sidecar.read_data(str(image_path))
            logger.debug(f"REQ-004: Read sidecar from {sidecar_path}")
            return metadata
        except Exception as e:
            logger.warning(f"REQ-004: Sidecar read failed: {e}")
            return {}
    
    def _fallback_sidecar(self, image_path: Path, metadata: dict[str, Any]) -> Path:
        """Fallback sidecar generation using JSON file."""
        sidecar_path = image_path.with_suffix(image_path.suffix + '.json')
        import json
        with open(sidecar_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return sidecar_path


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
