"""
Sidecar File Utilities

REQ-004: Utilities for reading sidecar files with fallback.
REQ-010: All code components directly linked to requirements.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_sidecar_metadata(
    sidecar_path: Path,
    sidecar_generator: Any | None = None,  # SidecarGenerator (avoid circular import)
) -> dict[str, Any]:
    """
    Read sidecar metadata with fallback to JSON.

    REQ-004: Read sidecar file using generator if available,
    otherwise fall back to direct JSON reading.

    Args:
        sidecar_path: Path to the sidecar file.
        sidecar_generator: Optional SidecarGenerator instance.

    Returns:
        Metadata dictionary from sidecar file.

    Raises:
        IOError: If sidecar file cannot be read.
    """
    if sidecar_generator:
        try:
            return sidecar_generator.read_sidecar(sidecar_path)
        except Exception as e:
            logger.debug(f"REQ-004: Sidecar generator read failed: {e}, using fallback")
            # Fall through to JSON fallback

    # Fallback: read JSON directly
    try:
        with open(sidecar_path) as f:
            metadata = json.load(f)
        logger.debug(f"REQ-004: Read sidecar via JSON fallback: {sidecar_path}")
        return metadata
    except Exception as e:
        logger.error(f"REQ-004: Failed to read sidecar {sidecar_path}: {e}")
        raise

