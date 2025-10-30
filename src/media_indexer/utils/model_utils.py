"""
Model Download Utilities

REQ-007, REQ-008, REQ-009: Model download utilities for YOLO and InsightFace models.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


def download_model_if_needed(model_path: str, url: str, requirement_id: str = "REQ") -> Path:
    """
    Download model file if it doesn't exist.

    REQ-007, REQ-008, REQ-009: Download model files for YOLO and InsightFace models.

    Args:
        model_path: Local path to model file.
        url: URL to download from.
        requirement_id: Requirement ID for logging (e.g., "REQ-007").

    Returns:
        Path to model file.

    Raises:
        RuntimeError: If download fails.
    """
    path = Path(model_path)
    
    if path.exists():
        return path
    
    logger.info(f"{requirement_id}: Downloading {model_path} from {url}")
    try:
        urlretrieve(url, path)
        logger.info(f"{requirement_id}: Successfully downloaded {path}")
        return path
    except Exception as e:
        logger.error(f"{requirement_id}: Failed to download {model_path}: {e}")
        raise RuntimeError(f"Failed to download model from {url}: {e}") from e

