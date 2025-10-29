"""
RAW Image Converter Module

REQ-040: Convert RAW image files to usable formats in memory for detection models.
REQ-010: All code components directly linked to requirements.
"""

import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def convert_raw_to_array(image_path: Path) -> tuple[np.ndarray | None, str]:
    """
    Convert RAW image file to numpy array in memory.

    REQ-040: Convert RAW images for YOLO processing.

    Args:
        image_path: Path to RAW image file.

    Returns:
        Tuple of (numpy array, format). Returns (None, "unknown") on failure.
    """
    try:
        import rawpy
    except ImportError:
        logger.warning("REQ-040: rawpy not available, cannot convert RAW images")
        return None, "unknown"

    raw_extensions = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef"}
    
    if image_path.suffix.lower() not in raw_extensions:
        logger.debug(f"REQ-040: {image_path} is not a RAW file")
        return None, "unknown"

    try:
        logger.debug(f"REQ-040: Converting RAW file {image_path} to numpy array")
        
        with rawpy.imread(str(image_path)) as raw:
            rgb = raw.postprocess()
        
        # Convert to uint8 if needed
        if rgb.dtype != np.uint8:
            rgb = (rgb / 255.0 * 255).astype(np.uint8)
        
        logger.debug(f"REQ-040: Successfully converted RAW file {image_path} to array shape {rgb.shape}")
        return rgb, "RGB"
    
    except Exception as e:
        logger.warning(f"REQ-040: Failed to convert RAW file {image_path}: {e}")
        return None, "unknown"


def convert_raw_to_pil(image_path: Path) -> Image.Image | None:
    """
    Convert RAW image file to PIL Image in memory.

    REQ-040: Convert RAW images for processing.

    Args:
        image_path: Path to RAW image file.

    Returns:
        PIL Image or None on failure.
    """
    rgb_array, _ = convert_raw_to_array(image_path)
    
    if rgb_array is None:
        return None
    
    try:
        # Convert numpy array to PIL Image
        image = Image.fromarray(rgb_array)
        logger.debug(f"REQ-040: Converted {image_path} to PIL Image shape {image.size}")
        return image
    except Exception as e:
        logger.warning(f"REQ-040: Failed to convert RAW array to PIL Image: {e}")
        return None


def convert_raw_to_temp_jpeg(image_path: Path) -> Path | None:
    """
    Convert RAW image to temporary JPEG file for YOLO processing.

    REQ-040: Convert RAW images for YOLO compatibility.

    Args:
        image_path: Path to RAW image file.

    Returns:
        Path to temporary JPEG file or None on failure.
    """
    image = convert_raw_to_pil(image_path)
    
    if image is None:
        return None
    
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        
        # Convert to RGB if needed (some cameras output RGBA)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Save as JPEG
        image.save(temp_path, "JPEG", quality=95)
        logger.debug(f"REQ-040: Saved RAW conversion to {temp_path}")
        return temp_path
    
    except Exception as e:
        logger.warning(f"REQ-040: Failed to create temporary JPEG from RAW: {e}")
        return None


def is_raw_file(image_path: Path) -> bool:
    """
    Check if image file is a RAW format.

    Args:
        image_path: Path to image file.

    Returns:
        True if file is a RAW format.
    """
    raw_extensions = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef", ".raw"}
    return image_path.suffix.lower() in raw_extensions


# Global registry for temporary files created during processing
_temp_file_registry: set[Path] = set()


def register_temp_file(temp_path: Path) -> Path:
    """
    Register a temporary file for later cleanup.

    Args:
        temp_path: Path to temporary file.

    Returns:
        The same temp_path for chaining.
    """
    _temp_file_registry.add(temp_path)
    return temp_path


def cleanup_temp_files() -> None:
    """
    Clean up all registered temporary files.

    REQ-040: Remove temporary files created for RAW conversion.
    """
    for temp_path in _temp_file_registry:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.warning(f"REQ-040: Failed to remove temp file {temp_path}: {e}")
    
    _temp_file_registry.clear()


def get_raw_image_source(image_path: Path) -> str:
    """
    Get a source that can be used by YOLO for RAW images.

    REQ-040: Return a usable image source for YOLO processing.

    Args:
        image_path: Path to RAW image file.

    Returns:
        Path to usable image file (temp file for RAW, original for non-RAW).
    """
    if is_raw_file(image_path):
        temp_jpeg = convert_raw_to_temp_jpeg(image_path)
        if temp_jpeg:
            register_temp_file(temp_jpeg)
            return str(temp_jpeg)
    return str(image_path)

