"""
RAW Image Converter Module

REQ-040: Convert RAW image files to usable formats in memory for detection models.
REQ-010: All code components directly linked to requirements.
"""

import logging
import os
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
from PIL import Image

from media_indexer.utils.file_utils import is_raw_file

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

        rawpy_available = True
    except ImportError:
        logger.warning("REQ-040: rawpy not available, will use PIL fallback for RAW images")
        rawpy_available = False

    if not is_raw_file(image_path):
        logger.debug(f"REQ-040: {image_path} is not a RAW file")
        return None, "unknown"

    # Try rawpy first if available
    if rawpy_available:
        try:
            logger.debug(f"REQ-040: Converting RAW file {image_path} to numpy array using rawpy")
            # REQ-016: Suppress CR2 corruption messages from rawpy (printed to stderr)
            # Use os.devnull for C library output suppression
            # REQ-015: Handle KeyboardInterrupt gracefully to prevent crashes
            try:
                with (
                    open(os.devnull, "w") as devnull,
                    redirect_stderr(devnull),
                    rawpy.imread(str(image_path)) as raw,
                ):
                    rgb = raw.postprocess()
            except KeyboardInterrupt:
                # REQ-015: Handle interrupt during rawpy processing
                logger.warning(f"REQ-015: RAW conversion interrupted for {image_path}")
                raise

            # Convert to uint8 if needed
            if rgb.dtype != np.uint8:
                rgb = (rgb / 255.0 * 255).astype(np.uint8)

            logger.debug(f"REQ-040: Successfully converted RAW file {image_path} to array shape {rgb.shape}")
            return rgb, "RGB"

        except KeyboardInterrupt:
            # REQ-015: Re-raise KeyboardInterrupt to propagate to signal handler
            raise
        except Exception as e:
            logger.debug(f"REQ-040: rawpy failed for {image_path}: {e}, trying PIL fallback")

    # Fall back to PIL (works for many CR2 files even when rawpy fails)
    try:
        logger.debug(f"REQ-040: Converting RAW file {image_path} to numpy array using PIL")
        # REQ-016: Suppress CR2 corruption messages from PIL (printed to stderr)
        # Use os.devnull for C library output suppression
        # REQ-015: Handle KeyboardInterrupt gracefully
        try:
            with (
                open(os.devnull, "w") as devnull,
                redirect_stderr(devnull),
            ):
                image = Image.open(str(image_path))
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
        except KeyboardInterrupt:
            # REQ-015: Handle interrupt during PIL processing
            logger.warning(f"REQ-015: RAW conversion interrupted for {image_path}")
            raise

        rgb_array = np.array(image)

        # Ensure uint8 dtype
        if rgb_array.dtype != np.uint8:
            rgb_array = rgb_array.astype(np.uint8)

        logger.debug(
            f"REQ-040: Successfully converted RAW file {image_path} to array shape {rgb_array.shape} using PIL"
        )
        return rgb_array, "RGB"

    except KeyboardInterrupt:
        # REQ-015: Re-raise KeyboardInterrupt to propagate to signal handler
        raise
    except Exception as e:
        logger.warning(f"REQ-040: Failed to convert RAW file {image_path} with both rawpy and PIL: {e}")
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
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

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


# Global registry for temporary files created during processing
_temp_file_registry: set[Path] = set()


def _remove_temp_file(temp_path: Path) -> None:
    """Remove a temporary file if it exists.

    Parameters
    ----------
    temp_path : Path
        Path to the temporary file.
    """

    if not temp_path.exists():
        return

    try:
        temp_path.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning("REQ-040: Failed to remove temp file %s: %s", temp_path, exc)


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
        _remove_temp_file(temp_path)

    _temp_file_registry.clear()


def get_raw_image_source(image_path: Path) -> str | None:
    """
    Get a source that can be used by YOLO for RAW images.

    REQ-040: Return a usable image source for YOLO processing.

    Args:
        image_path: Path to RAW image file.

    Returns:
        Path to usable image file (temp file for RAW, original for non-RAW), or None if conversion failed.
    """
    if is_raw_file(image_path):
        temp_jpeg = convert_raw_to_temp_jpeg(image_path)
        if temp_jpeg:
            register_temp_file(temp_jpeg)
            return str(temp_jpeg)
        else:
            # RAW conversion failed - return None to skip
            logger.warning(f"REQ-040: Skipping {image_path} - RAW conversion failed")
            return None
    return str(image_path)


def load_image_to_array(image_path: Path) -> np.ndarray | None:
    """
    Load an image file to numpy array using the best method available.

    REQ-040: Load both RAW and standard images to numpy arrays.

    This function tries multiple methods:
    1. For RAW files: Uses rawpy or PIL
    2. For standard formats: Uses PIL which has better format support

    Args:
        image_path: Path to image file.

    Returns:
        Numpy array with image data (RGB format) or None on failure.
    """
    # Try RAW conversion first
    if is_raw_file(image_path):
        rgb_array, _ = convert_raw_to_array(image_path)
        if rgb_array is not None:
            return rgb_array

    # Fall back to PIL for all image types
    try:
        # REQ-016: Suppress corruption messages from PIL (printed to stderr)
        # Use os.devnull for C library output suppression
        with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
            image = Image.open(image_path)
            # Convert to RGB if needed
            if image.mode in ("RGBA", "LA", "P"):
                # Create white background for alpha channel
                if image.mode == "RGBA":
                    rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                else:
                    rgb_image = image.convert("RGB")
            elif image.mode != "RGB":
                rgb_image = image.convert("RGB")
            else:
                rgb_image = image

        # Convert to numpy array
        rgb_array = np.array(rgb_image)
        logger.debug(f"REQ-040: Loaded image {image_path} to array shape {rgb_array.shape}")
        return rgb_array

    except Exception as e:
        logger.warning(f"REQ-040: Failed to load image {image_path}: {e}")
        return None
