"""
Human Pose Detection Module

REQ-009: Human pose detection using YOLOv11-pose model.
REQ-010: All code components directly linked to requirements.
"""

import logging
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from media_indexer.raw_converter import get_raw_image_source
from media_indexer.utils.image_utils import normalize_bbox, normalize_keypoints
from media_indexer.utils.model_utils import download_model_if_needed

logger = logging.getLogger(__name__)

# Model download URLs
YOLO11X_POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt"


class PoseDetector:
    """
    Human pose detector using YOLOv11-pose.

    REQ-009: Use YOLOv11-pose model for human pose detection.
    """

    def __init__(
        self, device: torch.device, model_path: str = "yolo11x-pose.pt", cache_dir: Path | None = None
    ) -> None:
        """
        Initialize pose detector.

        REQ-009: Initialize YOLOv11-pose model.

        Args:
            device: GPU device for model execution.
            model_path: Path to YOLO pose model file.
            cache_dir: Optional cache directory for model storage.

        Raises:
            RuntimeError: If model cannot be loaded.
        """
        self.device: torch.device = device

        # Setup model cache for centralized storage
        from media_indexer.model_cache import ModelCache

        cache = ModelCache(cache_dir)
        cache.setup_environment()

        try:
            logger.info(f"REQ-009: Loading YOLOv11-pose model from {model_path}")
            logger.debug(f"REQ-009: Model cache: {cache.yolo_cache}")
            # Download model if needed
            actual_path = download_model_if_needed(model_path, YOLO11X_POSE_URL, "REQ-009")
            # REQ-016: Suppress YOLO model summary output during initialization
            null_stream = StringIO()
            with redirect_stdout(null_stream), redirect_stderr(null_stream):
                self.model: YOLO = YOLO(str(actual_path))
                # Test model compatibility immediately to catch issues early
                # This prevents segmentation faults during inference
                try:
                    # Try to access model structure to verify compatibility
                    _ = self.model.model
                except (AttributeError, ModuleNotFoundError) as compat_error:
                    error_msg = str(compat_error)
                    if "ultralytics.yolo" in error_msg or "Conv" in error_msg or "bn" in error_msg:
                        logger.error(
                            f"REQ-009: YOLOv11-pose model is incompatible with current ultralytics version: {compat_error}. "
                            "The model appears to require an older ultralytics version or has compatibility issues."
                        )
                        raise RuntimeError(
                            f"REQ-009: Model compatibility error: {compat_error}. "
                            "Try updating ultralytics or using a compatible model version."
                        ) from compat_error
                    else:
                        raise
            logger.info("REQ-009: YOLOv11-pose model loaded successfully")
        except (ModuleNotFoundError, AttributeError) as e:
            error_msg_str = str(e)
            if "ultralytics.yolo" in error_msg_str or "Conv" in error_msg_str or "bn" in error_msg_str:
                error_msg = (
                    f"REQ-009: YOLOv11-pose model compatibility issue detected: {e}. "
                    "The model appears to require an older ultralytics version. "
                    "Try updating ultralytics: pip install --upgrade ultralytics"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            else:
                error_msg = f"REQ-009: Failed to load YOLOv11-pose model: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"REQ-009: Failed to load YOLOv11-pose model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def detect_poses(self, image_path: Path) -> list[dict[str, Any]]:
        """
        Detect human poses in an image.

        REQ-009: Detect human poses using YOLOv11-pose.
        REQ-040: Support RAW image files via in-memory conversion.

        Args:
            image_path: Path to the image file.

        Returns:
            List of detected poses with keypoints.
        """
        try:
            logger.debug(f"REQ-009: Detecting poses in {image_path}")
            # REQ-040: Convert RAW images to usable format
            source_path = get_raw_image_source(image_path)
            if source_path is None:
                logger.debug(f"REQ-009: Skipping {image_path} - no valid source")
                return []
            # REQ-009: Use YOLOv11-pose for pose detection
            results = self.model(source_path, device=self.device, verbose=False)

            poses: list[dict[str, Any]] = []
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                # Get image dimensions for normalization
                img_height, img_width = result.orig_shape

                for box, keypoint in zip(boxes, keypoints, strict=True):
                    confidence = float(box.conf.item())

                    # Normalize bbox to percentages (0.0-1.0)
                    bbox_absolute = box.xyxy[0].cpu().numpy().tolist()
                    bbox_normalized = normalize_bbox(bbox_absolute, img_width, img_height)

                    # Normalize keypoints to percentages (0.0-1.0)
                    keypoints_absolute = keypoint.xy[0].cpu().numpy().tolist()
                    keypoints_normalized = normalize_keypoints(keypoints_absolute, img_width, img_height)

                    poses.append(
                        {
                            "confidence": confidence,
                            "bbox": bbox_normalized,
                            "keypoints": keypoints_normalized,
                            "keypoints_conf": keypoint.conf[0].cpu().numpy().tolist(),
                        }
                    )

            logger.debug(f"REQ-009: Detected {len(poses)} poses in {image_path}")
            return poses

        except (AttributeError, ModuleNotFoundError) as e:
            error_msg = str(e)
            if "ultralytics.yolo" in error_msg or "'Conv' object has no attribute 'bn'" in error_msg or "'Conv' object has no attribute" in error_msg:
                logger.warning(
                    f"REQ-009: Pose detection failed for {image_path} due to model compatibility issue: {e}. "
                    "This may indicate a version mismatch between the model and ultralytics library. "
                    "Try updating ultralytics: pip install --upgrade ultralytics"
                )
            else:
                logger.warning(f"REQ-009: Pose detection failed for {image_path}: {e}")
            return []
        except Exception as e:
            import traceback
            error_msg = str(e)
            # Check for segmentation fault indicators or compatibility issues
            if "ultralytics.yolo" in error_msg or "Conv" in error_msg:
                logger.warning(
                    f"REQ-009: Pose detection failed for {image_path} due to compatibility issue: {e}. "
                    "This may indicate a version mismatch between the model and ultralytics library."
                )
            else:
                logger.warning(f"REQ-009: Pose detection failed for {image_path}: {e}")
                logger.debug(f"REQ-009: Pose detection traceback: {traceback.format_exc()}")
            return []


def get_pose_detector(device: torch.device, model_path: str = "yolo11x-pose.pt") -> PoseDetector:
    """
    Factory function to get pose detector instance.

    REQ-009: Factory function for pose detection.

    Args:
        device: GPU device.
        model_path: Path to YOLO pose model.

    Returns:
        PoseDetector: Configured pose detector.
    """
    return PoseDetector(device, model_path)
