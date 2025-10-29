"""
Human Pose Detection Module

REQ-009: Human pose detection using YOLOv11-pose model.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import torch
from ultralytics import YOLO

from media_indexer.raw_converter import get_raw_image_source

logger = logging.getLogger(__name__)

# Model download URLs
YOLO11X_POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt"


def download_model_if_needed(model_path: str, url: str) -> Path:
    """
    Download model file if it doesn't exist.

    Args:
        model_path: Local path to model file.
        url: URL to download from.

    Returns:
        Path to model file.
    """
    path = Path(model_path)
    
    if path.exists():
        return path
    
    logger.info(f"REQ-009: Downloading {model_path} from {url}")
    try:
        urlretrieve(url, path)
        logger.info(f"REQ-009: Successfully downloaded {path}")
        return path
    except Exception as e:
        logger.error(f"REQ-009: Failed to download {model_path}: {e}")
        raise RuntimeError(f"Failed to download model from {url}: {e}") from e


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
            actual_path = download_model_if_needed(model_path, YOLO11X_POSE_URL)
            self.model: YOLO = YOLO(str(actual_path))
            logger.info("REQ-009: YOLOv11-pose model loaded successfully")
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
            # REQ-009: Use YOLOv11-pose for pose detection
            results = self.model(source_path, device=self.device)

            poses: list[dict[str, Any]] = []
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints

                for box, keypoint in zip(boxes, keypoints, strict=True):
                    confidence = float(box.conf.item())

                    poses.append(
                        {
                            "confidence": confidence,
                            "bbox": box.xyxy[0].cpu().numpy().tolist(),
                            "keypoints": keypoint.xy[0].cpu().numpy().tolist(),
                            "keypoints_conf": keypoint.conf[0].cpu().numpy().tolist(),
                        }
                    )

            logger.debug(f"REQ-009: Detected {len(poses)} poses in {image_path}")
            return poses

        except Exception as e:
            logger.warning(f"REQ-009: Pose detection failed for {image_path}: {e}")
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
