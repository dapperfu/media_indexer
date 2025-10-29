"""
Human Pose Detection Module

REQ-009: Human pose detection using YOLOv11-pose model.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


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
            self.model: YOLO = YOLO(model_path)
            logger.info("REQ-009: YOLOv11-pose model loaded successfully")
        except Exception as e:
            error_msg = f"REQ-009: Failed to load YOLOv11-pose model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def detect_poses(self, image_path: Path) -> list[dict[str, Any]]:
        """
        Detect human poses in an image.

        REQ-009: Detect human poses using YOLOv11-pose.

        Args:
            image_path: Path to the image file.

        Returns:
            List of detected poses with keypoints.
        """
        try:
            logger.debug(f"REQ-009: Detecting poses in {image_path}")
            # REQ-009: Use YOLOv11-pose for pose detection
            results = self.model(str(image_path), device=self.device)

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
