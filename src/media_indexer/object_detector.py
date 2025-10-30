"""
Object Detection Module

REQ-008: Object detection using YOLOv12x model.
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
from media_indexer.utils.image_utils import normalize_bbox
from media_indexer.utils.model_utils import download_model_if_needed

logger = logging.getLogger(__name__)

# Model download URLs
YOLO12X_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt"


class ObjectDetector:
    """
    Object detector using YOLOv12x.

    REQ-008: Use YOLOv12x model for object detection.
    """

    def __init__(self, device: torch.device, model_path: str = "yolo12x.pt", cache_dir: Path | None = None) -> None:
        """
        Initialize object detector.

        REQ-008: Initialize YOLOv12x model with centralized cache.

        Args:
            device: GPU device for model execution.
            model_path: Path to YOLO model file.
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
            logger.info(f"REQ-008: Loading YOLOv12x model from {model_path}")
            logger.debug(f"REQ-008: Model cache: {cache.yolo_cache}")
            # Download model if needed
            actual_path = download_model_if_needed(model_path, YOLO12X_URL, "REQ-008")
            # REQ-016: Suppress YOLO model summary output during initialization
            null_stream = StringIO()
            with redirect_stdout(null_stream), redirect_stderr(null_stream):
                self.model: YOLO = YOLO(str(actual_path))
            logger.info("REQ-008: YOLOv12x model loaded successfully")
        except Exception as e:
            error_msg = f"REQ-008: Failed to load YOLOv12x model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def detect_objects(self, image_path: Path) -> list[dict[str, Any]]:
        """
        Detect objects in an image.

        REQ-008: Detect objects using YOLOv12x.
        REQ-040: Support RAW image files via in-memory conversion.

        Args:
            image_path: Path to the image file.

        Returns:
            List of detected objects with bounding boxes and class labels.
        """
        try:
            logger.debug(f"REQ-008: Detecting objects in {image_path}")
            # REQ-040: Convert RAW images to usable format
            source_path = get_raw_image_source(image_path)
            if source_path is None:
                logger.debug(f"REQ-008: Skipping {image_path} - no valid source")
                return []
            # REQ-008: Use YOLOv12x for object detection
            results = self.model(source_path, device=self.device, verbose=False)

            objects: list[dict[str, Any]] = []
            for result in results:
                boxes = result.boxes
                # Get image dimensions for normalization
                img_height, img_width = result.orig_shape
                
                for box in boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name = result.names[class_id]
                    
                    # Normalize bbox to percentages (0.0-1.0)
                    bbox_absolute = box.xyxy[0].cpu().numpy().tolist()
                    bbox_normalized = normalize_bbox(bbox_absolute, img_width, img_height)

                    objects.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": bbox_normalized,
                        }
                    )

            logger.debug(f"REQ-008: Detected {len(objects)} objects in {image_path}")
            return objects

        except Exception as e:
            logger.warning(f"REQ-008: Object detection failed for {image_path}: {e}")
            return []


def get_object_detector(device: torch.device, model_path: str = "yolo12x.pt") -> ObjectDetector:
    """
    Factory function to get object detector instance.

    REQ-008: Factory function for object detection.

    Args:
        device: GPU device.
        model_path: Path to YOLO model.

    Returns:
        ObjectDetector: Configured object detector.
    """
    return ObjectDetector(device, model_path)
