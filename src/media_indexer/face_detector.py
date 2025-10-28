"""
Face Detection Module

REQ-007: Face recognition using insightface, yolov8-face, and yolov11-face.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment, misc]

try:
    import insightface
except ImportError:
    insightface = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using insightface, yolov8-face, and yolov11-face.

    REQ-007: Use insightface library, yolov8-face model, and yolov11-face model.
    """

    def __init__(
        self,
        device: torch.device,
        model_path: str = "yolov8n-face.pt",
        model_path_v11: str = "yolov11n-face.pt",
    ) -> None:
        """
        Initialize face detector with multiple models.

        REQ-007: Initialize insightface, yolov8-face, and yolov11-face.

        Args:
            device: GPU device for model execution.
            model_path: Path to YOLOv8 face model.
            model_path_v11: Path to YOLOv11 face model.

        Raises:
            RuntimeError: If models cannot be loaded.
        """
        self.device: torch.device = device
        self.yolo_model: Any | None = None
        self.yolo_model_v11: Any | None = None
        self.insight_model: Any | None = None

        # REQ-007: Initialize yolov8-face
        if YOLO is not None:
            try:
                logger.info(f"REQ-007: Loading YOLOv8 face model from {model_path}")
                self.yolo_model = YOLO(model_path)
                logger.info("REQ-007: YOLOv8 face model loaded successfully")
            except Exception as e:
                logger.warning(f"REQ-007: Failed to load YOLOv8 face model: {e}")

        # REQ-007: Initialize yolov11-face
        if YOLO is not None:
            try:
                logger.info(f"REQ-007: Loading YOLOv11 face model from {model_path_v11}")
                self.yolo_model_v11 = YOLO(model_path_v11)
                logger.info("REQ-007: YOLOv11 face model loaded successfully")
            except Exception as e:
                logger.warning(f"REQ-007: Failed to load YOLOv11 face model: {e}")

        # REQ-007: Initialize insightface
        if insightface is not None:
            try:
                logger.info("REQ-007: Loading insightface model")
                self.insight_model = insightface.app.FaceAnalysis(
                    providers=["CUDAExecutionProvider"]
                )
                logger.info("REQ-007: insightface model loaded successfully")
            except Exception as e:
                logger.warning(f"REQ-007: Failed to load insightface model: {e}")

        if self.yolo_model is None and self.yolo_model_v11 is None and self.insight_model is None:
            raise RuntimeError("REQ-007: Failed to load any face detection model")

    def detect_faces(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Detect faces in an image using multiple models.

        REQ-007: Detect faces using insightface, yolov8-face, and yolov11-face.

        Args:
            image_path: Path to the image file.

        Returns:
            List of face detections with bounding boxes and embeddings.
        """
        faces: List[Dict[str, Any]] = []
        logger.debug(f"REQ-007: Detecting faces in {image_path}")

        # Load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"REQ-007: Could not load image {image_path}")
                return faces

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except Exception as e:
            logger.warning(f"REQ-007: Error loading image {image_path}: {e}")
            return faces

        # REQ-007: Detect with YOLOv8
        yolo_v8_results: List[Dict[str, Any]] = []
        if self.yolo_model is not None:
            try:
                yolo_detections = self.yolo_model(str(image_path), device=self.device)
                for detection in yolo_detections:
                    boxes = detection.boxes
                    for box in boxes:
                        yolo_v8_results.append(
                            {
                                "confidence": float(box.conf.item()),
                                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                                "model": "yolov8-face",
                            }
                        )
            except Exception as e:
                logger.warning(f"REQ-007: YOLOv8 detection failed: {e}")

        # REQ-007: Detect with YOLOv11
        yolo_v11_results: List[Dict[str, Any]] = []
        if self.yolo_model_v11 is not None:
            try:
                yolo_detections = self.yolo_model_v11(str(image_path), device=self.device)
                for detection in yolo_detections:
                    boxes = detection.boxes
                    for box in boxes:
                        yolo_v11_results.append(
                            {
                                "confidence": float(box.conf.item()),
                                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                                "model": "yolov11-face",
                            }
                        )
            except Exception as e:
                logger.warning(f"REQ-007: YOLOv11 detection failed: {e}")

        # REQ-007: Detect with insightface
        insight_results: List[Dict[str, Any]] = []
        if self.insight_model is not None:
            try:
                insight_faces = self.insight_model.get(image_rgb)
                for face in insight_faces:
                    insight_results.append(
                        {
                            "confidence": float(face.det_score),
                            "bbox": face.bbox.tolist(),
                            "embedding": face.embedding.tolist(),
                            "model": "insightface",
                        }
                    )
            except Exception as e:
                logger.warning(f"REQ-007: insightface detection failed: {e}")

        # Combine results
        faces.extend(yolo_v8_results)
        faces.extend(yolo_v11_results)
        faces.extend(insight_results)

        logger.debug(f"REQ-007: Detected {len(faces)} faces in {image_path}")
        return faces


def get_face_detector(
    device: torch.device,
    model_path: str = "yolov8n-face.pt",
    model_path_v11: str = "yolov11n-face.pt",
) -> FaceDetector:
    """
    Factory function to get face detector instance.

    REQ-007: Factory function for face detection.

    Args:
        device: GPU device.
        model_path: Path to YOLOv8 face model.
        model_path_v11: Path to YOLOv11 face model.

    Returns:
        FaceDetector: Configured face detector.
    """
    return FaceDetector(device, model_path, model_path_v11)


