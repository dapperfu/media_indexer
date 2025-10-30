"""
Face Detection Module

REQ-007: Face recognition using insightface, yolov8-face, and yolov11-face.
REQ-010: All code components directly linked to requirements.
"""

import logging
import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import torch

# REQ-016: Suppress ONNX Runtime verbose output before importing insightface
os.environ.setdefault("ORT_LOG_LEVEL", "3")  # 3 = ERROR level (suppress INFO/VERBOSE)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment, misc]

try:
    import insightface
except ImportError:
    insightface = None  # type: ignore[assignment, misc]

from media_indexer.raw_converter import load_image_to_array, get_raw_image_source

logger = logging.getLogger(__name__)

# Model download URLs
YOLOV12N_FACE_URL = "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12n-face.pt"
YOLOV8N_FACE_URL = "https://drive.google.com/uc?export=download&id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"
YOLOV11N_FACE_URL = "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov11n-face.pt"


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
    
    logger.info(f"REQ-007: Downloading {model_path} from {url}")
    try:
        urlretrieve(url, path)
        logger.info(f"REQ-007: Successfully downloaded {path}")
        return path
    except Exception as e:
        logger.error(f"REQ-007: Failed to download {model_path}: {e}")
        raise RuntimeError(f"Failed to download model from {url}: {e}") from e


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
                # Download if needed
                actual_path = download_model_if_needed(model_path, YOLOV8N_FACE_URL)
                self.yolo_model = YOLO(str(actual_path))
                logger.info("REQ-007: YOLOv8 face model loaded successfully")
            except Exception as e:
                logger.warning(f"REQ-007: Failed to load YOLOv8 face model: {e}")

        # REQ-007: Initialize yolov11-face
        if YOLO is not None:
            try:
                logger.info(f"REQ-007: Loading YOLOv11 face model from {model_path_v11}")
                # Download if needed
                actual_path = download_model_if_needed(model_path_v11, YOLOV11N_FACE_URL)
                self.yolo_model_v11 = YOLO(str(actual_path))
                logger.info("REQ-007: YOLOv11 face model loaded successfully")
            except Exception as e:
                logger.warning(f"REQ-007: Failed to load YOLOv11 face model: {e}")

        # REQ-007: Initialize insightface
        if insightface is not None:
            try:
                logger.info("REQ-007: Loading insightface model")
                # REQ-016: Suppress InsightFace verbose output (ONNX Runtime and print statements)
                import warnings
                from contextlib import redirect_stderr, redirect_stdout
                from io import StringIO
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Suppress InsightFace's print statements by redirecting stdout and stderr temporarily
                    null_stream = StringIO()
                    with redirect_stdout(null_stream), redirect_stderr(null_stream):
                        # Use buffalo_l model (best accuracy with embeddings)
                        self.insight_model = insightface.app.FaceAnalysis(name='buffalo_l')
                        # Prepare for GPU execution
                        self.insight_model.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("REQ-007: insightface model loaded successfully")
            except Exception as e:
                logger.warning(f"REQ-007: Failed to load insightface model: {e}")

        if self.yolo_model is None and self.yolo_model_v11 is None and self.insight_model is None:
            raise RuntimeError("REQ-007: Failed to load any face detection model")

    def detect_faces(self, image_path: Path) -> list[dict[str, Any]]:
        """
        Detect faces in an image using multiple models.

        REQ-007: Detect faces using insightface, yolov8-face, and yolov11-face.

        Args:
            image_path: Path to the image file.

        Returns:
            List of face detections with bounding boxes and embeddings.
        """
        faces: list[dict[str, Any]] = []
        logger.debug(f"REQ-007: Detecting faces in {image_path}")

        # Load image - using universal loader for RAW file support
        try:
            # Use universal loader that handles RAW files
            rgb_array = load_image_to_array(image_path)
            if rgb_array is None:
                logger.debug(f"REQ-007: Could not load image {image_path}")
                return faces
            
            # For insightface, we need the numpy array directly
            image_rgb = rgb_array

        except Exception as e:
            logger.debug(f"REQ-007: Error loading image {image_path}: {e}")
            return faces

        # REQ-007: Detect with YOLOv8
        yolo_v8_results: list[dict[str, Any]] = []
        if self.yolo_model is not None:
            try:
                # Use universal image source that handles RAW files
                source_path = get_raw_image_source(image_path)
                if source_path is None:
                    logger.debug(f"REQ-007: YOLOv8 - Skipping {image_path} - no valid source")
                else:
                    yolo_detections = self.yolo_model(source_path, device=self.device, verbose=False)
                    for detection in yolo_detections:
                        boxes = detection.boxes
                        # Get image dimensions for normalization
                        img_height, img_width = detection.orig_shape
                        
                        for box in boxes:
                            # Normalize bbox to percentages (0.0-1.0)
                            bbox_absolute = box.xyxy[0].cpu().numpy().tolist()
                            bbox_normalized = [
                                bbox_absolute[0] / img_width,   # x1
                                bbox_absolute[1] / img_height,  # y1
                                bbox_absolute[2] / img_width,  # x2
                                bbox_absolute[3] / img_height, # y2
                            ]
                            
                            yolo_v8_results.append(
                                {
                                    "confidence": float(box.conf.item()),
                                    "bbox": bbox_normalized,
                                    "model": "yolov8-face",
                                }
                            )
            except Exception as e:
                logger.warning(f"REQ-007: YOLOv8 detection failed: {e}")

        # REQ-007: Detect with YOLOv11
        yolo_v11_results: list[dict[str, Any]] = []
        if self.yolo_model_v11 is not None:
            try:
                # Use universal image source that handles RAW files
                source_path = get_raw_image_source(image_path)
                if source_path is None:
                    logger.debug(f"REQ-007: YOLOv11 - Skipping {image_path} - no valid source")
                else:
                    yolo_detections = self.yolo_model_v11(source_path, device=self.device, verbose=False)
                    for detection in yolo_detections:
                        boxes = detection.boxes
                        # Get image dimensions for normalization
                        img_height, img_width = detection.orig_shape
                        
                        for box in boxes:
                            # Normalize bbox to percentages (0.0-1.0)
                            bbox_absolute = box.xyxy[0].cpu().numpy().tolist()
                            bbox_normalized = [
                                bbox_absolute[0] / img_width,   # x1
                                bbox_absolute[1] / img_height,  # y1
                                bbox_absolute[2] / img_width,  # x2
                                bbox_absolute[3] / img_height, # y2
                            ]
                            
                            yolo_v11_results.append(
                                {
                                    "confidence": float(box.conf.item()),
                                    "bbox": bbox_normalized,
                                    "model": "yolov11-face",
                                }
                            )
            except Exception as e:
                logger.warning(f"REQ-007: YOLOv11 detection failed: {e}")

        # REQ-007: Detect with insightface
        insight_results: list[dict[str, Any]] = []
        if self.insight_model is not None:
            try:
                # Load image for insightface
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.debug(f"REQ-007: Could not load image {image_path} for insightface")
                else:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    insight_faces = self.insight_model.get(image_rgb)
                    
                    # Get image dimensions for normalization
                    img_height, img_width = image_rgb.shape[:2]
                    
                    for face in insight_faces:
                        # Normalize bbox to percentages (0.0-1.0)
                        # insightface bbox format is [x1, y1, x2, y2]
                        bbox_absolute = face.bbox.tolist()
                        bbox_normalized = [
                            bbox_absolute[0] / img_width,   # x1
                            bbox_absolute[1] / img_height,  # y1
                            bbox_absolute[2] / img_width,  # x2
                            bbox_absolute[3] / img_height, # y2
                        ]
                        
                        insight_results.append(
                            {
                                "confidence": float(face.det_score),
                                "bbox": bbox_normalized,
                                "embedding": face.embedding.tolist() if hasattr(face, 'embedding') else None,
                                "model": "insightface",
                            }
                        )
            except Exception as e:
                import traceback
                logger.warning(f"REQ-007: insightface detection failed: {e}")
                logger.debug(f"REQ-007: insightface traceback: {traceback.format_exc()}")

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
