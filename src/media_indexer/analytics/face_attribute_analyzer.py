"""Face attribute analysis utilities.

REQ-081: Enrich detected faces with age and emotion metadata.
REQ-010: All code components directly linked to requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import logging
from pathlib import Path
from typing import Any

import numpy as np

from media_indexer.raw_converter import load_image_to_array

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    from deepface import DeepFace

    _DEEPFACE_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    DeepFace = None  # type: ignore[assignment]
    _DEEPFACE_AVAILABLE = False


class AttributeSource(Enum):
    """Identify the origin of generated attributes."""

    DEEPFACE = auto()
    FALLBACK = auto()


@dataclass(slots=True)
class FaceAttributeResult:
    """Container for extracted face attributes.

    Parameters
    ----------
    age : float | None
        Estimated age in years.
    age_confidence : float | None
        Confidence score for the age estimate on a 0.0-1.0 scale.
    primary_emotion : str | None
        Dominant emotion label, e.g., ``"happy"``.
    emotion_confidence : float | None
        Confidence score for the dominant emotion on a 0.0-1.0 scale.
    emotion_scores : dict[str, float]
        Per-emotion probability distribution.
    source : AttributeSource
        Origin of the attribute estimation.
    error : str | None
        Optional error message describing why estimation failed.
    """

    age: float | None
    age_confidence: float | None
    primary_emotion: str | None
    emotion_confidence: float | None
    emotion_scores: dict[str, float]
    source: AttributeSource
    error: str | None = None


class FaceAttributeAnalyzer:
    """Analyze detected faces for age and emotion metadata (REQ-081)."""

    def __init__(self) -> None:
        """Construct the analyzer and verify DeepFace availability."""

        if not _DEEPFACE_AVAILABLE:
            msg = "REQ-081: DeepFace dependency is missing; install deepface to enable face attributes."
            raise RuntimeError(msg)

    def annotate_faces(self, image_path: Path, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return enriched face metadata for the provided detections.

        Parameters
        ----------
        image_path : Path
            Path to the source image.
        detections : list[dict[str, Any]]
            Face detections containing at minimum a ``bbox`` entry with
            normalized coordinates ``[x1, y1, x2, y2]``.

        Returns
        -------
        list[dict[str, Any]]
            New list of face metadata dictionaries with an ``attributes``
            entry appended for each detection.
        """

        if not detections:
            return []

        image_array = load_image_to_array(image_path)
        if image_array is None:
            logger.warning("REQ-081: Unable to load image %s for attribute analysis", image_path)
            return [self._attach_error(det, "image_load_failed") for det in detections]

        height, width = image_array.shape[0], image_array.shape[1]
        enriched_faces: list[dict[str, Any]] = []

        for detection in detections:
            bbox = detection.get("bbox")
            if not bbox:
                enriched_faces.append(self._attach_error(detection, "missing_bbox"))
                continue

            crop = self._crop_face(image_array, bbox, width, height)
            if crop is None:
                enriched_faces.append(self._attach_error(detection, "empty_crop"))
                continue

            prepared = self._prepare_face_crop(crop)
            if prepared is None:
                enriched_faces.append(self._attach_error(detection, "crop_prep_failed"))
                continue

            try:
                result = self._run_deepface(prepared)
                enriched_faces.append(self._apply_attributes(detection, result))
            except Exception as exc:  # noqa: BLE001
                logger.debug("REQ-081: DeepFace analysis failed: %s", exc)
                enriched_faces.append(self._attach_error(detection, "deepface_error"))

        return enriched_faces

    def _run_deepface(self, face_crop: np.ndarray) -> FaceAttributeResult:
        """Execute DeepFace analysis on a cropped face image."""

        import cv2  # Local import avoids unnecessary cv2 load for callers.

        bgr_face = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        analysis = DeepFace.analyze(  # type: ignore[misc]
            img_path=bgr_face,
            actions=["age", "emotion"],
            enforce_detection=False,
            detector_backend="skip",
            prog_bar=False,
        )

        payload = analysis[0] if isinstance(analysis, list) else analysis
        age = self._safe_float(payload.get("age"))
        emotion_scores = self._extract_emotion_scores(payload.get("emotion"))
        primary_emotion, emotion_confidence = self._max_emotion(emotion_scores)

        return FaceAttributeResult(
            age=age,
            age_confidence=None,
            primary_emotion=primary_emotion,
            emotion_confidence=emotion_confidence,
            emotion_scores=emotion_scores,
            source=AttributeSource.DEEPFACE,
        )

    @staticmethod
    def _prepare_face_crop(crop: np.ndarray) -> np.ndarray | None:
        """Resize the cropped face to DeepFace's expected dimensions."""

        if crop.ndim != 3 or crop.shape[2] != 3:
            return None

        import cv2

        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized)

    @staticmethod
    def _apply_attributes(detection: dict[str, Any], result: FaceAttributeResult) -> dict[str, Any]:
        """Attach attribute data to a detection dictionary."""

        updated = detection.copy()
        updated["attributes"] = {
            "age": result.age,
            "age_confidence": result.age_confidence,
            "primary_emotion": result.primary_emotion,
            "emotion_confidence": result.emotion_confidence,
            "emotion_scores": result.emotion_scores,
            "source": result.source.name.lower(),
            "error": result.error,
        }
        return updated

    @staticmethod
    def _attach_error(detection: dict[str, Any], error: str) -> dict[str, Any]:
        """Attach an error attribute payload to the detection."""

        return FaceAttributeAnalyzer._apply_attributes(
            detection,
            FaceAttributeResult(
                age=None,
                age_confidence=None,
                primary_emotion=None,
                emotion_confidence=None,
                emotion_scores={},
                source=AttributeSource.FALLBACK,
                error=error,
            ),
        )

    @staticmethod
    def _crop_face(
        image_array: np.ndarray,
        bbox: list[float],
        width: int,
        height: int,
    ) -> np.ndarray | None:
        """Crop a face region given a normalized bounding box."""

        x1 = max(int(bbox[0] * width), 0)
        y1 = max(int(bbox[1] * height), 0)
        x2 = min(int(bbox[2] * width), width)
        y2 = min(int(bbox[3] * height), height)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image_array[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert a value to float if possible."""

        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return None

    @staticmethod
    def _extract_emotion_scores(emotion_payload: Any) -> dict[str, float]:
        """Normalize DeepFace emotion output into a probability distribution."""

        if not isinstance(emotion_payload, dict):
            return {}

        scores: dict[str, float] = {}
        total = 0.0
        for key, value in emotion_payload.items():
            try:
                prob = float(value)
            except (TypeError, ValueError):
                continue
            scores[key.lower()] = prob
            total += prob

        if total > 0:
            scores = {label: prob / total for label, prob in scores.items()}

        return scores

    @staticmethod
    def _max_emotion(scores: dict[str, float]) -> tuple[str | None, float | None]:
        """Determine the dominant emotion from score distribution."""

        if not scores:
            return None, None

        label = max(scores, key=scores.get)
        return label, scores[label]


def get_face_attribute_analyzer() -> FaceAttributeAnalyzer:
    """Factory function returning a configured analyzer instance."""

    return FaceAttributeAnalyzer()


