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
    
    # REQ-081: Verify tf-keras is available (required by DeepFace with TensorFlow 2.20+)
    try:
        import tf_keras  # noqa: F401
    except ImportError:
        import warnings
        warnings.warn(
            "REQ-081: tf-keras package is required for DeepFace with TensorFlow 2.20+. "
            "Please install tf-keras: pip install tf-keras",
            UserWarning,
        )

    _DEEPFACE_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    DeepFace = None  # type: ignore[assignment]
    _DEEPFACE_AVAILABLE = False


class AttributeSource(Enum):
    """Identify the origin of generated attributes."""

    DEEPFACE = auto()
    INSIGHTFACE = auto()
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

    def __init__(self, face_detector: Any | None = None) -> None:
        """Construct the analyzer.

        Parameters
        ----------
        face_detector : Any | None
            Optional face detector instance for InsightFace fallback.
            If provided and DeepFace unavailable, will use InsightFace attributes.

        Raises
        ------
        RuntimeError
            If DeepFace is unavailable or not working. DeepFace is required for REQ-081.
        """
        self.face_detector = face_detector
        self._deepface_available = _DEEPFACE_AVAILABLE

        if not self._deepface_available:
            error_msg = (
                "REQ-081: DeepFace is unavailable or not working. "
                "DeepFace is required for face attribute analysis. "
                "Please ensure deepface and tf-keras packages are installed."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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

            result: FaceAttributeResult | None = None

            # REQ-081: DeepFace is required - always try it first
            if self._deepface_available:
                try:
                    result = self._run_deepface(prepared)
                except Exception as exc:  # noqa: BLE001
                    # REQ-081: If DeepFace fails, log error but allow InsightFace fallback
                    logger.warning("REQ-081: DeepFace analysis failed: %s", exc)
                    result = None
            else:
                # This should not happen if initialization succeeded, but handle gracefully
                logger.error("REQ-081: DeepFace unavailable during analysis (should have been caught at init)")
                result = None

            # REQ-081: Use InsightFace attributes if available (in addition to DeepFace)
            # InsightFace provides age and sex, which can supplement DeepFace's age and emotion
            if detection.get("model") == "insightface":
                insightface_result = self._extract_insightface_attributes(detection)
                if insightface_result is not None:
                    # If DeepFace result exists, prefer DeepFace for emotion but use InsightFace age/sex if better
                    # If DeepFace failed, use InsightFace result
                    if result is None:
                        result = insightface_result
                    else:
                        # Merge: prefer InsightFace age/sex if available, keep DeepFace emotion
                        if insightface_result.age is not None:
                            result.age = insightface_result.age
                        # Note: sex is not in DeepFace result structure, so we add it to attributes dict later
                        logger.debug("REQ-081: Merged InsightFace and DeepFace attributes")

            # Apply result or attach error
            if result is not None:
                enriched_faces.append(self._apply_attributes(detection, result))
            else:
                enriched_faces.append(self._attach_error(detection, "attribute_extraction_failed"))

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
        
        # REQ-081: Include sex from InsightFace if available
        sex = None
        insightface_attrs = detection.get("insightface_attributes")
        if insightface_attrs and isinstance(insightface_attrs, dict):
            sex = insightface_attrs.get("sex")
        
        updated["attributes"] = {
            "age": result.age,
            "age_confidence": result.age_confidence,
            "sex": sex,  # REQ-081: InsightFace provides sex (male/female)
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

    def _extract_insightface_attributes(self, detection: dict[str, Any]) -> FaceAttributeResult | None:
        """Extract age, sex, and emotion from InsightFace detection if available.

        REQ-081: Extract InsightFace attributes (age, sex) in addition to DeepFace.
        Note: InsightFace buffalo_l model provides age and gender, but not emotion.
        Emotion requires a separate model or DeepFace.

        Parameters
        ----------
        detection : dict[str, Any]
            Face detection dictionary from InsightFace model, containing
            ``insightface_attributes`` key with age and sex.

        Returns
        -------
        FaceAttributeResult | None
            Attribute result with age/sex if available, None otherwise.
        """
        insightface_attrs = detection.get("insightface_attributes")
        if not insightface_attrs or not isinstance(insightface_attrs, dict):
            return None

        age = self._safe_float(insightface_attrs.get("age"))
        sex = insightface_attrs.get("sex")  # "male" or "female"
        
        # InsightFace doesn't provide emotion directly from buffalo_l model
        # We set emotion scores to empty dict since InsightFace doesn't have emotion
        emotion_scores: dict[str, float] = {}

        return FaceAttributeResult(
            age=age,
            age_confidence=None,  # InsightFace doesn't provide confidence
            primary_emotion=None,  # InsightFace doesn't provide emotion
            emotion_confidence=None,
            emotion_scores=emotion_scores,
            source=AttributeSource.INSIGHTFACE,
        )


def get_face_attribute_analyzer(face_detector: Any | None = None) -> FaceAttributeAnalyzer:
    """Factory function returning a configured analyzer instance.

    Parameters
    ----------
    face_detector : Any | None
        Optional face detector for InsightFace fallback support.

    Returns
    -------
    FaceAttributeAnalyzer
        Configured analyzer instance.
    """
    return FaceAttributeAnalyzer(face_detector=face_detector)


