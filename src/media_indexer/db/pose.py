"""Pose database model.

REQ-022: PonyORM model for detected poses.
REQ-023: Pose model in separate file.
REQ-024: Pose entity linked to Image.
"""

import logging

from pony.orm import Json, Optional, Required

from media_indexer.db.connection import db
from media_indexer.db.image import Image

logger = logging.getLogger(__name__)


class Pose(db.Entity):
    """Human pose detection database model.

    REQ-024: Store pose detection results with relationship to Image.
    """

    # REQ-024: Foreign key to Image
    image = Required(Image, index=True)

    # Pose detection metadata
    confidence = Required(float, index=True)
    bbox = Required(Json)  # Bounding box [x1, y1, x2, y2] normalized (0.0-1.0)
    keypoints = Required(Json)  # Keypoint coordinates [[x, y], ...] normalized (0.0-1.0)
    keypoints_conf = Optional(Json)  # Keypoint confidence scores

    # Timestamp
    detected_at = Optional(float)  # Detection timestamp

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Pose(id={self.id}, image_id={self.image.id}, confidence={self.confidence:.2f})"

    @staticmethod
    def get_by_image(image: Image) -> list["Pose"]:
        """Get all poses for an image.

        REQ-024: Query poses by image.

        Args:
            image: Image entity.

        Returns:
            List of Pose entities.
        """
        return list(image.poses)

    @staticmethod
    def get_by_confidence(min_confidence: float) -> list["Pose"]:
        """Get poses with minimum confidence.

        REQ-028: Query poses by confidence threshold.

        Args:
            min_confidence: Minimum confidence threshold.

        Returns:
            List of Pose entities meeting confidence threshold.
        """
        return list(Pose.select(lambda p: p.confidence >= min_confidence))
