"""Object database model.

REQ-022: PonyORM model for detected objects.
REQ-023: Object model in separate file.
REQ-024: Object entity linked to Image.
"""

import logging

from pony.orm import Optional, Required, Json

from media_indexer.db.connection import db
from media_indexer.db.image import Image

logger = logging.getLogger(__name__)


class Object(db.Entity):
    """Object detection database model.

    REQ-024: Store object detection results with relationship to Image.
    """

    # REQ-024: Foreign key to Image
    image = Required(Image, index=True)

    # Object detection metadata
    class_id = Required(int, index=True)  # Object class ID
    class_name = Required(str, index=True)  # Object class name
    confidence = Required(float, index=True)
    bbox = Required(Json)  # Bounding box [x1, y1, x2, y2]

    # Timestamp
    detected_at = Optional(float)  # Detection timestamp

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Object(id={self.id}, class_name='{self.class_name}', confidence={self.confidence:.2f})"

    @staticmethod
    def get_by_image(image: Image) -> list["Object"]:
        """Get all objects for an image.

        REQ-024: Query objects by image.

        Args:
            image: Image entity.

        Returns:
            List of Object entities.
        """
        return list(image.objects)

    @staticmethod
    def get_by_class(class_name: str) -> list["Object"]:
        """Get objects by class name.

        REQ-028: Query objects by class name.

        Args:
            class_name: Object class name.

        Returns:
            List of Object entities.
        """
        return list(Object.select(lambda o: o.class_name == class_name))

    @staticmethod
    def get_by_confidence(min_confidence: float) -> list["Object"]:
        """Get objects with minimum confidence.

        REQ-028: Query objects by confidence threshold.

        Args:
            min_confidence: Minimum confidence threshold.

        Returns:
            List of Object entities meeting confidence threshold.
        """
        return list(Object.select(lambda o: o.confidence >= min_confidence))
