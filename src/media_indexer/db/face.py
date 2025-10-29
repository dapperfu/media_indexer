"""Face database model.

REQ-022: PonyORM model for detected faces.
REQ-023: Face model in separate file.
REQ-024: Face entity linked to Image.
"""

import logging

from pony.orm import Required
from pony.orm import Optional as PonyOptional
from pony.orm import db_json

from media_indexer.db.connection import db
from media_indexer.db.image import Image

logger = logging.getLogger(__name__)


class Face(db.Entity):
    """Face detection database model.

    REQ-024: Store face detection results with relationship to Image.
    """

    # Primary key
    id: int = Required(int, auto=True)

    # REQ-024: Foreign key to Image
    image: Image = Required(Image, index=True)

    # Face detection metadata
    confidence: float = Required(float, index=True)
    bbox: list[float] = Required(db_json)  # Bounding box [x1, y1, x2, y2]
    embedding: PonyOptional[list[float]] = None  # Face embedding vector
    model: str = Required(str, index=True)  # Detection model name

    # Timestamp
    detected_at: PonyOptional[float] = None  # Detection timestamp

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Face(id={self.id}, image_id={self.image.id}, confidence={self.confidence:.2f})"

    @staticmethod
    def get_by_image(image: Image) -> list["Face"]:
        """Get all faces for an image.

        REQ-024: Query faces by image.

        Args:
            image: Image entity.

        Returns:
            List of Face entities.
        """
        return list(image.faces)

    @staticmethod
    def get_by_confidence(min_confidence: float) -> list["Face"]:
        """Get faces with minimum confidence.

        REQ-028: Query faces by confidence threshold.

        Args:
            min_confidence: Minimum confidence threshold.

        Returns:
            List of Face entities meeting confidence threshold.
        """
        return list(Face.select(lambda f: f.confidence >= min_confidence))

    @staticmethod
    def get_by_model(model_name: str) -> list["Face"]:
        """Get faces detected by specific model.

        REQ-028: Query faces by detection model.

        Args:
            model_name: Detection model name.

        Returns:
            List of Face entities.
        """
        return list(Face.select(lambda f: f.model == model_name))
