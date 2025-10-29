"""EXIF data database model.

REQ-022: PonyORM model for EXIF data.
REQ-023: EXIF model in separate file.
REQ-024: EXIF entity linked to Image.
"""

import logging
from typing import Any, Optional

from pony.orm import Required
from pony.orm import Optional as PonyOptional
from pony.orm import db_json

from media_indexer.db.connection import db
from media_indexer.db.image import Image

logger = logging.getLogger(__name__)


class EXIFData(db.Entity):
    """EXIF data database model.

    REQ-024: Store EXIF metadata with relationship to Image.
    """

    # Primary key
    id: int = Required(int, auto=True)

    # REQ-024: Foreign key to Image (one-to-one)
    image: Image = Required(Image, unique=True, index=True)

    # EXIF data as JSON blob
    data: dict[str, Any] = Required(db_json)

    # Timestamp
    extracted_at: PonyOptional[float] = None  # Extraction timestamp

    def __repr__(self) -> str:
        """Return string representation."""
        return f"EXIFData(id={self.id}, image_id={self.image.id})"

    @staticmethod
    def get_by_image(image: Image) -> Optional["EXIFData"]:
        """Get EXIF data for an image.

        REQ-024: Query EXIF data by image.

        Args:
            image: Image entity.

        Returns:
            EXIFData entity if found, None otherwise.
        """
        return image.exif_data

    @staticmethod
    def get_by_tag(tag_name: str, tag_value: Any) -> list["EXIFData"]:
        """Get EXIF data by tag value.

        REQ-028: Query EXIF data by specific tag.

        Args:
            tag_name: EXIF tag name.
            tag_value: EXIF tag value.

        Returns:
            List of EXIFData entities.
        """
        # This is a simplified implementation
        # In practice, you might want to use JSON queries depending on the database
        return list(EXIFData.select(lambda e: tag_name in e.data and e.data[tag_name] == tag_value))
