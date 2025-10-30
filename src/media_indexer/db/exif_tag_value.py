"""EXIF tag value database model.

REQ-022: PonyORM model for EXIF tag values.
REQ-023: EXIF tag value model in separate file.
REQ-024: Relational EXIF value storage.
"""

import logging
from typing import TYPE_CHECKING, Any

from pony.orm import Json, Optional, Required

from media_indexer.db.connection import db
from media_indexer.db.exif_tag import EXIFTag
from media_indexer.db.image import Image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EXIFTagValue(db.Entity):
    """EXIF tag value database model.

    REQ-024: Store EXIF tag values in relational format.
    Each value links an image to a tag with its actual value.
    """

    # REQ-024: Foreign keys
    image = Required(Image, index=True)
    tag = Required(EXIFTag, index=True)

    # Value storage - multiple formats for efficient querying
    value_text = Optional(str, index=True)  # Text representation (always populated)
    value_numeric = Optional(float, index=True)  # Numeric value if applicable (for sorting/range queries)
    value_json = Optional(Json)  # Complex values (arrays, objects, rationals)

    # Timestamp
    extracted_at = Optional(float)  # Extraction timestamp

    def __repr__(self) -> str:
        """Return string representation."""
        return f"EXIFTagValue(id={self.id}, image_id={self.image.id}, tag='{self.tag.name}', value='{self.value_text}')"

    @staticmethod
    def get_by_image(image: Image) -> list["EXIFTagValue"]:
        """Get all EXIF tag values for an image.

        REQ-024: Query EXIF values by image.

        Parameters
        ----------
        image : Image
            Image entity.

        Returns
        -------
        list[EXIFTagValue]
            List of EXIF tag value entities.
        """
        return list(EXIFTagValue.select(lambda v: v.image == image))

    @staticmethod
    def get_by_tag(tag: EXIFTag) -> list["EXIFTagValue"]:
        """Get all EXIF tag values for a tag.

        REQ-024: Query EXIF values by tag.

        Parameters
        ----------
        tag : EXIFTag
            EXIF tag entity.

        Returns
        -------
        list[EXIFTagValue]
            List of EXIF tag value entities.
        """
        return list(EXIFTagValue.select(lambda v: v.tag == tag))

    @staticmethod
    def get_by_tag_name(tag_name: str) -> list["EXIFTagValue"]:
        """Get all EXIF tag values by tag name.

        REQ-024: Query EXIF values by tag name.

        Parameters
        ----------
        tag_name : str
            Tag name (e.g., "ImageWidth").

        Returns
        -------
        list[EXIFTagValue]
            List of EXIF tag value entities.
        """
        tag = EXIFTag.get_by_name(tag_name)
        if tag is None:
            return []
        return EXIFTagValue.get_by_tag(tag)

    @staticmethod
    def get_by_tag_value(tag_name: str, tag_value: Any) -> list["EXIFTagValue"]:
        """Get EXIF tag values by tag name and value.

        REQ-028: Query EXIF values by tag name and value.

        Parameters
        ----------
        tag_name : str
            Tag name (e.g., "ImageWidth").
        tag_value : Any
            Tag value to match.

        Returns
        -------
        list[EXIFTagValue]
            List of EXIF tag value entities matching the value.
        """
        tag = EXIFTag.get_by_name(tag_name)
        if tag is None:
            return []

        # Try numeric match first if value is numeric
        if isinstance(tag_value, (int, float)):
            return list(
                EXIFTagValue.select(
                    lambda v: v.tag == tag and v.value_numeric == float(tag_value)
                )
            )

        # Text match
        value_str = str(tag_value)
        return list(
            EXIFTagValue.select(lambda v: v.tag == tag and v.value_text == value_str)
        )

