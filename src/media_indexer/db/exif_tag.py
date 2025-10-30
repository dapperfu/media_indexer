"""EXIF tag database model.

REQ-022: PonyORM model for EXIF tag definitions.
REQ-023: EXIF tag model in separate file.
REQ-024: Relational EXIF tag storage.
"""

import logging
from typing import TYPE_CHECKING

from pony.orm import Optional, Required, Set

from media_indexer.db.connection import db

if TYPE_CHECKING:
    from media_indexer.db.exif_tag_value import EXIFTagValue

logger = logging.getLogger(__name__)


class EXIFTag(db.Entity):
    """EXIF tag definition database model.

    REQ-024: Store EXIF tag definitions in normalized relational format.
    Each tag represents a known EXIF tag name with metadata.
    """

    # Tag identifier
    name = Required(str, unique=True, index=True)  # e.g., "ImageWidth", "DateTimeOriginal"

    # Tag metadata
    group = Optional(str, index=True)  # e.g., "IFD0", "EXIF", "GPS", "Interop"
    description = Optional(str)  # Human-readable description
    tag_type = Optional(str)  # e.g., "string", "numeric", "date", "rational"

    # REQ-024: One-to-many relationship to tag values
    values = Set("EXIFTagValue", cascade_delete=True)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"EXIFTag(id={self.id}, name='{self.name}', group='{self.group}')"

    @staticmethod
    def get_or_create(name: str, group: str | None = None, description: str | None = None) -> "EXIFTag":
        """Get existing tag or create new one.

        REQ-024: Get or create EXIF tag definition.

        Parameters
        ----------
        name : str
            Tag name (e.g., "ImageWidth").
        group : str, optional
            Tag group (e.g., "IFD0", "EXIF").
        description : str, optional
            Human-readable description.

        Returns
        -------
        EXIFTag
            EXIF tag entity.
        """
        tag = EXIFTag.get(name=name)
        if tag is None:
            tag = EXIFTag(name=name, group=group, description=description)
        return tag

    @staticmethod
    def get_by_name(name: str) -> "EXIFTag | None":
        """Get tag by name.

        REQ-024: Query EXIF tag by name.

        Parameters
        ----------
        name : str
            Tag name.

        Returns
        -------
        EXIFTag | None
            EXIF tag entity if found, None otherwise.
        """
        return EXIFTag.get(name=name)

    @staticmethod
    def get_by_group(group: str) -> list["EXIFTag"]:
        """Get all tags in a group.

        REQ-024: Query EXIF tags by group.

        Parameters
        ----------
        group : str
            Tag group (e.g., "IFD0", "EXIF", "GPS").

        Returns
        -------
        list[EXIFTag]
            List of EXIF tag entities.
        """
        return list(EXIFTag.select(lambda t: t.group == group))

