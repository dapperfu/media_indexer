"""Image database model.

REQ-022: PonyORM model for images.
REQ-023: Image model in separate file.
REQ-024: Image entity with relationships to faces, objects, poses, EXIF.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from pony.orm import Optional, Required, Set

from media_indexer.db.connection import db

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Image(db.Entity):
    """Image database model.

    REQ-024: Store image metadata with relationships.
    """

    # Image file information
    path = Required(str, unique=True, index=True)
    file_hash = Required(str, index=True)  # Content hash for deduplication
    file_size = Required(int)  # File size in bytes
    width = Optional(int)
    height = Optional(int)

    # Timestamps
    created_at = Required(datetime, default=lambda: datetime.now())
    updated_at = Required(datetime, default=lambda: datetime.now())

    # REQ-024: One-to-many relationships
    faces = Set("Face", cascade_delete=True)
    objects = Set("Object", cascade_delete=True)
    poses = Set("Pose", cascade_delete=True)

    # REQ-024: One-to-one relationships
    exif_data = Optional("EXIFData", cascade_delete=True)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Image(id={self.id}, path='{self.path}')"

    @staticmethod
    def get_by_path(path: str):
        """Get image by file path.

        REQ-024: Query image by path.

        Args:
            path: Image file path.

        Returns:
            Image entity if found, None otherwise.
        """
        return Image.get(path=path)

    @staticmethod
    def get_all_images() -> list["Image"]:
        """Get all images.

        REQ-024: Query all images.

        Returns:
            List of all Image entities.
        """
        return list(Image.select())
