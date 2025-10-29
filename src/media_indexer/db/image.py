"""Image database model.

REQ-022: PonyORM model for images.
REQ-023: Image model in separate file.
REQ-024: Image entity with relationships to faces, objects, poses, EXIF.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pony.orm import Optional as PonyOptional
from pony.orm import Required, Set

from media_indexer.db.connection import db

if TYPE_CHECKING:
    from media_indexer.db.exif import EXIFData
    from media_indexer.db.face import Face
    from media_indexer.db.object import Object
    from media_indexer.db.pose import Pose

logger = logging.getLogger(__name__)


class Image(db.Entity):
    """Image database model.

    REQ-024: Store image metadata with relationships.
    """

    # Primary key
    id: int = Required(int, auto=True)

    # Image file information
    path: str = Required(str, unique=True, index=True)
    file_hash: str = Required(str, index=True)  # Content hash for deduplication
    file_size: int = Required(int)  # File size in bytes
    width: PonyOptional[int] = None  # Image width
    height: PonyOptional[int] = None  # Image height

    # Timestamps
    created_at: datetime = Required(datetime, default=datetime.now)
    updated_at: datetime = Required(datetime, default=datetime.now)

    # REQ-024: One-to-many relationships
    faces: Set["Face"] = Set("Face", cascade_delete=True)
    objects: Set["Object"] = Set("Object", cascade_delete=True)
    poses: Set["Pose"] = Set("Pose", cascade_delete=True)

    # REQ-024: One-to-one relationships
    exif_data: PonyOptional["EXIFData"] = PonyOptional("EXIFData", cascade_delete=True)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Image(id={self.id}, path='{self.path}')"

    @staticmethod
    def get_by_path(path: str) -> Optional["Image"]:
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
