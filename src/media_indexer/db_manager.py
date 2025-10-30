"""
Database Manager Module

Provides database statistics, queries, and management operations.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for statistics and manipulation.
    """

    def __init__(self, database_path: Path) -> None:
        """
        Initialize database manager.

        Args:
            database_path: Path to SQLite database.
        """
        self.database_path = Path(database_path)

        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")

        logger.info(f"Database manager initialized for {database_path}")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics.
        """
        from pony.orm import db_session

        from media_indexer.db.connection import DatabaseConnection

        # Connect to database first
        db_conn = DatabaseConnection(self.database_path)
        db_conn.connect()

        # Import models after connection is established
        from media_indexer.db.image import Image

        stats = {
            "total_images": 0,
            "with_faces": 0,
            "with_objects": 0,
            "with_poses": 0,
            "with_exif": 0,
            "total_faces": 0,
            "total_objects": 0,
            "total_poses": 0,
            "face_models": {},
            "object_classes": {},
        }

        try:
            with db_session:
                images = list(Image.select())
                stats["total_images"] = len(images)

                for image in images:
                    # Count features
                    if image.faces:
                        stats["with_faces"] += 1
                        stats["total_faces"] += len(image.faces)
                        for face in image.faces:
                            model = face.model
                            stats["face_models"][model] = stats["face_models"].get(model, 0) + 1

                    if image.objects:
                        stats["with_objects"] += 1
                        stats["total_objects"] += len(image.objects)
                        for obj in image.objects:
                            class_name = obj.class_name
                            stats["object_classes"][class_name] = stats["object_classes"].get(class_name, 0) + 1

                    if image.poses:
                        stats["with_poses"] += 1
                        stats["total_poses"] += len(image.poses)

                    if image.exif_data:
                        stats["with_exif"] += 1

        finally:
            db_conn.close()

        return stats

    def print_statistics(self) -> None:
        """Print database statistics."""
        stats = self.get_statistics()

        print("\n=== Database Statistics ===")
        print(f"Total Images: {stats['total_images']}")
        print("\nImages with features:")
        print(f"  Faces: {stats['with_faces']}")
        print(f"  Objects: {stats['with_objects']}")
        print(f"  Poses: {stats['with_poses']}")
        print(f"  EXIF: {stats['with_exif']}")

        print("\nTotal features:")
        print(f"  Faces: {stats['total_faces']}")
        print(f"  Objects: {stats['total_objects']}")
        print(f"  Poses: {stats['total_poses']}")

        if stats["face_models"]:
            print("\nFace detection models:")
            for model, count in sorted(stats["face_models"].items()):
                print(f"  {model}: {count} faces")

        if stats["object_classes"]:
            print("\nTop object classes:")
            sorted_classes = sorted(stats["object_classes"].items(), key=lambda x: x[1], reverse=True)[:10]
            for class_name, count in sorted_classes:
                print(f"  {class_name}: {count}")

    def search_images(self, query: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search images in database.

        Args:
            query: Optional search query.
            limit: Maximum number of results.

        Returns:
            List of matching images.
        """
        from pony.orm import db_session

        from media_indexer.db.connection import DatabaseConnection

        # Connect to database first
        db_conn = DatabaseConnection(self.database_path)
        db_conn.connect()

        # Import models after connection
        from media_indexer.db.image import Image

        try:
            with db_session:
                images = Image.select(lambda i: query in i.path)[:limit] if query else Image.select()[:limit]

                return [
                    {
                        "path": image.path,
                        "faces": len(image.faces),
                        "objects": len(image.objects),
                        "poses": len(image.poses),
                        "has_exif": image.exif_data is not None,
                        "width": image.width,
                        "height": image.height,
                        "file_size": image.file_size,
                    }
                    for image in images
                ]

        finally:
            db_conn.close()

    def clean_database(self) -> dict[str, int]:
        """
        Clean database by removing orphaned records.

        Returns:
            Dictionary with cleanup statistics.
        """
        from pony.orm import db_session

        from media_indexer.db.connection import DatabaseConnection

        # Connect to database first
        db_conn = DatabaseConnection(self.database_path)
        db_conn.connect()

        # Import models after connection
        from media_indexer.db.image import Image

        stats = {
            "images_checked": 0,
            "images_removed": 0,
            "files_not_found": 0,
        }

        try:
            with db_session:
                images = list(Image.select())
                stats["images_checked"] = len(images)

                for image in images:
                    image_path = Path(image.path)
                    if not image_path.exists():
                        logger.info(f"Removing missing image: {image.path}")
                        image.delete()
                        stats["images_removed"] += 1
                        stats["files_not_found"] += 1

        finally:
            db_conn.close()

        return stats
