"""
Metadata Converter for Database Operations

REQ-024: Convert metadata between dict format and database entities.
REQ-010: All code components directly linked to requirements.
"""

import logging
from datetime import datetime
from typing import Any

from pony.orm import db_session

from media_indexer.db.exif import EXIFData
from media_indexer.db.face import Face
from media_indexer.db.hash_util import calculate_file_hash, get_file_size
from media_indexer.db.image import Image
from media_indexer.db.object import Object
from media_indexer.db.pose import Pose

logger = logging.getLogger(__name__)


class MetadataConverter:
    """
    Convert metadata between dict format and database entities.

    REQ-024: Provides utilities for converting metadata dictionaries
    (from sidecar files or detectors) to database entities and vice versa.
    """

    @staticmethod
    def metadata_to_db_entities(
        image_path: str,
        db_image: Image,
        metadata: dict[str, Any],
    ) -> None:
        """
        Store metadata dict as database entities.

        REQ-024: Convert metadata dictionary to database entities
        (faces, objects, poses, EXIF).

        Args:
            image_path: Path to image file (for logging).
            db_image: Database Image entity.
            metadata: Metadata dictionary with faces, objects, poses, exif.
        """
        # REQ-024: Store faces
        if "faces" in metadata and metadata["faces"]:
            for face_data in metadata["faces"]:
                # REQ-066: Handle optional embedding field
                # PonyORM doesn't accept None for Optional(Json), so only set if present
                face_kwargs: dict[str, Any] = {
                    "image": db_image,
                    "confidence": face_data.get("confidence", 0.0),
                    "bbox": face_data.get("bbox", []),
                    "model": face_data.get("model", "unknown"),
                }
                embedding = face_data.get("embedding")
                if embedding is not None:
                    face_kwargs["embedding"] = embedding

                Face(**face_kwargs)

        # REQ-024: Store objects
        if "objects" in metadata and metadata["objects"]:
            for obj_data in metadata["objects"]:
                Object(
                    image=db_image,
                    class_id=obj_data.get("class_id", -1),
                    class_name=obj_data.get("class_name", "unknown"),
                    confidence=obj_data.get("confidence", 0.0),
                    bbox=obj_data.get("bbox", []),
                )

        # REQ-024: Store poses
        if "poses" in metadata and metadata["poses"]:
            for pose_data in metadata["poses"]:
                # REQ-066: Handle optional keypoints_conf field
                # PonyORM doesn't accept None for Optional(Json), so only set if present
                pose_kwargs: dict[str, Any] = {
                    "image": db_image,
                    "confidence": pose_data.get("confidence", 0.0),
                    "bbox": pose_data.get("bbox", []),
                    "keypoints": pose_data.get("keypoints", []),
                }
                keypoints_conf = pose_data.get("keypoints_conf")
                if keypoints_conf is not None:
                    pose_kwargs["keypoints_conf"] = keypoints_conf

                Pose(**pose_kwargs)

        # REQ-024: Store EXIF data
        if "exif" in metadata and metadata["exif"]:
            EXIFData(
                image=db_image,
                data=metadata["exif"],
            )

        logger.debug(f"REQ-024: Stored metadata for {image_path}")

    @staticmethod
    def db_entities_to_metadata(db_image: Image) -> dict[str, Any]:
        """
        Convert database entities to metadata dict.

        REQ-024: Convert database entities to metadata dictionary format
        (for sidecar file generation or export).

        Args:
            db_image: Database Image entity.

        Returns:
            Metadata dictionary with faces, objects, poses, exif.
        """
        metadata: dict[str, Any] = {
            "faces": [],
            "objects": [],
            "poses": [],
            "exif": None,
        }

        # Add faces
        for face in db_image.faces:
            metadata["faces"].append(
                {
                    "confidence": face.confidence,
                    "bbox": face.bbox,
                    "embedding": face.embedding,
                    "model": face.model,
                }
            )

        # Add objects
        for obj in db_image.objects:
            metadata["objects"].append(
                {
                    "class_id": obj.class_id,
                    "class_name": obj.class_name,
                    "confidence": obj.confidence,
                    "bbox": obj.bbox,
                }
            )

        # Add poses
        for pose in db_image.poses:
            metadata["poses"].append(
                {
                    "confidence": pose.confidence,
                    "keypoints": pose.keypoints,
                    "bbox": pose.bbox,
                    "keypoints_conf": pose.keypoints_conf,
                }
            )

        # Add EXIF data
        if db_image.exif_data:
            exif = db_image.exif_data
            metadata["exif"] = exif.data  # Extract from JSON blob

        return metadata

    @staticmethod
    def create_db_image(
        image_path: str,
        file_hash: str | None = None,
        file_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Image:
        """
        Create Image database entity.

        REQ-024: Create Image entity with file information.

        Args:
            image_path: Path to image file.
            file_hash: Optional file hash (calculated if not provided).
            file_size: Optional file size (calculated if not provided).
            width: Optional image width.
            height: Optional image height.

        Returns:
            Image database entity.
        """
        from pathlib import Path

        # Calculate hash and size if not provided
        if file_hash is None or file_size is None:
            path_obj = Path(image_path)
            if file_hash is None:
                file_hash = calculate_file_hash(path_obj)
            if file_size is None:
                file_size = get_file_size(path_obj) or 0

        # Extract dimensions from EXIF if available
        # (This would be done by caller if they have EXIF data)

        db_image = Image(
            path=image_path,
            file_hash=file_hash,
            file_size=file_size,
            width=width,
            height=height,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        return db_image

