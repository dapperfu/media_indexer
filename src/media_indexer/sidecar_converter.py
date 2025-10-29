"""
Sidecar Converter Module

REQ-032, REQ-033, REQ-034: Convert between sidecar and database formats.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def import_sidecars_to_database(
    input_dir: Path,
    database_path: Path,
    verbose: int = 20,
) -> None:
    """
    Import sidecar files into database.

    REQ-032: Import existing sidecar files into database.

    Args:
        input_dir: Directory containing images and sidecar files.
        database_path: Path to SQLite database.
        verbose: Verbosity level.
    """
    import image_sidecar_rust
    from pony.orm import db_session

    from media_indexer.db.connection import DatabaseConnection
    from media_indexer.db.exif import EXIFData
    from media_indexer.db.face import Face
    from media_indexer.db.hash_util import calculate_file_hash, get_file_size
    from media_indexer.db.image import Image
    from media_indexer.db.object import Object
    from media_indexer.db.pose import Pose

    logger.info("REQ-032: Initializing database connection")
    db_conn = DatabaseConnection(database_path)
    db_conn.connect()

    try:
        # REQ-032: Find all images and their corresponding sidecar files
        logger.info(f"REQ-032: Scanning {input_dir} for images and sidecar files")

        image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".raw"]
        processed = 0
        errors = 0

        for image_file in input_dir.rglob("*"):
            if image_file.suffix.lower() in image_extensions:
                # REQ-032: Check for corresponding sidecar file
                # Check for sidecar with .json extension
                sidecar_path = image_file.with_suffix('.json')
                
                if sidecar_path.exists():
                    try:
                        # REQ-032: Read sidecar file
                        logger.debug(f"REQ-032: Reading sidecar from {sidecar_path}")
                        import json
                        with open(sidecar_path) as f:
                            metadata = json.load(f)

                        # REQ-032: Import to database
                        with db_session:
                            # Check if image already exists
                            existing_image = Image.get(path=str(image_file))
                            if existing_image:
                                logger.debug(f"REQ-032: Image {image_file} already in database, skipping")
                                continue

                            # Calculate file hash
                            file_hash = calculate_file_hash(image_file)
                            file_size = get_file_size(image_file) or 0

                            # Get image dimensions from metadata if available
                            width = None
                            height = None
                            if "exif" in metadata and metadata["exif"]:
                                exif = metadata["exif"]
                                width = exif.get("imageWidth")
                                height = exif.get("imageHeight")

                            # Create Image entity
                            from datetime import datetime

                            db_image = Image(
                                path=str(image_file),
                                file_hash=file_hash,
                                file_size=file_size,
                                width=width,
                                height=height,
                                created_at=datetime.now(),
                                updated_at=datetime.now(),
                            )

                            # Store faces
                            if "faces" in metadata and metadata["faces"]:
                                for face_data in metadata["faces"]:
                                    # REQ-066: Handle optional embedding field
                                    face_kwargs = {
                                        "image": db_image,
                                        "confidence": face_data.get("confidence", 0.0),
                                        "bbox": face_data.get("bbox", []),
                                        "model": face_data.get("model", "unknown"),
                                    }
                                    embedding = face_data.get("embedding")
                                    if embedding is not None:
                                        face_kwargs["embedding"] = embedding
                                    
                                    Face(**face_kwargs)

                            # Store objects
                            if "objects" in metadata and metadata["objects"]:
                                for obj_data in metadata["objects"]:
                                    Object(
                                        image=db_image,
                                        class_id=obj_data.get("class_id", -1),
                                        class_name=obj_data.get("class_name", "unknown"),
                                        confidence=obj_data.get("confidence", 0.0),
                                        bbox=obj_data.get("bbox", []),
                                    )

                            # Store poses
                            if "poses" in metadata and metadata["poses"]:
                                for pose_data in metadata["poses"]:
                                    # REQ-066: Handle optional keypoints_conf field
                                    pose_kwargs = {
                                        "image": db_image,
                                        "confidence": pose_data.get("confidence", 0.0),
                                        "keypoints": pose_data.get("keypoints", []),
                                        "bbox": pose_data.get("bbox", []),
                                    }
                                    keypoints_conf = pose_data.get("keypoints_conf")
                                    if keypoints_conf is not None:
                                        pose_kwargs["keypoints_conf"] = keypoints_conf
                                    
                                    Pose(**pose_kwargs)

                            # Store EXIF data
                            if "exif" in metadata and metadata["exif"]:
                                exif_data = metadata["exif"]
                                EXIFData(
                                    image=db_image,
                                    data=exif_data,  # Store as JSON blob
                                )

                        processed += 1
                        if verbose <= 15:
                            logger.info(f"REQ-032: Imported {image_file}")

                    except Exception as e:
                        errors += 1
                        logger.error(f"REQ-032: Failed to import {image_file}: {e}")

        logger.info(f"REQ-032: Import completed - {processed} imported, {errors} errors")

    finally:
        db_conn.close()


def export_database_to_sidecars(
    database_path: Path,
    output_dir: Path,
    verbose: int = 20,
) -> None:
    """
    Export database contents to sidecar files.

    REQ-033: Export database contents to individual sidecar JSON files.

    Args:
        database_path: Path to SQLite database.
        output_dir: Directory for output sidecar files.
        verbose: Verbosity level.
    """
    from pony.orm import db_session

    from media_indexer.db.connection import DatabaseConnection
    from media_indexer.db.image import Image
    from media_indexer.sidecar_generator import SidecarGenerator

    logger.info("REQ-033: Initializing database connection")
    db_conn = DatabaseConnection(database_path)
    db_conn.connect()

    try:
        # REQ-033: Export all images from database to sidecar files
        logger.info(f"REQ-033: Exporting database to {output_dir}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sidecar_generator = SidecarGenerator(output_dir)

        with db_session:
            images = list(Image.select())
            logger.info(f"REQ-033: Exporting {len(images)} images")

            for db_image in images:
                try:
                    # REQ-033: Build metadata from database
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

                    # REQ-033: Generate sidecar file
                    image_path = Path(db_image.path)
                    sidecar_generator.generate_sidecar(image_path, metadata)

                    if verbose <= 15:
                        logger.info(f"REQ-033: Exported {db_image.path}")

                except Exception as e:
                    logger.error(f"REQ-033: Failed to export {db_image.path}: {e}")

            logger.info(f"REQ-033: Export completed for {len(images)} images")

    finally:
        db_conn.close()
