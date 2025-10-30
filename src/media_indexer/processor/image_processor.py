"""
Single Image Processing Module

REQ-002: Process single images and store results.
REQ-010: All code components directly linked to requirements.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pony.orm import db_session

from media_indexer.raw_converter import get_raw_image_source
from media_indexer.utils.file_utils import is_raw_file
from media_indexer.utils.sidecar_utils import read_sidecar_metadata

logger = logging.getLogger(__name__)


class ImageProcessorComponent:
    """
    Component for processing single images.

    REQ-002: Handles processing of individual images including detection
    and storage.
    """

    def __init__(
        self,
        exif_extractor: Any | None,
        face_detector: Any | None,
        object_detector: Any | None,
        pose_detector: Any | None,
        sidecar_generator: Any | None,
        database_connection: Any | None,
        disable_sidecar: bool,
    ) -> None:
        """
        Initialize image processor component.

        Args:
            exif_extractor: Optional EXIF extractor.
            face_detector: Optional face detector.
            object_detector: Optional object detector.
            pose_detector: Optional pose detector.
            sidecar_generator: Optional sidecar generator.
            database_connection: Optional database connection.
            disable_sidecar: Whether sidecar generation is disabled.
        """
        self.exif_extractor = exif_extractor
        self.face_detector = face_detector
        self.object_detector = object_detector
        self.pose_detector = pose_detector
        self.sidecar_generator = sidecar_generator
        self.database_connection = database_connection
        self.disable_sidecar = disable_sidecar

    def process_single_image(
        self,
        image_path: Path,
        missing_analyses: set[str],
        sidecar_path: Path | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Process a single image.

        REQ-002: Process image and extract requested analyses.
        REQ-015: Robust error handling.

        Args:
            image_path: Path to image file.
            missing_analyses: Set of analysis types to perform.
            sidecar_path: Optional path to existing sidecar file.

        Returns:
            Tuple of (success, detection_results). detection_results contains
            counts of faces, objects, poses.
        """
        try:
            # Load existing metadata from sidecar if it exists
            existing_metadata: dict[str, Any] = {}
            if sidecar_path and sidecar_path.exists():
                try:
                    existing_metadata = read_sidecar_metadata(sidecar_path, self.sidecar_generator)
                except Exception as e:
                    logger.debug(f"REQ-013: Failed to read existing sidecar: {e}")

            # Start with existing metadata or new dict
            metadata: dict[str, Any] = (
                existing_metadata.copy() if existing_metadata else {"image_path": str(image_path)}
            )
            detection_results: dict[str, Any] = {}

            # REQ-040: Check RAW conversion early to detect failures
            # If this is a RAW file and conversion fails, mark it and skip detections
            if is_raw_file(image_path):
                source_path = get_raw_image_source(image_path)
                if source_path is None:
                    # RAW conversion failed - mark in sidecar and skip processing
                    logger.warning(f"REQ-040: RAW conversion failed for {image_path}, marking in sidecar")
                    metadata["raw_conversion_failed"] = True
                    # Still save the sidecar with the flag
                    if not self.disable_sidecar and self.sidecar_generator is not None:
                        try:
                            self.sidecar_generator.generate_sidecar(image_path, metadata)
                            logger.debug(f"REQ-040: Updated sidecar with raw_conversion_failed flag for {image_path}")
                        except Exception as e:
                            logger.error(f"REQ-004: Sidecar generation failed: {e}")
                    return True, detection_results
                else:
                    # RAW conversion succeeded - clear the flag if it was previously set
                    if metadata.get("raw_conversion_failed"):
                        metadata.pop("raw_conversion_failed")
                        logger.debug(
                            f"REQ-040: RAW conversion succeeded for {image_path}, clearing raw_conversion_failed flag"
                        )

            # REQ-003: Extract EXIF if needed
            if "exif" in missing_analyses and self.exif_extractor is not None:
                try:
                    metadata["exif"] = self.exif_extractor.extract_from_path(image_path)
                except Exception as e:
                    logger.debug(f"REQ-003: EXIF extraction failed: {e}")

            # REQ-007: Detect faces if needed
            if "faces" in missing_analyses and self.face_detector is not None:
                try:
                    faces = self.face_detector.detect_faces(image_path)
                    metadata["faces"] = faces
                    detection_results["faces"] = len(faces)
                except Exception as e:
                    logger.debug(f"REQ-007: Face detection failed: {e}")
                    detection_results["faces"] = 0

            # REQ-008: Detect objects if needed
            if "objects" in missing_analyses and self.object_detector is not None:
                try:
                    objects = self.object_detector.detect_objects(image_path)
                    metadata["objects"] = objects
                    detection_results["objects"] = len(objects)
                    if objects:
                        class_counts = Counter(
                            obj.get("class_name", "").strip().lower()
                            for obj in objects
                            if obj.get("class_name")
                        )
                        if class_counts:
                            detection_results["object_class_counts"] = dict(class_counts)
                except Exception as e:
                    logger.debug(f"REQ-008: Object detection failed: {e}")
                    detection_results["objects"] = 0

            # REQ-009: Detect poses if needed
            if "poses" in missing_analyses and self.pose_detector is not None:
                try:
                    poses = self.pose_detector.detect_poses(image_path)
                    metadata["poses"] = poses
                    detection_results["poses"] = len(poses)
                except Exception as e:
                    logger.debug(f"REQ-009: Pose detection failed: {e}")
                    detection_results["poses"] = 0

            # REQ-025: Store to database
            if self.database_connection:
                db_success = self.store_to_database(image_path, metadata)
                if not db_success:
                    logger.error(f"REQ-025: Failed to store {image_path} to database")

            # REQ-027: Generate/update sidecar if not disabled
            if not self.disable_sidecar and self.sidecar_generator is not None:
                try:
                    self.sidecar_generator.generate_sidecar(image_path, metadata)
                    logger.debug(f"REQ-027: Generated/updated sidecar for {image_path}")
                except Exception as e:
                    logger.error(f"REQ-004: Sidecar generation failed: {e}")
                    if self.database_connection:
                        # If using database, sidecar failure is not fatal
                        return True, detection_results
                    else:
                        return False, {}
            elif self.disable_sidecar:
                logger.debug(f"REQ-026: Skipping sidecar generation for {image_path}")

            return True, detection_results

        except Exception as e:
            # REQ-015: Robust error handling
            logger.error(f"REQ-015: Error processing {image_path}: {e}")
            return False, {}

    def store_to_database(self, image_path: Path, metadata: dict[str, Any]) -> bool:
        """
        Store image metadata to database.

        REQ-025: Store image metadata to database using PonyORM.
        Uses MetadataConverter for conversion.

        Args:
            image_path: Path to the image file.
            metadata: Extracted metadata dictionary.

        Returns:
            True if successful, False otherwise.
        """
        if not self.database_connection:
            return True

        try:
            from media_indexer.db.hash_util import calculate_file_hash, get_file_size
            from media_indexer.db.image import Image as DBImage
            from media_indexer.db.metadata_converter import MetadataConverter

            with db_session:
                # REQ-028: Calculate file hash for deduplication
                file_hash = calculate_file_hash(image_path)
                file_size = get_file_size(image_path) or 0

                # Check if image already exists in database
                existing_image = DBImage.get_by_path(str(image_path))
                if existing_image:
                    logger.debug(f"REQ-024: Image {image_path} already exists in database, skipping")
                    return True

                # REQ-024: Create Image entity using converter
                db_image = MetadataConverter.create_db_image(
                    str(image_path),
                    file_hash=file_hash,
                    file_size=file_size,
                )

                # REQ-024: Store metadata using converter
                MetadataConverter.metadata_to_db_entities(str(image_path), db_image, metadata)

                logger.debug(f"REQ-025: Stored metadata for {image_path} in database")
                return True

        except Exception as e:
            logger.error(f"REQ-025: Failed to store to database: {e}")
            return False
