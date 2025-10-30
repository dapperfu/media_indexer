"""
Analysis Scanner Module

REQ-013: Scan sidecar files and database to determine required analyses.
REQ-010: All code components directly linked to requirements.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pony.orm import db_session

from media_indexer.utils.sidecar_utils import read_sidecar_metadata

logger = logging.getLogger(__name__)


class AnalysisScanner:
    """
    Scanner for determining required analyses.

    REQ-013: Scans sidecar files and database to determine which analyses
    are needed for each image.
    """

    def __init__(
        self,
        database_connection: Any | None,
        sidecar_generator: Any | None,
        disable_sidecar: bool,
        force: bool = False,
    ) -> None:
        """
        Initialize analysis scanner.

        Args:
            database_connection: Optional database connection.
            sidecar_generator: Optional sidecar generator.
            disable_sidecar: Whether sidecar generation is disabled.
            force: Whether to force reprocessing even if raw_conversion_failed flag is set.
        """
        self.database_connection = database_connection
        self.sidecar_generator = sidecar_generator
        self.disable_sidecar = disable_sidecar
        self.force = force

    def find_sidecar_path(self, image_path: Path) -> Path | None:
        """
        Find sidecar file path for an image.

        REQ-013: Find sidecar file to check if analysis already exists.
        Sidecar files are always located next to the image file.

        Args:
            image_path: Path to the image file.

        Returns:
            Path to sidecar file if found, None otherwise.
        """
        image_path_resolved = Path(image_path).resolve()

        # Sidecar files are always next to the image file
        sidecar_path = image_path_resolved.with_suffix(
            image_path_resolved.suffix + ".json"
        )

        if sidecar_path.exists():
            return sidecar_path

        return None

    def get_existing_analyses_from_database_batch(
        self,
        image_paths: list[Path],
    ) -> dict[str, set[str]]:
        """
        Get existing analyses from database for multiple images in batch.

        REQ-013: Query database for all images at once for better performance.

        Args:
            image_paths: List of image paths to check.

        Returns:
            Dictionary mapping image path (str) to set of existing analyses.
        """
        if not self.database_connection:
            return {}

        results: dict[str, set[str]] = {}

        try:
            from media_indexer.db.image import Image as DBImage
            
            with db_session:
                # Query all images at once
                # Convert to list for PonyORM compatibility (SQL IN clause)
                path_strs = [str(path) for path in image_paths]
                db_images = list(DBImage.select(lambda img: img.path in path_strs))

                # Build results map
                for db_image in db_images:
                    existing: set[str] = set()

                    if db_image.faces and len(db_image.faces) > 0:
                        existing.add("faces")
                    if db_image.objects and len(db_image.objects) > 0:
                        existing.add("objects")
                    if db_image.poses and len(db_image.poses) > 0:
                        existing.add("poses")
                    if db_image.exif_data is not None:
                        existing.add("exif")

                    results[db_image.path] = existing

                # Initialize empty sets for images not in database
                for path in image_paths:
                    path_str = str(path)
                    if path_str not in results:
                        results[path_str] = set()

        except Exception as e:
            logger.debug(f"REQ-025: Batch database query failed: {e}")

        return results

    def get_existing_analyses(
        self,
        image_path: Path,
        required_analyses: set[str],
    ) -> set[str]:
        """
        Get set of analyses already present in sidecar file or database.

        REQ-013: Check sidecar file and database to determine what analyses exist.
        REQ-040: Check for raw_conversion_failed flag and skip unless --force is used.

        Args:
            image_path: Path to the image file.
            required_analyses: Set of required analyses for early exit check.

        Returns:
            Set of analysis types present: 'exif', 'faces', 'objects', 'poses'.
            Returns set with all required analyses if raw_conversion_failed flag is set
            and force is False (effectively skipping processing).
        """
        existing: set[str] = set()

        # REQ-025: Check database first if available
        if self.database_connection:
            try:
                from media_indexer.db.image import Image as DBImage
                
                with db_session:
                    db_image = DBImage.get_by_path(str(image_path))
                    if db_image:
                        # Check if analyses exist in database
                        if db_image.faces and len(db_image.faces) > 0:
                            existing.add("faces")
                        if db_image.objects and len(db_image.objects) > 0:
                            existing.add("objects")
                        if db_image.poses and len(db_image.poses) > 0:
                            existing.add("poses")
                        if db_image.exif_data is not None:
                            existing.add("exif")

                        # If all analyses are in database, return early
                        if len(existing) == len(required_analyses):
                            return existing
            except Exception as e:
                logger.debug(f"REQ-025: Database check failed for {image_path}: {e}")

        # Also check sidecar files (complement database, not replace)
        if not self.disable_sidecar:
            sidecar_path = self.find_sidecar_path(image_path)
            if sidecar_path and sidecar_path.exists():
                try:
                    metadata = read_sidecar_metadata(
                        sidecar_path, self.sidecar_generator
                    )

                    # REQ-040: Check for raw_conversion_failed flag
                    # If flag is set and force is False, skip processing by returning all required analyses
                    if metadata.get("raw_conversion_failed") and not self.force:
                        logger.debug(
                            f"REQ-040: Skipping {image_path} - RAW conversion failed previously (use --force to retry)"
                        )
                        # Return all required analyses to indicate nothing needs processing
                        return required_analyses.copy()

                    # Add analyses from sidecar (union with database)
                    if metadata.get("exif"):
                        existing.add("exif")
                    if metadata.get("faces"):
                        existing.add("faces")
                    if metadata.get("objects"):
                        existing.add("objects")
                    if metadata.get("poses"):
                        existing.add("poses")
                except Exception as e:
                    logger.debug(f"REQ-013: Failed to read sidecar for {image_path}: {e}")

        return existing

