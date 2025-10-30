"""Feature extractor module.

REQ-075: Extract features from sidecar files and/or database to a specified output directory.
REQ-010: All code components directly linked to requirements.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extractor for reading from sidecars and/or database.

    REQ-075: Extract features from existing sidecars/database to explicit output directory.
    """

    def __init__(
        self,
        input_dir: Path | None = None,
        database_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize feature extractor.

        Parameters
        ----------
        input_dir : Path | None, optional
            Directory containing sidecar files.
        database_path : Path | None, optional
            Path to SQLite database.
        output_dir : Path | None, optional
            Directory for extracted features. Required for initialization.
        """
        self.input_dir = input_dir
        self.database_path = database_path

        if output_dir is None:
            msg = "REQ-075: output_dir must be provided for feature extraction."
            raise ValueError(msg)

        self.output_dir = Path(output_dir)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("REQ-075: Feature extractor initialized")
        logger.info(f"  Input directory: {input_dir}")
        logger.info(f"  Database: {database_path}")
        logger.info(f"  Output directory: {output_dir}")

    def extract_features(self) -> dict[str, int]:
        """Extract features from sidecar files and/or database.

        Returns
        -------
        dict[str, int]
            Dictionary with extraction statistics.
        """
        stats: dict[str, int] = {
            "total_images": 0,
            "extracted_images": 0,
            "error_images": 0,
            "faces": 0,
            "objects": 0,
            "poses": 0,
        }

        # Extract from database if provided
        if self.database_path:
            logger.info("REQ-075: Extracting from database")
            db_stats = self._extract_from_database()
            stats.update(db_stats)

        # Extract from sidecar files if provided
        if self.input_dir:
            logger.info("REQ-075: Extracting from sidecar files")
            sidecar_stats = self._extract_from_sidecars()
            # Merge stats
            for key in stats:
                stats[key] += sidecar_stats.get(key, 0)

        logger.info("REQ-075: Extraction complete")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Extracted: {stats['extracted_images']}")
        logger.info(f"  Errors: {stats['error_images']}")
        logger.info(f"  Total faces: {stats['faces']}")
        logger.info(f"  Total objects: {stats['objects']}")
        logger.info(f"  Total poses: {stats['poses']}")

        return stats

    def _process_database_image(self, db_image: Any, stats: dict[str, int]) -> None:
        """Process a single database image and update statistics.

        Parameters
        ----------
        db_image : Any
            PonyORM image entity to process.
        stats : dict[str, int]
            Aggregated statistics to update.
        """

        try:
            features = {
                "faces": len(db_image.faces),
                "objects": len(db_image.objects),
                "poses": len(db_image.poses),
                "has_exif": db_image.exif_data is not None,
            }

            image_path = Path(db_image.path)
            output_file = self.output_dir / f"{image_path.stem}.features.txt"

            with open(output_file, "w") as feature_file:
                feature_file.write(f"Image: {image_path}\n")
                feature_file.write("Features:\n")
                feature_file.write(f"  Faces: {features['faces']}\n")
                feature_file.write(f"  Objects: {features['objects']}\n")
                feature_file.write(f"  Poses: {features['poses']}\n")
                feature_file.write(f"  EXIF: {'Yes' if features['has_exif'] else 'No'}\n")

            stats["extracted_images"] += 1
            stats["faces"] += features["faces"]
            stats["objects"] += features["objects"]
            stats["poses"] += features["poses"]

        except Exception as exc:  # noqa: BLE001
            logger.error(f"REQ-075: Failed to extract {db_image.path}: {exc}")
            stats["error_images"] += 1

    def _process_sidecar_file(self, sidecar_path: Path, stats: dict[str, int]) -> None:
        """Process a single sidecar file and update statistics.

        Parameters
        ----------
        sidecar_path : Path
            Path to the sidecar JSON file.
        stats : dict[str, int]
            Aggregated statistics to update.
        """
        try:
            with open(sidecar_path) as sidecar_file:
                metadata = json.load(sidecar_file)

            features = {
                "faces": len(metadata.get("faces", [])),
                "objects": len(metadata.get("objects", [])),
                "poses": len(metadata.get("poses", [])),
                "has_exif": metadata.get("exif") is not None,
            }

            output_file = self.output_dir / f"{sidecar_path.stem}.features.txt"
            with open(output_file, "w") as feature_file:
                feature_file.write(f"Sidecar: {sidecar_path}\n")
                feature_file.write("Features:\n")
                feature_file.write(f"  Faces: {features['faces']}\n")
                feature_file.write(f"  Objects: {features['objects']}\n")
                feature_file.write(f"  Poses: {features['poses']}\n")
                feature_file.write(f"  EXIF: {'Yes' if features['has_exif'] else 'No'}\n")

            stats["extracted_images"] += 1
            stats["faces"] += features["faces"]
            stats["objects"] += features["objects"]
            stats["poses"] += features["poses"]

        except Exception as exc:  # noqa: BLE001
            logger.error(f"REQ-075: Failed to extract from {sidecar_path}: {exc}")
            stats["error_images"] += 1

    def _extract_from_database(self) -> dict[str, int]:
        """Extract features from database.

        Returns
        -------
        dict[str, int]
            Aggregated extraction statistics from the database.
        """
        from pony.orm import db_session

        from media_indexer.db.connection import DatabaseConnection
        from media_indexer.db.image import Image

        logger.info("REQ-075: Connecting to database")
        db_conn = DatabaseConnection(self.database_path)
        db_conn.connect()

        stats: dict[str, int] = {
            "total_images": 0,
            "extracted_images": 0,
            "error_images": 0,
            "faces": 0,
            "objects": 0,
            "poses": 0,
        }

        try:
            with db_session:
                images = list(Image.select())
                stats["total_images"] = len(images)

                for db_image in images:
                    self._process_database_image(db_image, stats)

        finally:
            db_conn.close()

        return stats

    def _extract_from_sidecars(self) -> dict[str, int]:
        """Extract features from sidecar files.

        Returns
        -------
        dict[str, int]
            Aggregated extraction statistics from sidecar files.
        """

        stats = {
            "total_images": 0,
            "extracted_images": 0,
            "error_images": 0,
            "faces": 0,
            "objects": 0,
            "poses": 0,
        }

        # Find all sidecar files
        sidecars = list(self.input_dir.rglob("*.json"))
        stats["total_images"] = len(sidecars)

        for sidecar_path in sidecars:
            self._process_sidecar_file(sidecar_path, stats)

        return stats


def get_feature_extractor(
    input_dir: Path | None = None,
    database_path: Path | None = None,
    output_dir: Path = Path("extracted_features"),
) -> FeatureExtractor:
    """Factory function to get feature extractor instance.

    Parameters
    ----------
    input_dir : Path | None, optional
        Directory containing sidecar files.
    database_path : Path | None, optional
        Path to SQLite database.
    output_dir : Path, optional
        Directory for extracted features. Defaults to ``extracted_features``.

    Returns
    -------
    FeatureExtractor
        Configured feature extractor.
    """
    return FeatureExtractor(input_dir, database_path, output_dir)
