"""
Feature Extractor Module

REQ-XXX: Extract features from sidecar files and/or database to an output directory.
REQ-010: All code components directly linked to requirements.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extractor for reading from sidecars and/or database.
    
    REQ-XXX: Extract features from existing sidecars/database.
    """
    
    def __init__(
        self,
        input_dir: Path | None = None,
        database_path: Path | None = None,
        output_dir: Path,
    ) -> None:
        """
        Initialize feature extractor.
        
        Args:
            input_dir: Directory containing sidecar files.
            database_path: Path to SQLite database.
            output_dir: Directory for extracted features.
        """
        self.input_dir = input_dir
        self.database_path = database_path
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"REQ-XXX: Feature extractor initialized")
        logger.info(f"  Input directory: {input_dir}")
        logger.info(f"  Database: {database_path}")
        logger.info(f"  Output directory: {output_dir}")
    
    def extract_features(self) -> dict[str, Any]:
        """
        Extract features from sidecar files and/or database.
        
        Returns:
            Dictionary with extraction statistics.
        """
        stats = {
            "total_images": 0,
            "extracted_images": 0,
            "error_images": 0,
            "faces": 0,
            "objects": 0,
            "poses": 0,
        }
        
        # Extract from database if provided
        if self.database_path:
            logger.info("REQ-XXX: Extracting from database")
            db_stats = self._extract_from_database()
            stats.update(db_stats)
        
        # Extract from sidecar files if provided
        if self.input_dir:
            logger.info("REQ-XXX: Extracting from sidecar files")
            sidecar_stats = self._extract_from_sidecars()
            # Merge stats
            for key in stats:
                stats[key] += sidecar_stats.get(key, 0)
        
        logger.info(f"REQ-XXX: Extraction complete")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Extracted: {stats['extracted_images']}")
        logger.info(f"  Errors: {stats['error_images']}")
        logger.info(f"  Total faces: {stats['faces']}")
        logger.info(f"  Total objects: {stats['objects']}")
        logger.info(f"  Total poses: {stats['poses']}")
        
        return stats
    
    def _extract_from_database(self) -> dict[str, Any]:
        """Extract features from database."""
        from pony.orm import db_session
        
        from media_indexer.db.connection import DatabaseConnection
        from media_indexer.db.image import Image
        
        logger.info("REQ-XXX: Connecting to database")
        db_conn = DatabaseConnection(self.database_path)
        db_conn.connect()
        
        stats = {
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
                    try:
                        # Build feature data
                        features = {
                            "faces": len(db_image.faces),
                            "objects": len(db_image.objects),
                            "poses": len(db_image.poses),
                            "has_exif": db_image.exif_data is not None,
                        }
                        
                        # Write to output file
                        image_path = Path(db_image.path)
                        output_file = self.output_dir / f"{image_path.stem}.features.txt"
                        
                        with open(output_file, "w") as f:
                            f.write(f"Image: {image_path}\n")
                            f.write(f"Features:\n")
                            f.write(f"  Faces: {features['faces']}\n")
                            f.write(f"  Objects: {features['objects']}\n")
                            f.write(f"  Poses: {features['poses']}\n")
                            f.write(f"  EXIF: {'Yes' if features['has_exif'] else 'No'}\n")
                        
                        stats["extracted_images"] += 1
                        stats["faces"] += features["faces"]
                        stats["objects"] += features["objects"]
                        stats["poses"] += features["poses"]
                        
                    except Exception as e:
                        logger.error(f"REQ-XXX: Failed to extract {db_image.path}: {e}")
                        stats["error_images"] += 1
        
        finally:
            db_conn.close()
        
        return stats
    
    def _extract_from_sidecars(self) -> dict[str, Any]:
        """Extract features from sidecar files."""
        import json
        
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
            try:
                # Read sidecar
                with open(sidecar_path) as f:
                    metadata = json.load(f)
                
                # Extract features
                features = {
                    "faces": len(metadata.get("faces", [])),
                    "objects": len(metadata.get("objects", [])),
                    "poses": len(metadata.get("poses", [])),
                    "has_exif": metadata.get("exif") is not None,
                }
                
                # Write to output file
                output_file = self.output_dir / f"{sidecar_path.stem}.features.txt"
                with open(output_file, "w") as f:
                    f.write(f"Sidecar: {sidecar_path}\n")
                    f.write(f"Features:\n")
                    f.write(f"  Faces: {features['faces']}\n")
                    f.write(f"  Objects: {features['objects']}\n")
                    f.write(f"  Poses: {features['poses']}\n")
                    f.write(f"  EXIF: {'Yes' if features['has_exif'] else 'No'}\n")
                
                stats["extracted_images"] += 1
                stats["faces"] += features["faces"]
                stats["objects"] += features["objects"]
                stats["poses"] += features["poses"]
                
            except Exception as e:
                logger.error(f"REQ-XXX: Failed to extract from {sidecar_path}: {e}")
                stats["error_images"] += 1
        
        return stats


def get_feature_extractor(
    input_dir: Path | None = None,
    database_path: Path | None = None,
    output_dir: Path = Path("extracted_features"),
) -> FeatureExtractor:
    """
    Factory function to get feature extractor instance.
    
    Args:
        input_dir: Directory containing sidecar files.
        database_path: Path to SQLite database.
        output_dir: Directory for extracted features.
    
    Returns:
        FeatureExtractor: Configured feature extractor.
    """
    return FeatureExtractor(input_dir, database_path, output_dir)
