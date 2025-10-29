"""
Main Image Processor

REQ-002: Process all images in collection and generate sidecar files.
REQ-006: GPU-only operation enforcement.
REQ-010: All code components directly linked to requirements.
REQ-011: Checkpoint/resume functionality.
REQ-012: Progress tracking and statistics.
REQ-013: Idempotent processing.
REQ-015: Robust error handling.
REQ-016: Multi-level verbosity logging.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import image_sidecar_rust
import tqdm

# REQ-022: Import database modules
from media_indexer.db.connection import DatabaseConnection
from media_indexer.db.hash_util import calculate_file_hash
from media_indexer.exif_extractor import EXIFExtractor, get_exif_extractor
from media_indexer.face_detector import FaceDetector, get_face_detector
from media_indexer.gpu_validator import GPUValidator, get_gpu_validator
from media_indexer.object_detector import ObjectDetector, get_object_detector
from media_indexer.pose_detector import PoseDetector, get_pose_detector
from media_indexer.raw_converter import cleanup_temp_files
from media_indexer.sidecar_generator import SidecarGenerator, get_sidecar_generator

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Main image processor orchestrating all components.

    REQ-002: Process images and generate sidecar files with metadata.
    REQ-011: Support checkpointing and resuming.
    REQ-012: Track progress and statistics.
    REQ-013: Implement idempotent processing.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        checkpoint_file: Path | None = None,
        verbose: int = 20,
        batch_size: int = 1,
        database_path: Path | None = None,
        disable_sidecar: bool = False,
        limit: int | None = None,
    ) -> None:
        """
        Initialize image processor.

        REQ-006: Ensure GPU-only operation.
        REQ-016: Multi-level verbosity.
        REQ-025: Support database storage.
        REQ-026: Support disabling sidecar generation.
        REQ-038: Support limiting number of images to process.

        Args:
            input_dir: Directory containing images to process.
            output_dir: Directory for sidecar files (defaults to input_dir).
            checkpoint_file: Path to checkpoint file (REQ-011).
            verbose: Verbosity level (REQ-016).
            batch_size: Batch size for processing (REQ-014).
            database_path: Path to database file (REQ-025).
            disable_sidecar: Disable sidecar generation when using database (REQ-026).
            limit: Maximum number of images to process (REQ-038).

        Raises:
            RuntimeError: If no GPU is available (REQ-006).
        """
        # REQ-006: Validate GPU availability
        self.gpu_validator: GPUValidator = get_gpu_validator()
        self.device = self.gpu_validator.get_device()

        # REQ-002: Setup input/output directories (resolve to absolute paths)
        self.input_dir: Path = Path(input_dir).resolve()
        if output_dir is None:
            self.output_dir: Path = self.input_dir
        else:
            self.output_dir = Path(output_dir).resolve()

        # REQ-011: Setup checkpoint file (resolve to absolute path)
        if checkpoint_file:
            self.checkpoint_file: Path = Path(checkpoint_file).resolve()
        else:
            self.checkpoint_file: Path = Path(".checkpoint.json").resolve()
        self.processed_files: set[str] = set()

        # REQ-016: Setup verbosity
        self.verbose = verbose

        # REQ-014: Batch size configuration
        self.batch_size = batch_size
        # REQ-020: Use optimal defaults for 12GB VRAM
        if self.batch_size == 1:
            self.batch_size = 4  # Default batch size for better GPU utilization

        # REQ-038: Setup image limit
        self.limit = limit
        if self.limit:
            logger.info(f"REQ-038: Limiting processing to {self.limit} images")

        # REQ-012: Statistics tracking
        self.stats: dict[str, Any] = {
            "total_images": 0,
            "processed_images": 0,
            "skipped_images": 0,
            "error_images": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
        }
        self.stats_lock: Lock = Lock()  # For thread-safe stat updates

        # REQ-002: Initialize components
        self.exif_extractor: EXIFExtractor | None = None
        self.face_detector: FaceDetector | None = None
        self.object_detector: ObjectDetector | None = None
        self.pose_detector: PoseDetector | None = None
        self.sidecar_generator: SidecarGenerator | None = None

        # REQ-025: Initialize database if specified
        self.database_path: Path | None = database_path
        self.disable_sidecar: bool = disable_sidecar
        self.database_connection: DatabaseConnection | None = None
        if self.database_path:
            logger.info(f"REQ-025: Database path specified: {self.database_path}")
            self.database_connection = DatabaseConnection(self.database_path)
            self.database_connection.connect()

        # REQ-026: Check if sidecar should be disabled
        if self.disable_sidecar and not self.database_path:
            logger.warning("REQ-026: --no-sidecar specified without --db, ignoring flag")
            self.disable_sidecar = False
        if self.disable_sidecar:
            logger.info("REQ-026: Sidecar generation disabled")

        # REQ-011: Load checkpoint if exists
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """
        Load checkpoint file (REQ-011).
        """
        if self.checkpoint_file.exists():
            logger.info(f"REQ-011: Loading checkpoint from {self.checkpoint_file}")
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                    self.processed_files = set(data.get("processed_files", []))
                    self.stats.update(data.get("stats", {}))
                    logger.info(f"REQ-011: Loaded {len(self.processed_files)} processed files from checkpoint")
            except Exception as e:
                logger.warning(f"REQ-011: Failed to load checkpoint: {e}")

    def _save_checkpoint(self) -> None:
        """
        Save checkpoint file (REQ-011).
        """
        try:
            checkpoint_data = {
                "processed_files": list(self.processed_files),
                "stats": self.stats,
            }
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f)
            if self.verbose <= 12:  # TRACE level
                logger.debug(f"REQ-011: Saved checkpoint to {self.checkpoint_file}")
        except Exception as e:
            logger.warning(f"REQ-011: Failed to save checkpoint: {e}")

    def _initialize_components(self) -> None:
        """
        Initialize all processing components (REQ-002).
        """
        logger.info("REQ-002: Initializing components...")

        # REQ-003: Initialize EXIF extractor
        try:
            self.exif_extractor = get_exif_extractor()
            logger.info("REQ-003: EXIF extractor initialized")
        except Exception as e:
            logger.warning(f"REQ-003: EXIF extractor not available: {e}")

        # REQ-007: Initialize face detector
        try:
            self.face_detector = get_face_detector(self.device)
            logger.info("REQ-007: Face detector initialized")
        except Exception as e:
            logger.warning(f"REQ-007: Face detector not available: {e}")

        # REQ-008: Initialize object detector
        try:
            self.object_detector = get_object_detector(self.device, "yolo12x.pt")
            logger.info("REQ-008: Object detector initialized")
        except Exception as e:
            logger.warning(f"REQ-008: Object detector not available: {e}")

        # REQ-009: Initialize pose detector
        try:
            self.pose_detector = get_pose_detector(self.device, "yolo11x-pose.pt")
            logger.info("REQ-009: Pose detector initialized")
        except Exception as e:
            logger.warning(f"REQ-009: Pose detector not available: {e}")

        # REQ-004: Initialize sidecar generator
        try:
            self.sidecar_generator = get_sidecar_generator(self.output_dir)
            logger.info("REQ-004: Sidecar generator initialized")
        except Exception as e:
            logger.warning(f"REQ-004: Sidecar generator not available: {e}")

    def _get_image_files(self) -> list[Path]:
        """
        Get list of image files to process (REQ-018, REQ-038).

        Returns:
            List of image file paths, optionally limited by --limit flag.
        """
        extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".raw", ".cr2", ".nef", ".arw"}
        images: list[Path] = []

        for ext in extensions:
            images.extend(self.input_dir.rglob(f"*{ext}"))
            images.extend(self.input_dir.rglob(f"*{ext.upper()}"))

        # REQ-038: Apply limit if specified
        if self.limit and len(images) > self.limit:
            images = images[: self.limit]
            logger.info(f"REQ-038: Limited to {self.limit} images from {len(images)} total")

        return images

    def _store_to_database(self, image_path: Path, metadata: dict[str, Any]) -> bool:
        """
        Store image metadata to database.

        REQ-025: Store image metadata to database using PonyORM.

        Args:
            image_path: Path to the image file.
            metadata: Extracted metadata dictionary.

        Returns:
            True if successful, False otherwise.
        """
        if not self.database_connection:
            return True

        try:
            from pony.orm import db_session

            from media_indexer.db.exif import EXIFData as DBEXIFData
            from media_indexer.db.face import Face as DBFace
            from media_indexer.db.hash_util import get_file_size
            from media_indexer.db.image import Image as DBImage
            from media_indexer.db.object import Object as DBObject
            from media_indexer.db.pose import Pose as DBPose

            with db_session:
                # REQ-028: Calculate file hash for deduplication
                file_hash = calculate_file_hash(image_path)
                file_size = get_file_size(image_path) or 0

                # Check if image already exists in database
                existing_image = DBImage.get_by_path(str(image_path))
                if existing_image:
                    logger.debug(f"REQ-024: Image {image_path} already exists in database, skipping")
                    return True

                # REQ-024: Create Image entity
                from datetime import datetime

                db_image = DBImage(
                    path=str(image_path),
                    file_hash=file_hash,
                    file_size=file_size,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

                # REQ-024: Store faces
                if "faces" in metadata:
                    for face_data in metadata["faces"]:
                        # REQ-066: Handle optional embedding field
                        # PonyORM doesn't accept None for Optional(Json), so only set if present
                        face_kwargs = {
                            "image": db_image,
                            "confidence": face_data.get("confidence", 0.0),
                            "bbox": face_data.get("bbox", []),
                            "model": face_data.get("model", "unknown"),
                        }
                        embedding = face_data.get("embedding")
                        if embedding is not None:
                            face_kwargs["embedding"] = embedding
                        
                        DBFace(**face_kwargs)

                # REQ-024: Store objects
                if "objects" in metadata:
                    for obj_data in metadata["objects"]:
                        DBObject(
                            image=db_image,
                            class_id=obj_data.get("class_id", -1),
                            class_name=obj_data.get("class_name", "unknown"),
                            confidence=obj_data.get("confidence", 0.0),
                            bbox=obj_data.get("bbox", []),
                        )

                # REQ-024: Store poses
                if "poses" in metadata:
                    for pose_data in metadata["poses"]:
                        # REQ-066: Handle optional keypoints_conf field
                        # PonyORM doesn't accept None for Optional(Json), so only set if present
                        pose_kwargs = {
                            "image": db_image,
                            "confidence": pose_data.get("confidence", 0.0),
                            "bbox": pose_data.get("bbox", []),
                            "keypoints": pose_data.get("keypoints", []),
                        }
                        keypoints_conf = pose_data.get("keypoints_conf")
                        if keypoints_conf is not None:
                            pose_kwargs["keypoints_conf"] = keypoints_conf
                        
                        DBPose(**pose_kwargs)

                # REQ-024: Store EXIF data
                if "exif" in metadata and metadata["exif"]:
                    DBEXIFData(
                        image=db_image,
                        data=metadata["exif"],
                    )

                logger.debug(f"REQ-025: Stored metadata for {image_path} in database")
                return True

        except Exception as e:
            logger.error(f"REQ-025: Failed to store to database: {e}")
            return False

    def _process_single_image(self, image_path: Path) -> bool:
        """
        Process a single image (REQ-002, REQ-015).

        Args:
            image_path: Path to image file.

        Returns:
            True if successful, False otherwise.
        """
        # REQ-013: Check if already processed (idempotent)
        # REQ-025: Check database if using database storage
        if self.database_connection:
            try:
                from pony.orm import db_session

                from media_indexer.db.image import Image as DBImage

                with db_session:
                    existing_image = DBImage.get_by_path(str(image_path))
                    if existing_image:
                        logger.debug(f"REQ-013: Image {image_path} already exists in database, skipping")
                        self.stats["skipped_images"] += 1
                        return True
            except Exception as e:
                logger.warning(f"REQ-025: Database check failed: {e}")

        # Check for sidecar in output directory - image-sidecar-rust uses .json extension
        sidecar_path = self.output_dir / (image_path.name + '.json')

        if sidecar_path.exists():
            logger.debug(f"REQ-013: Skipping already processed {image_path}")
            self.stats["skipped_images"] += 1
            return True

        try:
            metadata: dict[str, Any] = {"image_path": str(image_path)}

            # REQ-003: Extract EXIF
            if self.exif_extractor is not None:
                try:
                    metadata["exif"] = self.exif_extractor.extract_from_path(image_path)
                except Exception as e:
                    logger.debug(f"REQ-003: EXIF extraction failed: {e}")

            # REQ-007: Detect faces
            if self.face_detector is not None:
                try:
                    metadata["faces"] = self.face_detector.detect_faces(image_path)
                except Exception as e:
                    logger.debug(f"REQ-007: Face detection failed: {e}")

            # REQ-008: Detect objects
            if self.object_detector is not None:
                try:
                    metadata["objects"] = self.object_detector.detect_objects(image_path)
                except Exception as e:
                    logger.debug(f"REQ-008: Object detection failed: {e}")

            # REQ-009: Detect poses
            if self.pose_detector is not None:
                try:
                    metadata["poses"] = self.pose_detector.detect_poses(image_path)
                except Exception as e:
                    logger.debug(f"REQ-009: Pose detection failed: {e}")

            # REQ-025: Store to database
            if self.database_connection:
                db_success = self._store_to_database(image_path, metadata)
                if not db_success:
                    logger.error(f"REQ-025: Failed to store {image_path} to database")

            # REQ-027: Generate sidecar if not disabled
            if not self.disable_sidecar and self.sidecar_generator is not None:
                try:
                    sidecar_path = self.sidecar_generator.generate_sidecar(image_path, metadata)
                    logger.debug(f"REQ-027: Generated sidecar for {image_path}")
                except Exception as e:
                    logger.error(f"REQ-004: Sidecar generation failed: {e}")
                    if self.database_connection:
                        # If using database, sidecar failure is not fatal
                        self.processed_files.add(str(image_path))
                        return True
                    else:
                        return False
            elif self.disable_sidecar:
                logger.debug(f"REQ-026: Skipping sidecar generation for {image_path}")

            self.processed_files.add(str(image_path))
            return True

        except Exception as e:
            # REQ-015: Robust error handling
            logger.error(f"REQ-015: Error processing {image_path}: {e}")
            return False

    def process(self) -> dict[str, Any]:
        """
        Process all images with parallel/batch processing (REQ-002, REQ-012, REQ-014, REQ-020).

        Returns:
            Dictionary with processing statistics.
        """
        logger.info(f"REQ-002: Starting image processing with batch size {self.batch_size}")

        # Initialize components
        self._initialize_components()

        # Get image files
        images = self._get_image_files()
        self.stats["total_images"] = len(images)
        logger.info(f"REQ-002: Found {len(images)} images to process")

        # Filter out already processed images
        images_to_process = [img for img in images if str(img) not in self.processed_files]

        if not images_to_process:
            logger.info("REQ-013: All images already processed")
            return self.stats

        # REQ-012: Progress tracking with TQDM if verbose level <= 12
        progress_bar = tqdm.tqdm(total=len(images_to_process), desc="Processing images") if self.verbose <= 12 else None

        # REQ-020: Process images in batches with threading for I/O
        try:
            # Process in batches
            for i in range(0, len(images_to_process), self.batch_size):
                batch = images_to_process[i : i + self.batch_size]

                # REQ-015: Robust error handling with thread pool
                with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                    futures = {executor.submit(self._process_single_image, img): img for img in batch}

                    for future in as_completed(futures):
                        image_path = futures[future]
                        try:
                            success = future.result()
                            if success:
                                self._update_stats_increment("processed_images")
                            else:
                                self._update_stats_increment("error_images")
                        except Exception as e:
                            logger.error(f"REQ-015: Error processing {image_path}: {e}")
                            self._update_stats_increment("error_images")

                        # Update progress bar
                        if progress_bar:
                            progress_bar.update(1)

                # REQ-011: Save checkpoint periodically
                if self.stats["processed_images"] % 100 == 0:
                    self._save_checkpoint()
        finally:
            if progress_bar:
                progress_bar.close()

        # REQ-011: Final checkpoint save
        self._save_checkpoint()

        # REQ-012: Final statistics
        self.stats["end_time"] = datetime.now().isoformat()
        self._print_statistics()

        # REQ-040: Clean up temporary RAW conversion files
        cleanup_temp_files()

        # REQ-022: Close database connection
        if self.database_connection:
            self.database_connection.close()

        return self.stats

    def _update_stats_increment(self, key: str) -> None:
        """
        Thread-safe stats update.

        Args:
            key: Stat key to increment.
        """
        with self.stats_lock:
            self.stats[key] += 1

    def _print_statistics(self) -> None:
        """
        Print processing statistics (REQ-012).
        """
        logger.info("REQ-012: Processing complete")
        logger.info(f"  Total images: {self.stats['total_images']}")
        logger.info(f"  Processed: {self.stats['processed_images']}")
        logger.info(f"  Skipped: {self.stats['skipped_images']}")
        logger.info(f"  Errors: {self.stats['error_images']}")
        logger.info(f"  Start time: {self.stats['start_time']}")
        logger.info(f"  End time: {self.stats['end_time']}")
