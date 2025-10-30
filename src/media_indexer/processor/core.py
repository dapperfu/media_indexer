"""
Core Image Processor

REQ-002: Process all images in collection and generate sidecar files.
REQ-006: GPU-only operation enforcement.
REQ-010: All code components directly linked to requirements.
REQ-012: Progress tracking and statistics.
REQ-013: Idempotent processing using sidecar files.
REQ-015: Robust error handling.
REQ-016: Multi-level verbosity logging.
"""

import logging
from pathlib import Path
from typing import Any

from media_indexer.db.connection import DatabaseConnection
from media_indexer.exif_extractor import EXIFExtractor, get_exif_extractor
from media_indexer.face_detector import FaceDetector, get_face_detector
from media_indexer.gpu_validator import GPUValidator, get_gpu_validator
from media_indexer.object_detector import ObjectDetector, get_object_detector
from media_indexer.pose_detector import PoseDetector, get_pose_detector
from media_indexer.processor.analysis_scanner import AnalysisScanner
from media_indexer.processor.core_pipeline import run_processing
from media_indexer.processor.image_processor import ImageProcessorComponent
from media_indexer.processor.statistics import StatisticsTracker
from media_indexer.sidecar_generator import SidecarGenerator, get_sidecar_generator
from media_indexer.utils.cancellation import CancellationManager
from media_indexer.utils.file_utils import get_image_extensions

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Main image processor orchestrating all components.

    REQ-002: Process images and generate sidecar files with metadata.
    REQ-012: Track progress and statistics.
    REQ-013: Implement idempotent processing using sidecar files.
    """

    def __init__(
        self,
        input_dir: Path,
        verbose: int = 20,
        batch_size: int = 1,
        database_path: Path | None = None,
        disable_sidecar: bool = False,
        limit: int | None = None,
        force: bool = False,
        scan_workers: int = 8,
        enable_face_attributes: bool = True,
    ) -> None:
        """
        Initialize image processor.

        REQ-006: Ensure GPU-only operation.
        REQ-016: Multi-level verbosity.
        REQ-025: Support database storage.
        REQ-026: Support disabling sidecar generation.
        REQ-038: Support limiting number of images to process.
        REQ-020: Parallel processing with configurable workers.

        Args:
            input_dir: Directory containing images to process.
            verbose: Verbosity level (REQ-016).
            batch_size: Batch size for processing (REQ-014).
            database_path: Path to database file (REQ-025).
            disable_sidecar: Disable sidecar generation when using database.
            limit: Maximum number of images to process (REQ-038).
            force: Force reprocessing even if analyses already exist.
            scan_workers: Number of parallel workers for sidecar scanning.
            enable_face_attributes: Enable DeepFace age/emotion enrichment (REQ-081, default True).

        Raises:
            RuntimeError: If no GPU is available (REQ-006).
        """
        # REQ-006: Validate GPU availability
        self.gpu_validator: GPUValidator = get_gpu_validator()
        self.device = self.gpu_validator.get_device()

        # REQ-002: Setup input directory
        self.input_dir: Path = Path(input_dir).resolve()

        # REQ-016: Setup verbosity
        self.verbose = verbose

        # REQ-014: Batch size configuration
        self.batch_size = batch_size
        if self.batch_size == 1:
            self.batch_size = 4  # Default for better GPU utilization

        # REQ-038: Setup image limit
        self.limit = limit
        if self.limit:
            logger.info(f"REQ-038: Limiting processing to {self.limit} images")

        # REQ-013: Force reprocessing flag
        self.force = force

        # REQ-020: Workers for parallel sidecar scanning
        self.scan_workers = scan_workers
        self.enable_face_attributes = enable_face_attributes

        # REQ-012: Statistics tracking
        self.stats_tracker = StatisticsTracker()

        # REQ-002: Initialize components (will be initialized later)
        self.exif_extractor: EXIFExtractor | None = None
        self.face_detector: FaceDetector | None = None
        self.face_attribute_analyzer: Any | None = None
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

        # Initialize analysis scanner
        self.analysis_scanner = AnalysisScanner(
            self.database_connection,
            None,  # Will be set after initialization
            self.disable_sidecar,
            self.force,
        )

        # REQ-086: Propagate disable_sidecar flag to downstream components
        # Initialize image processor component
        self.image_processor = ImageProcessorComponent(
            None,  # exif_extractor - will be set after initialization
            None,  # face_detector - will be set after initialization
            None,  # object_detector - will be set after initialization
            None,  # pose_detector - will be set after initialization
            None,  # sidecar_generator - will be set after initialization
            self.database_connection,
            self.disable_sidecar,
            None,  # face_attribute_analyzer - will be set after initialization (optional)
        )

    def _get_required_analyses(self) -> set[str]:
        """
        Get set of required analyses based on initialized components.

        REQ-013: Determine which analyses should be performed.

        Returns:
            Set of analysis types: 'exif', 'faces', 'objects', 'poses'.
        """
        required: set[str] = set()
        if self.exif_extractor is not None:
            required.add("exif")
        if self.face_detector is not None:
            required.add("faces")
        if self.object_detector is not None:
            required.add("objects")
        if self.pose_detector is not None:
            required.add("poses")
        return required

    def _initialize_components(self) -> None:
        """
        Initialize all processing components.

        REQ-002: Initialize EXIF, face, object, pose detectors and sidecar generator.
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

        if self.enable_face_attributes and self.face_detector is not None:
            try:
                from media_indexer.analytics.face_attribute_analyzer import get_face_attribute_analyzer

                self.face_attribute_analyzer = get_face_attribute_analyzer()
                logger.info("REQ-081: Face attribute analyzer initialized")
            except Exception as e:
                logger.warning(f"REQ-081: Face attribute analyzer unavailable: {e}")

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
            self.sidecar_generator = get_sidecar_generator()
            logger.info("REQ-004: Sidecar generator initialized")
        except Exception as e:
            logger.warning(f"REQ-004: Sidecar generator not available: {e}")

        # Update components with initialized detectors
        self.analysis_scanner.sidecar_generator = self.sidecar_generator
        self.image_processor.exif_extractor = self.exif_extractor
        self.image_processor.face_detector = self.face_detector
        self.image_processor.face_attribute_analyzer = self.face_attribute_analyzer
        self.image_processor.object_detector = self.object_detector
        self.image_processor.pose_detector = self.pose_detector
        self.image_processor.sidecar_generator = self.sidecar_generator

    def _get_image_files(self) -> list[Path]:
        """
        Get list of image files to process.

        REQ-018, REQ-038: Find image files, optionally limited.

        Returns:
            List of image file paths, optionally limited by --limit flag.
        """
        extensions = get_image_extensions()
        images: list[Path] = []

        for ext in extensions:
            images.extend(self.input_dir.rglob(f"*{ext}"))
            images.extend(self.input_dir.rglob(f"*{ext.upper()}"))

        # REQ-038: Apply limit if specified
        if self.limit and len(images) > self.limit:
            images = images[: self.limit]
            logger.info(f"REQ-038: Limited to {self.limit} images from {len(images)} total")

        return images

    def _needs_processing(self, image_path: Path) -> tuple[bool, set[str]]:
        """
        Determine if image needs processing and which analyses are needed.

        REQ-013: Check if image needs processing based on existing data.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (needs_processing, missing_analyses).
        """
        if self.force:
            required = self._get_required_analyses()
            return True, required

        required = self._get_required_analyses()
        existing = self.analysis_scanner.get_existing_analyses(image_path, required)
        missing = required - existing

        return len(missing) > 0, missing

    def process(self) -> dict[str, Any]:
        """Run the full processing pipeline."""

        return run_processing(self)

    def _process_image_wrapper(
        self,
        image_path: Path,
        missing_analyses: set[str],
        cancellation_manager: CancellationManager,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Wrapper for processing single image with cancellation check.

        Args:
            image_path: Path to image file.
            missing_analyses: Set of analysis types to perform.
            cancellation_manager: Cancellation manager instance.

        Returns:
            Tuple of (success, detection_results).
        """
        if cancellation_manager.is_cancelled():
            return False, {}

        sidecar_path = self.analysis_scanner.find_sidecar_path(image_path)
        return self.image_processor.process_single_image(image_path, missing_analyses, sidecar_path)
