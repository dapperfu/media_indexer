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
from datetime import datetime
from pathlib import Path
from typing import Any

import tqdm

from media_indexer.exif_extractor import EXIFExtractor, get_exif_extractor
from media_indexer.face_detector import FaceDetector, get_face_detector
from media_indexer.gpu_validator import GPUValidator, get_gpu_validator
from media_indexer.object_detector import ObjectDetector, get_object_detector
from media_indexer.pose_detector import PoseDetector, get_pose_detector
from media_indexer.sidecar_generator import SidecarGenerator, get_sidecar_generator

try:
    import image_sidecar_rust
except ImportError:
    image_sidecar_rust = None  # type: ignore[assignment, misc]

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
    ) -> None:
        """
        Initialize image processor.

        REQ-006: Ensure GPU-only operation.
        REQ-016: Multi-level verbosity.

        Args:
            input_dir: Directory containing images to process.
            output_dir: Directory for sidecar files (defaults to input_dir).
            checkpoint_file: Path to checkpoint file (REQ-011).
            verbose: Verbosity level (REQ-016).
            batch_size: Batch size for processing (REQ-014).

        Raises:
            RuntimeError: If no GPU is available (REQ-006).
        """
        # REQ-006: Validate GPU availability
        self.gpu_validator: GPUValidator = get_gpu_validator()
        self.device = self.gpu_validator.get_device()

        # REQ-002: Setup input/output directories
        self.input_dir: Path = Path(input_dir)
        if output_dir is None:
            self.output_dir: Path = self.input_dir
        else:
            self.output_dir = Path(output_dir)

        # REQ-011: Setup checkpoint file
        self.checkpoint_file: Path = checkpoint_file if checkpoint_file else Path(".checkpoint.json")
        self.processed_files: set[str] = set()

        # REQ-016: Setup verbosity
        self.verbose = verbose

        # REQ-014: Batch size configuration
        self.batch_size = batch_size

        # REQ-012: Statistics tracking
        self.stats: dict[str, Any] = {
            "total_images": 0,
            "processed_images": 0,
            "skipped_images": 0,
            "error_images": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
        }

        # REQ-002: Initialize components
        self.exif_extractor: EXIFExtractor | None = None
        self.face_detector: FaceDetector | None = None
        self.object_detector: ObjectDetector | None = None
        self.pose_detector: PoseDetector | None = None
        self.sidecar_generator: SidecarGenerator | None = None

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
                    logger.info(
                        f"REQ-011: Loaded {len(self.processed_files)} processed files from checkpoint"
                    )
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
            self.pose_detector = get_pose_detector(self.device, "yolo12-pose.pt")
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
        Get list of image files to process (REQ-018).

        Returns:
            List of image file paths.
        """
        extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".raw", ".cr2", ".nef", ".arw"}
        images: list[Path] = []

        for ext in extensions:
            images.extend(self.input_dir.glob(f"*{ext}"))
            images.extend(self.input_dir.glob(f"*{ext.upper()}"))

        return images

    def _process_single_image(self, image_path: Path) -> bool:
        """
        Process a single image (REQ-002, REQ-015).

        Args:
            image_path: Path to image file.

        Returns:
            True if successful, False otherwise.
        """
        # REQ-013: Check if already processed (idempotent)
        # Check for sidecar in output directory using image-sidecar-rust's naming
        # The library determines the sidecar filename based on format
        try:
            sidecar_filename = image_sidecar_rust.get_sidecar_filename(str(image_path))  # type: ignore[misc, arg-type]
            sidecar_path = self.output_dir / sidecar_filename
        except Exception:
            # Fallback if library not available
            image_filename = image_path.name
            sidecar_path = self.output_dir / image_filename
            
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

            # REQ-004: Generate sidecar
            if self.sidecar_generator is not None:
                try:
                    sidecar_path = self.sidecar_generator.generate_sidecar(image_path, metadata)
                    logger.debug(f"REQ-004: Generated sidecar for {image_path}")
                except Exception as e:
                    logger.error(f"REQ-004: Sidecar generation failed: {e}")
                    return False

            self.processed_files.add(str(image_path))
            self.stats["processed_images"] += 1
            return True

        except Exception as e:
            # REQ-015: Robust error handling
            logger.error(f"REQ-015: Error processing {image_path}: {e}")
            self.stats["error_images"] += 1
            return False

    def process(self) -> dict[str, Any]:
        """
        Process all images (REQ-002, REQ-012).

        Returns:
            Dictionary with processing statistics.
        """
        logger.info("REQ-002: Starting image processing")

        # Initialize components
        self._initialize_components()

        # Get image files
        images = self._get_image_files()
        self.stats["total_images"] = len(images)
        logger.info(f"REQ-002: Found {len(images)} images to process")

        # REQ-012: Progress tracking with TQDM if verbose level <= 12
        if self.verbose <= 12:
            images_iter = tqdm.tqdm(images, desc="Processing images")
        else:
            images_iter = images

        # Process images
        for image_path in images_iter:
            # REQ-013: Skip if already processed
            if str(image_path) in self.processed_files:
                continue

            self._process_single_image(image_path)

            # REQ-011: Save checkpoint periodically
            if self.stats["processed_images"] % 100 == 0:
                self._save_checkpoint()

        # REQ-011: Final checkpoint save
        self._save_checkpoint()

        # REQ-012: Final statistics
        self.stats["end_time"] = datetime.now().isoformat()
        self._print_statistics()

        return self.stats

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

