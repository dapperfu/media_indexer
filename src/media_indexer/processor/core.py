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
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from media_indexer.db.connection import DatabaseConnection
from media_indexer.exif_extractor import EXIFExtractor, get_exif_extractor
from media_indexer.face_detector import FaceDetector, get_face_detector
from media_indexer.gpu_validator import GPUValidator, get_gpu_validator
from media_indexer.object_detector import ObjectDetector, get_object_detector
from media_indexer.pose_detector import PoseDetector, get_pose_detector
from media_indexer.raw_converter import cleanup_temp_files
from media_indexer.sidecar_generator import SidecarGenerator, get_sidecar_generator
from media_indexer.utils.cancellation import CancellationManager
from media_indexer.utils.file_utils import get_image_extensions
from media_indexer.utils.sidecar_utils import read_sidecar_metadata

from media_indexer.processor.analysis_scanner import AnalysisScanner
from media_indexer.processor.image_processor import ImageProcessorComponent
from media_indexer.processor.progress import create_rich_progress_bar
from media_indexer.processor.statistics import StatisticsTracker

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

        # REQ-012: Statistics tracking
        self.stats_tracker = StatisticsTracker()

        # REQ-002: Initialize components (will be initialized later)
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
            logger.warning(
                "REQ-026: --no-sidecar specified without --db, ignoring flag"
            )
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

        # Initialize image processor component
        self.image_processor = ImageProcessorComponent(
            None,  # Will be set after initialization
            None,
            None,
            None,
            None,  # Will be set after initialization
            self.database_connection,
            self.disable_sidecar,
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
            logger.info(
                f"REQ-038: Limited to {self.limit} images from {len(images)} total"
            )

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
        existing = self.analysis_scanner.get_existing_analyses(
            image_path, required
        )
        missing = required - existing

        return len(missing) > 0, missing

    def process(self) -> dict[str, Any]:
        """
        Process all images with parallel/batch processing.

        REQ-002, REQ-012, REQ-014, REQ-020: Process images with batching.
        REQ-013: Scan sidecar files to determine which analyses are needed.
        REQ-015: Handle user interruption gracefully with quick shutdown.

        Returns:
            Dictionary with processing statistics.
        """
        # REQ-015: Setup cancellation manager
        cancellation_manager = CancellationManager()
        cancellation_manager.setup_signal_handler()

        try:
            logger.info(
                f"REQ-002: Starting image processing with batch size {self.batch_size}"
            )

            # Initialize components first
            self._initialize_components()

            # Get image files
            images = self._get_image_files()
            self.stats_tracker.set("total_images", len(images))
            logger.info(f"REQ-002: Found {len(images)} images")

            # REQ-013: Scan sidecar files or database to determine which images need processing
            images_to_process: list[tuple[Path, set[str]]] = []
            skipped_count = 0

            # REQ-025: Use database batch query if database is available
            if self.database_connection:
                logger.info("REQ-013: Querying database for existing analyses...")
                # REQ-012: Use Rich progress bar for multi-line display
                use_rich_query = self.verbose >= 15
                if use_rich_query:
                    progress_query, display_query, live_query = create_rich_progress_bar(
                        total=len(images),
                        desc="Querying database",
                        unit="file",
                        verbose=self.verbose,
                        show_detections=False,
                    )
                    task_id_query = progress_query.add_task(
                        "Querying database",
                        total=len(images),
                        avg_speed="0.0 file/s",
                    )
                    live_query.start()
                    progress_bar_query = None
                else:
                    progress_query = None
                    display_query = None
                    live_query = None
                    task_id_query = None
                    progress_bar_query = None

                try:
                    # Query database in batch
                    db_analyses = (
                        self.analysis_scanner.get_existing_analyses_from_database_batch(
                            images
                        )
                    )

                    required = self._get_required_analyses()

                    for image_path in images:
                        if cancellation_manager.is_cancelled():
                            logger.warning("REQ-015: Processing interrupted by user")
                            break

                        path_str = str(image_path)
                        existing = db_analyses.get(path_str, set()).copy()  # Copy to avoid modifying shared set

                        # Also check sidecar files if not disabled
                        if not self.disable_sidecar:
                            sidecar_path = self.analysis_scanner.find_sidecar_path(
                                image_path
                            )
                            if sidecar_path and sidecar_path.exists():
                                try:
                                    metadata = read_sidecar_metadata(
                                        sidecar_path, self.sidecar_generator
                                    )

                                    # REQ-040: Check for raw_conversion_failed flag
                                    # If flag is set and force is False, skip processing
                                    if metadata.get("raw_conversion_failed") and not self.force:
                                        logger.debug(
                                            f"REQ-040: Skipping {image_path} - RAW conversion failed previously (use --force to retry)"
                                        )
                                        skipped_count += 1
                                        if use_rich_query and progress_query and display_query:
                                            elapsed = time.time() - progress_query._start_time  # type: ignore[attr-defined]
                                            progress_query._processed_count += 1  # type: ignore[attr-defined]
                                            if elapsed > 0:
                                                avg_speed = progress_query._processed_count / elapsed  # type: ignore[attr-defined]
                                                avg_str = f"{avg_speed:.1f} file/s"
                                            else:
                                                avg_str = "0.0 file/s"
                                            img_name = image_path.name
                                            if len(img_name) > 50:
                                                img_name = "..." + img_name[-47:]
                                            progress_query.update(
                                                task_id_query,
                                                advance=1,
                                                avg_speed=avg_str,
                                            )
                                            display_query.update_info(
                                                current_file=f"📁 {img_name}",
                                                avg_speed=avg_str,
                                            )
                                            live_query.update(display_query)
                                        continue

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
                                    logger.debug(
                                        f"REQ-013: Failed to read sidecar for {image_path}: {e}"
                                    )

                        missing = required - existing

                        if len(missing) > 0:
                            images_to_process.append((image_path, missing))
                        else:
                            skipped_count += 1

                        if use_rich_query and progress_query and display_query:
                            elapsed = time.time() - progress_query._start_time  # type: ignore[attr-defined]
                            progress_query._processed_count += 1  # type: ignore[attr-defined]
                            if elapsed > 0:
                                avg_speed = progress_query._processed_count / elapsed  # type: ignore[attr-defined]
                                avg_str = f"{avg_speed:.1f} file/s"
                            else:
                                avg_str = "0.0 file/s"
                            img_name = image_path.name
                            if len(img_name) > 50:
                                img_name = "..." + img_name[-47:]
                            progress_query.update(
                                task_id_query,
                                advance=1,
                                avg_speed=avg_str,
                            )
                            display_query.update_info(
                                current_file=f"📁 {img_name}",
                                avg_speed=avg_str,
                            )
                            live_query.update(display_query)
                finally:
                    if use_rich_query and live_query:
                        live_query.stop()
                    elif progress_bar_query:
                        progress_bar_query.close()
            else:
                # REQ-020: Parallel scanning with progress bar (file-based)
                logger.info("REQ-013: Scanning sidecar files...")
                scan_workers = min(self.scan_workers, len(images))
                # REQ-012: Use Rich progress bar for multi-line display
                use_rich_scan = self.verbose >= 15
                if use_rich_scan:
                    progress_scan, display_scan, live_scan = create_rich_progress_bar(
                        total=len(images),
                        desc="Scanning sidecars",
                        unit="file",
                        verbose=self.verbose,
                        show_detections=False,
                    )
                    task_id_scan = progress_scan.add_task(
                        "Scanning sidecars",
                        total=len(images),
                        avg_speed="0.0 file/s",
                    )
                    live_scan.start()
                    progress_bar_scan = None
                else:
                    progress_scan = None
                    display_scan = None
                    live_scan = None
                    task_id_scan = None
                    progress_bar_scan = None

                try:
                    if scan_workers == 1:
                        # Sequential scanning
                        for image_path in images:
                            if cancellation_manager.is_cancelled():
                                logger.warning(
                                    "REQ-015: Processing interrupted by user"
                                )
                                break
                            needs_processing, missing_analyses = (
                                self._needs_processing(image_path)
                            )
                            if needs_processing:
                                images_to_process.append((image_path, missing_analyses))
                            else:
                                skipped_count += 1
                            if use_rich_scan and progress_scan and display_scan:
                                elapsed = time.time() - progress_scan._start_time  # type: ignore[attr-defined]
                                progress_scan._processed_count += 1  # type: ignore[attr-defined]
                                if elapsed > 0:
                                    avg_speed = progress_scan._processed_count / elapsed  # type: ignore[attr-defined]
                                    avg_str = f"{avg_speed:.1f} file/s"
                                else:
                                    avg_str = "0.0 file/s"
                                img_name = image_path.name
                                if len(img_name) > 50:
                                    img_name = "..." + img_name[-47:]
                                progress_scan.update(
                                    task_id_scan,
                                    advance=1,
                                    avg_speed=avg_str,
                                )
                                display_scan.update_info(
                                    current_file=f"📁 {img_name}",
                                    avg_speed=avg_str,
                                )
                                live_scan.update(display_scan)
                    else:
                        # REQ-020: Parallel scanning with thread pool
                        executor: ThreadPoolExecutor | None = None
                        try:
                            executor = ThreadPoolExecutor(max_workers=scan_workers)
                            futures = {
                                executor.submit(self._needs_processing, image_path): image_path
                                for image_path in images
                            }

                            for future in as_completed(futures):
                                if cancellation_manager.is_cancelled():
                                    logger.warning(
                                        "REQ-015: Processing interrupted by user"
                                    )
                                    for f in futures:
                                        if not f.done():
                                            f.cancel()
                                    break

                                image_path = futures[future]
                                try:
                                    needs_processing, missing_analyses = (
                                        future.result()
                                    )
                                    if needs_processing:
                                        images_to_process.append(
                                            (image_path, missing_analyses)
                                        )
                                    else:
                                        skipped_count += 1
                                except CancelledError:
                                    pass
                                except Exception as e:
                                    logger.debug(
                                        f"REQ-013: Failed to scan {image_path}: {e}"
                                    )
                                    images_to_process.append(
                                        (image_path, self._get_required_analyses())
                                    )
                                finally:
                                    if use_rich_scan and progress_scan and display_scan:
                                        elapsed = time.time() - progress_scan._start_time  # type: ignore[attr-defined]
                                        progress_scan._processed_count += 1  # type: ignore[attr-defined]
                                        if elapsed > 0:
                                            avg_speed = progress_scan._processed_count / elapsed  # type: ignore[attr-defined]
                                            avg_str = f"{avg_speed:.1f} file/s"
                                        else:
                                            avg_str = "0.0 file/s"
                                        img_name = image_path.name
                                        if len(img_name) > 50:
                                            img_name = "..." + img_name[-47:]
                                        progress_scan.update(
                                            task_id_scan,
                                            advance=1,
                                            avg_speed=avg_str,
                                        )
                                        display_scan.update_info(
                                            current_file=f"📁 {img_name}",
                                            avg_speed=avg_str,
                                        )
                                        live_scan.update(display_scan)
                        finally:
                            if executor:
                                if cancellation_manager.is_cancelled():
                                    executor.shutdown(wait=False, cancel_futures=True)
                                else:
                                    executor.shutdown(wait=True)
                except KeyboardInterrupt:
                    logger.warning("REQ-015: Processing interrupted by user")
                    cancellation_manager.cancel()
                finally:
                    if use_rich_scan and live_scan:
                        live_scan.stop()
                    elif progress_bar_scan:
                        progress_bar_scan.close()

            self.stats_tracker.set("skipped_images", skipped_count)

            # REQ-013: Show summary before processing
            required_analyses = self._get_required_analyses()
            analysis_names = {
                "exif": "EXIF",
                "faces": "faces",
                "objects": "objects",
                "poses": "poses",
            }
            required_str = ", ".join(
                analysis_names.get(a, a) for a in sorted(required_analyses)
            )
            logger.info(f"REQ-013: Required analyses: {required_str}")
            logger.info(f"REQ-013: Already analyzed: {skipped_count} images")
            logger.info(f"REQ-013: Needs processing: {len(images_to_process)} images")

            if not images_to_process:
                logger.info("REQ-013: All images already have complete analyses")
                self.stats_tracker.finalize()
                self.stats_tracker.print_statistics()
                return self.stats_tracker.get_stats()

            # REQ-012: Progress tracking
            def format_detection_summary(detections: dict[str, Any]) -> str:
                """Format detection info for display."""
                parts = []
                if detections.get("faces", 0) > 0:
                    parts.append(
                        f"{detections['faces']} face{'s' if detections['faces'] != 1 else ''}"
                    )
                if detections.get("objects", 0) > 0:
                    parts.append(
                        f"{detections['objects']} object{'s' if detections['objects'] != 1 else ''}"
                    )
                if detections.get("poses", 0) > 0:
                    parts.append(
                        f"{detections['poses']} pose{'s' if detections['poses'] != 1 else ''}"
                    )
                return ", ".join(parts) if parts else "no detections"

            # REQ-012: Progress tracking with Rich for multi-line detection info
            # REQ-016: Use Rich when verbose >= 15 for progress display
            use_rich = self.verbose >= 15
            if use_rich:
                progress, display, live = create_rich_progress_bar(
                    total=len(images_to_process),
                    desc="Processing images",
                    unit="img",
                    verbose=self.verbose,
                    show_detections=True,
                )
                progress_bar = None
                task_id = progress.add_task(
                    "Processing images",
                    total=len(images_to_process),
                    avg_speed="0.0 img/s",
                )
                live.start()
            else:
                # No progress bar when verbose < 15
                progress = None
                display = None
                live = None
                progress_bar = None
                task_id = None

            # REQ-020: Process images in batches with threading for I/O
            try:
                for i in range(0, len(images_to_process), self.batch_size):
                    if cancellation_manager.is_cancelled():
                        logger.warning("REQ-015: Processing interrupted by user")
                        break

                    batch = images_to_process[i : i + self.batch_size]

                    executor: ThreadPoolExecutor | None = None
                    try:
                        executor = ThreadPoolExecutor(max_workers=self.batch_size)
                        futures = {
                            executor.submit(
                                self._process_image_wrapper,
                                img_path,
                                missing,
                                cancellation_manager,
                            ): img_path
                            for img_path, missing in batch
                        }

                        for future in as_completed(futures):
                            if cancellation_manager.is_cancelled():
                                logger.warning(
                                    "REQ-015: Processing interrupted by user"
                                )
                                for f in futures:
                                    if not f.done():
                                        f.cancel()
                                break

                            image_path = futures[future]
                            try:
                                success, detections = future.result()
                                if success:
                                    self.stats_tracker.update_increment(
                                        "processed_images"
                                    )
                                    img_name = image_path.name
                                    if len(img_name) > 50:
                                        img_name = "..." + img_name[-47:]
                                    det_summary = format_detection_summary(detections)

                                    if use_rich and progress and display:
                                        elapsed = time.time() - progress._start_time  # type: ignore[attr-defined]
                                        progress._processed_count += 1  # type: ignore[attr-defined]
                                        if elapsed > 0:
                                            avg_speed = (
                                                progress._processed_count / elapsed
                                            )  # type: ignore[attr-defined]
                                            avg_str = f"{avg_speed:.1f} img/s"
                                        else:
                                            avg_str = "0.0 img/s"
                                        progress.update(
                                            task_id,
                                            advance=1,
                                            avg_speed=avg_str,
                                        )
                                        display.update_info(
                                            current_file=f"📷 {img_name}",
                                            detections=f"  👤 {det_summary}"
                                            if det_summary
                                            else "  ✓ Processed",
                                            avg_speed=avg_str,
                                        )
                                        live.update(display)
                                    elif progress_bar:
                                        progress_bar.set_description(
                                            f"Image: {img_name}"
                                        )
                                        progress_bar.set_postfix_str(
                                            f"Detected: {det_summary}", refresh=False
                                        )
                                        progress_bar.update(1)
                                else:
                                    self.stats_tracker.update_increment("error_images")
                                    img_name = image_path.name
                                    if len(img_name) > 50:
                                        img_name = "..." + img_name[-47:]
                                    if use_rich and progress and display:
                                        elapsed = time.time() - progress._start_time  # type: ignore[attr-defined]
                                        progress._processed_count += 1  # type: ignore[attr-defined]
                                        if elapsed > 0:
                                            avg_speed = (
                                                progress._processed_count / elapsed
                                            )  # type: ignore[attr-defined]
                                            avg_str = f"{avg_speed:.1f} img/s"
                                        else:
                                            avg_str = "0.0 img/s"
                                        progress.update(
                                            task_id,
                                            advance=1,
                                            avg_speed=avg_str,
                                        )
                                        display.update_info(
                                            current_file=f"📷 {img_name}",
                                            detections="  ❌ ERROR",
                                            avg_speed=avg_str,
                                        )
                                        live.update(display)
                                    elif progress_bar:
                                        progress_bar.set_description(
                                            f"Image: {img_name}"
                                        )
                                        progress_bar.set_postfix_str(
                                            "ERROR", refresh=False
                                        )
                                        progress_bar.update(1)
                            except Exception as e:
                                logger.error(
                                    f"REQ-015: Error processing {image_path}: {e}"
                                )
                                self.stats_tracker.update_increment("error_images")
                                img_name = image_path.name
                                if len(img_name) > 50:
                                    img_name = "..." + img_name[-47:]
                                if use_rich and progress and display:
                                    elapsed = time.time() - progress._start_time  # type: ignore[attr-defined]
                                    progress._processed_count += 1  # type: ignore[attr-defined]
                                    if elapsed > 0:
                                        avg_speed = (
                                            progress._processed_count / elapsed
                                        )  # type: ignore[attr-defined]
                                        avg_str = f"{avg_speed:.1f} img/s"
                                    else:
                                        avg_str = "0.0 img/s"
                                    progress.update(
                                        task_id,
                                        advance=1,
                                        avg_speed=avg_str,
                                    )
                                    display.update_info(
                                        current_file=f"📷 {img_name}",
                                        detections=f"  ❌ FAILED: {str(e)[:50]}",
                                        avg_speed=avg_str,
                                    )
                                    live.update(display)
                                elif progress_bar:
                                    progress_bar.set_description(f"Image: {img_name}")
                                    progress_bar.set_postfix_str(
                                        "FAILED", refresh=False
                                    )
                                    progress_bar.update(1)
                            except CancelledError:
                                logger.debug(
                                    f"REQ-015: Processing cancelled for {image_path}"
                                )
                                self.stats_tracker.update_increment("error_images")
                    finally:
                        if executor:
                            if cancellation_manager.is_cancelled():
                                executor.shutdown(wait=False, cancel_futures=True)
                            else:
                                executor.shutdown(wait=True)
            except KeyboardInterrupt:
                logger.warning("REQ-015: Processing interrupted by user")
                cancellation_manager.cancel()
            finally:
                if use_rich and live:
                    live.stop()
                elif progress_bar:
                    progress_bar.close()

                # REQ-012: Final statistics
                self.stats_tracker.finalize()
                self.stats_tracker.print_statistics()

        finally:
            # REQ-040: Clean up temporary RAW conversion files
            cleanup_temp_files()

            # REQ-022: Close database connection
            if self.database_connection:
                self.database_connection.close()

            # REQ-015: Cleanup cancellation manager
            cancellation_manager.cleanup()

        return self.stats_tracker.get_stats()

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
        return self.image_processor.process_single_image(
            image_path, missing_analyses, sidecar_path
        )

