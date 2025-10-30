"""
Main Image Processor

REQ-002: Process all images in collection and generate sidecar files.
REQ-006: GPU-only operation enforcement.
REQ-010: All code components directly linked to requirements.
REQ-012: Progress tracking and statistics.
REQ-013: Idempotent processing using sidecar files.
REQ-015: Robust error handling.
REQ-016: Multi-level verbosity logging.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import image_sidecar_rust
import tqdm
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from media_indexer.db.connection import DatabaseConnection
from media_indexer.db.hash_util import calculate_file_hash
from media_indexer.exif_extractor import EXIFExtractor, get_exif_extractor
from media_indexer.face_detector import FaceDetector, get_face_detector
from media_indexer.gpu_validator import GPUValidator, get_gpu_validator
from media_indexer.object_detector import ObjectDetector, get_object_detector
from media_indexer.pose_detector import PoseDetector, get_pose_detector
from media_indexer.raw_converter import cleanup_temp_files
from media_indexer.sidecar_generator import SidecarGenerator, get_sidecar_generator
from media_indexer.utils.file_utils import get_image_extensions
from media_indexer.utils.progress import (
    create_progress_bar_with_global_speed,
    create_rich_progress_bar,
)

logger = logging.getLogger(__name__)
console = Console()


class AvgSpeedColumn(ProgressColumn):
    """Custom column to display average speed safely."""
    
    def __init__(self, unit: str = "item") -> None:
        super().__init__()
        self.unit = unit
    
    def render(self, task: Any) -> Text:
        """Render average speed safely."""
        avg_speed = task.fields.get("avg_speed", "0.0")
        if avg_speed:
            return Text(f"[cyan]avg: {avg_speed}[/cyan]", style="progress.data.speed")
        return Text(f"[cyan]avg: 0.0 {self.unit}/s[/cyan]", style="progress.data.speed")


class CurrentFileColumn(ProgressColumn):
    """Custom column to display current file on second line."""
    
    def render(self, task: Any) -> Text:
        """Render current file name."""
        current_file = task.fields.get("current_file", "")
        if current_file:
            return Text(f"\n[dim]{current_file}[/dim]")
        return Text("")


class DetectionsColumn(ProgressColumn):
    """Custom column to display detection information on third line."""
    
    def render(self, task: Any) -> Text:
        """Render detection information."""
        detections = task.fields.get("detections", "")
        if detections:
            return Text(f"\n[dim]{detections}[/dim]")
        return Text("")


def create_rich_progress_bar(
    total: int,
    desc: str,
    unit: str = "item",
    verbose: int = 20,
    show_detections: bool = False,
) -> Progress | None:
    """
    Create a Rich progress bar with multi-line support for detection information.

    REQ-012: Progress tracking with both instantaneous and global/average speed.
    Supports multi-line display for detection information.

    Args:
        total: Total number of items to process.
        desc: Description for the progress bar.
        unit: Unit label for items (e.g., "file", "img").
        verbose: Verbosity level (only create if >= 15).
        show_detections: If True, add a second line for detection information.

    Returns:
        Rich Progress instance or None if verbosity is too low.
    """
    if verbose < 15:
        return None

    # REQ-012: Create Rich progress bar with custom columns
    # Use custom columns to safely handle None values
    if show_detections:
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn(f"{unit}"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[progress.speed]{task.speed:>8.1f}", style="progress.speed"),
            TextColumn(f"{unit}/s"),
            TextColumn("•"),
            AvgSpeedColumn(unit=unit),
            CurrentFileColumn(),
            DetectionsColumn(),
        ]
    else:
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn(f"{unit}"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[progress.speed]{task.speed:>8.1f}", style="progress.speed"),
            TextColumn(f"{unit}/s"),
            TextColumn("•"),
            AvgSpeedColumn(unit=unit),
        ]

    progress = Progress(*columns, console=console, transient=False)
    
    # Store processed count and start time
    progress._processed_count = 0  # type: ignore[attr-defined]
    progress._start_time = time.time()  # type: ignore[attr-defined]
    
    return progress


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
            input_dir: Directory containing images to process (sidecar files created alongside).
            verbose: Verbosity level (REQ-016).
            batch_size: Batch size for processing (REQ-014).
            database_path: Path to database file (REQ-025).
            disable_sidecar: Disable sidecar generation when using database (REQ-026).
            limit: Maximum number of images to process (REQ-038).
            force: Force reprocessing even if analyses already exist (REQ-013).
            scan_workers: Number of parallel workers for sidecar scanning (REQ-020).

        Raises:
            RuntimeError: If no GPU is available (REQ-006).
        """
        # REQ-006: Validate GPU availability
        self.gpu_validator: GPUValidator = get_gpu_validator()
        self.device = self.gpu_validator.get_device()

        # REQ-002: Setup input directory (resolve to absolute path)
        # Sidecar files are always created alongside image files
        self.input_dir: Path = Path(input_dir).resolve()

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

        # REQ-013: Force reprocessing flag
        self.force = force

        # REQ-020: Workers for parallel sidecar scanning
        self.scan_workers = scan_workers

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
            logger.warning("REQ-026: --no-sidecar specified without --db, ignoring flag")
            self.disable_sidecar = False
        if self.disable_sidecar:
            logger.info("REQ-026: Sidecar generation disabled")

    def _find_sidecar_path(self, image_path: Path) -> Path | None:
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
        sidecar_path = image_path_resolved.with_suffix(image_path_resolved.suffix + '.json')
        
        if sidecar_path.exists():
            return sidecar_path
        
        return None

    def _get_required_analyses(self) -> set[str]:
        """
        Get set of required analyses based on initialized components.

        REQ-013: Determine which analyses should be performed.

        Returns:
            Set of analysis types: 'exif', 'faces', 'objects', 'poses'.
        """
        required: set[str] = set()
        if self.exif_extractor is not None:
            required.add('exif')
        if self.face_detector is not None:
            required.add('faces')
        if self.object_detector is not None:
            required.add('objects')
        if self.pose_detector is not None:
            required.add('poses')
        return required

    def _get_existing_analyses(self, image_path: Path) -> set[str]:
        """
        Get set of analyses already present in sidecar file or database.

        REQ-013: Check sidecar file and database to determine what analyses exist.

        Args:
            image_path: Path to the image file.

        Returns:
            Set of analysis types present: 'exif', 'faces', 'objects', 'poses'.
        """
        existing: set[str] = set()
        
        # REQ-025: Check database first if available
        if self.database_connection:
            try:
                from pony.orm import db_session

                from media_indexer.db.image import Image as DBImage

                with db_session:
                    db_image = DBImage.get_by_path(str(image_path))
                    if db_image:
                        # Check if analyses exist in database
                        if db_image.faces and len(db_image.faces) > 0:
                            existing.add('faces')
                        if db_image.objects and len(db_image.objects) > 0:
                            existing.add('objects')
                        if db_image.poses and len(db_image.poses) > 0:
                            existing.add('poses')
                        if db_image.exif_data is not None:
                            existing.add('exif')
                        
                        # If all analyses are in database, return early
                        if len(existing) == len(self._get_required_analyses()):
                            return existing
            except Exception as e:
                logger.debug(f"REQ-025: Database check failed for {image_path}: {e}")
        
        # Also check sidecar files (complement database, not replace)
        if not self.disable_sidecar:
            sidecar_path = self._find_sidecar_path(image_path)
            if sidecar_path and sidecar_path.exists():
                try:
                    if self.sidecar_generator:
                        metadata = self.sidecar_generator.read_sidecar(sidecar_path)
                    else:
                        # Fallback: read JSON directly
                        with open(sidecar_path) as f:
                            metadata = json.load(f)
                    
                    # Add analyses from sidecar (union with database)
                    if metadata.get('exif'):
                        existing.add('exif')
                    if metadata.get('faces'):
                        existing.add('faces')
                    if metadata.get('objects'):
                        existing.add('objects')
                    if metadata.get('poses'):
                        existing.add('poses')
                except Exception as e:
                    logger.debug(f"REQ-013: Failed to read sidecar for {image_path}: {e}")
        
        return existing

    def _needs_processing(self, image_path: Path) -> tuple[bool, set[str]]:
        """
        Determine if image needs processing and which analyses are needed.

        REQ-013: Check if image needs processing based on existing sidecar data.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (needs_processing, missing_analyses).
        """
        if self.force:
            required = self._get_required_analyses()
            return True, required
        
        required = self._get_required_analyses()
        existing = self._get_existing_analyses(image_path)
        missing = required - existing
        
        return len(missing) > 0, missing

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
            self.sidecar_generator = get_sidecar_generator()
            logger.info("REQ-004: Sidecar generator initialized")
        except Exception as e:
            logger.warning(f"REQ-004: Sidecar generator not available: {e}")

    def _get_image_files(self) -> list[Path]:
        """
        Get list of image files to process (REQ-018, REQ-038).

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

    def _process_single_image(self, image_path: Path, missing_analyses: set[str]) -> tuple[bool, dict[str, Any]]:
        """
        Process a single image (REQ-002, REQ-015).

        Args:
            image_path: Path to image file.
            missing_analyses: Set of analysis types to perform.

        Returns:
            Tuple of (success, detection_results). detection_results contains counts of faces, objects, poses.
        """
        try:
            # Load existing metadata from sidecar if it exists
            existing_metadata: dict[str, Any] = {}
            sidecar_path = self._find_sidecar_path(image_path)
            if sidecar_path and sidecar_path.exists():
                try:
                    if self.sidecar_generator:
                        existing_metadata = self.sidecar_generator.read_sidecar(sidecar_path)
                    else:
                        with open(sidecar_path) as f:
                            existing_metadata = json.load(f)
                except Exception as e:
                    logger.debug(f"REQ-013: Failed to read existing sidecar: {e}")
            
            # Start with existing metadata or new dict
            metadata: dict[str, Any] = existing_metadata.copy() if existing_metadata else {"image_path": str(image_path)}
            detection_results: dict[str, Any] = {}

            # REQ-003: Extract EXIF if needed
            if 'exif' in missing_analyses and self.exif_extractor is not None:
                try:
                    metadata["exif"] = self.exif_extractor.extract_from_path(image_path)
                except Exception as e:
                    logger.debug(f"REQ-003: EXIF extraction failed: {e}")

            # REQ-007: Detect faces if needed
            if 'faces' in missing_analyses and self.face_detector is not None:
                try:
                    faces = self.face_detector.detect_faces(image_path)
                    metadata["faces"] = faces
                    detection_results["faces"] = len(faces)
                except Exception as e:
                    logger.debug(f"REQ-007: Face detection failed: {e}")
                    detection_results["faces"] = 0

            # REQ-008: Detect objects if needed
            if 'objects' in missing_analyses and self.object_detector is not None:
                try:
                    objects = self.object_detector.detect_objects(image_path)
                    metadata["objects"] = objects
                    detection_results["objects"] = len(objects)
                except Exception as e:
                    logger.debug(f"REQ-008: Object detection failed: {e}")
                    detection_results["objects"] = 0

            # REQ-009: Detect poses if needed
            if 'poses' in missing_analyses and self.pose_detector is not None:
                try:
                    poses = self.pose_detector.detect_poses(image_path)
                    metadata["poses"] = poses
                    detection_results["poses"] = len(poses)
                except Exception as e:
                    logger.debug(f"REQ-009: Pose detection failed: {e}")
                    detection_results["poses"] = 0

            # REQ-025: Store to database
            if self.database_connection:
                db_success = self._store_to_database(image_path, metadata)
                if not db_success:
                    logger.error(f"REQ-025: Failed to store {image_path} to database")

            # REQ-027: Generate/update sidecar if not disabled
            if not self.disable_sidecar and self.sidecar_generator is not None:
                try:
                    sidecar_path = self.sidecar_generator.generate_sidecar(image_path, metadata)
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

    def process(self) -> dict[str, Any]:
        """
        Process all images with parallel/batch processing (REQ-002, REQ-012, REQ-014, REQ-020).

        REQ-013: Scan sidecar files to determine which analyses are needed.

        Returns:
            Dictionary with processing statistics.
        """
        logger.info(f"REQ-002: Starting image processing with batch size {self.batch_size}")

        # Initialize components first (needed to determine required analyses)
        self._initialize_components()

        # Get image files
        images = self._get_image_files()
        self.stats["total_images"] = len(images)
        logger.info(f"REQ-002: Found {len(images)} images")

        # REQ-013: Scan sidecar files and determine which images need processing
        logger.info("REQ-013: Scanning sidecar files to determine required analyses...")
        images_to_process: list[tuple[Path, set[str]]] = []
        skipped_count = 0
        
        # REQ-020: Parallel scanning with progress bar and global speed
        scan_workers = min(self.scan_workers, len(images))  # Use configured workers for I/O-bound scanning
        progress_bar = create_progress_bar_with_global_speed(
            total=len(images),
            desc="Scanning sidecars",
            unit="file",
            verbose=self.verbose,
        )
        
        try:
            if scan_workers == 1:
                # Sequential scanning for small sets
                for image_path in images:
                    needs_processing, missing_analyses = self._needs_processing(image_path)
                    if needs_processing:
                        images_to_process.append((image_path, missing_analyses))
                    else:
                        skipped_count += 1
                    if progress_bar:
                        progress_bar.update(1)
            else:
                # REQ-020: Parallel scanning with thread pool
                with ThreadPoolExecutor(max_workers=scan_workers) as executor:
                    futures = {
                        executor.submit(self._needs_processing, image_path): image_path
                        for image_path in images
                    }
                    
                    for future in as_completed(futures):
                        image_path = futures[future]
                        try:
                            needs_processing, missing_analyses = future.result()
                            if needs_processing:
                                images_to_process.append((image_path, missing_analyses))
                            else:
                                skipped_count += 1
                        except Exception as e:
                            # If scanning fails, assume needs processing
                            logger.debug(f"REQ-013: Failed to scan {image_path}: {e}")
                            images_to_process.append((image_path, self._get_required_analyses()))
                        finally:
                            if progress_bar:
                                progress_bar.update(1)
        finally:
            if progress_bar:
                progress_bar.close()
        
        self.stats["skipped_images"] = skipped_count
        
        # REQ-013: Show summary before processing
        required_analyses = self._get_required_analyses()
        analysis_names = {
            'exif': 'EXIF',
            'faces': 'faces',
            'objects': 'objects',
            'poses': 'poses'
        }
        required_str = ', '.join(analysis_names.get(a, a) for a in sorted(required_analyses))
        logger.info(f"REQ-013: Required analyses: {required_str}")
        logger.info(f"REQ-013: Already analyzed: {skipped_count} images")
        logger.info(f"REQ-013: Needs processing: {len(images_to_process)} images")
        
        if not images_to_process:
            logger.info("REQ-013: All images already have complete analyses")
            self.stats["end_time"] = datetime.now().isoformat()
            self._print_statistics()
            return self.stats

        # REQ-012: Progress tracking with TQDM
        # Show progress bars by default (WARNING level) and above, disable only at very verbose levels
        def format_detection_summary(detections: dict[str, Any]) -> str:
            """Format detection info for display."""
            parts = []
            if detections.get("faces", 0) > 0:
                parts.append(f"{detections['faces']} face{'s' if detections['faces'] != 1 else ''}")
            if detections.get("objects", 0) > 0:
                parts.append(f"{detections['objects']} object{'s' if detections['objects'] != 1 else ''}")
            if detections.get("poses", 0) > 0:
                parts.append(f"{detections['poses']} pose{'s' if detections['poses'] != 1 else ''}")
            return ", ".join(parts) if parts else "no detections"
        
        # REQ-012: Progress tracking with Rich for multi-line detection info
        use_rich = self.verbose >= 15
        if use_rich:
            progress = create_rich_progress_bar(
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
                current_file="",
                detections="",
                avg_speed="0.0 img/s",
            )
            progress.start()
        else:
            progress = None
            progress_bar = create_progress_bar_with_global_speed(
                total=len(images_to_process),
                desc="Processing images",
                unit="img",
                verbose=self.verbose,
            )
            task_id = None

        # REQ-020: Process images in batches with threading for I/O
        try:
            # Process in batches
            for i in range(0, len(images_to_process), self.batch_size):
                batch = images_to_process[i : i + self.batch_size]

                # REQ-015: Robust error handling with thread pool
                with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                    futures = {
                        executor.submit(self._process_single_image, img_path, missing): img_path 
                        for img_path, missing in batch
                    }

                    for future in as_completed(futures):
                        image_path = futures[future]
                        try:
                            success, detections = future.result()
                            if success:
                                self._update_stats_increment("processed_images")
                                # Display detection results  
                                img_name = image_path.name
                                if len(img_name) > 50:
                                    img_name = "..." + img_name[-47:]
                                det_summary = format_detection_summary(detections)
                                
                                if use_rich and progress:
                                    # Update Rich progress bar with multi-line info
                                    elapsed = time.time() - progress._start_time  # type: ignore[attr-defined]
                                    progress._processed_count += 1  # type: ignore[attr-defined]
                                    if elapsed > 0:
                                        avg_speed = progress._processed_count / elapsed  # type: ignore[attr-defined]
                                        avg_str = f"{avg_speed:.1f} img/s"
                                    else:
                                        avg_str = "0.0 img/s"
                                    progress.update(
                                        task_id,
                                        advance=1,
                                        current_file=f"📷 {img_name}",
                                        detections=f"  👤 {det_summary}" if det_summary else "  ✓ Processed",
                                        avg_speed=avg_str,
                                    )
                                elif progress_bar:
                                    # Show image name and detections (tqdm fallback)
                                    progress_bar.set_description(f"Image: {img_name}")
                                    progress_bar.set_postfix_str(f"Detected: {det_summary}", refresh=False)
                                    progress_bar.update(1)
                            else:
                                self._update_stats_increment("error_images")
                                img_name = image_path.name
                                if len(img_name) > 50:
                                    img_name = "..." + img_name[-47:]
                                if use_rich and progress:
                                    elapsed = time.time() - progress._start_time  # type: ignore[attr-defined]
                                    progress._processed_count += 1  # type: ignore[attr-defined]
                                    if elapsed > 0:
                                        avg_speed = progress._processed_count / elapsed  # type: ignore[attr-defined]
                                        avg_str = f"{avg_speed:.1f} img/s"
                                    else:
                                        avg_str = "0.0 img/s"
                                    progress.update(
                                        task_id,
                                        advance=1,
                                        current_file=f"📷 {img_name}",
                                        detections="  ❌ ERROR",
                                        avg_speed=avg_str,
                                    )
                                elif progress_bar:
                                    progress_bar.set_description(f"Image: {img_name}")
                                    progress_bar.set_postfix_str("ERROR", refresh=False)
                                    progress_bar.update(1)
                        except Exception as e:
                            logger.error(f"REQ-015: Error processing {image_path}: {e}")
                            self._update_stats_increment("error_images")
                            img_name = image_path.name
                            if len(img_name) > 50:
                                img_name = "..." + img_name[-47:]
                            if use_rich and progress:
                                elapsed = time.time() - progress._start_time  # type: ignore[attr-defined]
                                progress._processed_count += 1  # type: ignore[attr-defined]
                                if elapsed > 0:
                                    avg_speed = progress._processed_count / elapsed  # type: ignore[attr-defined]
                                    avg_str = f"{avg_speed:.1f} img/s"
                                else:
                                    avg_str = "0.0 img/s"
                                progress.update(
                                    task_id,
                                    advance=1,
                                    current_file=f"📷 {img_name}",
                                    detections=f"  ❌ FAILED: {str(e)[:50]}",
                                    avg_speed=avg_str,
                                )
                            elif progress_bar:
                                progress_bar.set_description(f"Image: {img_name}")
                                progress_bar.set_postfix_str("FAILED", refresh=False)
                                progress_bar.update(1)
        finally:
            if use_rich and progress:
                progress.stop()
            elif progress_bar:
                progress_bar.close()

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
