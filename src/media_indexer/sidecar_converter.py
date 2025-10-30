"""
Sidecar Converter Module

REQ-032, REQ-033, REQ-034: Convert between sidecar and database formats.
REQ-010: All code components directly linked to requirements.
REQ-020: Parallel processing with thread-based I/O operations.
REQ-015: Handle user interruption gracefully with quick shutdown.
"""

import json
import logging
import signal
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import tqdm
from pony.orm import db_session

from media_indexer.db.connection import DatabaseConnection
from media_indexer.db.exif import EXIFData
from media_indexer.db.face import Face
from media_indexer.db.hash_util import calculate_file_hash, get_file_size
from media_indexer.db.image import Image
from media_indexer.db.object import Object
from media_indexer.db.pose import Pose

logger = logging.getLogger(__name__)

# REQ-015: Global cancellation flag for quick shutdown
_shutdown_flag = threading.Event()


def create_progress_bar_with_global_speed(
    total: int,
    desc: str,
    unit: str = "item",
    verbose: int = 20,
) -> tqdm.tqdm | None:
    """
    Create a tqdm progress bar with global speed calculation.

    REQ-012: Progress tracking with both instantaneous and global/average speed.

    Args:
        total: Total number of items to process.
        desc: Description for the progress bar.
        unit: Unit label for items (e.g., "file", "img").
        verbose: Verbosity level (only create if >= 15).

    Returns:
        tqdm progress bar instance or None if verbosity is too low.
    """
    if verbose < 15:
        return None

    # REQ-012: Create progress bar with compact format showing both speeds
    progress_bar = tqdm.tqdm(
        total=total,
        desc=desc,
        unit=unit,
        bar_format='{desc}: {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]',
    )

    # Store start time for global speed calculation
    progress_bar.start_time = time.time()  # type: ignore[attr-defined]
    progress_bar._custom_postfix = ""  # type: ignore[attr-defined]
    
    # Store original methods BEFORE wrapping
    original_update = progress_bar.update
    original_set_postfix = progress_bar.set_postfix_str

    def update_with_global_speed(n: int = 1) -> None:
        """Update progress bar with global speed calculation."""
        # Call original update first
        original_update(n)
        # Calculate global speed: processed / elapsed_time
        elapsed = time.time() - progress_bar.start_time  # type: ignore[attr-defined]
        if elapsed > 0 and progress_bar.n > 0:
            global_speed = progress_bar.n / elapsed
            # Format speed nicely
            if global_speed >= 1:
                speed_str = f"{global_speed:.1f} {unit}/s"
            else:
                speed_str = f"{1/global_speed:.1f}s/{unit}"
            # Combine with custom postfix if any
            custom = progress_bar._custom_postfix  # type: ignore[attr-defined]
            if custom:
                combined = f"{custom} | avg: {speed_str}"
            else:
                combined = f"avg: {speed_str}"
            # Use original set_postfix to avoid recursion
            original_set_postfix(combined, refresh=False)
    
    def set_postfix_with_global_speed(postfix: str | None = None, refresh: bool = True) -> None:
        """Set postfix while preserving global speed."""
        if postfix:
            progress_bar._custom_postfix = postfix  # type: ignore[attr-defined]
            # Recalculate global speed and combine
            elapsed = time.time() - progress_bar.start_time  # type: ignore[attr-defined]
            if elapsed > 0 and progress_bar._processed_count > 0:  # type: ignore[attr-defined]
                global_speed = progress_bar._processed_count / elapsed  # type: ignore[attr-defined]
                if global_speed >= 1:
                    speed_str = f"{global_speed:.1f} {unit}/s"
                else:
                    speed_str = f"{1/global_speed:.1f}s/{unit}"
                combined = f"{postfix} | avg: {speed_str}"
                original_set_postfix(combined, refresh=refresh)
            else:
                original_set_postfix(postfix, refresh=refresh)
        else:
            original_set_postfix(postfix, refresh=refresh)
    
    # Replace methods AFTER defining both functions
    progress_bar.set_postfix_str = set_postfix_with_global_speed  # type: ignore[method-assign]
    progress_bar.update = update_with_global_speed  # type: ignore[method-assign]

    return progress_bar


def _import_single_sidecar(
    image_file: Path,
    database_path: Path,
    verbose: int,
) -> tuple[bool, int]:
    """
    Import a single sidecar file into database.

    REQ-032: Import a single sidecar file into database.
    REQ-020: Parallel processing support for sidecar import.
    REQ-015: Check cancellation flag for quick shutdown.

    Args:
        image_file: Path to image file.
        database_path: Path to SQLite database.
        verbose: Verbosity level.

    Returns:
        Tuple of (success: bool, items_processed: int) where items_processed is 1 on success, 0 on skip/error.
    """
    # REQ-015: Check for cancellation before processing
    if _shutdown_flag.is_set():
        return False, 0
    
    sidecar_path = image_file.with_suffix('.json')
    
    if not sidecar_path.exists():
        return False, 0
    
    try:
        # REQ-015: Check for cancellation again before expensive operations
        if _shutdown_flag.is_set():
            return False, 0
        # REQ-032: Read sidecar file
        logger.debug(f"REQ-032: Reading sidecar from {sidecar_path}")
        with open(sidecar_path) as f:
            metadata = json.load(f)

        # REQ-032: Import to database
        # Each thread needs its own db_session
        with db_session:
            # Check if image already exists
            existing_image = Image.get(path=str(image_file))
            if existing_image:
                logger.debug(f"REQ-032: Image {image_file} already in database, skipping")
                return False, 0

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

        if verbose <= 15:
            logger.info(f"REQ-032: Imported {image_file}")
        
        return True, 1

    except Exception as e:
        logger.error(f"REQ-032: Failed to import {image_file}: {e}")
        return False, 0


def _signal_handler(signum: int, frame: Any) -> None:
    """
    Signal handler for SIGINT (KeyboardInterrupt).

    REQ-015: Set shutdown flag when user interrupts processing.

    Args:
        signum: Signal number.
        frame: Current stack frame.
    """
    logger.warning("REQ-015: Processing interrupted by user")
    _shutdown_flag.set()


def import_sidecars_to_database(
    input_dir: Path,
    database_path: Path,
    verbose: int = 20,
    workers: int = 1,
) -> None:
    """
    Import sidecar files into database.

    REQ-032: Import existing sidecar files into database.
    REQ-020: Parallel processing with thread-based I/O operations.
    REQ-015: Handle user interruption gracefully with quick shutdown.

    Args:
        input_dir: Directory containing images and sidecar files.
        database_path: Path to SQLite database.
        verbose: Verbosity level.
        workers: Number of parallel workers for processing (default: 1).
    """
    # REQ-015: Reset shutdown flag and register signal handler
    _shutdown_flag.clear()
    original_handler = signal.signal(signal.SIGINT, _signal_handler)
    
    logger.info("REQ-032: Initializing database connection")
    db_conn = DatabaseConnection(database_path)
    db_conn.connect()

    try:
        # REQ-032: Find all images and their corresponding sidecar files
        logger.info(f"REQ-032: Scanning {input_dir} for images and sidecar files")

        image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".raw"]
        image_files: list[Path] = []
        
        for image_file in input_dir.rglob("*"):
            if image_file.suffix.lower() in image_extensions:
                # REQ-032: Check for corresponding sidecar file
                sidecar_path = image_file.with_suffix('.json')
                if sidecar_path.exists():
                    image_files.append(image_file)

        logger.info(f"REQ-032: Found {len(image_files)} images with sidecar files")

        if not image_files:
            logger.info("REQ-032: No sidecar files found to import")
            return

        # REQ-012: Progress tracking with TQDM and global speed
        progress_bar = create_progress_bar_with_global_speed(
            total=len(image_files),
            desc="Importing sidecars",
            unit="file",
            verbose=verbose,
        )

        # REQ-020: Process with parallel workers
        processed = 0
        errors = 0
        executor: ThreadPoolExecutor | None = None

        try:
            if workers == 1:
                # Sequential processing for single worker
                for image_file in image_files:
                    # REQ-015: Check for cancellation
                    if _shutdown_flag.is_set():
                        logger.warning("REQ-015: Processing interrupted by user")
                        break
                    success, items_processed = _import_single_sidecar(
                        image_file, database_path, verbose
                    )
                    if success:
                        processed += items_processed
                    else:
                        errors += 1
                    if progress_bar:
                        progress_bar.update(1)
            else:
                # REQ-020: Parallel processing with thread pool
                executor = ThreadPoolExecutor(max_workers=workers)
                try:
                    futures = {
                        executor.submit(
                            _import_single_sidecar,
                            image_file,
                            database_path,
                            verbose,
                        ): image_file
                        for image_file in image_files
                    }

                    for future in as_completed(futures):
                        # REQ-015: Check for cancellation
                        if _shutdown_flag.is_set():
                            logger.warning("REQ-015: Processing interrupted by user")
                            # Cancel remaining futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break
                        
                        image_file = futures[future]
                        try:
                            success, items_processed = future.result()
                            if success:
                                processed += items_processed
                            else:
                                errors += 1
                        except CancelledError:
                            # REQ-015: Future was cancelled, skip
                            errors += 1
                        except Exception as e:
                            errors += 1
                            logger.error(f"REQ-032: Unexpected error processing {image_file}: {e}")
                        finally:
                            if progress_bar:
                                progress_bar.update(1)
                finally:
                    # REQ-015: Shutdown executor quickly when interrupted
                    if executor:
                        if _shutdown_flag.is_set():
                            # Quick shutdown: cancel futures and don't wait
                            executor.shutdown(wait=False, cancel_futures=True)
                        else:
                            # Normal shutdown: wait for completion
                            executor.shutdown(wait=True)
        except KeyboardInterrupt:
            # REQ-015: Handle KeyboardInterrupt at loop level
            logger.warning("REQ-015: Processing interrupted by user")
            _shutdown_flag.set()
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
        finally:
            if progress_bar:
                progress_bar.close()
            # REQ-015: Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

        logger.info(f"REQ-032: Import completed - {processed} imported, {errors} errors")

    finally:
        db_conn.close()


def _export_single_image(
    image_path: str,
    database_path: Path,
    verbose: int,
) -> tuple[bool, int]:
    """
    Export a single image from database to sidecar file.

    REQ-033: Export a single image from database to sidecar file.
    REQ-020: Parallel processing support for sidecar export.
    REQ-015: Check cancellation flag for quick shutdown.

    Args:
        image_path: Path to image file (as string from database).
        database_path: Path to SQLite database.
        verbose: Verbosity level.

    Returns:
        Tuple of (success: bool, items_processed: int) where items_processed is 1 on success, 0 on error.
    """
    # REQ-015: Check for cancellation before processing
    if _shutdown_flag.is_set():
        return False, 0
    
    from pony.orm import db_session

    from media_indexer.db.image import Image
    from media_indexer.sidecar_generator import get_sidecar_generator

    try:
        # REQ-015: Check for cancellation again before expensive operations
        if _shutdown_flag.is_set():
            return False, 0
        sidecar_generator = get_sidecar_generator()

        # REQ-033: Fetch image data in its own db_session
        with db_session:
            db_image = Image.get(path=image_path)
            if not db_image:
                logger.warning(f"REQ-033: Image not found in database: {image_path}")
                return False, 0

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

        # REQ-033: Generate sidecar file (next to image file)
        # Do this outside db_session to avoid holding session during I/O
        image_path_obj = Path(image_path)
        sidecar_generator.generate_sidecar(image_path_obj, metadata)

        if verbose <= 15:
            logger.info(f"REQ-033: Exported {image_path}")

        return True, 1

    except Exception as e:
        logger.error(f"REQ-033: Failed to export {image_path}: {e}")
        return False, 0


def export_database_to_sidecars(
    database_path: Path,
    output_dir: Path | None = None,
    verbose: int = 20,
    workers: int = 1,
) -> None:
    """
    Export database contents to sidecar files.

    REQ-033: Export database contents to individual sidecar JSON files.
    Sidecar files are created next to the image files.
    REQ-020: Parallel processing with thread-based I/O operations.
    REQ-015: Handle user interruption gracefully with quick shutdown.

    Args:
        database_path: Path to SQLite database.
        output_dir: Ignored (sidecar files are always created next to image files).
        verbose: Verbosity level.
        workers: Number of parallel workers for processing (default: 1).
    """
    from pony.orm import db_session

    from media_indexer.db.connection import DatabaseConnection
    from media_indexer.db.image import Image

    # REQ-015: Reset shutdown flag and register signal handler
    _shutdown_flag.clear()
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    logger.info("REQ-033: Initializing database connection")
    db_conn = DatabaseConnection(database_path)
    db_conn.connect()

    try:
        # REQ-033: Export all images from database to sidecar files
        # Sidecar files are created next to the image files
        logger.info("REQ-033: Exporting database to sidecar files (next to image files)")

        # Fetch all image paths in one session
        with db_session:
            images = list(Image.select())
            image_paths = [str(img.path) for img in images]

        logger.info(f"REQ-033: Exporting {len(image_paths)} images")

        if not image_paths:
            logger.info("REQ-033: No images found in database to export")
            return

        # REQ-012: Progress tracking with TQDM and global speed
        progress_bar = create_progress_bar_with_global_speed(
            total=len(image_paths),
            desc="Exporting sidecars",
            unit="file",
            verbose=verbose,
        )

        # REQ-020: Process with parallel workers
        processed = 0
        errors = 0
        executor: ThreadPoolExecutor | None = None

        try:
            if workers == 1:
                # Sequential processing for single worker
                for image_path in image_paths:
                    # REQ-015: Check for cancellation
                    if _shutdown_flag.is_set():
                        logger.warning("REQ-015: Processing interrupted by user")
                        break
                    success, items_processed = _export_single_image(
                        image_path, database_path, verbose
                    )
                    if success:
                        processed += items_processed
                    else:
                        errors += 1
                    if progress_bar:
                        progress_bar.update(1)
            else:
                # REQ-020: Parallel processing with thread pool
                executor = ThreadPoolExecutor(max_workers=workers)
                try:
                    futures = {
                        executor.submit(
                            _export_single_image,
                            image_path,
                            database_path,
                            verbose,
                        ): image_path
                        for image_path in image_paths
                    }

                    for future in as_completed(futures):
                        # REQ-015: Check for cancellation
                        if _shutdown_flag.is_set():
                            logger.warning("REQ-015: Processing interrupted by user")
                            # Cancel remaining futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break
                        
                        image_path = futures[future]
                        try:
                            success, items_processed = future.result()
                            if success:
                                processed += items_processed
                            else:
                                errors += 1
                        except CancelledError:
                            # REQ-015: Future was cancelled, skip
                            errors += 1
                        except Exception as e:
                            errors += 1
                            logger.error(f"REQ-033: Unexpected error processing {image_path}: {e}")
                        finally:
                            if progress_bar:
                                progress_bar.update(1)
                finally:
                    # REQ-015: Shutdown executor quickly when interrupted
                    if executor:
                        if _shutdown_flag.is_set():
                            # Quick shutdown: cancel futures and don't wait
                            executor.shutdown(wait=False, cancel_futures=True)
                        else:
                            # Normal shutdown: wait for completion
                            executor.shutdown(wait=True)
        except KeyboardInterrupt:
            # REQ-015: Handle KeyboardInterrupt at loop level
            logger.warning("REQ-015: Processing interrupted by user")
            _shutdown_flag.set()
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
        finally:
            if progress_bar:
                progress_bar.close()
            # REQ-015: Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

        logger.info(f"REQ-033: Export completed - {processed} exported, {errors} errors")

    finally:
        db_conn.close()
