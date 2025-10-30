"""Sidecar-to-database migration utilities.

REQ-072: Extracted from the legacy monolith to keep modules under 500 lines.
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from pony.orm import db_session

from media_indexer.db.connection import DatabaseConnection
from media_indexer.db.exif import EXIFData
from media_indexer.db.face import Face
from media_indexer.db.hash_util import calculate_file_hash, get_file_size
from media_indexer.db.image import Image
from media_indexer.db.object import Object
from media_indexer.db.pose import Pose
from media_indexer.processor.progress import create_rich_progress_bar

logger = logging.getLogger(__name__)

_shutdown_flag = threading.Event()


def _import_single_sidecar(
    image_file: Path,
    database_path: Path,
    verbose: int,
) -> tuple[bool, int]:
    """Import a single sidecar file into the database."""

    if _shutdown_flag.is_set():
        return False, 0

    sidecar_path = image_file.with_suffix(".json")
    if not sidecar_path.exists():
        return False, 0

    try:
        if _shutdown_flag.is_set():
            return False, 0

        logger.debug("REQ-032: Reading sidecar from %s", sidecar_path)
        with open(sidecar_path) as handle:
            metadata = json.load(handle)

        with db_session:
            existing_image = Image.get(path=str(image_file))
            if existing_image:
                logger.debug("REQ-032: Image %s already in database, skipping", image_file)
                return False, 0

            file_hash = calculate_file_hash(image_file)
            file_size = get_file_size(image_file) or 0

            width = None
            height = None
            if metadata.get("exif"):
                width = metadata["exif"].get("imageWidth")
                height = metadata["exif"].get("imageHeight")

            db_image = Image(
                path=str(image_file),
                file_hash=file_hash,
                file_size=file_size,
                width=width,
                height=height,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            if metadata.get("faces"):
                for face_data in metadata["faces"]:
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

            if metadata.get("objects"):
                for obj_data in metadata["objects"]:
                    Object(
                        image=db_image,
                        class_id=obj_data.get("class_id", -1),
                        class_name=obj_data.get("class_name", "unknown"),
                        confidence=obj_data.get("confidence", 0.0),
                        bbox=obj_data.get("bbox", []),
                    )

            if metadata.get("poses"):
                for pose_data in metadata["poses"]:
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

            if metadata.get("exif"):
                EXIFData(image=db_image, data=metadata["exif"])

        if verbose <= 15:
            logger.info("REQ-032: Imported %s", image_file)

        return True, 1
    except Exception as exc:  # noqa: BLE001
        logger.error("REQ-032: Failed to import %s: %s", image_file, exc)
        return False, 0


def _signal_handler(signum: int, frame: Any) -> None:  # noqa: D401 - signal contract
    logger.warning("REQ-015: Processing interrupted by user")
    _shutdown_flag.set()


def import_sidecars_to_database(
    input_dir: Path,
    database_path: Path,
    verbose: int = 20,
    workers: int = 1,
) -> None:
    """Import existing sidecar files into the database."""

    _shutdown_flag.clear()
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    logger.info("REQ-032: Initializing database connection")
    db_conn = DatabaseConnection(database_path)
    db_conn.connect()

    try:
        logger.info("REQ-032: Scanning %s for images and sidecar files", input_dir)
        image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".raw"]
        image_files = [
            path
            for path in input_dir.rglob("*")
            if path.suffix.lower() in image_extensions and path.with_suffix(path.suffix + ".json").exists()
        ]

        logger.info("REQ-032: Found %s images with sidecar files", len(image_files))
        if not image_files:
            logger.info("REQ-032: No sidecar files found to import")
            return

        context = _init_progress(len(image_files), "Importing sidecars", verbose)
        processed = 0
        errors = 0
        executor: ThreadPoolExecutor | None = None

        try:
            if workers == 1:
                for image_file in image_files:
                    if _shutdown_flag.is_set():
                        logger.warning("REQ-015: Processing interrupted by user")
                        break
                    success, items_processed = _import_single_sidecar(image_file, database_path, verbose)
                    processed += items_processed
                    if not success:
                        errors += 1
                    _advance_progress(context, image_file.name)
            else:
                executor = ThreadPoolExecutor(max_workers=workers)
                futures = {
                    executor.submit(_import_single_sidecar, image_file, database_path, verbose): image_file
                    for image_file in image_files
                }

                for future in as_completed(futures):
                    if _shutdown_flag.is_set():
                        logger.warning("REQ-015: Processing interrupted by user")
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        break

                    image_file = futures[future]
                    try:
                        success, items_processed = future.result()
                        processed += items_processed
                        if not success:
                            errors += 1
                    except CancelledError:
                        errors += 1
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        logger.error(
                            "REQ-032: Unexpected error processing %s: %s",
                            image_file,
                            exc,
                        )
                    finally:
                        _advance_progress(context, image_file.name)
        except KeyboardInterrupt:
            logger.warning("REQ-015: Processing interrupted by user")
            _shutdown_flag.set()
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
        finally:
            if executor:
                if _shutdown_flag.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=True)
            _stop_progress(context)
            signal.signal(signal.SIGINT, original_handler)

        logger.info("REQ-032: Import completed - %s imported, %s errors", processed, errors)
    finally:
        db_conn.close()


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def _init_progress(total: int, desc: str, verbose: int) -> dict[str, Any] | None:
    if verbose < 15 or total == 0:
        return None

    progress, display, live = create_rich_progress_bar(
        total=total,
        desc=desc,
        unit="file",
        verbose=verbose,
        show_detections=False,
    )

    if not all((progress, display, live)):
        return None

    task_id = progress.add_task(desc, total=total, avg_speed="0.0 file/s")
    live.start()
    return {
        "progress": progress,
        "display": display,
        "live": live,
        "task_id": task_id,
        "start_time": time.time(),
        "processed": 0,
    }


def _advance_progress(context: dict[str, Any] | None, filename: str) -> None:
    if not context:
        return

    context["processed"] += 1
    elapsed = time.time() - context["start_time"]
    avg_speed = context["processed"] / elapsed if elapsed > 0 else 0.0
    avg_text = f"{avg_speed:.1f} file/s"

    truncated = filename
    if len(truncated) > 50:
        truncated = "..." + truncated[-47:]

    context["progress"].update(
        context["task_id"],
        advance=1,
        avg_speed=avg_text,
    )
    context["display"].update_info(
        current_file=f"📁 {truncated}",
        detections="",
        avg_speed=avg_text,
    )
    context["live"].update(context["display"])


def _stop_progress(context: dict[str, Any] | None) -> None:
    if context:
        context["live"].stop()


__all__ = ["import_sidecars_to_database"]
