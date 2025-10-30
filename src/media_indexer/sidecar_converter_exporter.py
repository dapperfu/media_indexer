"""Database-to-sidecar export utilities.

REQ-072: Separated from the legacy converter to satisfy module size limits.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pony.orm import db_session

from media_indexer.db.connection import DatabaseConnection
from media_indexer.db.image import Image
from media_indexer.processor.progress import create_rich_progress_bar
from media_indexer.sidecar_generator import get_sidecar_generator

logger = logging.getLogger(__name__)

_shutdown_flag = threading.Event()


def _export_single_image(
    image_path: str,
    database_path: Path,
    verbose: int,
) -> tuple[bool, int]:
    """Export a single database image record to a sidecar file."""

    if _shutdown_flag.is_set():
        return False, 0

    try:
        sidecar_generator = get_sidecar_generator()
        with db_session:
            db_image = Image.get(path=image_path)
            if not db_image:
                logger.warning("REQ-033: Image not found in database: %s", image_path)
                return False, 0

            metadata: dict[str, Any] = {
                "faces": [],
                "objects": [],
                "poses": [],
                "exif": None,
            }

            for face in db_image.faces:
                metadata["faces"].append(
                    {
                        "confidence": face.confidence,
                        "bbox": face.bbox,
                        "embedding": face.embedding,
                        "model": face.model,
                    }
                )

            for obj in db_image.objects:
                metadata["objects"].append(
                    {
                        "class_id": obj.class_id,
                        "class_name": obj.class_name,
                        "confidence": obj.confidence,
                        "bbox": obj.bbox,
                    }
                )

            for pose in db_image.poses:
                metadata["poses"].append(
                    {
                        "confidence": pose.confidence,
                        "keypoints": pose.keypoints,
                        "bbox": pose.bbox,
                        "keypoints_conf": pose.keypoints_conf,
                    }
                )

            if db_image.exif_data:
                metadata["exif"] = db_image.exif_data.data

        image_path_obj = Path(image_path)
        sidecar_generator.generate_sidecar(image_path_obj, metadata)

        if verbose <= 15:
            logger.info("REQ-033: Exported %s", image_path)

        return True, 1
    except Exception as exc:  # noqa: BLE001
        logger.error("REQ-033: Failed to export %s: %s", image_path, exc)
        return False, 0


def _signal_handler(signum: int, frame: Any) -> None:  # noqa: D401
    logger.warning("REQ-015: Processing interrupted by user")
    _shutdown_flag.set()


def export_database_to_sidecars(
    database_path: Path,
    output_dir: Path | None = None,
    verbose: int = 20,
    workers: int = 1,
) -> None:
    """Export database contents to sidecar JSON files."""

    del output_dir  # Sidecars are always emitted alongside images.

    _shutdown_flag.clear()
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    logger.info("REQ-033: Initializing database connection")
    db_conn = DatabaseConnection(database_path)
    db_conn.connect()

    try:
        with db_session:
            image_paths = [str(img.path) for img in Image.select()]

        logger.info("REQ-033: Exporting %s images", len(image_paths))
        if not image_paths:
            logger.info("REQ-033: No images found in database to export")
            return

        context = _init_progress(len(image_paths), "Exporting sidecars", verbose)
        processed = 0
        errors = 0
        executor: ThreadPoolExecutor | None = None

        try:
            if workers == 1:
                for image_path in image_paths:
                    if _shutdown_flag.is_set():
                        logger.warning("REQ-015: Processing interrupted by user")
                        break
                    success, items_processed = _export_single_image(image_path, database_path, verbose)
                    processed += items_processed
                    if not success:
                        errors += 1
                    _advance_progress(context, Path(image_path).name)
            else:
                executor = ThreadPoolExecutor(max_workers=workers)
                futures = {
                    executor.submit(_export_single_image, image_path, database_path, verbose): image_path
                    for image_path in image_paths
                }

                for future in as_completed(futures):
                    if _shutdown_flag.is_set():
                        logger.warning("REQ-015: Processing interrupted by user")
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        break

                    image_path = futures[future]
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
                            "REQ-033: Unexpected error processing %s: %s",
                            image_path,
                            exc,
                        )
                    finally:
                        _advance_progress(context, Path(image_path).name)
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

        logger.info("REQ-033: Export completed - %s exported, %s errors", processed, errors)
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

    context["progress"].update(context["task_id"], advance=1, avg_speed=avg_text)
    context["display"].update_info(current_file=f"📁 {truncated}", detections="", avg_speed=avg_text)
    context["live"].update(context["display"])


def _stop_progress(context: dict[str, Any] | None) -> None:
    if context:
        context["live"].stop()


__all__ = ["export_database_to_sidecars"]
