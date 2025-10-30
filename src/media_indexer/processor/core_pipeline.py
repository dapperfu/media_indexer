"""Image processing orchestration helpers.

REQ-072: Splits the processing pipeline into dedicated helpers below the
500-line limit, improving maintainability while retaining original behaviour.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from media_indexer.processor.object_emoji import get_object_emoji
from media_indexer.processor.progress import create_rich_progress_bar
from media_indexer.raw_converter import cleanup_temp_files
from media_indexer.utils.cancellation import CancellationManager
from media_indexer.utils.sidecar_utils import read_sidecar_metadata

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from media_indexer.processor.core import ImageProcessor

logger = logging.getLogger(__name__)


_DB_QUERY_BATCH_SIZE = 256


@dataclass(slots=True)
class RichProgressContext:
    """Wraps Rich-based progress tracking for reuse across phases."""

    total: int
    unit: str
    progress: Any
    display: Any
    live: Any
    task_id: Any
    start_time: float
    processed: int = 0

    def stop(self) -> None:
        """Tear down the live display safely."""

        self.live.stop()


def run_processing(processor: ImageProcessor) -> dict[str, Any]:
    """Coordinate the end-to-end processing pipeline."""

    cancellation_manager = CancellationManager()
    cancellation_manager.setup_signal_handler()

    try:
        logger.info(
            "REQ-002: Starting image processing with batch size %s",
            processor.batch_size,
        )

        processor._initialize_components()

        images = processor._get_image_files()
        processor.stats_tracker.set("total_images", len(images))
        logger.info("REQ-002: Found %s images", len(images))

        images_to_process, skipped_count = collect_images_to_process(processor, images, cancellation_manager)
        processor.stats_tracker.set("skipped_images", skipped_count)

        required_analyses = processor._get_required_analyses()
        _log_analysis_summary(required_analyses, skipped_count, images_to_process)

        if not images_to_process:
            processor.stats_tracker.finalize()
            processor.stats_tracker.print_statistics()
            return processor.stats_tracker.get_stats()

        process_image_batches(processor, images_to_process, cancellation_manager)

        processor.stats_tracker.finalize()
        processor.stats_tracker.print_statistics()
    finally:
        cleanup_temp_files()
        if processor.database_connection:
            processor.database_connection.close()
        cancellation_manager.cleanup()

    return processor.stats_tracker.get_stats()


def collect_images_to_process(
    processor: ImageProcessor,
    images: list[Path],
    cancellation_manager: CancellationManager,
) -> tuple[list[tuple[Path, set[str]]], int]:
    """Determine which images require additional analyses."""

    if processor.database_connection:
        return _collect_with_database(processor, images, cancellation_manager)

    return _collect_with_sidecars(processor, images, cancellation_manager)


def process_image_batches(
    processor: ImageProcessor,
    images_to_process: list[tuple[Path, set[str]]],
    cancellation_manager: CancellationManager,
) -> None:
    """Execute detection pipelines for each required image."""

    context = _init_rich_context(
        total=len(images_to_process),
        desc="Processing images",
        unit="img",
        verbose=processor.verbose,
        show_detections=True,
    )

    batch_size = processor.batch_size
    try:
        for start in range(0, len(images_to_process), batch_size):
            if cancellation_manager.is_cancelled():
                logger.warning("REQ-015: Processing interrupted by user")
                break

            batch = images_to_process[start : start + batch_size]
            executor: ThreadPoolExecutor | None = None
            try:
                executor = ThreadPoolExecutor(max_workers=len(batch))
                futures = {
                    executor.submit(
                        processor._process_image_wrapper,
                        image_path,
                        missing,
                        cancellation_manager,
                    ): image_path
                    for image_path, missing in batch
                }

                for future in as_completed(futures):
                    if cancellation_manager.is_cancelled():
                        logger.warning("REQ-015: Processing interrupted by user")
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        break

                    image_path = futures[future]
                    try:
                        success, detections = future.result()
                    except CancelledError:
                        success = False
                        detections = {}
                    except Exception as exc:  # noqa: BLE001
                        logger.error("REQ-015: Error processing %s: %s", image_path, exc)
                        processor.stats_tracker.update_increment("error_images")
                        _advance_progress(
                            context,
                            image_path.name,
                            detections=f"  ❌ FAILED: {str(exc)[:50]}",
                        )
                        continue

                    if success:
                        processor.stats_tracker.update_increment("processed_images")
                        summary = _format_detection_summary(detections)
                        detection_line = f"  👤 {summary}" if summary else "  ✓ Processed"
                        _advance_progress(context, image_path.name, detection_line)
                    else:
                        processor.stats_tracker.update_increment("error_images")
                        _advance_progress(
                            context,
                            image_path.name,
                            detections="  ❌ ERROR",
                        )
            finally:
                if executor:
                    if cancellation_manager.is_cancelled():
                        executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        executor.shutdown(wait=True)
    finally:
        _stop_context(context)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_with_database(
    processor: ImageProcessor,
    images: list[Path],
    cancellation_manager: CancellationManager,
) -> tuple[list[tuple[Path, set[str]]], int]:
    """Collect images requiring processing when database storage is enabled."""

    context = _init_rich_context(
        total=len(images),
        desc="Querying database",
        unit="file",
        verbose=processor.verbose,
    )

    images_to_process: list[tuple[Path, set[str]]] = []
    skipped_count = 0
    batch_size = max(1, min(_DB_QUERY_BATCH_SIZE, len(images)))
    required = processor._get_required_analyses()

    try:
        for start in range(0, len(images), batch_size):
            if cancellation_manager.is_cancelled():
                logger.warning("REQ-015: Processing interrupted by user")
                break

            batch = images[start : start + batch_size]
            db_analyses = processor.analysis_scanner.get_existing_analyses_from_database_batch(batch)

            for image_path in batch:
                if cancellation_manager.is_cancelled():
                    logger.warning("REQ-015: Processing interrupted by user")
                    break

                existing = set(db_analyses.get(str(image_path), set()))

                if not processor.disable_sidecar:
                    sidecar_existing = _load_sidecar_metadata(
                        processor,
                        image_path,
                        required,
                        cancellation_manager,
                    )
                    if sidecar_existing is None:
                        skipped_count += 1
                        _advance_progress(context, image_path.name)
                        continue
                    existing.update(sidecar_existing)

                missing = required - existing
                if missing:
                    images_to_process.append((image_path, missing))
                else:
                    skipped_count += 1

                _advance_progress(context, image_path.name)
    finally:
        _stop_context(context)

    return images_to_process, skipped_count


def _collect_with_sidecars(
    processor: ImageProcessor,
    images: list[Path],
    cancellation_manager: CancellationManager,
) -> tuple[list[tuple[Path, set[str]]], int]:
    """Collect images requiring processing using sidecar scanning."""

    images_to_process: list[tuple[Path, set[str]]] = []
    skipped_count = 0

    context = _init_rich_context(
        total=len(images),
        desc="Scanning sidecars",
        unit="file",
        verbose=processor.verbose,
    )

    scan_workers = min(processor.scan_workers, len(images)) or 1

    try:
        if scan_workers == 1:
            for image_path in images:
                if cancellation_manager.is_cancelled():
                    logger.warning("REQ-015: Processing interrupted by user")
                    break

                needs_processing, missing = processor._needs_processing(image_path)
                if needs_processing:
                    images_to_process.append((image_path, missing))
                else:
                    skipped_count += 1

                _advance_progress(context, image_path.name)
        else:
            executor: ThreadPoolExecutor | None = None
            try:
                executor = ThreadPoolExecutor(max_workers=scan_workers)
                futures = {executor.submit(processor._needs_processing, path): path for path in images}

                for future in as_completed(futures):
                    if cancellation_manager.is_cancelled():
                        logger.warning("REQ-015: Processing interrupted by user")
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        break

                    image_path = futures[future]
                    try:
                        needs_processing, missing = future.result()
                        if needs_processing:
                            images_to_process.append((image_path, missing))
                        else:
                            skipped_count += 1
                    except CancelledError:
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("REQ-013: Failed to scan %s: %s", image_path, exc)
                        images_to_process.append((image_path, processor._get_required_analyses()))
                    finally:
                        _advance_progress(context, image_path.name)
            finally:
                if executor:
                    if cancellation_manager.is_cancelled():
                        executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        executor.shutdown(wait=True)
    finally:
        _stop_context(context)

    return images_to_process, skipped_count


def _load_sidecar_metadata(
    processor: ImageProcessor,
    image_path: Path,
    required: set[str],
    cancellation_manager: CancellationManager,
) -> set[str] | None:
    """Load metadata from sidecar files when database data is insufficient."""

    if processor.disable_sidecar:
        return set()

    sidecar_path = processor.analysis_scanner.find_sidecar_path(image_path)
    if not sidecar_path or not sidecar_path.exists():
        return set()

    try:
        metadata = read_sidecar_metadata(sidecar_path, processor.sidecar_generator)
    except Exception as exc:  # noqa: BLE001
        logger.debug("REQ-013: Failed to read sidecar for %s: %s", image_path, exc)
        return set()

    if metadata.get("raw_conversion_failed") and not processor.force:
        logger.debug(
            "REQ-040: Skipping %s - RAW conversion failed previously (use --force to retry)",
            image_path,
        )
        return None

    existing: set[str] = set()
    if metadata.get("exif"):
        existing.add("exif")
    if metadata.get("faces"):
        existing.add("faces")
    if metadata.get("objects"):
        existing.add("objects")
    if metadata.get("poses"):
        existing.add("poses")

    return existing


def _init_rich_context(
    *,
    total: int,
    desc: str,
    unit: str,
    verbose: int,
    show_detections: bool = False,
) -> RichProgressContext | None:
    """Create a Rich progress context when verbose output is enabled."""

    if verbose < 15 or total == 0:
        return None

    progress, display, live = create_rich_progress_bar(
        total=total,
        desc=desc,
        unit=unit,
        verbose=verbose,
        show_detections=show_detections,
    )

    if not all((progress, display, live)):
        return None

    task_id = progress.add_task(desc, total=total, avg_speed=f"0.0 {unit}/s")
    live.start()
    return RichProgressContext(
        total=total,
        unit=unit,
        progress=progress,
        display=display,
        live=live,
        task_id=task_id,
        start_time=time.time(),
    )


def _advance_progress(
    context: RichProgressContext | None,
    current_file: str,
    detections: str | None = None,
) -> None:
    """Advance progress bars when enabled."""

    if not context:
        return

    context.processed += 1
    elapsed = time.time() - context.start_time
    avg_speed = context.processed / elapsed if elapsed > 0 else 0.0
    avg_text = f"{avg_speed:.1f} {context.unit}/s"

    truncated = current_file
    if len(truncated) > 50:
        truncated = "..." + truncated[-47:]

    context.progress.update(context.task_id, advance=1, avg_speed=avg_text)
    context.display.update_info(
        current_file=f"📁 {truncated}" if context.unit == "file" else f"📷 {truncated}",
        detections=detections or "",
        avg_speed=avg_text,
    )
    if detections:
        normalized = detections.strip()
        if "❌" in normalized or "⚠" in normalized:
            # REQ-076: Persist visible error details alongside the live progress bar.
            event_message = f"{truncated}: {normalized}"
            context.display.add_event(event_message)
    context.live.update(context.display)


def _stop_context(context: RichProgressContext | None) -> None:
    """Stop and dispose of a progress context."""

    if context:
        context.stop()


def _log_analysis_summary(
    required_analyses: Iterable[str],
    skipped_count: int,
    images_to_process: Iterable[tuple[Path, set[str]]],
) -> None:
    """Emit summary log entries for required analyses."""

    analysis_names = {
        "exif": "EXIF",
        "faces": "faces",
        "objects": "objects",
        "poses": "poses",
    }
    required_str = ", ".join(analysis_names.get(a, a) for a in sorted(required_analyses))
    logger.info("REQ-013: Required analyses: %s", required_str)
    logger.info("REQ-013: Already analyzed: %s images", skipped_count)
    logger.info("REQ-013: Needs processing: %s images", len(images_to_process))


def _format_detection_summary(detections: dict[str, Any]) -> str:
    """Format detection counts for human-readable output."""

    parts: list[str] = []
    faces = detections.get("faces", 0)
    objects = detections.get("objects", 0)
    poses = detections.get("poses", 0)
    object_class_counts = detections.get("object_class_counts", {})

    if faces:
        parts.append(f"{faces} face{'s' if faces != 1 else ''}")
    if objects:
        detail = ""
        if isinstance(object_class_counts, dict) and object_class_counts:
            sorted_classes = sorted(
                object_class_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
            labels: list[str] = []
            for label, count in sorted_classes:
                emoji = get_object_emoji(label)
                display_label = label.replace("_", " ")
                snippet = f"{emoji} {display_label}" if emoji else display_label
                if count > 1:
                    snippet = f"{snippet}×{count}"
                labels.append(snippet)
            if labels:
                detail = f" ({', '.join(labels)})"

        parts.append(f"{objects} object{'s' if objects != 1 else ''}{detail}")
    if poses:
        parts.append(f"{poses} pose{'s' if poses != 1 else ''}")

    return ", ".join(parts)
