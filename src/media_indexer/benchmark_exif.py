"""
Benchmark script for EXIF extraction and storage performance.

REQ-010: All code components directly linked to requirements.
Benchmarks relational EXIF storage with different file counts.
"""

import logging
import random
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_image_files(picture_dir: Path, limit: int | None = None) -> list[Path]:
    """Collect image files from directory.

    Parameters
    ----------
    picture_dir : Path
        Directory containing images.
    limit : int, optional
        Maximum number of files to collect.

    Returns
    -------
    list[Path]
        List of image file paths.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".raw", ".cr2", ".nef", ".arw"]
    image_files: list[Path] = []

    for ext in image_extensions:
        image_files.extend(picture_dir.rglob(f"*{ext}"))
        image_files.extend(picture_dir.rglob(f"*{ext.upper()}"))

    # Shuffle and limit if needed
    if limit and len(image_files) > limit:
        random.shuffle(image_files)
        image_files = image_files[:limit]

    return image_files


def benchmark_exif_extraction(
    image_files: list[Path],
    exif_extractor: Any,
) -> tuple[float, int, int]:
    """Benchmark EXIF extraction only.

    Parameters
    ----------
    image_files : list[Path]
        List of image file paths.
    exif_extractor : Any
        EXIF extractor instance.

    Returns
    -------
    tuple[float, int, int]
        (elapsed_time, success_count, failure_count)
    """
    logger.info(f"Benchmarking EXIF extraction for {len(image_files)} files...")

    success_count = 0
    failure_count = 0
    start_time = time.time()

    for image_path in image_files:
        try:
            exif_data = exif_extractor.extract_from_path(image_path)
            if exif_data:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            logger.debug(f"EXIF extraction failed for {image_path}: {e}")
            failure_count += 1

    elapsed_time = time.time() - start_time

    return elapsed_time, success_count, failure_count


def benchmark_exif_storage(
    image_files: list[Path],
    exif_extractor: Any,
    database_path: Path,
) -> tuple[float, int, int]:
    """Benchmark EXIF extraction and relational storage.

    Parameters
    ----------
    image_files : list[Path]
        List of image file paths.
    exif_extractor : Any
        EXIF extractor instance.
    database_path : Path
        Path to database file.

    Returns
    -------
    tuple[float, int, int]
        (elapsed_time, success_count, failure_count)
    """
    from pony.orm import db_session

    from media_indexer.db.connection import DatabaseConnection
    from media_indexer.db.hash_util import calculate_file_hash, get_file_size
    from media_indexer.db.image import Image as DBImage
    from media_indexer.db.metadata_converter import MetadataConverter

    logger.info(f"Benchmarking EXIF extraction + storage for {len(image_files)} files...")

    # Connect to database
    db_conn = DatabaseConnection(database_path)
    db_conn.connect()

    success_count = 0
    failure_count = 0
    start_time = time.time()

    try:
        with db_session:
            for image_path in image_files:
                try:
                    # Extract EXIF
                    exif_data = exif_extractor.extract_from_path(image_path)
                    if not exif_data:
                        failure_count += 1
                        continue

                    # Create image entity
                    file_hash = calculate_file_hash(image_path)
                    file_size = get_file_size(image_path) or 0

                    # Check if exists
                    existing_image = DBImage.get_by_path(str(image_path))
                    if existing_image:
                        # Delete existing EXIF tag values for re-benchmarking
                        from media_indexer.db.exif_tag_value import EXIFTagValue

                        existing_values = EXIFTagValue.get_by_image(existing_image)
                        for value in existing_values:
                            value.delete()
                        db_image = existing_image
                    else:
                        db_image = MetadataConverter.create_db_image(
                            str(image_path),
                            file_hash=file_hash,
                            file_size=file_size,
                        )

                    # Store EXIF relationally
                    metadata = {"exif": exif_data}
                    MetadataConverter.metadata_to_db_entities(
                        str(image_path),
                        db_image,
                        metadata,
                    )

                    success_count += 1
                except Exception as e:
                    logger.debug(f"EXIF storage failed for {image_path}: {e}")
                    failure_count += 1

            # Commit all changes
            db_session.commit()

    finally:
        db_conn.close()

    elapsed_time = time.time() - start_time

    return elapsed_time, success_count, failure_count


def run_benchmark(
    picture_dir: Path,
    database_path: Path,
    file_counts: list[int] = [100, 1000, 10000],
) -> None:
    """Run EXIF benchmark for different file counts.

    Parameters
    ----------
    picture_dir : Path
        Directory containing images.
    database_path : Path
        Path to database file.
    file_counts : list[int]
        List of file counts to benchmark.
    """
    from media_indexer.exif_extractor import get_exif_extractor

    logger.info(f"Collecting image files from {picture_dir}...")
    all_image_files = collect_image_files(picture_dir)

    if not all_image_files:
        logger.error(f"No image files found in {picture_dir}")
        return

    logger.info(f"Found {len(all_image_files)} total image files")

    exif_extractor = get_exif_extractor()

    results: list[dict[str, Any]] = []

    for count in file_counts:
        if count > len(all_image_files):
            logger.warning(
                f"Skipping {count} files benchmark - only {len(all_image_files)} files available"
            )
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"BENCHMARKING {count} FILES")
        logger.info(f"{'=' * 60}")

        # Select subset
        sample_files = all_image_files[:count]

        # Benchmark extraction only
        extract_time, extract_success, extract_fail = benchmark_exif_extraction(
            sample_files,
            exif_extractor,
        )

        # Create fresh database for storage benchmark
        storage_db_path = database_path.parent / f"{database_path.stem}_{count}{database_path.suffix}"
        if storage_db_path.exists():
            storage_db_path.unlink()

        # Benchmark extraction + storage
        storage_time, storage_success, storage_fail = benchmark_exif_storage(
            sample_files,
            exif_extractor,
            storage_db_path,
        )

        results.append(
            {
                "file_count": count,
                "extract_time": extract_time,
                "extract_success": extract_success,
                "extract_fail": extract_fail,
                "storage_time": storage_time,
                "storage_success": storage_success,
                "storage_fail": storage_fail,
            }
        )

        logger.info(f"\nExtraction only:")
        logger.info(f"  Time:      {extract_time:.3f}s")
        logger.info(f"  Success:   {extract_success}")
        logger.info(f"  Failures:  {extract_fail}")
        logger.info(f"  Speed:     {count / extract_time:.1f} files/s")

        logger.info(f"\nExtraction + Storage:")
        logger.info(f"  Time:      {storage_time:.3f}s")
        logger.info(f"  Success:   {storage_success}")
        logger.info(f"  Failures:  {storage_fail}")
        logger.info(f"  Speed:     {count / storage_time:.1f} files/s")
        logger.info(f"  Overhead: {storage_time - extract_time:.3f}s ({((storage_time - extract_time) / extract_time * 100):.1f}%)")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")

    logger.info(f"\n{'Files':<10} {'Extract (s)':<15} {'Storage (s)':<15} {'Extract/s':<15} {'Storage/s':<15}")
    logger.info("-" * 70)

    for result in results:
        logger.info(
            f"{result['file_count']:<10} "
            f"{result['extract_time']:<15.3f} "
            f"{result['storage_time']:<15.3f} "
            f"{result['file_count'] / result['extract_time']:<15.1f} "
            f"{result['file_count'] / result['storage_time']:<15.1f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark EXIF extraction and storage")
    parser.add_argument(
        "picture_dir",
        type=Path,
        help="Directory containing images (e.g., /tun/pictures/)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("benchmark_exif.db"),
        help="Database file path (default: benchmark_exif.db)",
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[100, 1000, 10000],
        help="File counts to benchmark (default: 100 1000 10000)",
    )

    args = parser.parse_args()

    if not args.picture_dir.exists():
        logger.error(f"Directory not found: {args.picture_dir}")
        exit(1)

    run_benchmark(
        picture_dir=args.picture_dir,
        database_path=args.db,
        file_counts=args.counts,
    )

