"""
Benchmark script to compare file-based vs database-based sidecar scanning.

REQ-010: All code components directly linked to requirements.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_file_based_scanning(
    input_dir: Path,
    image_paths: list[Path],
    workers: int = 8,
) -> tuple[float, dict[str, set[str]]]:
    """
    Benchmark file-based sidecar scanning.
    
    Args:
        input_dir: Input directory containing images.
        image_paths: List of image paths to check.
        workers: Number of parallel workers.
    
    Returns:
        Tuple of (elapsed_time, results_dict) where results_dict maps path -> set of analyses.
    """
    logger.info(f"Benchmarking file-based scanning with {workers} workers...")
    
    results: dict[str, set[str]] = {}
    
    def check_sidecar(image_path: Path) -> tuple[str, set[str]]:
        """Check sidecar file for existing analyses."""
        existing: set[str] = set()
        sidecar_path = image_path.with_suffix(image_path.suffix + '.json')
        
        if sidecar_path.exists():
            try:
                with open(sidecar_path) as f:
                    metadata = json.load(f)
                
                if metadata.get('exif'):
                    existing.add('exif')
                if metadata.get('faces'):
                    existing.add('faces')
                if metadata.get('objects'):
                    existing.add('objects')
                if metadata.get('poses'):
                    existing.add('poses')
            except Exception as e:
                logger.debug(f"Failed to read sidecar {sidecar_path}: {e}")
        
        return str(image_path), existing
    
    start_time = time.time()
    
    if workers == 1:
        # Sequential
        for image_path in image_paths:
            path_str, existing = check_sidecar(image_path)
            results[path_str] = existing
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(check_sidecar, image_path): image_path
                for image_path in image_paths
            }
            
            for future in as_completed(futures):
                try:
                    path_str, existing = future.result()
                    results[path_str] = existing
                except Exception as e:
                    logger.debug(f"Failed to scan: {e}")
    
    elapsed_time = time.time() - start_time
    
    return elapsed_time, results


def benchmark_database_based_scanning(
    database_path: Path,
    image_paths: list[Path],
) -> tuple[float, dict[str, set[str]]]:
    """
    Benchmark database-based sidecar scanning.
    
    Args:
        database_path: Path to database file.
        image_paths: List of image paths to check.
    
    Returns:
        Tuple of (elapsed_time, results_dict) where results_dict maps path -> set of analyses.
    """
    logger.info("Benchmarking database-based scanning...")
    
    from media_indexer.db.connection import DatabaseConnection
    from pony.orm import db_session
    
    results: dict[str, set[str]] = {}
    
    # Connect to database first (required before importing models)
    db_conn = DatabaseConnection(database_path)
    db_conn.connect()
    
    # Import models after connection is established
    from media_indexer.db.image import Image as DBImage
    
    try:
        start_time = time.time()
        
        with db_session:
            # Query all images at once
            # Convert to list for PonyORM compatibility (SQL IN clause)
            path_strs = [str(path) for path in image_paths]
            db_images = list(DBImage.select(lambda img: img.path in path_strs))
            
            # Build results map
            for db_image in db_images:
                existing: set[str] = set()
                
                if db_image.faces and len(db_image.faces) > 0:
                    existing.add('faces')
                if db_image.objects and len(db_image.objects) > 0:
                    existing.add('objects')
                if db_image.poses and len(db_image.poses) > 0:
                    existing.add('poses')
                if db_image.exif_data is not None:
                    existing.add('exif')
                
                results[db_image.path] = existing
            
            # Add empty sets for images not in database
            for path in image_paths:
                path_str = str(path)
                if path_str not in results:
                    results[path_str] = set()
        
        elapsed_time = time.time() - start_time
        
    finally:
        db_conn.close()
    
    return elapsed_time, results


def run_benchmark(
    input_dir: Path,
    database_path: Path,
    workers: int = 8,
    iterations: int = 3,
) -> None:
    """
    Run benchmark comparing file-based vs database-based scanning.
    
    Args:
        input_dir: Input directory containing images.
        database_path: Path to database file.
        workers: Number of parallel workers for file-based scanning.
        iterations: Number of iterations to run (for averaging).
    """
    from media_indexer.utils.file_utils import get_image_extensions
    
    # Get image files
    extensions = get_image_extensions()
    image_paths: list[Path] = []
    
    for ext in extensions:
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(image_paths)} images to benchmark")
    
    if not image_paths:
        logger.error("No images found!")
        return
    
    if not database_path.exists():
        logger.error(f"Database not found: {database_path}")
        return
    
    # Run file-based benchmark
    file_times: list[float] = []
    file_results: dict[str, set[str]] = {}
    
    logger.info(f"\n{'='*60}")
    logger.info("FILE-BASED SCANNING BENCHMARK")
    logger.info(f"{'='*60}")
    
    for i in range(iterations):
        elapsed, results = benchmark_file_based_scanning(input_dir, image_paths, workers)
        file_times.append(elapsed)
        if i == 0:
            file_results = results
        logger.info(f"Iteration {i+1}: {elapsed:.3f}s")
    
    avg_file_time = sum(file_times) / len(file_times)
    min_file_time = min(file_times)
    max_file_time = max(file_times)
    
    # Run database-based benchmark
    db_times: list[float] = []
    db_results: dict[str, set[str]] = {}
    
    logger.info(f"\n{'='*60}")
    logger.info("DATABASE-BASED SCANNING BENCHMARK")
    logger.info(f"{'='*60}")
    
    for i in range(iterations):
        elapsed, results = benchmark_database_based_scanning(database_path, image_paths)
        db_times.append(elapsed)
        if i == 0:
            db_results = results
        logger.info(f"Iteration {i+1}: {elapsed:.3f}s")
    
    avg_db_time = sum(db_times) / len(db_times)
    min_db_time = min(db_times)
    max_db_time = max(db_times)
    
    # Compare results
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS COMPARISON")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nFile-based scanning:")
    logger.info(f"  Average: {avg_file_time:.3f}s")
    logger.info(f"  Min:     {min_file_time:.3f}s")
    logger.info(f"  Max:     {max_file_time:.3f}s")
    logger.info(f"  Speed:   {len(image_paths)/avg_file_time:.1f} images/s")
    
    logger.info(f"\nDatabase-based scanning:")
    logger.info(f"  Average: {avg_db_time:.3f}s")
    logger.info(f"  Min:     {min_db_time:.3f}s")
    logger.info(f"  Max:     {max_db_time:.3f}s")
    logger.info(f"  Speed:   {len(image_paths)/avg_db_time:.1f} images/s")
    
    speedup = avg_file_time / avg_db_time if avg_db_time > 0 else float('inf')
    logger.info(f"\nSpeedup: {speedup:.2f}x {'(database faster)' if speedup > 1 else '(files faster)'}")
    
    # Verify results match
    logger.info(f"\n{'='*60}")
    logger.info("VERIFICATION")
    logger.info(f"{'='*60}")
    
    mismatches = 0
    for path_str in file_results:
        if path_str not in db_results:
            logger.warning(f"Path {path_str} not in database results")
            mismatches += 1
        elif file_results[path_str] != db_results[path_str]:
            logger.warning(
                f"Mismatch for {path_str}: "
                f"file={file_results[path_str]}, db={db_results[path_str]}"
            )
            mismatches += 1
    
    if mismatches == 0:
        logger.info("✓ Results match perfectly!")
    else:
        logger.warning(f"⚠ Found {mismatches} mismatches")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark sidecar scanning methods")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing images",
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Database file path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for file-based scanning",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run",
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        input_dir=args.input_dir,
        database_path=args.db,
        workers=args.workers,
        iterations=args.iterations,
    )

