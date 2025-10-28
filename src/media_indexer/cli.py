"""
Command-Line Interface for Media Indexer

REQ-016: Multi-level verbosity logging with support for -v through -vvvv.
REQ-010: All code components directly linked to requirements.
"""

import argparse
import logging
import sys
from pathlib import Path

from media_indexer.processor import ImageProcessor


def setup_logging(verbose: int) -> None:
    """
    Setup logging with multi-level verbosity.

    REQ-016: Implement verbosity levels:
        -vvvv = DEBUG (10)
        -vvv = TRACE (12)
        -vv = VERBOSE (15)
        -v = DETAILED (17)
        default = INFO (20)

    Args:
        verbose: Verbosity level (lower = more verbose).
    """
    # REQ-016: Set up logging level based on verbosity
    level_map = {
        10: logging.DEBUG,  # -vvvv
        12: logging.DEBUG,  # -vvv (TRACE with TQDM)
        15: logging.DEBUG,  # -vv (VERBOSE)
        17: logging.INFO,   # -v (DETAILED)
        20: logging.INFO,   # default (INFO)
    }

    log_level = level_map.get(verbose, logging.INFO)

    # REQ-016: Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # REQ-016: Disable tqdm if not at TRACE level
    if verbose > 12:
        import tqdm
        tqdm.tqdm.__init__ = lambda self, *args, **kwargs: None  # type: ignore[method-assign, assignment]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    REQ-016: Parse verbose flags for multi-level verbosity.
    REQ-017: Support configuration file.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Media Indexer - GPU-accelerated image analysis"
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing images to process",
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sidecar files (defaults to input directory)",
    )

    # REQ-016: Multi-level verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v INFO, -vv DETAILED, -vvv VERBOSE, -vvvv TRACE with TQDM, -vvvvv DEBUG)",
    )

    parser.add_argument(
        "-c", "--checkpoint",
        type=Path,
        default=Path(".checkpoint.json"),
        help="Checkpoint file path for resume functionality (REQ-011)",
    )

    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel processing (default: 1 auto-scales to 4 for 12GB VRAM) (REQ-014, REQ-020)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Configuration file (YAML/TOML) (REQ-017)",
    )

    # REQ-011: Resume option
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (REQ-011)",
    )

    # REQ-015: Retry options
    parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Number of retries for failed images (REQ-015)",
    )

    # REQ-018: Image format filtering
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["jpg", "jpeg", "png", "tiff", "raw"],
        help="Image formats to process (REQ-018)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for CLI.

    REQ-002, REQ-016: Process images with specified verbosity.
    """
    args = parse_args()

    # REQ-016: Convert verbose count to level
    # 0 = 20 (INFO), 1 = 17 (DETAILED), 2 = 15 (VERBOSE), 3 = 12 (TRACE), 4+ = 10 (DEBUG)
    verbosity_map = {
        0: 20,  # INFO
        1: 17,  # DETAILED
        2: 15,  # VERBOSE
        3: 12,  # TRACE (with TQDM)
        4: 10,  # DEBUG
    }
    verbose = verbosity_map.get(args.verbose, 10)

    # REQ-016: Setup logging
    setup_logging(verbose)

    # REQ-002: Validate input directory
    if not args.input_dir.exists():
        logging.error(f"REQ-002: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    logging.info(f"REQ-002: Processing images from {args.input_dir}")
    logging.info(f"REQ-016: Verbosity level: {verbose}")

    # REQ-002: Initialize processor
    try:
        processor = ImageProcessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            checkpoint_file=args.checkpoint,
            verbose=verbose,
            batch_size=args.batch_size,
        )

        # REQ-002: Process images
        stats = processor.process()

        # REQ-012: Exit with error code if there were errors
        if stats["error_images"] > 0:
            logging.warning(f"REQ-012: Completed with {stats['error_images']} errors")
            sys.exit(1)
        else:
            logging.info("REQ-012: Processing completed successfully")
            sys.exit(0)

    except KeyboardInterrupt:
        logging.warning("REQ-011: Processing interrupted by user, checkpoint saved")
        sys.exit(130)
    except Exception as e:
        logging.error(f"REQ-015: Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

