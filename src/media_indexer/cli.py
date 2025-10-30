"""
Command-Line Interface for Media Indexer

REQ-016: Multi-level verbosity logging with support for -v through -vvvv.
REQ-010: All code components directly linked to requirements.
REQ-029: Subcommand-based CLI operation with extract, annotate, and convert commands.
REQ-067: Database management operations organized under 'db' command group.
"""

import argparse
import logging
import sys
from pathlib import Path

from media_indexer.cli_handlers import (
    process_analyze,
    process_convert,
    process_db,
    process_extract,
)


def setup_logging(verbose: int) -> None:
    """
    Setup logging with multi-level verbosity.

    REQ-016: Implement verbosity levels:
        -vvvvv = DEBUG (10)
        -vvvv = TRACE (12)
        -vvv = VERBOSE (15)
        -vv = DETAILED (17)
        -v = INFO (20)
        default = WARNING (30)

    Args:
        verbose: Verbosity level (lower = more verbose).
    """
    # REQ-016: Set up logging level based on verbosity
    level_map = {
        10: logging.DEBUG,  # -vvvvv
        12: logging.DEBUG,  # -vvvv (TRACE with TQDM)
        15: logging.DEBUG,  # -vvv (VERBOSE)
        17: logging.INFO,  # -vv (DETAILED)
        20: logging.INFO,  # -v (INFO)
        30: logging.WARNING,  # default (WARNING)
    }

    log_level = level_map.get(verbose, logging.WARNING)

    # REQ-016: Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # REQ-016: OpenCV warnings are suppressed globally at module import time
    # (see top of file for OpenCV warning suppression)


def add_common_args(parser: argparse.ArgumentParser, include_db: bool = False) -> None:
    """
    Add common arguments to a parser.

    REQ-016: Multi-level verbosity.
    REQ-017: Configuration file support.
    REQ-025: Database storage (optional).

    Args:
        parser: Argument parser to add arguments to.
        include_db: If True, include --db and --no-sidecar arguments.
    """
    # REQ-016: Multi-level verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v INFO, -vv DETAILED, -vvv VERBOSE, -vvvv TRACE with TQDM, -vvvvv DEBUG). Default shows only warnings and errors.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Configuration file (YAML/TOML) (REQ-017)",
    )

    # REQ-025: Database storage option (only for subcommands that need it)
    if include_db:
        parser.add_argument(
            "--db",
            type=Path,
            default=None,
            help="Database file path for storing metadata (SQLite)",
        )

        # REQ-026: No sidecar option
        parser.add_argument(
            "--no-sidecar",
            action="store_true",
            help="Disable sidecar file generation when using --db (REQ-026)",
        )

        # Option to use sidecars as source for existing data queries
        parser.add_argument(
            "--use-sidecars-for-existing",
            action="store_true",
            help="Use sidecar files as source for existing data queries instead of database (useful when sidecars are more complete)",
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    REQ-016: Parse verbose flags for multi-level verbosity.
    REQ-017: Support configuration file.
    REQ-029: Support subcommand-based operation.

    Returns:
        Parsed arguments.
    """
    # REQ-029: Create main parser with subcommands
    parser = argparse.ArgumentParser(
        description="Media Indexer - GPU-accelerated image analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add common arguments (without --db at top level)
    add_common_args(parser, include_db=False)

    # REQ-029: Create subparsers for subcommands
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # REQ-030: Extract subcommand
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract features from images (REQ-030)",
        description="Extract features (faces, objects, poses, EXIF) from images.",
    )
    extract_parser.add_argument(
        "input_dirs",
        type=Path,
        nargs="+",
        help="Input directories or files containing images to process (REQ-078)",
    )
    add_common_args(extract_parser, include_db=True)
    extract_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel processing (default: 1 auto-scales to 4 for 12GB VRAM) (REQ-014, REQ-020)",
    )
    extract_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force reprocessing even if analyses already exist (REQ-013)",
    )
    extract_parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Number of retries for failed images (REQ-015)",
    )
    extract_parser.add_argument(
        "--formats",
        nargs="+",
        default=["jpg", "jpeg", "png", "tiff", "raw"],
        help="Image formats to process (REQ-018)",
    )
    extract_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing on small subsets) (REQ-038)",
    )

    # REQ-031: Annotate subcommand (same as extract functionally)
    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Add features to images (REQ-031)",
        description="Process images and add features (faces, objects, poses, EXIF) as annotations.",
    )

    # Analyze subcommand (alias for extract/annotate)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze images to extract features (REQ-030)",
        description="Analyze images to extract features (faces, objects, poses, EXIF data).",
    )
    annotate_parser.add_argument(
        "input_dirs",
        type=Path,
        nargs="+",
        help="Input directories or files containing images to process (REQ-078)",
    )
    add_common_args(annotate_parser, include_db=True)
    annotate_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel processing (default: 1 auto-scales to 4 for 12GB VRAM) (REQ-014, REQ-020)",
    )
    annotate_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force reprocessing even if analyses already exist (REQ-013)",
    )
    annotate_parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Number of retries for failed images (REQ-015)",
    )
    annotate_parser.add_argument(
        "--formats",
        nargs="+",
        default=["jpg", "jpeg", "png", "tiff", "raw"],
        help="Image formats to process (REQ-018)",
    )
    annotate_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing on small subsets) (REQ-038)",
    )
    annotate_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for sidecar scanning (default: 8) (REQ-020)",
    )
    annotate_parser.add_argument(
        "--no-face-attributes",
        action="store_true",
        help="Disable DeepFace age and emotion analysis (enabled by default) (REQ-081)",
    )

    # Duplicate all arguments for analyze subcommand (same as extract)
    analyze_parser.add_argument(
        "input_dirs",
        type=Path,
        nargs="+",
        help="Input directories or files containing images to process (REQ-078)",
    )
    add_common_args(analyze_parser, include_db=True)
    analyze_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel processing (default: 1 auto-scales to 4 for 12GB VRAM) (REQ-014, REQ-020)",
    )
    analyze_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force reprocessing even if analyses already exist (REQ-013)",
    )
    analyze_parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Number of retries for failed images (REQ-015)",
    )
    analyze_parser.add_argument(
        "--formats",
        nargs="+",
        default=["jpg", "jpeg", "png", "tiff", "raw"],
        help="Image formats to process (REQ-018)",
    )
    analyze_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing on small subsets) (REQ-038)",
    )
    analyze_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for sidecar scanning (default: 8) (REQ-020)",
    )
    analyze_parser.add_argument(
        "--no-face-attributes",
        action="store_true",
        help="Disable DeepFace age and emotion analysis (enabled by default) (REQ-081)",
    )

    # REQ-067: Database management subcommand group
    db_parser = subparsers.add_parser(
        "db",
        help="Database management operations (REQ-067)",
        description="Manage database: initialize, view statistics, search, and clean.",
    )
    db_parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Database file path",
    )
    # REQ-067: Create sub-subcommands under 'db'
    db_subparsers = db_parser.add_subparsers(dest="db_command")

    # REQ-067: 'db stats' subcommand
    db_subparsers.add_parser("stats", help="Display database statistics")

    # REQ-067: 'db init' subcommand
    db_subparsers.add_parser("init", help="Initialize database with required tables")

    # REQ-067: 'db migrate' subcommand
    db_subparsers.add_parser("migrate", help="Run database migrations to update schema")

    # REQ-067: 'db search' subcommand
    search_parser = db_subparsers.add_parser("search", help="Search for images")
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of results",
    )

    # REQ-067: 'db clean' subcommand
    db_subparsers.add_parser("clean", help="Remove orphaned records")

    # REQ-032, REQ-033, REQ-034: Convert subcommand
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert between sidecar and database formats (REQ-032, REQ-033)",
        description="Migrate data between sidecar files and database storage.",
    )
    convert_parser.add_argument(
        "input_dirs",
        type=Path,
        nargs="+",
        help="Input directories or files containing images or sidecar files (REQ-078)",
    )
    add_common_args(convert_parser, include_db=True)
    # REQ-034: Direction flag
    convert_parser.add_argument(
        "--direction",
        type=str,
        choices=["to-db", "to-sidecar"],
        required=True,
        help="Conversion direction: 'to-db' (import sidecar to database) or 'to-sidecar' (export database to sidecar) (REQ-034)",
    )
    convert_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing (default: 1) (REQ-020)",
    )

    args = parser.parse_args()

    # REQ-029: If no subcommand specified, show help
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    return args


def main() -> None:
    """
    Main entry point for CLI.

    REQ-002, REQ-016, REQ-029: Handle subcommands with specified verbosity.
    REQ-038: Lazy loading for optimal startup performance.
    """
    args = parse_args()

    # REQ-016: Convert verbose count to level
    # 0 = 30 (WARNING), 1 = 20 (INFO), 2 = 17 (DETAILED), 3 = 15 (VERBOSE), 4 = 12 (TRACE), 5+ = 10 (DEBUG)
    verbosity_map = {
        0: 30,  # WARNING (default, quiet)
        1: 20,  # INFO
        2: 17,  # DETAILED
        3: 15,  # VERBOSE
        4: 12,  # TRACE (with TQDM)
        5: 10,  # DEBUG
    }
    verbose = verbosity_map.get(args.verbose, 10)

    # REQ-016: Setup logging
    setup_logging(verbose)

    # REQ-016, REQ-038: Suppress ONNX Runtime and OpenCV verbose output only when needed
    # (lazy load to avoid importing cv2 at startup for commands that don't need it)
    if args.command in ["extract", "annotate", "analyze"]:
        from media_indexer.utils.suppression import setup_suppression

        setup_suppression()

    # REQ-029: Route to appropriate subcommand handler
    if args.command == "extract":
        sys.exit(process_extract(args, verbose))
    elif args.command in ["annotate", "analyze"]:
        sys.exit(process_analyze(args, verbose))
    elif args.command == "convert":
        sys.exit(process_convert(args, verbose))
    elif args.command == "db":
        # REQ-067: Route to database management handler
        sys.exit(process_db(args, verbose))
    else:
        # Should not reach here due to handling in parse_args
        logging.error("REQ-029: Unknown subcommand")
        sys.exit(1)


if __name__ == "__main__":
    main()
