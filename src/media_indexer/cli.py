"""
Command-Line Interface for Media Indexer

REQ-016: Multi-level verbosity logging with support for -v through -vvvv.
REQ-010: All code components directly linked to requirements.
REQ-029: Subcommand-based CLI operation with extract, annotate, and convert commands.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

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


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common arguments to a parser.

    REQ-016: Multi-level verbosity.
    REQ-017: Configuration file support.
    REQ-025: Database storage.

    Args:
        parser: Argument parser to add arguments to.
    """
    # REQ-016: Multi-level verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v INFO, -vv DETAILED, -vvv VERBOSE, -vvvv TRACE with TQDM, -vvvvv DEBUG)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Configuration file (YAML/TOML) (REQ-017)",
    )

    # REQ-025: Database storage option
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

    # Add common arguments
    add_common_args(parser)

    # REQ-029: Create subparsers for subcommands
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # REQ-030: Extract subcommand
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract features from images (REQ-030)",
        description="Extract features (faces, objects, poses, EXIF) from images.",
    )
    extract_parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing images to process",
    )
    extract_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sidecar files (defaults to input directory)",
    )
    add_common_args(extract_parser)
    extract_parser.add_argument(
        "-c", "--checkpoint",
        type=Path,
        default=Path(".checkpoint.json"),
        help="Checkpoint file path for resume functionality (REQ-011)",
    )
    extract_parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel processing (default: 1 auto-scales to 4 for 12GB VRAM) (REQ-014, REQ-020)",
    )
    extract_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (REQ-011)",
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
    annotate_parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing images to process",
    )
    annotate_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sidecar files (defaults to input directory)",
    )
    add_common_args(annotate_parser)
    annotate_parser.add_argument(
        "-c", "--checkpoint",
        type=Path,
        default=Path(".checkpoint.json"),
        help="Checkpoint file path for resume functionality (REQ-011)",
    )
    annotate_parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel processing (default: 1 auto-scales to 4 for 12GB VRAM) (REQ-014, REQ-020)",
    )
    annotate_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (REQ-011)",
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

    # REQ-032, REQ-033, REQ-034: Convert subcommand
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert between sidecar and database formats (REQ-032, REQ-033)",
        description="Migrate data between sidecar files and database storage.",
    )
    convert_parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing images or sidecar files",
    )
    add_common_args(convert_parser)
    # REQ-034: Direction flag
    convert_parser.add_argument(
        "--direction",
        type=str,
        choices=["to-db", "to-sidecar"],
        required=True,
        help="Conversion direction: 'to-db' (import sidecar to database) or 'to-sidecar' (export database to sidecar) (REQ-034)",
    )
    convert_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sidecar files (when direction=to-sidecar)",
    )

    args = parser.parse_args()

    # REQ-029: Default to 'extract' if no subcommand specified (backwards compatibility)
    if args.command is None:
        # Parse again with extract as default for backwards compatibility
        # This allows old-style usage: media-indexer /path/to/images
        try:
            # Try to parse without subcommand
            import sys
            old_args = sys.argv[1:]
            if old_args and not old_args[0] in ['extract', 'annotate', 'convert', '-h', '--help']:
                # Insert 'extract' as the subcommand
                args = parser.parse_args(['extract'] + old_args)
        except:
            # If parsing fails, show help
            parser.print_help()
            sys.exit(1)

    return args


def process_extract_or_annotate(args: argparse.Namespace, verbose: int) -> int:
    """
    Handle extract or annotate subcommands.

    REQ-002, REQ-016, REQ-030, REQ-031: Process images with specified verbosity.

    Args:
        args: Parsed arguments.
        verbose: Verbosity level.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # REQ-002: Validate input directory
    if not args.input_dir.exists():
        logging.error(f"REQ-002: Input directory does not exist: {args.input_dir}")
        return 1

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
            database_path=args.db,
            disable_sidecar=args.no_sidecar,
            limit=args.limit,
        )

        # REQ-002: Process images
        stats = processor.process()

        # REQ-012: Exit with error code if there were errors
        if stats["error_images"] > 0:
            logging.warning(f"REQ-012: Completed with {stats['error_images']} errors")
            return 1
        else:
            logging.info("REQ-012: Processing completed successfully")
            return 0

    except KeyboardInterrupt:
        logging.warning("REQ-011: Processing interrupted by user, checkpoint saved")
        return 130
    except Exception as e:
        logging.error(f"REQ-015: Processing failed: {e}")
        return 1


def process_convert(args: argparse.Namespace, verbose: int) -> int:
    """
    Handle convert subcommand.

    REQ-032, REQ-033, REQ-034: Convert between sidecar and database formats.

    Args:
        args: Parsed arguments.
        verbose: Verbosity level.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # REQ-034: Validate direction and setup
    if args.direction == "to-db":
        # REQ-032: Import sidecar files to database
        if not args.db:
            logging.error("REQ-032: --db required for to-db conversion")
            return 1
        if not args.input_dir.exists():
            logging.error(f"REQ-032: Input directory does not exist: {args.input_dir}")
            return 1

        logging.info(f"REQ-032: Converting sidecar files to database from {args.input_dir}")
        try:
            from media_indexer.sidecar_converter import import_sidecars_to_database
            import_sidecars_to_database(
                input_dir=args.input_dir,
                database_path=args.db,
                verbose=verbose,
            )
            logging.info("REQ-032: Sidecar import to database completed successfully")
            return 0
        except Exception as e:
            logging.error(f"REQ-032: Sidecar import failed: {e}")
            return 1

    elif args.direction == "to-sidecar":
        # REQ-033: Export database to sidecar files
        if not args.db:
            logging.error("REQ-033: --db required for to-sidecar conversion")
            return 1
        if not args.db.exists():
            logging.error(f"REQ-033: Database does not exist: {args.db}")
            return 1

        output_dir = args.output_dir or args.input_dir
        logging.info(f"REQ-033: Converting database to sidecar files, output: {output_dir}")
        try:
            from media_indexer.sidecar_converter import export_database_to_sidecars
            export_database_to_sidecars(
                database_path=args.db,
                output_dir=output_dir,
                verbose=verbose,
            )
            logging.info("REQ-033: Database export to sidecar completed successfully")
            return 0
        except Exception as e:
            logging.error(f"REQ-033: Database export failed: {e}")
            return 1

    return 1


def main() -> None:
    """
    Main entry point for CLI.

    REQ-002, REQ-016, REQ-029: Handle subcommands with specified verbosity.
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

    # REQ-029: Route to appropriate subcommand handler
    if args.command in ["extract", "annotate"]:
        sys.exit(process_extract_or_annotate(args, verbose))
    elif args.command == "convert":
        sys.exit(process_convert(args, verbose))
    else:
        # Should not reach here due to default handling in parse_args
        logging.error("REQ-029: Unknown subcommand")
        sys.exit(1)


if __name__ == "__main__":
    main()

