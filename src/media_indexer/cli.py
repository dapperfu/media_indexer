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
        17: logging.INFO,  # -v (DETAILED)
        20: logging.INFO,  # default (INFO)
    }

    log_level = level_map.get(verbose, logging.INFO)

    # REQ-016: Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # REQ-016: Disable tqdm if verbosity is too high (DEBUG level)
    if verbose < 20:  # Only disable tqdm at DEBUG level (10) or below
        import tqdm

        tqdm.tqdm.__init__ = lambda _self, *_args, **_kwargs: None  # type: ignore[method-assign, assignment]


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
        "-v",
        "--verbose",
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
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sidecar files (defaults to input directory)",
    )
    add_common_args(extract_parser)
    extract_parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=Path(".checkpoint.json"),
        help="Checkpoint file path for resume functionality (REQ-011)",
    )
    extract_parser.add_argument(
        "-b",
        "--batch-size",
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
    
    # Analyze subcommand (alias for extract/annotate)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze images to extract features (REQ-030)",
        description="Analyze images to extract features (faces, objects, poses, EXIF data).",
    )
    annotate_parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing images to process",
    )
    annotate_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sidecar files (defaults to input directory)",
    )
    add_common_args(annotate_parser)
    annotate_parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=Path(".checkpoint.json"),
        help="Checkpoint file path for resume functionality (REQ-011)",
    )
    annotate_parser.add_argument(
        "-b",
        "--batch-size",
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
    
    # Duplicate all arguments for analyze subcommand (same as extract)
    analyze_parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing images to process",
    )
    analyze_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sidecar files (defaults to input directory)",
    )
    add_common_args(analyze_parser)
    analyze_parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=Path(".checkpoint.json"),
        help="Checkpoint file path for resume functionality (REQ-011)",
    )
    analyze_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel processing (default: 1 auto-scales to 4 for 12GB VRAM) (REQ-014, REQ-020)",
    )
    analyze_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (REQ-011)",
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
    stats_parser = db_subparsers.add_parser("stats", help="Display database statistics")
    
    # REQ-067: 'db init' subcommand
    init_parser = db_subparsers.add_parser("init", help="Initialize database with required tables")
    
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
    clean_parser = db_subparsers.add_parser("clean", help="Remove orphaned records")
    
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
        "-o",
        "--output-dir",
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
            if old_args and old_args[0] not in ["extract", "annotate", "convert", "db", "-h", "--help"]:
                # Insert 'extract' as the subcommand
                args = parser.parse_args(["extract"] + old_args)
        except Exception:
            # If parsing fails, show help
            parser.print_help()
            sys.exit(1)

    return args


def process_extract(args: argparse.Namespace, verbose: int) -> int:
    """
    Handle extract subcommand (read from sidecars/database and export).
    
    REQ-030: Extract features from existing sidecar files and/or database.
    
    Args:
        args: Parsed arguments.
        verbose: Verbosity level.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # REQ-030: Validate that we have either input_dir or database
    if not args.input_dir and not args.db:
        logging.error("REQ-030: Either --input-dir or --db must be specified")
        return 1
    
    if args.input_dir and not args.input_dir.exists():
        logging.error(f"REQ-030: Input directory does not exist: {args.input_dir}")
        return 1
    
    if not args.output_dir:
        logging.error("REQ-030: --output-dir is required")
        return 1
    
    logging.info("REQ-030: Extracting features from sidecars/database")
    logging.info(f"REQ-016: Verbosity level: {verbose}")
    
    # REQ-030: Initialize feature extractor (lazy import)
    try:
        from media_indexer.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor(
            input_dir=args.input_dir,
            database_path=args.db,
            output_dir=args.output_dir,
        )
        
        # REQ-030: Extract features
        stats = extractor.extract_features()
        
        # REQ-030: Exit with error code if there were errors
        if stats["error_images"] > 0:
            logging.warning(f"REQ-030: Completed with {stats['error_images']} errors")
            return 1
        else:
            logging.info("REQ-030: Extraction completed successfully")
            return 0
    
    except Exception as e:
        logging.error(f"REQ-030: Extraction failed: {e}")
        return 1


def process_analyze(args: argparse.Namespace, verbose: int) -> int:
    """
    Handle analyze subcommand (analyze images and populate sidecars/database).
    
    REQ-031: Analyze images and extract features.
    
    Args:
        args: Parsed arguments.
        verbose: Verbosity level.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # REQ-031: Validate input directory
    if not args.input_dir.exists():
        logging.error(f"REQ-031: Input directory does not exist: {args.input_dir}")
        return 1
    
    logging.info(f"REQ-031: Analyzing images from {args.input_dir}")
    logging.info(f"REQ-016: Verbosity level: {verbose}")
    
    # REQ-031: Initialize processor (lazy import)
    try:
        from media_indexer.processor import ImageProcessor
        
        processor = ImageProcessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir or args.input_dir,
            checkpoint_file=args.checkpoint,
            verbose=verbose,
            batch_size=args.batch_size,
            database_path=args.db,
            disable_sidecar=args.no_sidecar,
            limit=args.limit,
        )
        
        # REQ-031: Process images
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


def process_db(args: argparse.Namespace, verbose: int) -> int:
    """
    Handle database management subcommand.
    
    REQ-067: Database management operations organized under 'db' command group.
    
    Args:
        args: Parsed arguments.
        verbose: Verbosity level.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Validate database path
    if not args.db:
        logging.error("REQ-067: --db is required for db command")
        return 1
    
    # Get the db subcommand
    db_command = getattr(args, 'db_command', None)
    
    # REQ-067: Handle 'db init' subcommand
    if db_command == "init":
        try:
            from media_indexer.db.connection import DatabaseConnection
            
            logging.info(f"REQ-067: Initializing database at {args.db}")
            db_conn = DatabaseConnection(args.db)
            db_conn.connect()
            logging.info(f"REQ-067: Database initialized successfully at {args.db}")
            return 0
        except Exception as e:
            logging.error(f"REQ-067: Database initialization failed: {e}")
            return 1
    
    # For other commands, database must exist
    if not args.db.exists():
        logging.error(f"REQ-067: Database does not exist: {args.db}. Run 'media-indexer db init --db <path>' first.")
        return 1
    
    try:
        from media_indexer.db_manager import DatabaseManager
        
        db_manager = DatabaseManager(args.db)
        
        # REQ-067: Handle 'db stats' subcommand
        if db_command == "stats":
            db_manager.print_statistics()
            return 0
        
        # REQ-067: Handle 'db search' subcommand
        elif db_command == "search":
            if not args.query:
                logging.error("REQ-067: --query required for search command")
                return 1
            
            limit = getattr(args, 'limit', 10)
            results = db_manager.search_images(args.query, limit)
            print(f"\n=== Search Results (showing {len(results)} of {limit}) ===")
            for result in results:
                print(f"\n{result['path']}")
                print(f"  Faces: {result['faces']}, Objects: {result['objects']}, Poses: {result['poses']}")
                print(f"  Size: {result['file_size']} bytes")
                if result['width'] and result['height']:
                    print(f"  Dimensions: {result['width']}x{result['height']}")
            return 0
        
        # REQ-067: Handle 'db clean' subcommand
        elif db_command == "clean":
            logging.info("REQ-067: Cleaning database...")
            stats = db_manager.clean_database()
            print(f"\n=== Cleanup Statistics ===")
            print(f"Images checked: {stats['images_checked']}")
            print(f"Images removed: {stats['images_removed']}")
            print(f"Files not found: {stats['files_not_found']}")
            return 0
        
        else:
            logging.error("REQ-067: No db subcommand specified. Use: stats, init, search, or clean")
            return 1
    
    except Exception as e:
        logging.error(f"REQ-067: Database command failed: {e}")
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
        # Should not reach here due to default handling in parse_args
        logging.error("REQ-029: Unknown subcommand")
        sys.exit(1)


if __name__ == "__main__":
    main()
