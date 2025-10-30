"""CLI command handlers for Media Indexer subcommands.

REQ-072: Maintains modular structure with sub-500 line modules.
"""

from __future__ import annotations

import argparse
import logging


def process_extract(args: argparse.Namespace, verbose: int) -> int:
    """Handle the ``extract`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the extract subcommand.
    verbose : int
        Derived verbosity level.

    Returns
    -------
    int
        Exit code where ``0`` indicates success.
    """
    input_dirs = getattr(args, "input_dirs", [])
    if not input_dirs and not args.db:
        logging.error("REQ-030: Either input directories or --db must be specified")
        return 1

    # Validate all input directories exist
    for input_dir in input_dirs:
        if not input_dir.exists():
            logging.error("REQ-030: Input directory does not exist: %s", input_dir)
            return 1

    logging.info("REQ-030: Extracting features from sidecars/database")
    logging.info("REQ-016: Verbosity level: %s", verbose)

    total_errors = 0
    try:
        from media_indexer.feature_extractor import FeatureExtractor

        for input_dir in input_dirs:
            logging.info("REQ-030: Processing directory: %s", input_dir)
            extractor = FeatureExtractor(
                input_dir=input_dir,
                database_path=args.db,
                output_dir=input_dir,
            )

            stats = extractor.extract_features()

            if stats["error_images"] > 0:
                total_errors += stats["error_images"]
                logging.warning(
                    "REQ-030: Directory %s completed with %s errors",
                    input_dir,
                    stats["error_images"],
                )

        if total_errors > 0:
            logging.warning("REQ-030: Completed with %s total errors", total_errors)
            return 1

        logging.info("REQ-030: Extraction completed successfully")
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("REQ-030: Extraction failed: %s", exc)
        return 1


def process_analyze(args: argparse.Namespace, verbose: int) -> int:
    """Handle the ``analyze`` and ``annotate`` subcommands.

    REQ-078: Supports multiple input directories/files for batch processing.
    """

    input_dirs = getattr(args, "input_dirs", [])
    if not input_dirs:
        logging.error("REQ-031: At least one input directory must be specified")
        return 1

    # Validate all input directories exist
    for input_dir in input_dirs:
        if not input_dir.exists():
            logging.error("REQ-031: Input directory does not exist: %s", input_dir)
            return 1

    logging.info("REQ-031: Analyzing images from %s directories", len(input_dirs))
    logging.info("REQ-016: Verbosity level: %s", verbose)

    total_errors = 0
    try:
        from media_indexer.processor import ImageProcessor

        for input_dir in input_dirs:
            logging.info("REQ-031: Processing directory: %s", input_dir)
            processor = ImageProcessor(
                input_dir=input_dir,
                verbose=verbose,
                batch_size=args.batch_size,
                database_path=args.db,
                disable_sidecar=args.no_sidecar,
                limit=args.limit,
                force=getattr(args, "force", False),
                scan_workers=getattr(args, "workers", 8),
            )

            stats = processor.process()

            if stats["error_images"] > 0:
                total_errors += stats["error_images"]
                logging.warning(
                    "REQ-012: Directory %s completed with %s errors",
                    input_dir,
                    stats["error_images"],
                )

        if total_errors > 0:
            logging.warning("REQ-012: Completed with %s total errors", total_errors)
            return 1

        logging.info("REQ-012: Processing completed successfully")
        return 0
    except KeyboardInterrupt:
        logging.warning("REQ-015: Processing interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.error("REQ-015: Processing failed: %s", exc)
        return 1


def process_db(args: argparse.Namespace, verbose: int) -> int:
    """Handle ``db`` command group operations."""

    del verbose  # Verbosity is already configured globally.

    if not args.db:
        logging.error("REQ-067: --db is required for db command")
        return 1

    db_command = getattr(args, "db_command", None)

    if db_command == "init":
        try:
            from media_indexer.db.connection import DatabaseConnection

            logging.info("REQ-067: Initializing database at %s", args.db)
            db_conn = DatabaseConnection(args.db)
            db_conn.connect()
            logging.info("REQ-067: Database initialized successfully at %s", args.db)
            return 0
        except Exception as exc:  # noqa: BLE001
            logging.error("REQ-067: Database initialization failed: %s", exc)
            return 1

    if not args.db.exists():
        logging.error(
            "REQ-067: Database does not exist: %s. Run 'media-indexer db init --db <path>' first.",
            args.db,
        )
        return 1

    try:
        from media_indexer.db_manager import DatabaseManager

        db_manager = DatabaseManager(args.db)

        if db_command == "stats":
            db_manager.print_statistics()
            return 0

        if db_command == "search":
            if not args.query:
                logging.error("REQ-067: --query required for search command")
                return 1

            limit = getattr(args, "limit", 10)
            results = db_manager.search_images(args.query, limit)
            print(f"\n=== Search Results (showing {len(results)} of {limit}) ===")
            for result in results:
                print(f"\n{result['path']}")
                print(f"  Faces: {result['faces']}, Objects: {result['objects']}, Poses: {result['poses']}")
                print(f"  Size: {result['file_size']} bytes")
                if result["width"] and result["height"]:
                    print(f"  Dimensions: {result['width']}x{result['height']}")
            return 0

        if db_command == "clean":
            logging.info("REQ-067: Cleaning database...")
            stats = db_manager.clean_database()
            print("\n=== Cleanup Statistics ===")
            print(f"Images checked: {stats['images_checked']}")
            print(f"Images removed: {stats['images_removed']}")
            print(f"Files not found: {stats['files_not_found']}")
            return 0

        logging.error("REQ-067: No db subcommand specified. Use: stats, init, search, or clean")
        return 1
    except Exception as exc:  # noqa: BLE001
        logging.error("REQ-067: Database command failed: %s", exc)
        return 1


def process_convert(args: argparse.Namespace, verbose: int) -> int:
    """Handle ``convert`` operations between sidecars and database storage."""

    input_dirs = getattr(args, "input_dirs", [])
    if args.direction == "to-db":
        if not args.db:
            logging.error("REQ-032: --db required for to-db conversion")
            return 1
        if not input_dirs:
            logging.error("REQ-032: At least one input directory must be specified")
            return 1

        # Validate all input directories exist
        for input_dir in input_dirs:
            if not input_dir.exists():
                logging.error("REQ-032: Input directory does not exist: %s", input_dir)
                return 1

        total_errors = 0
        for input_dir in input_dirs:
            logging.info("REQ-032: Converting sidecar files to database from %s", input_dir)
            try:
                from media_indexer.sidecar_converter import import_sidecars_to_database

                import_sidecars_to_database(
                    input_dir=input_dir,
                    database_path=args.db,
                    verbose=verbose,
                    workers=args.workers,
                )
                logging.info("REQ-032: Sidecar import to database completed successfully for %s", input_dir)
            except Exception as exc:  # noqa: BLE001
                logging.error("REQ-032: Sidecar import failed for %s: %s", input_dir, exc)
                total_errors += 1

        if total_errors > 0:
            logging.error("REQ-032: Sidecar import completed with %s errors", total_errors)
            return 1
        return 0

    if args.direction == "to-sidecar":
        if not args.db:
            logging.error("REQ-033: --db required for to-sidecar conversion")
            return 1
        if not args.db.exists():
            logging.error("REQ-033: Database does not exist: %s", args.db)
            return 1

        logging.info("REQ-033: Converting database to sidecar files next to image files")
        try:
            from media_indexer.sidecar_converter import export_database_to_sidecars

            export_database_to_sidecars(
                database_path=args.db,
                output_dir=None,
                verbose=verbose,
                workers=args.workers,
            )
            logging.info("REQ-033: Database export to sidecar completed successfully")
            return 0
        except Exception as exc:  # noqa: BLE001
            logging.error("REQ-033: Database export failed: %s", exc)
            return 1

    logging.error("REQ-034: Unknown conversion direction: %s", args.direction)
    return 1


__all__ = [
    "process_extract",
    "process_analyze",
    "process_db",
    "process_convert",
]
