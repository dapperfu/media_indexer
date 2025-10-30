"""
Media Indexer - GPU-accelerated image analysis tool for extracting metadata,
faces, objects, and poses from large image collections.

REQ-010: All code components directly linked to requirements.
"""

__version__ = "0.1.0"


def main() -> int:
    """Run the Media Indexer command-line interface.

    Returns
    -------
    int
        Exit code propagated from the CLI handler. A ``0`` is used when the
        handler completes without explicitly returning an exit status.

    Notes
    -----
    REQ-029
        Maintains the setuptools entry point while preserving lazy imports
        for optimal startup performance (REQ-038).
    """

    from media_indexer.cli import main as cli_main

    exit_code = cli_main()
    return int(exit_code) if exit_code is not None else 0


__all__ = ["main"]
