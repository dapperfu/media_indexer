"""
Media Indexer - GPU-accelerated image analysis tool for extracting metadata,
faces, objects, and poses from large image collections.

REQ-010: All code components directly linked to requirements.
"""

__version__ = "0.1.0"


def main():
    """Main entry point for CLI."""
    from media_indexer.cli import main as cli_main
    return cli_main()


__all__ = ["main"]
