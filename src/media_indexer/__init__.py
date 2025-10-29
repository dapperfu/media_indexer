"""
Media Indexer - GPU-accelerated image analysis tool for extracting metadata,
faces, objects, and poses from large image collections.

REQ-010: All code components directly linked to requirements.
"""

__version__ = "0.1.0"

from media_indexer.cli import main
from media_indexer.processor import ImageProcessor

__all__ = ["ImageProcessor", "main"]
