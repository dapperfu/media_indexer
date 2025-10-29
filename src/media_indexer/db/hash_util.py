"""File hashing utility using xxhash.

REQ-028: Fast file hashing for deduplication using xxhash.
"""

import logging
from pathlib import Path

import xxhash

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate xxhash for a file.

    REQ-028: Use xxhash for fast file hashing without cryptographic security requirements.

    Args:
        file_path: Path to the file to hash.
        chunk_size: Size of chunks to read from file (default: 8KB).

    Returns:
        xxhash hex digest as string.

    Raises:
        IOError: If file cannot be read.
    """
    try:
        xxh = xxhash.xxh64()

        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                xxh.update(chunk)

        return xxh.hexdigest()

    except Exception as e:
        logger.error(f"REQ-028: Failed to calculate hash for {file_path}: {e}")
        raise


def get_file_size(file_path: Path) -> int | None:
    """Get file size in bytes.

    Args:
        file_path: Path to the file.

    Returns:
        File size in bytes, or None if file cannot be accessed.
    """
    try:
        return file_path.stat().st_size
    except Exception as e:
        logger.warning(f"Failed to get file size for {file_path}: {e}")
        return None
