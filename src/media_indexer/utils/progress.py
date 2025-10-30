"""
Progress Bar Utilities

REQ-012: Progress tracking with Rich support.
REQ-010: All code components directly linked to requirements.

Note: This module exists for backward compatibility. New code should use
media_indexer.processor.progress.create_rich_progress_bar instead.
"""

import logging
from typing import Any

# Import from processor.progress for consistency
from media_indexer.processor.progress import create_rich_progress_bar

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = ["create_rich_progress_bar"]

