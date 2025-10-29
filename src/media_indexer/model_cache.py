"""
Model Cache Configuration

Central location for model weight storage and management.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Centralized model cache management.

    Stores models in a central location to avoid re-downloading on every install.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """
        Initialize model cache.

        Args:
            cache_dir: Custom cache directory. Defaults to ~/.media_indexer/models
        """
        if cache_dir is None:
            self.cache_dir: Path = Path.home() / ".media_indexer" / "models"
        else:
            self.cache_dir = Path(cache_dir)

        # REQ-008, REQ-009: YOLO model cache
        self.yolo_cache: Path = self.cache_dir / "yolo"
        self.yolo_cache.mkdir(parents=True, exist_ok=True)

        # REQ-007: InsightFace model cache
        self.insightface_cache: Path = self.cache_dir / "insightface"
        self.insightface_cache.mkdir(parents=True, exist_ok=True)

        logger.info(f"Model cache directory: {self.cache_dir}")
        logger.info(f"YOLO cache: {self.yolo_cache}")
        logger.info(f"InsightFace cache: {self.insightface_cache}")

    def setup_environment(self) -> None:
        """
        Setup environment variables for centralized model storage.

        REQ-008, REQ-009: Configure YOLO to use central cache.
        REQ-007: Configure InsightFace to use central cache.
        """
        # Set Ultralytics/YOLO cache directory
        os.environ["ULTRALYTICS_CACHE_DIR"] = str(self.yolo_cache)
        logger.debug(f"REQ-008, REQ-009: Set YOLO cache to {self.yolo_cache}")

        # Set InsightFace cache directory
        os.environ["INSIGHTFACE_CACHE_DIR"] = str(self.insightface_cache)
        logger.debug(f"REQ-007: Set InsightFace cache to {self.insightface_cache}")

    def get_yolo_model_path(self, model_name: str = "yolo12x.pt") -> Path:
        """
        Get path to YOLO model in cache.

        REQ-008, REQ-009: Returns cached model path.

        Args:
            model_name: Name of the model file.

        Returns:
            Path to cached model.
        """
        # Use Ultralytics cache structure
        model_path = self.yolo_cache / model_name
        return model_path

    def get_model_info(self) -> dict[str, any]:
        """
        Get information about cached models.

        Returns:
            Dictionary with cache information.
        """
        models = {}
        model_files = list(self.yolo_cache.glob("*.pt"))
        for model_file in model_files:
            size = model_file.stat().st_size / (1024 * 1024)  # MB
            models[str(model_file.name)] = {
                "size_mb": round(size, 2),
                "path": str(model_file),
            }
        return models

    @staticmethod
    def get_default_cache() -> Path:
        """
        Get default cache directory.

        Returns:
            Default cache path.
        """
        return Path.home() / ".media_indexer" / "models"


def setup_model_cache(cache_dir: Path | None = None) -> ModelCache:
    """
    Setup and configure model cache.

    REQ-008, REQ-009, REQ-007: Configure centralized model storage.

    Args:
        cache_dir: Custom cache directory.

    Returns:
        Configured ModelCache instance.
    """
    cache = ModelCache(cache_dir)
    cache.setup_environment()
    return cache


# Initialize global cache on import
DEFAULT_CACHE: ModelCache = setup_model_cache()
