"""
Model Download Utility

Pre-download models to central cache location.
"""

import argparse
import logging
from pathlib import Path

from media_indexer.model_cache import ModelCache

logger = logging.getLogger(__name__)


def download_models(args: argparse.Namespace) -> None:
    """
    Download models to central cache.

    REQ-007, REQ-008, REQ-009: Download YOLO models.
    REQ-007: Download InsightFace models.
    """
    cache = ModelCache(Path(args.cache_dir) if args.cache_dir else None)
    cache.setup_environment()

    logger.info(f"Central model cache: {cache.cache_dir}")

    if args.list:
        # List cached models
        models = cache.get_model_info()
        if models:
            print("Cached models:")
            for name, info in models.items():
                print(f"  {name}: {info['size_mb']} MB")
        else:
            print("No models cached yet.")
        return

    if args.download_yolo:
        # REQ-007, REQ-008, REQ-009: Download YOLO models
        logger.info("Downloading YOLO models...")
        try:
            from ultralytics import YOLO

            models_to_download = ["yolo12x.pt", "yolo11x-pose.pt", "yolov8n-face.pt", "yolov11n-face.pt"]
            for model_name in models_to_download:
                logger.info(f"Downloading {model_name}...")
                YOLO(model_name)
                logger.info(f"{model_name} downloaded to {cache.yolo_cache}")
        except Exception as e:
            logger.error(f"Failed to download YOLO models: {e}")
            raise

    if args.download_insightface:
        # REQ-007: Download InsightFace models
        logger.info("Downloading InsightFace models...")
        try:
            import insightface

            insightface.app.FaceAnalysis(providers=["CUDAExecutionProvider"])
            logger.info(f"InsightFace models downloaded to {cache.insightface_cache}")
        except Exception as e:
            logger.error(f"Failed to download InsightFace models: {e}")
            raise


def main() -> None:
    """CLI entry point for model download utility."""
    parser = argparse.ArgumentParser(description="Download models to central cache")
    parser.add_argument("--cache-dir", help="Custom cache directory")
    parser.add_argument("--list", action="store_true", help="List cached models")
    parser.add_argument("--download-yolo", action="store_true", help="Download YOLO models (REQ-007, REQ-008, REQ-009)")
    parser.add_argument("--download-insightface", action="store_true", help="Download InsightFace models (REQ-007)")

    args = parser.parse_args()

    if args.list or args.download_yolo or args.download_insightface:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        download_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
