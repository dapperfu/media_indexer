"""
Model Download Utilities

REQ-007, REQ-008, REQ-009: Model download utilities for YOLO and InsightFace models.
REQ-010: All code components directly linked to requirements.
"""

import logging
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)


def download_model_if_needed(model_path: str, url: str, requirement_id: str = "REQ") -> Path:
    """Ensure a model file exists locally, downloading it when necessary.

    Parameters
    ----------
    model_path : str
        Destination path for the downloaded model.
    url : str
        Remote URL hosting the model artifact.
    requirement_id : str, default="REQ"
        Requirement identifier emitted alongside log messages (e.g.,
        ``REQ-008`` for YOLO object detection downloads).

    Returns
    -------
    Path
        Filesystem path to the available model artifact.

    Raises
    ------
    RuntimeError
        Raised when the remote download fails or the payload cannot be
        persisted locally.

    Notes
    -----
    REQ-007, REQ-008, REQ-009
        Maintains reliable delivery of InsightFace and YOLO model weights.
    REQ-071
        Streams downloads to a temporary file and promotes the completed
        payload atomically to avoid exposing partially written artefacts.
    """

    path = Path(model_path)

    if path.exists():
        logger.debug("%s: Reusing cached model at %s", requirement_id, path)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("%s: Downloading %s from %s", requirement_id, model_path, url)
    temp_name: str | None = None

    try:
        with (
            urlopen(url) as response,  # nosec: B310 validated external URL by requirement
            NamedTemporaryFile(delete=False, dir=str(path.parent)) as buffered_dest,
        ):
            shutil.copyfileobj(response, buffered_dest)
            temp_name = buffered_dest.name

        temp_path = Path(temp_name)
        temp_path.replace(path)
        logger.info("%s: Successfully downloaded %s", requirement_id, path)
        return path
    except (HTTPError, URLError) as exc:
        logger.error("%s: Failed to download %s: %s", requirement_id, model_path, exc)
        raise RuntimeError(f"Failed to download model from {url}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("%s: Unexpected error downloading %s: %s", requirement_id, model_path, exc)
        raise RuntimeError(f"Failed to download model from {url}: {exc}") from exc
    finally:
        if temp_name is not None:
            residual = Path(temp_name)
            if residual.exists() and not path.exists():
                residual.unlink(missing_ok=True)
