"""
Image Processing Utilities

REQ-007, REQ-008, REQ-009: Bounding box normalization utilities.
REQ-010: All code components directly linked to requirements.
"""

from typing import Any


def normalize_bbox(
    bbox_absolute: list[float],
    img_width: int,
    img_height: int,
) -> list[float]:
    """
    Normalize bounding box coordinates to percentages (0.0-1.0).

    REQ-007, REQ-008, REQ-009: Normalize bounding boxes for face, object, and pose detection.

    Args:
        bbox_absolute: Absolute bounding box coordinates [x1, y1, x2, y2].
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Normalized bounding box coordinates [x1, y1, x2, y2] in range 0.0-1.0.
    """
    return [
        bbox_absolute[0] / img_width,   # x1
        bbox_absolute[1] / img_height,   # y1
        bbox_absolute[2] / img_width,    # x2
        bbox_absolute[3] / img_height,   # y2
    ]


def normalize_keypoints(
    keypoints_absolute: list[list[float]],
    img_width: int,
    img_height: int,
) -> list[list[float]]:
    """
    Normalize keypoint coordinates to percentages (0.0-1.0).

    REQ-009: Normalize keypoints for pose detection.

    Args:
        keypoints_absolute: Absolute keypoint coordinates [[x, y], ...].
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Normalized keypoint coordinates [[x, y], ...] in range 0.0-1.0.
    """
    return [[kp[0] / img_width, kp[1] / img_height] for kp in keypoints_absolute]

