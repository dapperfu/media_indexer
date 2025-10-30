"""Analytics utilities for enriched metadata.

REQ-081: Face attribute analysis utilities.
REQ-010: All code components directly linked to requirements.
"""

__all__ = ["FaceAttributeAnalyzer", "FaceAttributeResult", "AttributeSource"]

from .face_attribute_analyzer import (  # noqa: F401
    AttributeSource,
    FaceAttributeAnalyzer,
    FaceAttributeResult,
)

