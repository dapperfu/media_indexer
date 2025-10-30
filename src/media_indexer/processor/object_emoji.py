"""
Emoji helpers for object detection summaries.

REQ-077: Provide intuitive iconography for detected object classes in progress output.
REQ-010: Maintain explicit traceability to requirements.
"""

from __future__ import annotations

from typing import Final

_OBJECT_EMOJI_MAP: Final[dict[str, str]] = {
    "person": "🧍",
    "bicycle": "🚲",
    "car": "🚗",
    "motorcycle": "🏍️",
    "airplane": "✈️",
    "bus": "🚌",
    "train": "🚆",
    "truck": "🚚",
    "boat": "⛵",
    "traffic light": "🚦",
    "fire hydrant": "🚒",
    "stop sign": "🛑",
    "parking meter": "🅿️",
    "bench": "🪑",
    "bird": "🐦",
    "cat": "🐱",
    "dog": "🐶",
    "horse": "🐎",
    "sheep": "🐑",
    "cow": "🐄",
    "elephant": "🐘",
    "bear": "🐻",
    "zebra": "🦓",
    "giraffe": "🦒",
    "backpack": "🎒",
    "umbrella": "☔",
    "handbag": "👜",
    "tie": "👔",
    "suitcase": "💼",
    "frisbee": "🥏",
    "skis": "🎿",
    "snowboard": "🏂",
    "sports ball": "⚽",
    "kite": "🪁",
    "baseball bat": "⚾",
    "baseball glove": "🥎",
    "skateboard": "🛹",
    "surfboard": "🏄",
    "tennis racket": "🎾",
    "bottle": "🍾",
    "wine glass": "🍷",
    "cup": "☕",
    "fork": "🍴",
    "knife": "🔪",
    "spoon": "🥄",
    "bowl": "🥣",
    "banana": "🍌",
    "apple": "🍎",
    "sandwich": "🥪",
    "orange": "🍊",
    "broccoli": "🥦",
    "carrot": "🥕",
    "hot dog": "🌭",
    "pizza": "🍕",
    "donut": "🍩",
    "cake": "🍰",
    "chair": "🪑",
    "sofa": "🛋️",
    "potted plant": "🪴",
    "bed": "🛏️",
    "dining table": "🍽️",
    "toilet": "🚽",
    "tv": "📺",
    "laptop": "💻",
    "mouse": "🖱️",
    "remote": "🕹️",
    "keyboard": "⌨️",
    "cell phone": "📱",
    "microwave": "📡",
    "oven": "🍽️",
    "toaster": "🍞",
    "sink": "🚰",
    "refrigerator": "🧊",
    "book": "📚",
    "clock": "⏰",
    "vase": "🏺",
    "scissors": "✂️",
    "teddy bear": "🧸",
    "hair drier": "💇",
    "toothbrush": "🪥",
}


def get_object_emoji(label: str) -> str:
    """Return emoji representing a detected object class.

    Parameters
    ----------
    label : str
        Object class label reported by the detector.

    Returns
    -------
    str
        Emoji string or an empty string when no mapping is defined.
    """

    key = label.strip().lower()
    return _OBJECT_EMOJI_MAP.get(key, "")


