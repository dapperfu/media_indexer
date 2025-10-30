"""
Progress Bar Utilities

REQ-012: Progress tracking with Rich and TQDM support.
REQ-010: All code components directly linked to requirements.
"""

import logging
import time
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


class AvgSpeedColumn(ProgressColumn):
    """
    Custom column to display average speed safely.

    REQ-012: Shows average processing speed in progress bar.
    """

    def __init__(self, unit: str = "item") -> None:
        """
        Initialize average speed column.

        Args:
            unit: Unit label for items (e.g., "file", "img").
        """
        super().__init__()
        self.unit = unit

    def render(self, task: Any) -> Text:
        """
        Render average speed safely.

        Args:
            task: Progress task.

        Returns:
            Formatted text for average speed.
        """
        avg_speed = task.fields.get("avg_speed", "0.0")
        if avg_speed:
            return Text(f"[cyan]avg: {avg_speed}[/cyan]", style="progress.data.speed")
        return Text(f"[cyan]avg: 0.0 {self.unit}/s[/cyan]", style="progress.data.speed")


class CurrentFileColumn(ProgressColumn):
    """
    Custom column to display current file on second line.

    REQ-012: Shows current file being processed.
    """

    def render(self, task: Any) -> Text:
        """
        Render current file name.

        Args:
            task: Progress task.

        Returns:
            Formatted text for current file.
        """
        current_file = task.fields.get("current_file", "")
        if current_file:
            return Text(f"\n[dim]{current_file}[/dim]")
        return Text("")


class DetectionsColumn(ProgressColumn):
    """
    Custom column to display detection information on third line.

    REQ-012: Shows detection results (faces, objects, poses).
    """

    def render(self, task: Any) -> Text:
        """
        Render detection information.

        Args:
            task: Progress task.

        Returns:
            Formatted text for detections.
        """
        detections = task.fields.get("detections", "")
        if detections:
            return Text(f"\n[dim]{detections}[/dim]")
        return Text("")


class SpeedColumn(ProgressColumn):
    """
    Custom column to display instantaneous speed safely.

    REQ-012: Shows instantaneous processing speed in progress bar.
    REQ-015: Handles None values to prevent format errors.
    """

    def __init__(self, unit: str = "item") -> None:
        """
        Initialize speed column.

        Args:
            unit: Unit label for items (e.g., "file", "img").
        """
        super().__init__()
        self.unit = unit

    def render(self, task: Any) -> Text:
        """
        Render instantaneous speed safely.

        Args:
            task: Progress task.

        Returns:
            Formatted text for speed.
        """
        speed = getattr(task, "speed", None)
        if speed is not None:
            return Text(
                f"[progress.speed]{speed:>8.1f}[/progress.speed]",
                style="progress.speed",
            )
        return Text("[progress.speed]       0.0[/progress.speed]", style="progress.speed")


class PercentageColumn(ProgressColumn):
    """
    Custom column to display percentage safely.

    REQ-012: Shows percentage complete in progress bar.
    REQ-015: Handles None values to prevent format errors.
    """

    def render(self, task: Any) -> Text:
        """
        Render percentage safely.

        Args:
            task: Progress task.

        Returns:
            Formatted text for percentage.
        """
        percentage = getattr(task, "percentage", None)
        if percentage is not None:
            return Text(f"{percentage:>3.0f}%")
        return Text("  0%")


def create_rich_progress_bar(
    total: int,
    desc: str,
    unit: str = "item",
    verbose: int = 20,
    show_detections: bool = False,
) -> Progress | None:
    """
    Create a Rich progress bar with multi-line support.

    REQ-012: Progress tracking with both instantaneous and global/average speed.
    Supports multi-line display for detection information.

    Args:
        total: Total number of items to process.
        desc: Description for the progress bar.
        unit: Unit label for items (e.g., "file", "img").
        verbose: Verbosity level (only create if >= 15).
        show_detections: If True, add a second line for detection information.

    Returns:
        Rich Progress instance or None if verbosity is too low.
    """
    if verbose < 15:
        return None

    # REQ-012: Create Rich progress bar with custom columns
    # REQ-015: Use custom columns to safely handle None values for speed and percentage
    if show_detections:
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            PercentageColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn(f"{unit}"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("•"),
            SpeedColumn(unit=unit),
            TextColumn(f"{unit}/s"),
            TextColumn("•"),
            AvgSpeedColumn(unit=unit),
            CurrentFileColumn(),
            DetectionsColumn(),
        ]
    else:
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            PercentageColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn(f"{unit}"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("•"),
            SpeedColumn(unit=unit),
            TextColumn(f"{unit}/s"),
            TextColumn("•"),
            AvgSpeedColumn(unit=unit),
        ]

    progress = Progress(*columns, console=console, transient=False)

    # Store processed count and start time
    progress._processed_count = 0  # type: ignore[attr-defined]
    progress._start_time = time.time()  # type: ignore[attr-defined]

    return progress

