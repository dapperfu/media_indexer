"""
Progress Bar Utilities

REQ-012: Progress tracking with Rich and TQDM support.
REQ-010: All code components directly linked to requirements.
"""

import logging
import time
from typing import Any

from rich.console import Console, RenderableType
from rich.live import Live
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
        text = Text()
        text.append("avg: ", style="cyan")
        if avg_speed and avg_speed != "0.0":
            text.append(avg_speed, style="cyan")
        else:
            text.append(f"0.0 {self.unit}/s", style="cyan")
        return text


class MultiLineProgressDisplay:
    """
    Custom renderable that combines Progress bar with multi-line info below it.

    REQ-012: Shows progress bar with current file and detections on separate lines.
    Uses Rich's Live display pattern for proper multi-line rendering.
    """

    def __init__(self, progress: Progress, show_detections: bool = False) -> None:
        """
        Initialize multi-line progress display.

        Args:
            progress: Rich Progress instance.
            show_detections: If True, show detection information.
        """
        self.progress = progress
        self.show_detections = show_detections
        self.current_file: str = ""
        self.detections: str = ""
        self.avg_speed: str = "0.0"

    def __rich__(self) -> RenderableType:
        """
        Render the multi-line display.

        Returns:
            Renderable containing progress bar and info lines.
        """
        from rich.console import Group

        # Create info lines below progress bar
        info_lines: list[RenderableType] = [self.progress]
        
        if self.current_file or self.detections:
            info_text = Text()
            if self.current_file:
                info_text.append(self.current_file, style="dim")
            if self.show_detections and self.detections:
                if self.current_file:
                    info_text.append("\n")
                info_text.append(self.detections, style="dim")
            if len(info_text) > 0:
                info_lines.append(info_text)
        
        return Group(*info_lines)

    def update_info(self, current_file: str = "", detections: str = "", avg_speed: str = "0.0") -> None:
        """
        Update the info lines.

        Args:
            current_file: Current file being processed.
            detections: Detection information.
            avg_speed: Average processing speed.
        """
        self.current_file = current_file
        self.detections = detections
        self.avg_speed = avg_speed


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
            return Text(f"{speed:>8.1f}", style="progress.speed")
        return Text("       0.0", style="progress.speed")


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
) -> tuple[Progress, MultiLineProgressDisplay, Live] | tuple[None, None, None]:
    """
    Create a Rich progress bar with multi-line support using Live display.

    REQ-012: Progress tracking with both instantaneous and global/average speed.
    Supports multi-line display for detection information using Rich's Live display.

    Args:
        total: Total number of items to process.
        desc: Description for the progress bar.
        unit: Unit label for items (e.g., "file", "img").
        verbose: Verbosity level (only create if >= 15).
        show_detections: If True, add a second line for detection information.

    Returns:
        Tuple of (Progress instance, MultiLineProgressDisplay, Live instance) or (None, None, None).
    """
    if verbose < 15:
        return None, None, None

    # REQ-012: Create Rich progress bar with custom columns
    # REQ-015: Use custom columns to safely handle None values for speed and percentage
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

    progress = Progress(
        *columns,
        console=console,
        transient=False,
    )

    # Create multi-line display wrapper
    display = MultiLineProgressDisplay(progress, show_detections=show_detections)

    # Create Live display for proper multi-line rendering
    live = Live(display, console=console, refresh_per_second=10, screen=False)

    # Store processed count and start time
    progress._processed_count = 0  # type: ignore[attr-defined]
    progress._start_time = time.time()  # type: ignore[attr-defined]

    return progress, display, live

