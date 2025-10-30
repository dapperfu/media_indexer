"""
Progress Bar Utilities

REQ-012: Progress tracking with Rich and TQDM support.
REQ-010: All code components directly linked to requirements.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
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

    def __init__(self, progress: Progress, show_detections: bool = False, max_events: int = 5) -> None:
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
        self.recent_events: deque[str] = deque(maxlen=max_events)

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

        if self.recent_events:
            events_text = Text("Recent events:\n", style="bold")
            for message in self.recent_events:
                events_text.append(message)
                events_text.append("\n")
            info_lines.append(events_text)

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

    def add_event(self, message: str) -> None:
        """Record a recent event for persistent display.

        REQ-076: Preserve transient error messages alongside the live progress bar.

        Parameters
        ----------
        message : str
            Message describing the recent event.
        """

        self.recent_events.appendleft(message)


class ProgressLogHandler(logging.Handler):
    """Bridge warnings and errors into the live progress display.

    REQ-076: Persist transient warning/error messages beneath the progress bar
    so that brief console output does not obscure actionable issues.
    """

    def __init__(self, display: MultiLineProgressDisplay, live: Live) -> None:
        """Initialize handler with target display and live renderer.

        Parameters
        ----------
        display : MultiLineProgressDisplay
            Display instance that aggregates recent events.
        live : Live
            Live renderer responsible for refreshing the terminal output.
        """

        super().__init__(level=logging.WARNING)
        self.display = display
        self.live = live
        self._last_event: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """Handle log records reaching WARNING level or higher.

        Parameters
        ----------
        record : logging.LogRecord
            Logging record to display.
        """

        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001
            message = record.getMessage()

        sanitized = message.strip().replace("\n", " ⏎ ")
        if len(sanitized) > 160:
            sanitized = f"{sanitized[:157]}..."

        symbol = "❌" if record.levelno >= logging.ERROR else "⚠️"
        event_text = f"{symbol} {sanitized}"

        if event_text == self._last_event:
            return

        self._last_event = event_text

        try:
            self.display.add_event(event_text)
            self.live.console.call_from_thread(self._refresh)
        except Exception as exc:  # noqa: BLE001
            self.handleError(record)
            logger.debug("REQ-076: ProgressLogHandler refresh failed: %s", exc)

    def _refresh(self) -> None:
        """Refresh the live progress display to surface new events.

        Notes
        -----
        Invoked on the console's render thread via
        :meth:`rich.console.Console.call_from_thread` to preserve Rich's thread
        safety guarantees while updating the shared live display (REQ-076).
        """

        self.live.update(self.display)


@dataclass(slots=True)
class RichProgressContext:
    """Encapsulate Rich progress machinery for reuse across pipeline phases.

    REQ-012: Provide reusable tracking components for processing phases.
    REQ-076: Manage the lifecycle of warning-aware handlers for live output.

    Attributes
    ----------
    total : int
        Total number of work items in the current phase.
    unit : str
        Unit label describing the work items (e.g., ``"img"``).
    progress : Any
        Rich ``Progress`` instance displaying the bar.
    display : Any
        Multi-line wrapper combining the bar with supplemental text.
    live : Any
        Live renderer responsible for updating the terminal output.
    task_id : Any
        Identifier returned by Rich for the registered task.
    start_time : float
        Epoch timestamp recorded when tracking began.
    processed : int, optional
        Count of processed items updated by callers. Defaults to ``0``.
    log_handler : logging.Handler | None, optional
        Active handler mirroring warnings into the progress display.
    """

    total: int
    unit: str
    progress: Any
    display: Any
    live: Any
    task_id: Any
    start_time: float
    processed: int = 0
    log_handler: logging.Handler | None = None

    def stop(self) -> None:
        """Dispose of the live progress display and detach log forwarding.

        Notes
        -----
        The attached progress log handler is removed prior to stopping the
        live renderer to prevent duplicate warning delivery across subsequent
        processing phases (REQ-076).
        """

        if self.log_handler:
            logging.getLogger().removeHandler(self.log_handler)
            self.log_handler.close()
            self.log_handler = None

        self.live.stop()

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

