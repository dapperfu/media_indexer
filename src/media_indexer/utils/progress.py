"""
Progress Bar Utilities

REQ-012: Progress tracking with both instantaneous and global/average speed.
REQ-010: All code components directly linked to requirements.
"""

import logging
import time
from typing import Any

import tqdm
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


def create_progress_bar_with_global_speed(
    total: int,
    desc: str,
    unit: str = "item",
    verbose: int = 20,
) -> tqdm.tqdm | None:
    """
    Create a tqdm progress bar with global speed calculation.

    REQ-012: Progress tracking with both instantaneous and global/average speed.

    Args:
        total: Total number of items to process.
        desc: Description for the progress bar.
        unit: Unit label for items (e.g., "file", "img").
        verbose: Verbosity level (only create if >= 17 to match CLI tqdm disable threshold).

    Returns:
        tqdm progress bar instance or None if verbosity is too low.
    """
    # REQ-016: Match CLI threshold - tqdm is disabled when verbose < 17
    if verbose < 17:
        return None

    # REQ-012: Create progress bar with compact format showing both speeds
    progress_bar = tqdm.tqdm(
        total=total,
        desc=desc,
        unit=unit,
        bar_format='{desc}: {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]',
    )

    # Store start time for global speed calculation
    progress_bar.start_time = time.time()  # type: ignore[attr-defined]
    progress_bar._custom_postfix = ""  # type: ignore[attr-defined]
    progress_bar._processed_count = 0  # type: ignore[attr-defined]
    
    # Store original methods BEFORE wrapping
    original_update = progress_bar.update
    original_set_postfix = progress_bar.set_postfix_str

    def update_with_global_speed(n: int = 1) -> None:
        """Update progress bar with global speed calculation."""
        # Increment our tracked count
        progress_bar._processed_count += n  # type: ignore[attr-defined]
        # Call original update
        original_update(n)
        # Calculate global speed: processed / elapsed_time
        elapsed = time.time() - progress_bar.start_time  # type: ignore[attr-defined]
        if elapsed > 0 and progress_bar._processed_count > 0:  # type: ignore[attr-defined]
            global_speed = progress_bar._processed_count / elapsed  # type: ignore[attr-defined]
            # Format speed nicely
            if global_speed >= 1:
                speed_str = f"{global_speed:.1f} {unit}/s"
            else:
                speed_str = f"{1/global_speed:.1f}s/{unit}"
            # Combine with custom postfix if any
            custom = progress_bar._custom_postfix  # type: ignore[attr-defined]
            if custom:
                combined = f"{custom} | avg: {speed_str}"
            else:
                combined = f"avg: {speed_str}"
            # Use original set_postfix to avoid recursion
            original_set_postfix(combined, refresh=False)
    
    def set_postfix_with_global_speed(postfix: str | None = None, refresh: bool = True) -> None:
        """Set postfix while preserving global speed."""
        if postfix:
            progress_bar._custom_postfix = postfix  # type: ignore[attr-defined]
            # Recalculate global speed and combine
            elapsed = time.time() - progress_bar.start_time  # type: ignore[attr-defined]
            if elapsed > 0 and progress_bar._processed_count > 0:  # type: ignore[attr-defined]
                global_speed = progress_bar._processed_count / elapsed  # type: ignore[attr-defined]
                if global_speed >= 1:
                    speed_str = f"{global_speed:.1f} {unit}/s"
                else:
                    speed_str = f"{1/global_speed:.1f}s/{unit}"
                combined = f"{postfix} | avg: {speed_str}"
                original_set_postfix(combined, refresh=refresh)
            else:
                original_set_postfix(postfix, refresh=refresh)
        else:
            original_set_postfix(postfix, refresh=refresh)
    
    # Replace methods AFTER defining both functions
    progress_bar.set_postfix_str = set_postfix_with_global_speed  # type: ignore[method-assign]
    progress_bar.update = update_with_global_speed  # type: ignore[method-assign]

    return progress_bar


def create_rich_progress_bar(
    total: int,
    desc: str,
    unit: str = "item",
    verbose: int = 20,
    show_detections: bool = False,
) -> Progress | None:
    """
    Create a Rich progress bar with multi-line support for detection information.

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
    if show_detections:
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn(f"{unit}"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[progress.speed]{task.speed:>8.1f}"),
            TextColumn(f"{unit}/s"),
            TextColumn("•"),
            TextColumn("[cyan]avg: {task.fields[avg_speed]}[/cyan]"),
            TextColumn("\n[dim]{task.fields[current_file]}[/dim]"),
            TextColumn("\n[dim]{task.fields[detections]}[/dim]"),
        ]
    else:
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn(f"{unit}"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[progress.speed]{task.speed:>8.1f}"),
            TextColumn(f"{unit}/s"),
            TextColumn("•"),
            TextColumn("[cyan]avg: {task.fields[avg_speed]}[/cyan]"),
        ]

    progress = Progress(*columns, console=console, transient=False)
    
    # Store processed count and start time
    progress._processed_count = 0  # type: ignore[attr-defined]
    progress._start_time = time.time()  # type: ignore[attr-defined]
    
    return progress

