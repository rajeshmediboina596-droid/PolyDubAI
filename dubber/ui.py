"""
Terminal UI and progress rendering using Rich.
"""

import sys
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    FileSizeColumn,
    TotalFileSizeColumn,
    TransferSpeedColumn,
)
from rich.text import Text

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)


def print_banner():
    """Print the system startup banner."""
    title = Text("IDEALABS DIGITAL", style="bold cyan")
    subtitle = Text("Automated Video Dubbing System • Multi-Language to English", style="italic white")
    banner_content = Text.assemble(title, "\n", subtitle)
    console.print(Panel(banner_content, border_style="bright_blue", expand=False))


def print_video_info(info: Dict[str, Any]):
    """Print parsed video metadata in a formatted card."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold magenta")
    table.add_column("Value", style="bright_white")

    duration_min = round(info.get("duration", 0) / 60, 1)
    table.add_row("Title", str(info.get("title", "Unknown")))
    table.add_row("Uploader", str(info.get("uploader", "Unknown")))
    table.add_row("Duration", f"{info.get('duration', 0)}s (~{duration_min} min)")
    table.add_row("URL", str(info.get("url", "")))

    console.print(Panel(table, title="[bold green]Source Video Information[/bold green]", border_style="green"))


def print_step(step_number: int, total_steps: int, title: str):
    """Print a clean step heading."""
    console.print(f"\n[bold yellow]Step {step_number}/{total_steps}:[/bold yellow] [bold white]{title}[/bold white]")


def print_segments_preview(segments: List[Dict[str, Any]], max_rows: int = 6):
    """Display a formatted preview table of transcribed & translated segments with speaker info."""
    table = Table(title="Translated Segments Preview (Adaptive Speaker Dubbing)", border_style="blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Time Window", style="cyan", width=18)
    table.add_column("Speaker", style="yellow", width=18)
    table.add_column("Duration", style="magenta", width=9)
    table.add_column("English Dubbed Text", style="bright_white")

    display_segs = segments[:max_rows]
    for seg in display_segs:
        start_fmt = f"{seg['start']:.2f}s"
        end_fmt = f"{seg['end']:.2f}s"
        dur_fmt = f"{seg['duration']:.2f}s"
        gender = seg.get("gender")
        voice_short = seg.get("voice", "").split("-")[-1].replace("Neural", "")

        if gender == "female":
            speaker_badge = f"[bold magenta]♀ Female[/bold magenta] ({voice_short})"
        elif gender == "male":
            speaker_badge = f"[bold cyan]♂ Male[/bold cyan] ({voice_short})"
        else:
            speaker_badge = f"[dim]{voice_short or 'Default'}[/dim]"

        table.add_row(str(seg["id"]), f"{start_fmt} -> {end_fmt}", speaker_badge, dur_fmt, seg["text"])

    if len(segments) > max_rows:
        table.add_row("...", "...", "...", "...", f"[dim]and {len(segments) - max_rows} more segments...[/dim]")

    console.print(table)


def print_summary(summary_data: Dict[str, Any]):
    """Print the final run execution summary table."""
    table = Table(title="[bold green]Dubbing Pipeline Completed Successfully![/bold green]", border_style="green")
    table.add_column("Metric / Stage", style="bold cyan")
    table.add_column("Details", style="bright_white")

    table.add_row("Detected Source Language", f"[bold yellow]{summary_data.get('language', 'N/A').upper()}[/bold yellow] (confidence: {summary_data.get('lang_confidence', 0)*100:.1f}%)")
    table.add_row("Total Speech Segments", str(summary_data.get("segments_count", 0)))
    if summary_data.get("auto_gender"):
        table.add_row(
            "Speaker Classification",
            f"[bold cyan]{summary_data.get('male_segments', 0)} Male ♂[/bold cyan]  •  [bold magenta]{summary_data.get('female_segments', 0)} Female ♀[/bold magenta]"
        )
    table.add_row("Neural Voice Applied", str(summary_data.get("voice", "N/A")))
    table.add_row("Video Duration", f"{summary_data.get('video_duration', 0):.1f} seconds")
    table.add_row("Total Processing Time", f"[bold green]{summary_data.get('total_time', 0):.2f} seconds[/bold green]")
    if summary_data.get('video_duration', 0) > 0:
        rtf = summary_data.get('total_time', 0) / summary_data.get('video_duration', 1)
        table.add_row("Real-Time Factor (RTF)", f"{rtf:.2f}x (Lower is faster)")

    table.add_row("Output Video File", f"[bold magenta]{summary_data.get('output_video', 'N/A')}[/bold magenta]")
    table.add_row("Output File Size", f"{summary_data.get('file_size_mb', 0)} MB")

    console.print("\n")
    console.print(Panel(table, border_style="bright_green"))


def get_download_progress() -> Progress:
    """Create a customized Rich progress bar for downloads."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        FileSizeColumn(),
        TextColumn("/"),
        TotalFileSizeColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def get_general_progress() -> Progress:
    """Create a general progress bar for transcription and synthesis."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )
