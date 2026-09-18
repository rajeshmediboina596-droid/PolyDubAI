"""
Main CLI entry point for the Automated Video Dubbing System.
IDEALABS DIGITAL Assignment.
"""

import argparse
import sys
from rich.console import Console
from rich.table import Table

from dubber.config import AVAILABLE_VOICES, DubbingConfig
from dubber.pipeline import DubbingPipeline
from dubber.ui import print_banner

console = Console()


def display_voice_catalog():
    """Display a rich table of all available neural voices."""
    table = Table(title="Available Neural Voices (edge-tts)", border_style="cyan")
    table.add_column("Voice ID", style="bold green")
    table.add_column("Gender", style="magenta")
    table.add_column("Locale", style="yellow")
    table.add_column("Description", style="white")

    for voice_id, details in AVAILABLE_VOICES.items():
        table.add_row(voice_id, details["gender"], details["locale"], details["description"])

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Automated Video Dubbing System - Turn foreign YouTube videos into English-dubbed videos."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="YouTube video URL or local video file path (optional if using interactive prompt)",
    )
    parser.add_argument(
        "--url",
        dest="flag_url",
        default=None,
        help="YouTube video URL (alternative to positional argument)",
    )
    parser.add_argument(
        "-v", "--voice",
        default="en-US-ChristopherNeural",
        help=f"Edge TTS voice to use for English speech (default: en-US-ChristopherNeural)",
    )
    parser.add_argument(
        "-l", "--target-lang",
        dest="target_lang",
        default="en",
        help="Target language for dubbing (e.g. te for Telugu, hi for Hindi, ta for Tamil, es for Spanish, fr for French, etc.)",
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper transcription model size (default: base)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Destination directory for output video and subtitles (default: output)",
    )
    parser.add_argument(
        "--mix-original",
        type=float,
        default=0.0,
        help="Volume of ducked original background audio to mix in (0.0 = full replacement, 0.1 = 10%% background)",
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Do not embed English subtitles into final video",
    )
    parser.add_argument(
        "--no-auto-gender",
        dest="auto_gender",
        action="store_false",
        default=True,
        help="Disable automatic speaker gender detection (use static voice for all speakers)",
    )
    parser.add_argument(
        "--male-voice",
        default="en-US-ChristopherNeural",
        help="Voice for detected male speakers (default: en-US-ChristopherNeural)",
    )
    parser.add_argument(
        "--female-voice",
        default="en-US-JennyNeural",
        help="Voice for detected female speakers (default: en-US-JennyNeural)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Retain intermediate extracted WAVs and segment audio files",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all supported natural voices and exit",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the interactive Dubber Studio Web Application in your browser",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web server (default: 8000)",
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in interactive CLI mode instead of web interface",
    )

    args = parser.parse_args()

    if args.web or (not args.cli and not args.url and not args.flag_url and not args.list_voices):
        import run_web
        sys.argv = [sys.argv[0]] + (["--port", str(args.port)] if args.port != 8000 else [])
        run_web.main()
        sys.exit(0)

    if args.list_voices:
        display_voice_catalog()
        sys.exit(0)

    target_url = args.url or args.flag_url
    if not target_url:
        print_banner()
        target_url = console.input("[bold yellow]Enter YouTube Video URL (or local video file path): [/bold yellow]").strip()

    if not target_url:
        console.print("[bold red]Error: No URL or video path provided.[/bold red]")
        sys.exit(1)

    # Validate voice
    if args.voice not in AVAILABLE_VOICES:
        console.print(
            f"[bold red]Warning: Voice '{args.voice}' is not in the recommended catalogue.[/bold red]"
        )
        console.print("[dim]Proceeding with specified voice name in edge-tts...[/dim]")

    config = DubbingConfig(
        target_lang=args.target_lang,
        output_dir=args.output,
        whisper_model=args.model,
        voice=args.voice,
        auto_gender=args.auto_gender,
        male_voice=args.male_voice,
        female_voice=args.female_voice,
        mix_original_volume=args.mix_original,
        embed_subtitles=not args.no_subtitles,
        keep_temp=args.keep_temp,
    )

    pipeline = DubbingPipeline(config)

    try:
        pipeline.run(target_url)
    except KeyboardInterrupt:
        console.print("\n[bold red]Dubbing process interrupted by user.[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Pipeline Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
