"""
Benchmarking and profiling script for the Automated Video Dubbing System.
Tracks execution times, stage breakdowns, and generates submission-ready reports
for careers@idealabsdigital.com.
"""

import argparse
import datetime
import json
import os
import platform
import time
from typing import List
from rich.console import Console
from rich.table import Table

from dubber.config import DubbingConfig
from dubber.pipeline import DubbingPipeline

console = Console()


def get_system_specs():
    """Collect host environment details for benchmark reporting."""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "processor": platform.processor() or "Unknown CPU",
        "python_version": platform.python_version(),
    }


def run_benchmark(urls: List[str], output_dir: str = "output_benchmarks", model: str = "base", voice: str = "en-US-ChristopherNeural"):
    """Run dubbing benchmark on a set of videos and output a comprehensive report."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    config = DubbingConfig(
        output_dir=output_dir,
        whisper_model=model,
        voice=voice,
        keep_temp=True,
    )
    pipeline = DubbingPipeline(config)

    console.print(f"[bold cyan]Starting Benchmark Run for {len(urls)} video(s)...[/bold cyan]\n")

    for i, url in enumerate(urls, 1):
        console.print(f"[bold magenta]============================================================[/bold magenta]")
        console.print(f"[bold magenta] BENCHMARK VIDEO {i}/{len(urls)}: {url}[/bold magenta]")
        console.print(f"[bold magenta]============================================================[/bold magenta]\n")

        try:
            start_wall = datetime.datetime.now()
            metrics = pipeline.run(url)
            end_wall = datetime.datetime.now()

            record = {
                "benchmark_id": f"benchmark_{i}",
                "start_time": start_wall.isoformat(),
                "end_time": end_wall.isoformat(),
                "source_url": url,
                "title": metrics.get("title"),
                "source_language": metrics.get("language"),
                "language_confidence": metrics.get("lang_confidence"),
                "video_duration_seconds": metrics.get("video_duration"),
                "video_duration_formatted": f"{metrics.get('video_duration', 0)/60:.1f} mins",
                "segments_count": metrics.get("segments_count"),
                "voice_applied": metrics.get("voice"),
                "stage_timings_seconds": {
                    "download_and_extract": metrics.get("download_time"),
                    "whisper_transcription_and_translation": metrics.get("transcribe_time"),
                    "edge_tts_synthesis": metrics.get("synth_time"),
                    "timing_synchronization": metrics.get("sync_time"),
                    "lossless_remuxing": metrics.get("remux_time"),
                },
                "total_processing_seconds": metrics.get("total_time"),
                "total_processing_formatted": f"{metrics.get('total_time', 0)/60:.2f} mins",
                "real_time_factor": round(metrics.get("total_time", 0) / max(1.0, metrics.get("video_duration", 1)), 2),
                "output_video_path": os.path.abspath(metrics.get("output_video", "")),
                "output_file_size_mb": metrics.get("file_size_mb"),
                "status": "SUCCESS",
            }
            results.append(record)

        except Exception as e:
            console.print(f"[bold red]Benchmark run failed for {url}: {e}[/bold red]")
            results.append({
                "benchmark_id": f"benchmark_{i}",
                "source_url": url,
                "status": "FAILED",
                "error": str(e),
            })

    # Save structured JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "system": get_system_specs(),
            "timestamp": datetime.datetime.now().isoformat(),
            "benchmarks": results,
        }, f, indent=2)

    # Generate Markdown Report for Email Submission
    md_path = os.path.join(output_dir, "benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Automated Video Dubbing - Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**System Specs:** {platform.system()} {platform.release()} ({platform.machine()}) | Python {platform.python_version()}\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Video Title | Duration | Language | Total Processing Time | Real-Time Factor (RTF) | Output File |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for r in results:
            if r.get("status") == "SUCCESS":
                f.write(
                    f"| {r.get('title')} | {r.get('video_duration_formatted')} | {r.get('source_language', '').upper()} | "
                    f"**{r.get('total_processing_formatted')}** | {r.get('real_time_factor')}x | `{os.path.basename(r.get('output_video_path'))}` |\n"
                )
            else:
                f.write(f"| {r.get('source_url')} | FAILED | - | - | - | Error: {r.get('error')} |\n")

        f.write("\n## Stage Timing Breakdown\n\n")
        f.write("| Benchmark | Download | Whisper Transcribe & Translate | Edge TTS Synthesis | Audio Sync & Stretch | FFmpeg Remux |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            if r.get("status") == "SUCCESS":
                st = r.get("stage_timings_seconds", {})
                f.write(
                    f"| {r.get('benchmark_id')} | {st.get('download_and_extract')}s | {st.get('whisper_transcription_and_translation')}s | "
                    f"{st.get('edge_tts_synthesis')}s | {st.get('timing_synchronization')}s | {st.get('lossless_remuxing')}s |\n"
                )

    console.print(f"\n[bold green]✓ Benchmark complete! Report written to {md_path} and {json_path}[/bold green]")


RECOMMENDED_BENCHMARK_URLS = [
    {
        "label": "30-Minute Foreign Video",
        "url": "https://www.youtube.com/watch?v=GnNUFPdEnts",
        "desc": "ARTE French Documentary: 1347 La Peste Noire (~26.2 mins)",
    },
    {
        "label": "2-Hour Foreign Video",
        "url": "https://www.youtube.com/watch?v=8GFujvNsllU",
        "desc": "ARTE German Documentary: Asbest (~92.5 mins)",
    },
]


def ensure_demo_video() -> str:
    """Ensure a sample foreign video exists for instant benchmark verification."""
    demo_path = os.path.join("output", "temp", "sample_french_source.mp4")
    if not os.path.isfile(demo_path):
        from create_sample_dub import create_sample_media
        create_sample_media(demo_path)
    return demo_path


def main():
    parser = argparse.ArgumentParser(description="Run dubbing benchmarks and generate submission reports.")
    parser.add_argument("urls", nargs="*", help="YouTube URLs or video paths to benchmark")
    parser.add_argument("-o", "--output", default="output_benchmarks", help="Output directory for benchmark artifacts")
    parser.add_argument("-m", "--model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("-v", "--voice", default="en-US-ChristopherNeural", help="Neural voice name")
    parser.add_argument("--demo", action="store_true", help="Run benchmark on a local foreign demonstration video")
    parser.add_argument(
        "--recommended",
        action="store_true",
        help="Benchmark the recommended 30-min and long-form foreign documentary videos",
    )

    args = parser.parse_args()

    if args.recommended:
        urls = [item["url"] for item in RECOMMENDED_BENCHMARK_URLS]
        console.print("[bold green]Running benchmark on recommended foreign documentary videos:[/bold green]")
        for item in RECOMMENDED_BENCHMARK_URLS:
            console.print(f"  • {item['label']}: {item['desc']} ({item['url']})")
        run_benchmark(urls, output_dir=args.output, model=args.model, voice=args.voice)
        return

    if args.demo:
        demo_video = ensure_demo_video()
        run_benchmark([demo_video], output_dir=args.output, model=args.model, voice=args.voice)
        return

    urls = args.urls
    if not urls:
        console.print("[bold yellow]No benchmark URLs passed via CLI arguments.[/bold yellow]")
        console.print("Options:")
        console.print("  [cyan]1[/cyan]: Run instant demo benchmark on sample foreign video")
        console.print("  [cyan]2[/cyan]: Run benchmark on recommended 30-min & long-form foreign documentaries")
        console.print("  [cyan]Enter URL(s)[/cyan]: Benchmark specific YouTube URLs")
        line = console.input("\n[bold yellow]Select option (1/2) or enter URL(s) [Default: 1 (Demo)]: [/bold yellow]").strip()

        if not line or line == "1":
            demo_video = ensure_demo_video()
            urls = [demo_video]
        elif line == "2":
            urls = [item["url"] for item in RECOMMENDED_BENCHMARK_URLS]
        else:
            urls = line.split()

    run_benchmark(urls, output_dir=args.output, model=args.model, voice=args.voice)


if __name__ == "__main__":
    main()
