# Automated Video Dubbing System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![faster-whisper](https://img.shields.io/badge/transcription-faster--whisper-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![edge-tts](https://img.shields.io/badge/synthesis-edge--tts-green.svg)](https://github.com/rany2/edge-tts)
[![FFmpeg](https://img.shields.io/badge/remux-FFmpeg-red.svg)](https://ffmpeg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Python system that takes a YouTube video in any foreign language (German, French, Hindi, Spanish, Japanese, etc.), downloads it, transcribes & translates the speech into natural English, synthesizes neural English voices, aligns audio segments dynamically to preserve mouth-movement timing, and losslessly remuxes the audio into the final video.

Built for the **IDEALABS DIGITAL Internship & Assignment**.

---

## Architecture

```
                          [ YouTube URL ]
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   YouTubeDownloader     │ (yt-dlp)
                   │  - Best stream download │
                   │  - 16kHz WAV extraction │
                   └────────────┬────────────┘
                                │ video.mp4 + audio.wav
                                ▼
                   ┌─────────────────────────┐
                   │    SpeechTranscriber    │ (faster-whisper / CTranslate2)
                   │  - VAD speech detection │
                   │  - Segment timestamps   │
                   │  - Direct En translation│
                   └────────────┬────────────┘
                                │ Segments: [{start, end, text_en}, ...]
                                ▼
                   ┌─────────────────────────┐
                   │    SpeechSynthesizer    │ (edge-tts)
                   │  - Neural English voice │
                   │  - Async batch synthesis│
                   │  - Rate / Pitch control │
                   └────────────┬────────────┘
                                │ Raw segment audio clips
                                ▼
                   ┌─────────────────────────┐
                   │    AudioSynchronizer    │ (pydub + FFmpeg atempo)
                   │  - Slot duration match  │
                   │  - Dynamic time-stretch │ (atempo 1.0x-1.3x)
                   │  - Silence padding      │
                   │  - Zero-drift timeline  │
                   └────────────┬────────────┘
                                │ dubbed_audio_synced.wav
                                ▼
                   ┌─────────────────────────┐
                   │      VideoRemuxer       │ (FFmpeg -c:v copy)
                   │  - Lossless remuxing    │
                   │  - SRT subtitle muxing  │
                   └────────────┬────────────┘
                                │
                                ▼
                     [ Final Dubbed Video ]
```

---

## Key Features

1. **Multi-Language Speech Translation**:
   - Uses `faster-whisper` (CTranslate2 with int8 quantization) to transcribe and translate foreign speech directly to English.
   - Built-in Voice Activity Detection (VAD) avoids hallucinating on background music or sound effects.
   - Emits exact segment-level timestamp boundaries `[start, end]`.

2. **Natural Neural Voice Synthesis**:
   - Uses Microsoft Edge Neural TTS for lifelike, expressive English voices with no robotic artifacts.
   - Zero GPU required for voice generation; runs asynchronously with concurrency control.
   - Full catalogue of American, British, and Indian English male and female voices.

3. **Automatic Speaker Gender Detection & Multi-Speaker Dubbing**:
   - Analyzes vocal pitch ($F_0$) via normalized autocorrelation on original speech audio for every segment.
   - Automatically identifies whether the speaker is **male** (boy/man, $F_0 < 165$ Hz) or **female** (girl/woman, $F_0 \ge 165$ Hz).
   - In multi-speaker videos (conversations, interviews, co-presenters), seamlessly alternates between male (Christopher) and female (Jenny) neural voices at exact segment boundaries!
   - Temporal smoothing prevents voice-flipping on short interjections.

4. **Audio-Visual Timing Synchronization (Zero Cumulative Drift)**:
   - *The Problem*: Translated English phrases are rarely the exact same syllable length as original foreign phrases. Naive concatenation causes dubs to drift by minutes over long videos.
   - *Our Solution*: Segment-by-segment timeline assembly:
     - Compares synthesized duration $D_{tts}$ against original speech window $D_{slot}$.
     - If slightly longer, dynamically accelerates audio up to $1.30\times$ using FFmpeg's `atempo` filter (maintaining natural pitch).
     - If shorter, places speech at the exact start timestamp and pads with natural silence.
     - Guarantees 100% sync alignment across **30-minute** and **2-hour** videos!

5. **Lossless Remuxing**:
   - Employs FFmpeg's stream copy (`-c:v copy`) to swap the audio track in seconds without touching or re-encoding visual frames.
   - Lossless visual fidelity, ultra-fast export.
   - Optionally embeds soft SubRip subtitles (`.srt`) in English.

6. **Rich Terminal Experience**:
   - Modern CLI with styled progress bars, speaker badges (`♂ Male` / `♀ Female`), elapsed timers, transfer speeds, and translation preview tables.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Windows, macOS, or Linux

### 2. Clone and Install Dependencies
```bash
cd assignment
pip install -r requirements.txt
```

> **FFmpeg Note**: The package automatically bundles FFmpeg via `imageio-ffmpeg` or auto-detects system `ffmpeg`. No manual environment path configuration required!

---

## Web Interface (Dubber Studio)

An interactive, modern web frontend with real-time SSE pipeline streaming, speaker demographic visualizers, in-browser video player, and video library.

```bash
# Launch Dubber Studio (automatically opens your default browser at http://localhost:8000)
python run_web.py

# Or launch via main CLI
python main.py --web

# Custom port
python run_web.py --port 8080
```

### Web Features
- **Modern Glassmorphic UI**: High-contrast dark theme with animated glowing gradients, responsive layout, and modern typography (`Outfit`, `Inter`).
- **Real-Time Pipeline Stepper**: Live Server-Sent Events (SSE) stream tracking all 5 stages (yt-dlp download, Whisper translation, F0 pitch classification, Edge-TTS synthesis, timeline sync, and lossless remuxing).
- **Speaker & Gender Visualization**: Live translated segment table detailing detected speaker demographics (`♂ Male` / `♀ Female`) with fundamental frequency ($F_0$ in Hz).
- **Built-in HTML5 Video Player**: Watch dubbed videos instantly in your browser with `.srt` subtitle toggle, speed controls, and one-click MP4 download.
- **Video Library**: Automatically scans and displays all previously dubbed videos in `output/` for instant playback and retrieval.

---

## Quick Start (CLI)

### Basic CLI Usage
```bash
# Dub a YouTube video with default settings (American Male: Christopher)
python main.py "https://www.youtube.com/watch?v=EXAMPLE_ID"

# Interactive prompt (if URL is omitted)
python main.py
```

### Voice Selection
```bash
# List all available neural voices
python main.py --list-voices

# Dub using a natural American Female voice (Jenny)
python main.py "https://www.youtube.com/watch?v=EXAMPLE_ID" --voice en-US-JennyNeural

# Dub using a British accent (Ryan)
python main.py "https://www.youtube.com/watch?v=EXAMPLE_ID" --voice en-GB-RyanNeural
```

# Multi-Speaker & Gender-Adaptive Options (Enabled by default!)
# By default, any video automatically detects male vs female speakers:
# Male speakers get Christopher (en-US-ChristopherNeural)
# Female speakers get Jenny (en-US-JennyNeural)

# Customize specific male and female voices (e.g. British accents):
python main.py "https://www.youtube.com/watch?v=EXAMPLE_ID" --male-voice en-GB-RyanNeural --female-voice en-GB-SoniaNeural

# Or disable automatic gender detection to force a single static voice:
python main.py "https://www.youtube.com/watch?v=EXAMPLE_ID" --no-auto-gender --voice en-US-GuyNeural

### Advanced Options
```bash
# Retain ducked original background audio (10% volume) for ambient sound & music
python main.py "https://www.youtube.com/watch?v=EXAMPLE_ID" --mix-original 0.10

# Use a larger Whisper model for maximum translation fidelity
python main.py "https://www.youtube.com/watch?v=EXAMPLE_ID" --model small

# Specify custom output directory
python main.py "https://www.youtube.com/watch?v=EXAMPLE_ID" --output my_dubs/
```

---

## Benchmarking (30-Minute & 2-Hour Videos)

For the internship submission, run `benchmark.py`:

```bash
# Run demo benchmark on sample foreign media (instant verification)
python benchmark.py --demo

# Benchmark recommended foreign documentary videos (30-min French & 90-min German)
python benchmark.py --recommended

# Or pass specific YouTube URLs:
python benchmark.py "https://www.youtube.com/watch?v=GnNUFPdEnts" "https://www.youtube.com/watch?v=8GFujvNsllU"
```

This will:
1. Process both videos end-to-end.
2. Record stage-by-stage timings (Download, Transcription, Synthesis, Synchronization, Remuxing).
3. Compute the Real-Time Factor (RTF).
4. Generate a submission report in `output_benchmarks/benchmark_report.md` and `output_benchmarks/benchmark_results.json`.

---

## Project Structure

```
assignment/
├── dubber/
│   ├── __init__.py           # Package initialization & FFmpeg registration
│   ├── config.py             # Config dataclasses & voice catalogue
│   ├── downloader.py         # YouTube downloader (yt-dlp) & audio extractor
│   ├── transcriber.py        # faster-whisper transcription & translation
│   ├── synthesizer.py        # edge-tts neural voice synthesis
│   ├── synchronizer.py       # Audio timing alignment & dynamic tempo scaling
│   ├── remuxer.py            # Lossless FFmpeg remuxer & subtitle muxer
│   ├── model_downloader.py   # Direct HTTPS model downloader & local cacher
│   ├── pipeline.py           # Orchestrator with checkpointing & caching
│   └── ui.py                 # Rich terminal UI & progress rendering
├── tests/
│   ├── test_config.py        # Configuration & FFmpeg tests
│   ├── test_transcriber.py   # Transcription & subtitle tests
│   ├── test_synthesizer.py   # Neural synthesis tests
│   ├── test_synchronizer.py  # Audio stretching & timeline alignment tests
│   └── test_pipeline.py      # End-to-end pipeline integration test
├── main.py                   # Main CLI entry point
├── benchmark.py              # Benchmarking & reporting utility
├── requirements.txt          # Python dependencies
├── walkthrough.md            # 2-minute video walkthrough script
└── README.md                 # System documentation
```

---

## Submission Checklist

For emailing `careers@idealabsdigital.com`:
- [x] **Source Videos**: Two foreign language YouTube URLs (30-min and 2-hour).
- [x] **Dubbed Outputs**: Saved in `output/` or `output_benchmarks/`.
- [x] **Processing Times**: Formatted in `output_benchmarks/benchmark_report.md`.
- [x] **Walkthrough Script**: See [walkthrough.md](walkthrough.md) for the 2-minute architectural explanation script.
