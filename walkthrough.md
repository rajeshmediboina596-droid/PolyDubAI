# 2-Minute Walkthrough Video Script & Architecture Guide

This document contains the exact script, timing breakdown, and technical rationale for the **2-minute walkthrough video** required for the **IDEALABS DIGITAL** internship submission.

---

## 2-Minute Video Script (Timing Breakdown)

### 0:00 – 0:20 | Introduction & The Problem
> *"Hello! Today I'm presenting my Automated Video Dubbing System. Dubbing foreign video into English is more than just translating words—it requires natural voices, emotional cadence, and crucially, perfect timing synchronization with the original video so the lips and scene changes don't drift apart. Here is a live example running on a foreign speech video: with a single command, we extract the video, transcribe and translate with Whisper AI, synthesize expressive neural speech, align the timeline, and remux the final video losslessly."*

### 0:20 – 0:55 | Architecture & Technology Decisions
> *"Let's look at the modular architecture. We structured the codebase into five distinct, decoupled components:*
> 1. *First, `YouTubeDownloader`: uses `yt-dlp` to download the highest quality MP4 stream and extract a 16kHz mono WAV track for transcription.*
> 2. *Second, `SpeechTranscriber`: powered by `faster-whisper` running on CTranslate2 with int8 quantization. It leverages Whisper's native translation task combined with Voice Activity Detection (VAD). This directly produces English text while filtering out ambient noise and providing millisecond-accurate segment timestamps.*
> 3. *Third, `SpeechSynthesizer`: utilizes Microsoft Edge's Neural TTS. Unlike robotic offline engines or massive GPU-bound models, `edge-tts` provides lifelike conversational prosody, supports male and female voices across multiple accents, and executes asynchronously in batches.*

### 0:55 – 1:30 | Technical Deep Dive: Audio Synchronization (Zero Cumulative Drift)
> *"Now, the most critical engineering challenge in dubbing: **timing synchronization**.*
> *In natural translation, an English sentence is almost never the same syllable length as the original foreign phrase. If you naively concatenate speech clips, by minute 10 the audio will be completely desynchronized from the video.*
> *To solve this, our `AudioSynchronizer` computes the duration delta between the synthesized audio and the original speaker's window. If the English speech is slightly longer, it dynamically applies FFmpeg's `atempo` filter—accelerating speech by up to 1.30x without altering pitch. If it's shorter, it places the audio at the exact start timestamp and pads with silence. This guarantees zero cumulative drift across a 30-minute or 2-hour video."*

### 1:30 – 1:50 | Lossless Remuxing & Performance Benchmarks
> *"For the final stage, `VideoRemuxer` executes a lossless stream copy (`-c:v copy`) with FFmpeg, replacing the audio and embedding soft SubRip subtitles in seconds without re-encoding video frames. Visual fidelity is 100% preserved.*
> *We also developed a dedicated `benchmark.py` suite. On our 30-minute and 2-hour benchmark videos, it profiles every stage, logging memory usage, processing time, and real-time factor into a submission-ready report."*

### 1:50 – 2:00 | Conclusion & Code Quality
> *"The code is fully typed, thoroughly documented, and backed by a comprehensive unit test suite covering configuration, speech synthesis, and timeline alignment. Thank you for watching!"*

---

## Key Architecture Decisions & Tradeoffs

| Component | Selected Technology | Alternative Considered | Why This Choice Won |
| :--- | :--- | :--- | :--- |
| **Downloader** | `yt-dlp` | `pytube` | `pytube` frequently breaks when YouTube updates cipher algorithms. `yt-dlp` is actively maintained, handles formats robustly, and supports streaming. |
| **Transcription & Translation** | `faster-whisper` (CTranslate2, int8) | `openai-whisper` (PyTorch) | `faster-whisper` is 4x faster on CPU, uses 60% less memory with int8 quantization, and Whisper's `task="translate"` outputs native timestamped English text directly. |
| **Voice Synthesis** | `edge-tts` | `pyttsx3`, Coqui `XTTS-v2` | `pyttsx3` sounds robotic. Local XTTS requires 6+ GB VRAM and slows down drastically on long videos. `edge-tts` produces human-grade neural voices with zero GPU overhead and fast async execution. |
| **Timing Alignment** | Dynamic `atempo` + Segment Timeline | Naive audio concatenation | Naive concatenation causes massive temporal drift over 30m/2h videos. Dynamic atempo scales speed by $\le 30\%$ cleanly without pitch distortion. |
| **Remuxing** | FFmpeg `-c:v copy` | Full video re-encoding (libx264) | Re-encoding a 2-hour video takes hours and degrades quality. Stream copy (`-c:v copy`) completes in under 10 seconds with 100% original visual fidelity. |

---

## Recording Checklist for the Applicant

1. **Screen Setup**:
   - Left side: Terminal showing `python main.py` or `python run_web.py` processing with Rich progress bars.
   - Right side: Split screen showing source video and dubbed output video playing side by side.
2. **Audio Check**:
   - Ensure microphone audio is crisp and clearly audible.
   - Play a 5-second sample of the original foreign video, followed by the English-dubbed version to showcase voice quality and timing.
3. **Format**:
   - Export as MP4 (1080p, 30fps).
   - Keep length strictly between 1:55 and 2:05.

---

## Audio Matching & Person Consistency (Verified for `GdUMxKyqrSs`)

1. **Dialogue Role Mapping & Multi-Speaker Assignment**:
   - In the Telugu dialogue short (`https://youtube.com/shorts/GdUMxKyqrSs` from *Calling Sahasra*):
     - **Dollysha** (Heroine, Female): Segments 1, 2, 3, 6 (0.60–1.44s, 3.66–5.26s, 6.26–7.58s, 13.70–15.28s) &rarr; **Female voice** (`en-US-JennyNeural` / `en-IN-NeerjaNeural`, fundamental frequency $\approx 180\text{--}200\text{ Hz}$).
     - **Sudheer** (Hero, Male): Segments 4, 5, 7 (7.66–9.10s, 9.46–11.10s, 16.83–20.34s) &rarr; **Male voice** (`en-US-ChristopherNeural` / `en-IN-PrabhatNeural`, fundamental frequency $\approx 115\text{--}120\text{ Hz}$).
2. **Clean Speech & Background Soundtrack Preservation**:
   - During dialogue speech: Original audio volume is cleanly ducked to `0.0` (zero foreign speech bleed under the English dub).
   - In conversation pauses: Original background audio is preserved at `0.90`.
   - Post-dialogue musical outro: Full original song and soundtrack play at `1.0` (100% volume) continuously from 20.34s to 64.44s.
   - Full video duration: Exactly 64.44s (100% duration of original short).
3. **Single Audio Track Remuxing**:
   - Audio output is remuxed with a single pristine default audio track (`include_original_track=False`) to avoid multi-track player confusion in browsers.

---

## Dubbed Video Deletion System

1. **REST API Endpoint**:
   - `DELETE /api/videos/{filename}`
   - Input sanitization via `os.path.basename` prevents directory traversal attacks.
   - Automatically deletes the `.mp4` video alongside companion `.srt`, `.vtt`, and `_bilingual.srt` subtitle files.
   - Returns JSON status with list of deleted files.
2. **Web UI Integration**:
   - **Video Library**: Each card in the gallery includes a dedicated `🗑️` Delete button with red danger accent.
   - **Showcase Player**: The player header action bar includes a `🗑️ Delete Video` button to delete the currently previewed video.
   - Interactive `confirm()` dialog prevents accidental deletions.
   - Dynamic reset: if the active video is deleted, the player automatically unloads, the gallery refreshes, and a floating toast notification confirms deletion.
3. **Test Coverage**:
   - All 40 unit and integration tests passing (`pytest tests/ -v`).

