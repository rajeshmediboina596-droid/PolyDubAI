# PolyDubAI — AI-Powered Multilingual Video Dubbing with Voice Preservation and Lip Synchronization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![faster-whisper](https://img.shields.io/badge/transcription-faster--whisper-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![edge-tts](https://img.shields.io/badge/synthesis-edge--tts-green.svg)](https://github.com/rany2/edge-tts)
[![Wav2Lip](https://img.shields.io/badge/lipsync-Wav2Lip--FaceSync-purple.svg)](https://github.com/Rudrabha/Wav2Lip)
[![FFmpeg](https://img.shields.io/badge/remux-FFmpeg-red.svg)](https://ffmpeg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PolyDubAI** is an advanced, modular AI video dubbing system that translates any foreign video URL or upload into any user-selected language (15+ Indian & Global languages). The system accurately **preserves the original speaker's vocal pitch, tone, style, and speaking cadence** while **synchronizing the speaker's lip movements** with the translated speech.

Built with an **8 GB RAM laptop-optimized streaming pipeline** that ensures complete stability without Out-Of-Memory (OOM) crashes.

---

## 🌟 Key Features

1. **Video URL Input & Upload**:
   - Accepts standard YouTube URLs, YouTube Shorts, or local video file uploads.
2. **Speech Transcription & Alignment**:
   - Extracts original speech using Faster-Whisper (int8 quantized) with millisecond-accurate word/segment timestamps.
3. **Multilingual Translation**:
   - Translates speech into any user-selected target language:
     - **Indian Languages**: Telugu, Hindi, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Urdu, Indian English.
     - **Global Languages**: English (US/UK), Spanish, French, German, Japanese, Korean, Mandarin, Arabic, Portuguese, Russian, Italian.
4. **Speaker Voice Preservation (Phase 3)**:
   - Analyzes original audio acoustic characteristics ($F_0$ pitch contour, median pitch, speaking tempo, vocal energy).
   - Dynamically modulates neural speech synthesis via SSML prosody parameters (`pitch="+XHz"`, `rate="+Y%"`) and formant frequency shaping to replicate the speaker's natural vocal timbre.
5. **AI Lip Synchronization (Phase 4 — Wav2Lip / Face-Sync)**:
   - Tracks speaker face ROI and lower-third mouth coordinates with exponential moving average (EMA) smoothing to eliminate jitter.
   - Computes 80-channel mel-spectrogram windows to align mouth aperture, lip closure (bilabials), and inner oral cavity depth to the translated audio.
   - Uses **Gaussian feathered elliptical alpha blending** for seamless integration with zero visible square seams or artifacts.
6. **Final Video Delivery & Subtitles**:
   - Lossless audio/video multiplexing via FFmpeg.
   - Preserves background ambient soundtrack with zero foreign speech bleed (`duck_vol = 0.0` during dialogue, `0.90` during pauses, `1.0` during musical outros).
   - Generates and embeds switchable Soft Subtitles (.SRT/.VTT) and bilingual dual-language subtitles.
7. **8 GB RAM Laptop Protection**:
   - Frame-streaming micro-batches ($N=4$) — never loads uncompressed full video into memory.
   - Speech-selective processing: only processes video frames during active speech timestamps, skipping 60%+ non-speech/music frames.

---

## 🏗️ Architecture & 5-Phase Pipeline

```
                              [ Video URL / Upload ]
                                        │
                                        ▼
                             [ YouTubeDownloader ]
                                  (yt-dlp)
                                  ┌─────┴─────┐
                             Video (MP4)   Audio (WAV)
                                  │           │
                                  │           ▼
                                  │    [ SpeechTranscriber ]
                                  │    (Faster-Whisper int8)
                                  │           │
                                  │     Segments + Timestamps
                                  │           │
                                  │           ▼
                                  │    [ MultilingualTranslator ]
                                  │    (Whisper + Deep-Translator)
                                  │           │
                   ┌──────────────┴───────────┼───────────────────┐
                   │                          ▼                   │
                   │               [ SpeakerVoicePreserver ]      │
                   │               (Phase 3: Voice Cloner)        │
                   │               - F0 Pitch & Timbre Profiling  │
                   │               - Tone & Cadence Transfer      │
                   │               - Neural SSML Prosody Deltas   │
                   │                          │                   │
                   │                          ▼                   │
                   │                 [ AudioSynchronizer ]        │
                   │                 - Dynamic Tempo Scaling      │
                   │                 - Zero Foreign Speech Bleed  │
                   │                 - Background Music Retention │
                   │                          │                   │
                   │                          ▼                   │
                   │                   Dubbed Audio Track         │
                   │                          │                   │
                   ▼                          ▼                   │
          [ AILipSynchronizer ] ◄─────────────┘                   │
          (Phase 4: Wav2Lip Face-Sync)                            │
          - Face Landmark & ROI Detection                         │
          - 80-Channel Mel-Windows                                │
          - Streaming Frame Generation                            │
          - Seamless Gaussian Feathering                          │
                   │                                              │
                   ▼                                              ▼
           Lip-Synced Video                              Subtitles (.SRT/.VTT)
                   │                                              │
                   └──────────────────────┬───────────────────────┘
                                          ▼
                                   [ VideoRemuxer ]
                                  (FFmpeg Stream-Mux)
                                          │
                                          ▼
                             [ Final PolyDubAI Video ]
```

---

## 💻 8 GB RAM Optimization Strategy

| Component | Unoptimized (Crash Risk) | PolyDubAI 8 GB RAM Safe Design |
|---|---|---|
| **Video Decoding** | Full uncompressed frames in RAM ($> 5\text{ GB}$) | Streaming frame generator with `cv2.VideoCapture` ($< 150\text{ MB}$) |
| **Face-Sync Inference** | Batch size 32–64 (GPU VRAM OOM) | Micro-batch size $N=4$ on CPU / quantized PyTorch |
| **Frame Scope** | Processing every frame (0 to 60s) | **Speech-Selective**: processes frames ONLY during dialogue; skips pauses & music |
| **Whisper Transcription** | Float32 Whisper Large ($> 6\text{ GB}$) | Int8 Quantized CTranslate2 ($< 1\text{ GB}$ RAM) |
| **Voice Preservation** | 16 GB VRAM diffusion voice cloning | Acoustic $F_0$ Autocorrelation + SSML Prosody & Formant Transfer ($0\text{ MB}$ VRAM) |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/rajeshmediboina596-droid/PolyDubAI.git
cd PolyDubAI

# Create and activate Python virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Web Studio (Recommended)

```bash
python run_web.py --port 8000
```
Open **`http://localhost:8000`** in your web browser.

### 3. CLI Usage

```bash
# Basic Dubbing (YouTube Shorts / Video URL)
python main.py "https://youtube.com/shorts/GdUMxKyqrSs" --target-lang en

# Dub into Telugu with Voice Preservation and AI Lip-Sync
python main.py "https://youtube.com/shorts/GdUMxKyqrSs" --target-lang te --voice-preservation adaptive_prosody --enable-lipsync

# Dub into Hindi with Bilingual Subtitles
python main.py "https://youtube.com/shorts/GdUMxKyqrSs" --target-lang hi --subtitle-mode dual
```

---

## 🧪 Running Automated Tests

Run the full automated test suite (52+ passing tests covering all 5 phases):

```bash
pytest tests/ -v
```

Test coverage includes:
- `tests/test_voice_cloner.py`: Fundamental pitch ($F_0$) estimation, pitch deltas, prosody adaptation, and timbre formant transfer.
- `tests/test_lipsync.py`: Mel-filterbank calculation, window extraction, jitter-free bounding box smoothing, feathered blending, and streaming lip-sync.
- `tests/test_server.py`: FastAPI endpoints, video deletion, language catalog, and SSE streaming.
- `tests/test_multilingual.py`: Translation across Indian and global languages.
- `tests/test_synchronizer.py`: Time-stretching, zero speech bleed, and background music retention.

---

## 📁 Repository Structure

```
PolyDubAI/
├── dubber/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration & language catalog
│   ├── downloader.py         # YouTube downloader (yt-dlp)
│   ├── transcriber.py        # Faster-Whisper transcription & subtitles
│   ├── classifier.py         # Speaker pitch & gender classifier
│   ├── visual_classifier.py  # OpenCV face & vision classifier
│   ├── voice_cloner.py       # Phase 3: Speaker Voice Preservation & Timbre Transfer
│   ├── synthesizer.py        # Edge-TTS neural speech synthesis
│   ├── synchronizer.py       # Audio timing alignment & dynamic tempo scaling
│   ├── lipsync.py            # Phase 4: AI Lip Synchronization (Wav2Lip / Face-Sync)
│   ├── remuxer.py            # Lossless FFmpeg video/audio remuxer
│   └── ui.py                 # Rich terminal output formatting
├── web/
│   ├── index.html            # PolyDubAI Web Studio interface
│   ├── app.js                # SSE streaming & interactive UI logic
│   └── style.css             # Glassmorphism dark-theme styling
├── tests/                    # Pytest automated test suite
├── server.py                 # FastAPI backend with REST & SSE endpoints
├── main.py                   # CLI entrypoint
├── Dockerfile                # Deployment configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation
```

---

## 📜 License & Compliance

MIT License. Designed and engineered for academic and internship evaluation at **IDEALABS DIGITAL**.
