"""
FastAPI Server for Automated AI Video Dubbing Studio.
Provides REST endpoints, Server-Sent Events (SSE) real-time pipeline streaming,
and serves modern web UI and generated output media.
"""

import asyncio
from datetime import datetime
import json
import os
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dubber.config import AVAILABLE_VOICES, DubbingConfig, SUPPORTED_LANGUAGES, find_ffmpeg_executable
from dubber.pipeline import DubbingPipeline
from dubber.transcriber import SpeechTranscriber
from dubber.synthesizer import SpeechSynthesizer, get_media_duration
from dubber.synchronizer import AudioSynchronizer
from dubber.remuxer import VideoRemuxer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(WEB_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(
    title="Automated AI Video Dubbing Studio",
    description="Real-time web frontend & API for multi-speaker AI video dubbing",
    version="1.0.0",
)

# CORS middleware for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for active jobs
# job_id -> {"queue": asyncio.Queue, "status": str, "loop": asyncio.AbstractEventLoop, "history": list}
jobs: Dict[str, Dict[str, Any]] = {}


class DubRequest(BaseModel):
    url: str = Field(..., description="YouTube URL, YouTube Shorts URL, or local video path")
    source_lang: str = Field("auto", description="Original speech language (auto for auto-detection)")
    target_lang: str = Field("en", description="Target language code: te, hi, ta, kn, ml, es, fr, etc.")
    model: str = Field("small", description="Whisper model name: tiny, base, small, medium")
    auto_gender: bool = Field(True, description="Enable automated speaker gender & multi-speaker dubbing")
    speaker_mode: str = Field("auto", description="Speaker assignment mode: auto (person-matching lock), male_only, female_only, multi_speaker")
    male_voice: str = Field("en-US-ChristopherNeural", description="Voice ID for male speakers")
    female_voice: str = Field("en-US-JennyNeural", description="Voice ID for female speakers")
    voice: str = Field("en-US-JennyNeural", description="Fallback single voice when auto_gender is false")
    mix_original: float = Field(0.0, ge=0.0, le=1.0, description="Volume level (0.0 - 1.0) of original background audio (default 0.0 = clean dub)")
    embed_subtitles: bool = Field(True, description="Whether to embed subtitles into the MP4 container")
    subtitle_mode: str = Field("soft", description="Subtitle mode: soft, burn, dual, none")
    lip_sync_mode: str = Field("visual_adaptive", description="Lip sync mode: visual_adaptive, strict_lock, natural")
    enable_voice_preservation: bool = Field(True, description="Preserve original speaker vocal pitch, tone, and cadence")
    voice_preservation_mode: str = Field("adaptive_prosody", description="Voice preservation mode: adaptive_prosody, formant_transfer, none")
    enable_lipsync: bool = Field(True, description="Enable AI Lip Synchronization")
    lipsync_model: str = Field("wav2lip", description="Lip sync model: wav2lip, visual_adaptive")
    proofread_mode: bool = Field(False, description="Whether to pause for interactive script proofreading")


class PacingRequest(BaseModel):
    text: str
    target_duration: float
    target_lang: str = "en"


class RedubRequest(BaseModel):
    video_filename: str = Field(..., description="Target video filename or base name e.g. GdUMxKyqrSs_dubbed_en.mp4")
    segments: List[Dict[str, Any]] = Field(..., description="Updated segments list with voice, gender, text, start, end")
    mix_original: float = Field(0.0, ge=0.0, le=1.0, description="Volume level (0.0 - 1.0) of original background audio")
    target_lang: str = Field("en", description="Target language code")
    burn_subtitles: bool = Field(False, description="Whether to burn subtitles directly into video frames")


@app.post("/api/optimize_pacing")
async def optimize_pacing_endpoint(req: PacingRequest):
    """Auto-optimize script phrasing to match visual speech duration."""
    opt = SpeechTranscriber.optimize_pacing(req.text, req.target_duration, req.target_lang)
    return {"optimized_text": opt, "original_text": req.text}


@app.post("/api/redub")
async def redub_endpoint(req: RedubRequest):
    """
    Instant 1-Click Re-Dubbing for Script Proofreader.
    Re-synthesizes speech with modified voices/genders, synchronizes onto exact video duration,
    duck-mixes original background music, and remuxes output MP4 and subtitles in seconds.
    """
    t0 = time.time()
    filename = req.video_filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="video_filename is required")

    filename = os.path.basename(filename)
    if "?" in filename:
        filename = filename.split("?")[0]

    stem = os.path.splitext(filename)[0]
    base_id = stem.split("_dubbed_")[0]

    # Find source video
    candidate_videos = [
        os.path.join(OUTPUT_DIR, "temp", f"{base_id}_source.mp4"),
        os.path.join(OUTPUT_DIR, filename),
        os.path.join(OUTPUT_DIR, f"{base_id}_dubbed_en.mp4"),
    ]
    temp_root = os.path.join(OUTPUT_DIR, "temp")
    if os.path.exists(temp_root):
        for sub in os.scandir(temp_root):
            if sub.is_dir() and sub.name.startswith("job_"):
                candidate_videos.append(os.path.join(sub.path, "source.mp4"))
                candidate_videos.append(os.path.join(sub.path, f"{base_id}_source.mp4"))

    source_video = None
    for cand in candidate_videos:
        if os.path.isfile(cand):
            source_video = cand
            break

    if not source_video:
        raise HTTPException(status_code=404, detail=f"Source video for '{filename}' not found.")

    # Find source audio
    candidate_audios = [
        os.path.join(OUTPUT_DIR, "temp", f"{base_id}_source_audio.wav"),
        os.path.join(OUTPUT_DIR, f"temp_inspect_{base_id}.wav"),
    ]
    if os.path.exists(temp_root):
        for sub in os.scandir(temp_root):
            if sub.is_dir() and sub.name.startswith("job_"):
                candidate_audios.append(os.path.join(sub.path, "source_audio.wav"))

    source_audio = None
    for cand in candidate_audios:
        if os.path.isfile(cand):
            source_audio = cand
            break

    ffmpeg_path = find_ffmpeg_executable()
    actual_dur = get_media_duration(source_video, ffmpeg_path)
    if actual_dur <= 0:
        actual_dur = max((s.get("end", 0.0) for s in req.segments), default=10.0)

    # 1. Synthesize segments with voices
    target_info = SUPPORTED_LANGUAGES.get(req.target_lang, {})
    default_male = target_info.get("male_voice", "en-US-ChristopherNeural")
    default_female = target_info.get("female_voice", "en-US-JennyNeural")

    for idx, s in enumerate(req.segments):
        s["id"] = idx + 1
        if not s.get("voice") or s.get("voice") == "custom":
            s["voice"] = default_female if s.get("gender") == "female" else default_male
        elif s.get("gender") == "female" and "Christopher" in str(s.get("voice")):
            s["voice"] = default_female
        elif s.get("gender") == "male" and "Jenny" in str(s.get("voice")):
            s["voice"] = default_male

    redub_temp = os.path.join(OUTPUT_DIR, "temp", f"redub_{base_id}_{int(time.time())}")
    os.makedirs(redub_temp, exist_ok=True)
    tts_dir = os.path.join(redub_temp, "tts")

    synthesizer = SpeechSynthesizer(ffmpeg_path=ffmpeg_path)
    synth_segments = synthesizer.synthesize(req.segments, output_dir=tts_dir)

    # 2. Synchronize timeline (preserving post-dialogue audio at full volume)
    synchronizer = AudioSynchronizer(ffmpeg_path=ffmpeg_path)
    dubbed_wav = os.path.join(redub_temp, "dubbed.wav")
    synchronizer.synchronize(
        segments=synth_segments,
        output_path=dubbed_wav,
        total_duration=actual_dur,
        temp_dir=os.path.join(redub_temp, "sync"),
        original_audio_path=source_audio,
        mix_original_volume=req.mix_original,
    )

    # 3. Export updated Subtitles
    out_mp4_name = f"{base_id}_dubbed_{req.target_lang}.mp4"
    out_mp4_path = os.path.join(OUTPUT_DIR, out_mp4_name)
    srt_path = os.path.join(OUTPUT_DIR, f"{base_id}_subtitles_{req.target_lang}.srt")
    vtt_path = os.path.join(OUTPUT_DIR, f"{base_id}_subtitles_{req.target_lang}.vtt")
    bilingual_srt_path = os.path.join(OUTPUT_DIR, f"{base_id}_subtitles_bilingual.srt")

    SpeechTranscriber.export_srt(req.segments, srt_path)
    SpeechTranscriber.export_vtt(req.segments, vtt_path)
    SpeechTranscriber.export_bilingual_srt(req.segments, bilingual_srt_path)

    # 4. Remux final video with both Dubbed English (Track 1) and Original Audio (Track 2)
    remuxer = VideoRemuxer(ffmpeg_path=ffmpeg_path)
    remuxer.remux(
        video_path=source_video,
        audio_path=dubbed_wav,
        output_path=out_mp4_path,
        subtitle_path=srt_path,
        burn_subtitles=req.burn_subtitles,
        original_audio_path=source_audio,
        include_original_track=False,
    )

    elapsed = round(time.time() - t0, 2)
    return {
        "status": "success",
        "video_url": f"/output/{out_mp4_name}",
        "srt_url": f"/output/{os.path.basename(srt_path)}",
        "vtt_url": f"/output/{os.path.basename(vtt_path)}",
        "bilingual_srt_url": f"/output/{os.path.basename(bilingual_srt_path)}",
        "duration": round(actual_dur, 2),
        "segments": req.segments,
        "elapsed": elapsed,
        "message": f"Successfully re-dubbed {len(req.segments)} segments in {elapsed}s",
    }


@app.get("/api/languages")
async def get_languages():
    """Return all supported Indian and global languages with default male/female voice pairs."""
    lang_list = []
    for code, info in SUPPORTED_LANGUAGES.items():
        lang_list.append({
            "code": code,
            "name": info.get("name", code),
            "locale": info.get("locale", code),
            "region": info.get("region", "Global"),
            "male_voice": info.get("male_voice"),
            "female_voice": info.get("female_voice"),
        })
    return {"languages": lang_list}


@app.get("/api/voices")
async def get_voices():
    """Return all available neural voices categorized by gender and locale."""
    voice_list = []
    for voice_id, details in AVAILABLE_VOICES.items():
        voice_list.append({
            "id": voice_id,
            "gender": details.get("gender", "Unknown"),
            "locale": details.get("locale", "en-US"),
            "language": details.get("language", ""),
            "description": details.get("description", ""),
        })
    return {"voices": voice_list}


@app.get("/api/videos")
async def get_videos():
    """List all previously dubbed MP4 videos in the output directory."""
    videos = []
    if os.path.exists(OUTPUT_DIR):
        for entry in os.scandir(OUTPUT_DIR):
            if entry.is_file() and entry.name.endswith(".mp4"):
                stem = os.path.splitext(entry.name)[0]
                # Look for matching .srt and .vtt files
                srt_url = None
                vtt_url = None
                bilingual_srt_url = None

                for srt_cand in [f"{stem}.srt", entry.name.replace(".mp4", ".srt"), f"{entry.name.split('_dubbed_')[0]}_subtitles_en.srt"]:
                    if os.path.isfile(os.path.join(OUTPUT_DIR, srt_cand)):
                        srt_url = f"/output/{srt_cand}"
                        break

                vtt_cand = srt_cand.replace(".srt", ".vtt") if srt_url else None
                if vtt_cand and os.path.isfile(os.path.join(OUTPUT_DIR, vtt_cand)):
                    vtt_url = f"/output/{vtt_cand}"

                bilingual_cand = f"{entry.name.split('_dubbed_')[0]}_subtitles_bilingual.srt"
                if os.path.isfile(os.path.join(OUTPUT_DIR, bilingual_cand)):
                    bilingual_srt_url = f"/output/{bilingual_cand}"

                stat = entry.stat()
                videos.append({
                    "filename": entry.name,
                    "url": f"/output/{entry.name}",
                    "srt_url": srt_url,
                    "vtt_url": vtt_url,
                    "bilingual_srt_url": bilingual_srt_url,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": stat.st_mtime,
                })
    # Sort descending by creation date
    videos.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"videos": videos}


@app.delete("/api/videos/{filename}")
async def delete_video(filename: str):
    """
    Delete a dubbed MP4 video and its associated subtitle and metadata files from the output directory.
    Validates the filename to prevent directory traversal and ensure safety.
    """
    cleaned = filename.strip().split("?")[0]
    cleaned = os.path.basename(cleaned)
    if not cleaned or not cleaned.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid video filename. Must be an .mp4 file.")

    video_path = os.path.join(OUTPUT_DIR, cleaned)
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Video '{cleaned}' not found in output directory.")

    deleted_files = []
    try:
        os.remove(video_path)
        deleted_files.append(cleaned)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete video file: {str(e)}")

    # Clean up associated subtitles and sidecar files
    stem = os.path.splitext(cleaned)[0]
    base_id = stem.split("_dubbed_")[0]

    candidates = {
        f"{stem}.srt",
        f"{stem}.vtt",
        f"{base_id}_subtitles_bilingual.srt",
    }
    if os.path.exists(OUTPUT_DIR):
        for entry in os.scandir(OUTPUT_DIR):
            if entry.is_file():
                if entry.name.startswith(f"{base_id}_subtitles_") and (entry.name.endswith(".srt") or entry.name.endswith(".vtt")):
                    candidates.add(entry.name)

    for cand in candidates:
        cand_path = os.path.join(OUTPUT_DIR, cand)
        if os.path.isfile(cand_path):
            try:
                os.remove(cand_path)
                deleted_files.append(cand)
            except OSError:
                pass

    return {
        "status": "success",
        "message": f"Successfully deleted '{cleaned}' and {len(deleted_files) - 1} associated subtitle files.",
        "filename": cleaned,
        "deleted_files": deleted_files,
    }


@app.post("/api/dub")
async def start_dubbing(req: DubRequest, request: Request):
    """Start a dubbing job asynchronously and return a tracking job_id."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL or file path cannot be empty")

    job_id = str(uuid.uuid4())[:8]
    loop = asyncio.get_running_loop()
    job_queue: asyncio.Queue = asyncio.Queue()

    jobs[job_id] = {
        "queue": job_queue,
        "status": "starting",
        "loop": loop,
        "history": [],
    }

    def run_job_thread():
        try:
            # Build configuration for this run
            config = DubbingConfig(
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                whisper_model=req.model,
                models_dir=MODELS_DIR,
                output_dir=OUTPUT_DIR,
                temp_dir=os.path.join(OUTPUT_DIR, "temp", f"job_{job_id}"),
                voice=req.voice,
                auto_gender=req.auto_gender,
                speaker_mode=req.speaker_mode,
                male_voice=req.male_voice,
                female_voice=req.female_voice,
                mix_original_volume=req.mix_original,
                embed_subtitles=req.embed_subtitles,
                subtitle_mode=req.subtitle_mode,
                lip_sync_mode=req.lip_sync_mode,
                voice_preservation_mode=req.voice_preservation_mode,
                enable_voice_preservation=req.enable_voice_preservation,
                enable_lipsync=req.enable_lipsync,
                lipsync_model=req.lipsync_model,
                proofread_mode=req.proofread_mode,
                keep_temp=False,
            )

            pipeline = DubbingPipeline(config=config)

            def safe_put_event(item):
                if loop and not loop.is_closed():
                    try:
                        asyncio.run_coroutine_threadsafe(job_queue.put(item), loop)
                    except (RuntimeError, Exception):
                        pass

            def event_callback(evt: Dict[str, Any]):
                # Store in history so late-connecting clients get state
                jobs[job_id]["history"].append(evt)
                safe_put_event(evt)

            pipeline.run(url_or_path=url, event_callback=event_callback)
            jobs[job_id]["status"] = "finished"

        except Exception as e:
            trace = traceback.format_exc()
            err_evt = {
                "stage": -1,
                "status": "error",
                "error": str(e) or "An error occurred during dubbing execution.",
                "traceback": trace,
            }
            jobs[job_id]["history"].append(err_evt)
            jobs[job_id]["status"] = "failed"
            safe_put_event(err_evt)

        finally:
            # Sentinel to close stream
            safe_put_event(None)

    thread = threading.Thread(target=run_job_thread, daemon=True, name=f"dubber-job-{job_id}")
    thread.start()

    return {"job_id": job_id, "status": "started"}


@app.get("/api/dub/status/{job_id}")
async def get_job_status(job_id: str):
    """Return the current status and latest history event for a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job_data = jobs[job_id]
    history = job_data.get("history", [])
    latest_evt = history[-1] if history else {}
    return {
        "job_id": job_id,
        "status": job_data.get("status", "unknown"),
        "latest_event": latest_evt,
        "events_count": len(history),
    }


@app.get("/api/dub/stream/{job_id}")
async def stream_dubbing_progress(job_id: str):
    """Server-Sent Events (SSE) endpoint providing real-time pipeline stage progress."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = jobs[job_id]
    job_queue = job_data["queue"]

    async def event_generator():
        # First yield connection ack
        yield f"data: {json.dumps({'status': 'connected', 'job_id': job_id})}\n\n"

        # Replay any history if already emitted
        for past_evt in job_data.get("history", []):
            yield f"data: {json.dumps(past_evt)}\n\n"
            if past_evt.get("status") in ("complete", "error"):
                return

        while True:
            try:
                evt = await asyncio.wait_for(job_queue.get(), timeout=20.0)
                if evt is None:
                    break
                yield f"data: {json.dumps(evt)}\n\n"
                if evt.get("status") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                # SSE heartbeat keep-alive comment
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Serve static web assets
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Mount output folder to support HTML5 video player range streaming
if os.path.exists(OUTPUT_DIR):
    app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Health check endpoint for cloud deployments and monitors."""
    return {"status": "ok", "app": "PolyDubAI", "version": "1.0.0"}


@app.get("/")
async def root():
    """Serve the single-page application."""
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>Web frontend is initializing...</h2>")

