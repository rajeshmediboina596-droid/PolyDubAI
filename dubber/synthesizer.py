"""
Speech synthesis module using Microsoft Edge TTS neural voices.
"""

import asyncio
import os
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional
import edge_tts

from .config import AVAILABLE_VOICES, find_ffmpeg_executable


def get_media_duration(file_path: str, ffmpeg_path: str) -> float:
    """Extract audio duration in seconds using FFmpeg without needing ffprobe."""
    if not os.path.isfile(file_path):
        return 0.0

    if file_path.lower().endswith(".wav"):
        try:
            import wave
            with wave.open(file_path, "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            pass

    cmd = [ffmpeg_path, "-i", file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if match:
        hours = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds
    return 0.0


class SpeechSynthesizer:
    """Generates natural-sounding speech from text using neural voices."""

    def __init__(
        self,
        voice: str = "en-US-ChristopherNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        ffmpeg_path: Optional[str] = None,
        max_concurrent: int = 4,
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self.ffmpeg_path = ffmpeg_path or find_ffmpeg_executable()
        self.max_concurrent = max_concurrent

    def _synthesize_sapi_fallback(self, text: str, output_path: str, voice: Optional[str] = None) -> float:
        """Fallback to Windows built-in SAPI speech synthesis if Edge-TTS is blocked or offline."""
        try:
            temp_wav = output_path if output_path.lower().endswith(".wav") else output_path + ".temp.wav"
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.SetOutputToWaveFile('{os.path.abspath(temp_wav).replace(chr(92), chr(92)*2).replace(chr(39), chr(39)*2)}'); "
                f"$s.Speak('{text.replace(chr(39), chr(39)*2)}'); "
                "$s.Dispose()"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if os.path.isfile(temp_wav) and os.path.getsize(temp_wav) > 0:
                if temp_wav != output_path:
                    subprocess.run(
                        [self.ffmpeg_path, "-y", "-i", temp_wav, output_path],
                        capture_output=True,
                    )
                    try:
                        os.remove(temp_wav)
                    except OSError:
                        pass
                return get_media_duration(output_path, self.ffmpeg_path)
        except Exception:
            pass
        return 0.0

    async def _synthesize_single(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        retries: int = 3,
    ) -> float:
        """Synthesize a single text string to an audio file with retry logic and offline SAPI fallback."""
        v = voice or self.voice

        for attempt in range(retries):
            try:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=v,
                    rate=self.rate,
                    pitch=self.pitch,
                    volume=self.volume,
                )
                await communicate.save(output_path)
                if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    dur = get_media_duration(output_path, self.ffmpeg_path)
                    if dur > 0:
                        return dur
            except Exception as e:
                if attempt == retries - 1:
                    # Gracefully fallback to Windows SAPI
                    sapi_dur = self._synthesize_sapi_fallback(text, output_path, v)
                    if sapi_dur > 0:
                        return sapi_dur
                    raise RuntimeError(
                        f"Failed to synthesize text after {retries} attempts: '{text}'. Error: {e}"
                    )
                await asyncio.sleep(0.5 * (attempt + 1))

        return 0.0

    async def _synthesize_batch(
        self,
        segments: List[Dict[str, Any]],
        output_dir: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously synthesize audio for all segments with concurrency control."""
        os.makedirs(output_dir, exist_ok=True)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        total_segments = len(segments)
        completed_count = 0

        async def worker(idx: int, seg: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal completed_count
            seg_copy = dict(seg)
            text = seg.get("text", "").strip()
            if not text:
                seg_copy["audio_path"] = None
                seg_copy["tts_duration"] = 0.0
                return seg_copy

            out_filename = f"segment_{idx:05d}.mp3"
            out_filepath = os.path.join(output_dir, out_filename)

            seg_voice = seg.get("voice") or self.voice

            async with semaphore:
                dur = await self._synthesize_single(text, out_filepath, voice=seg_voice)

            seg_copy["voice"] = seg_voice
            seg_copy["audio_path"] = out_filepath
            seg_copy["tts_duration"] = round(dur, 3)

            completed_count += 1
            if progress_callback:
                progress_callback({
                    "completed": completed_count,
                    "total": total_segments,
                    "percent": (completed_count / total_segments) * 100.0 if total_segments else 0.0,
                    "segment": seg_copy,
                })

            return seg_copy

        tasks = [worker(i, seg) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        return list(results)

    def synthesize(
        self,
        segments: List[Dict[str, Any]],
        output_dir: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for batch synthesis, safe within existing event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self._synthesize_batch(segments, output_dir, progress_callback)
                ).result()
        else:
            return asyncio.run(
                self._synthesize_batch(segments, output_dir, progress_callback)
            )

    @staticmethod
    def get_available_voices() -> Dict[str, Dict[str, str]]:
        """Return list of supported natural neural voices."""
        return AVAILABLE_VOICES
