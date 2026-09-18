"""
YouTube video and audio downloader using yt-dlp.
"""

import os
import subprocess
from typing import Any, Callable, Dict, Optional
import yt_dlp

from .config import find_ffmpeg_executable


class YouTubeDownloader:
    """Downloads YouTube videos and extracts clean audio for transcription."""

    def __init__(self, temp_dir: str = "output/temp", ffmpeg_path: Optional[str] = None):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.ffmpeg_path = ffmpeg_path or find_ffmpeg_executable()

    def get_info(self, url: str) -> Dict[str, Any]:
        """Fetch video metadata without downloading."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "id": info.get("id"),
                "title": info.get("title", "Unknown Title"),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", "Unknown"),
                "view_count": info.get("view_count", 0),
                "thumbnail": info.get("thumbnail"),
                "url": url,
            }

    def download(
        self,
        url: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Download the video in best compatible MP4 format and extract clean 16kHz WAV audio.
        Returns paths to both video_path and audio_path along with metadata.
        """
        info = self.get_info(url)
        video_id = info.get("id", "video")
        video_template = os.path.join(self.temp_dir, f"{video_id}_source.%(ext)s")

        def hook(d: Dict[str, Any]):
            if progress_callback and d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
                downloaded = d.get("downloaded_bytes", 0)
                speed = d.get("speed", 0)
                eta = d.get("eta", 0)
                progress_callback({
                    "status": "downloading",
                    "percent": (downloaded / total) * 100 if total else 0,
                    "downloaded": downloaded,
                    "total": total,
                    "speed": speed,
                    "eta": eta,
                })
            elif progress_callback and d.get("status") == "finished":
                progress_callback({"status": "finished"})

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": video_template,
            "progress_hooks": [hook],
            "quiet": True,
            "no_warnings": True,
            "ffmpeg_location": os.path.dirname(self.ffmpeg_path) if self.ffmpeg_path else None,
            "merge_output_format": "mp4",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        final_video_path = os.path.join(self.temp_dir, f"{video_id}_source.mp4")
        if not os.path.isfile(final_video_path):
            # Check if saved with another extension
            for f in os.listdir(self.temp_dir):
                if f.startswith(f"{video_id}_source."):
                    final_video_path = os.path.join(self.temp_dir, f)
                    break

        if not os.path.isfile(final_video_path):
            raise FileNotFoundError(f"Downloaded video file not found in {self.temp_dir}")

        # Extract 16kHz mono WAV for Whisper
        audio_path = os.path.join(self.temp_dir, f"{video_id}_source_audio.wav")
        self._extract_audio(final_video_path, audio_path)

        return {
            "metadata": info,
            "video_path": final_video_path,
            "audio_path": audio_path,
        }

    def _extract_audio(self, video_path: str, audio_path: str) -> None:
        """Extract a 16kHz mono WAV audio track using FFmpeg."""
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract audio with FFmpeg: {result.stderr}")
