"""
Video remuxer using FFmpeg.
Replaces the audio track losslessly without re-encoding video streams,
optionally embedding English subtitles.
"""

import os
import subprocess
from typing import Any, Dict, Optional

from .config import find_ffmpeg_executable


class VideoRemuxer:
    """Remuxes final dubbed audio into the original video stream losslessly."""

    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_path = ffmpeg_path or find_ffmpeg_executable()

    def remux(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        subtitle_path: Optional[str] = None,
        burn_subtitles: bool = False,
        original_audio_path: Optional[str] = None,
        include_original_track: bool = False,
    ) -> Dict[str, Any]:
        """
        Merge video stream from video_path with audio_path.
        Supports:
          - Soft embedded subtitles (mov_text)
          - Burn-in hardsubs rendered directly onto video frames with stylish font & outline
          - Multi-audio MP4 (Track 1: Dubbed Audio, Track 2: Original Audio)
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Dubbed audio file not found: {audio_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Case 1: Hardsub burn-in (re-encodes video with subtitle filter)
        if burn_subtitles and subtitle_path and os.path.isfile(subtitle_path):
            # Format subtitle path for FFmpeg subtitles filter on Windows
            sub_escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")
            # Stylish subtitle style: Yellow/white text with black border
            sub_filter = f"subtitles='{sub_escaped}':force_style='FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=1,MarginV=25'"

            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", video_path,
                "-i", audio_path,
                "-vf", sub_filter,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", "192k",
                output_path,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                return {"output_path": output_path, "file_size_mb": round(file_size_mb, 2), "burned_subtitles": True}

        # Case 2: Soft subtitles / lossless remux (c:v copy)
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", video_path,
            "-i", audio_path,
        ]

        has_second_audio = include_original_track and original_audio_path and os.path.isfile(original_audio_path)
        if has_second_audio:
            cmd.extend(["-i", original_audio_path])

        if subtitle_path and os.path.isfile(subtitle_path):
            cmd.extend(["-i", subtitle_path])
            sub_idx = 3 if has_second_audio else 2
            maps = [
                "-map", "0:v:0",
                "-map", "1:a:0",
            ]
            if has_second_audio:
                maps.extend(["-map", "2:a:0"])
            maps.extend([
                "-map", f"{sub_idx}:s:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-c:s", "mov_text",
                "-metadata:s:a:0", "title=Dubbed Audio",
                "-metadata:s:a:0", "language=eng",
            ])
            if has_second_audio:
                maps.extend([
                    "-metadata:s:a:1", "title=Original Audio",
                    "-metadata:s:a:1", "language=und",
                ])
            maps.extend([
                "-metadata:s:s:0", "language=eng",
                "-metadata:s:s:0", "title=Dubbed Subtitles",
            ])
            cmd.extend(maps)
        else:
            maps = [
                "-map", "0:v:0",
                "-map", "1:a:0",
            ]
            if has_second_audio:
                maps.extend(["-map", "2:a:0"])
            maps.extend([
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-metadata:s:a:0", "title=Dubbed Audio",
            ])
            if has_second_audio:
                maps.extend(["-metadata:s:a:1", "title=Original Audio"])
            cmd.extend(maps)

        cmd.append(output_path)

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg remuxing failed: {result.stderr}")

        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Remuxed output video file was not created properly: {output_path}")

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        return {
            "output_path": output_path,
            "file_size_mb": round(file_size_mb, 2),
            "burned_subtitles": False,
        }
