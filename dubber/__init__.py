"""
Automated Video Dubbing System
A complete pipeline to translate and dub foreign videos into natural English with synchronized audio.
"""

import os
from .config import DubbingConfig, AVAILABLE_VOICES, find_ffmpeg_executable

# Ensure FFmpeg directory is in PATH for pydub and subprocesses
try:
    _ffmpeg_exe = find_ffmpeg_executable()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
    if _ffmpeg_dir and _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    from pydub import AudioSegment
    AudioSegment.converter = _ffmpeg_exe
except Exception:
    pass

from .pipeline import DubbingPipeline

__all__ = ["DubbingConfig", "DubbingPipeline", "AVAILABLE_VOICES"]
