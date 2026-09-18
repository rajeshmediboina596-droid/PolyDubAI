"""
Tests for configuration and environment discovery.
"""

import os
import pytest
from dubber.config import DubbingConfig, AVAILABLE_VOICES, find_ffmpeg_executable


def test_find_ffmpeg():
    """Verify that FFmpeg executable is discovered and exists on disk."""
    exe = find_ffmpeg_executable()
    assert exe is not None
    assert os.path.isfile(exe)


def test_dubbing_config_defaults():
    """Verify default configuration parameters."""
    cfg = DubbingConfig()
    assert cfg.whisper_model in ("base", "small")
    assert cfg.target_lang == "en"
    assert cfg.voice == "en-US-ChristopherNeural"
    assert cfg.max_stretch_factor in (1.30, 2.20)
    assert cfg.min_stretch_factor == 0.85
    assert cfg.output_dir == "output"
    assert os.path.isdir(cfg.output_dir)
    assert os.path.isdir(cfg.temp_dir)


def test_voice_catalog():
    """Verify neural voice catalogue contains key voices."""
    assert "en-US-ChristopherNeural" in AVAILABLE_VOICES
    assert "en-US-JennyNeural" in AVAILABLE_VOICES
    assert "en-IN-PrabhatNeural" in AVAILABLE_VOICES
    assert AVAILABLE_VOICES["en-US-ChristopherNeural"]["gender"] == "Male"
