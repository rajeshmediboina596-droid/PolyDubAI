"""
Tests for transcription, timestamp formatting, and SRT generation.
"""

import os
import tempfile
from dubber.transcriber import format_timestamp, SpeechTranscriber


def test_format_timestamp():
    """Verify SRT and VTT timestamp conversions."""
    # 65.5 seconds -> 00:01:05,500
    srt_ts = format_timestamp(65.5, srt_format=True)
    assert srt_ts == "00:01:05,500"

    vtt_ts = format_timestamp(65.5, srt_format=False)
    assert vtt_ts == "00:01:05.500"

    # 3661.123 -> 01:01:01,123
    long_ts = format_timestamp(3661.123, srt_format=True)
    assert long_ts == "01:01:01,123"


def test_export_srt():
    """Verify proper subtitle file generation."""
    segments = [
        {"id": 1, "start": 0.0, "end": 2.5, "text": "Hello world"},
        {"id": 2, "start": 3.0, "end": 6.2, "text": "This is dubbed speech"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
        tmp_srt = f.name

    try:
        SpeechTranscriber.export_srt(segments, tmp_srt)
        assert os.path.isfile(tmp_srt)
        with open(tmp_srt, "r", encoding="utf-8") as f:
            content = f.read()
        assert "1\n00:00:00,000 --> 00:00:02,500\nHello world" in content
        assert "2\n00:00:03,000 --> 00:00:06,200\nThis is dubbed speech" in content
    finally:
        if os.path.isfile(tmp_srt):
            os.remove(tmp_srt)
