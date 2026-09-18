"""
Tests for Subtitle Studio, Language Selection, and Script Pacing Optimizer.
"""

import os
import tempfile
import pytest
from starlette.testclient import TestClient
from server import app
from dubber.transcriber import SpeechTranscriber
from dubber.config import DubbingConfig


client = TestClient(app)


def test_optimize_pacing_logic():
    """Verify that optimize_pacing shortens wordy sentences with filler words when duration is tight."""
    text_with_fillers = "Um actually you know we basically have to do all this work today"
    # Target duration is 1.0 second (very fast)
    opt = SpeechTranscriber.optimize_pacing(text_with_fillers, target_duration=1.0, target_lang="en")
    assert "um" not in opt.lower()
    assert "actually" not in opt.lower()
    assert len(opt.split()) < len(text_with_fillers.split())

    # When duration is comfortable (e.g. 5.0s), natural text is preserved
    natural_text = "Good morning everyone"
    opt_natural = SpeechTranscriber.optimize_pacing(natural_text, target_duration=5.0, target_lang="en")
    assert opt_natural == natural_text


def test_export_bilingual_srt_and_vtt():
    """Verify bilingual stacked subtitle export and WebVTT formatting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        segments = [
            {
                "id": 1,
                "start": 1.5,
                "end": 3.8,
                "text": "Hello world, welcome to our presentation.",
                "orig_text": "Bonjour le monde, bienvenue à notre présentation.",
            },
            {
                "id": 2,
                "start": 4.0,
                "end": 6.2,
                "text": "Thank you for watching.",
                "orig_text": "Merci d'avoir regardé.",
            },
        ]

        srt_path = os.path.join(tmpdir, "bilingual.srt")
        vtt_path = os.path.join(tmpdir, "output.vtt")

        SpeechTranscriber.export_bilingual_srt(segments, srt_path)
        SpeechTranscriber.export_vtt(segments, vtt_path)

        assert os.path.isfile(srt_path)
        with open(srt_path, "r", encoding="utf-8") as f:
            content_srt = f.read()
            assert "Bonjour le monde" in content_srt
            assert "Hello world" in content_srt

        assert os.path.isfile(vtt_path)
        with open(vtt_path, "r", encoding="utf-8") as f:
            content_vtt = f.read()
            assert "WEBVTT" in content_vtt
            assert "00:00:01.500 --> 00:00:03.800" in content_vtt


def test_api_optimize_pacing_endpoint():
    """Verify POST /api/optimize_pacing endpoint."""
    res = client.post("/api/optimize_pacing", json={
        "text": "Um like basically we should go there right now",
        "target_duration": 1.2,
        "target_lang": "en"
    })
    assert res.status_code == 200
    data = res.json()
    assert "optimized_text" in data
    assert "original_text" in data


def test_dub_request_with_language_and_subtitles():
    """Verify validation of DubRequest with source_lang, subtitle_mode, and lip_sync_mode."""
    res = client.post("/api/dub", json={
        "url": "https://www.youtube.com/watch?v=sample",
        "source_lang": "te",
        "target_lang": "en",
        "subtitle_mode": "dual",
        "lip_sync_mode": "visual_adaptive",
    })
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data
    assert data["status"] == "started"
