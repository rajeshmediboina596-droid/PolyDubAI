"""
Unit tests for FastAPI Web Server & REST/SSE Endpoints.
"""

import os
import pytest
from starlette.testclient import TestClient

from server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint_serves_html(client):
    """Test that GET / returns the Dubber Studio HTML single-page app."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Dubber Studio" in response.text
    assert "id=\"videoUrlInput\"" in response.text


def test_api_voices_endpoint(client):
    """Test that GET /api/voices returns catalog of male and female voices."""
    response = client.get("/api/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    voices = data["voices"]
    assert len(voices) > 0

    # Ensure both male and female voices exist
    genders = {v["gender"] for v in voices}
    assert "Male" in genders
    assert "Female" in genders

    # Verify key voice IDs
    voice_ids = {v["id"] for v in voices}
    assert "en-US-ChristopherNeural" in voice_ids
    assert "en-US-JennyNeural" in voice_ids


def test_api_videos_endpoint(client):
    """Test that GET /api/videos lists mp4 files from output directory."""
    response = client.get("/api/videos")
    assert response.status_code == 200
    data = response.json()
    assert "videos" in data
    videos = data["videos"]
    assert isinstance(videos, list)

    # Check structure of each video entry
    for vid in videos:
        assert "filename" in vid
        assert "url" in vid
        assert "size_mb" in vid
        assert "created_at" in vid
        assert vid["url"].startswith("/output/")


def test_api_dub_validation(client):
    """Test validation errors for empty URL."""
    response = client.post("/api/dub", json={"url": "   "})
    assert response.status_code == 400


def test_stream_nonexistent_job(client):
    """Test that streaming a non-existent job returns 404."""
    response = client.get("/api/dub/stream/invalid-job-id")
    assert response.status_code == 404


def test_api_redub_endpoint(client):
    """Test rapid 1-click re-dub endpoint with mock video."""
    # Test validation when video doesn't exist
    resp = client.post("/api/redub", json={
        "video_filename": "non_existent_video.mp4",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello", "gender": "male"}],
    })
    assert resp.status_code in (404, 400)


def test_delete_video_endpoint_success_and_cleanup(client):
    """Test DELETE /api/videos/{filename} deletes video and associated subtitles."""
    from server import OUTPUT_DIR
    dummy_mp4 = os.path.join(OUTPUT_DIR, "dummy_unit_test_dubbed_en.mp4")
    dummy_srt = os.path.join(OUTPUT_DIR, "dummy_unit_test_subtitles_en.srt")
    dummy_vtt = os.path.join(OUTPUT_DIR, "dummy_unit_test_subtitles_en.vtt")

    # Create dummy files
    with open(dummy_mp4, "wb") as f:
        f.write(b"fake mp4 content")
    with open(dummy_srt, "w", encoding="utf-8") as f:
        f.write("1\n00:00:00,000 --> 00:00:01,000\nHello")
    with open(dummy_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHello")

    assert os.path.exists(dummy_mp4)
    assert os.path.exists(dummy_srt)

    resp = client.delete("/api/videos/dummy_unit_test_dubbed_en.mp4")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "dummy_unit_test_dubbed_en.mp4" in data["deleted_files"]
    assert "dummy_unit_test_subtitles_en.srt" in data["deleted_files"]

    # Verify deleted on filesystem
    assert not os.path.exists(dummy_mp4)
    assert not os.path.exists(dummy_srt)
    assert not os.path.exists(dummy_vtt)


def test_delete_video_not_found(client):
    """Test DELETE /api/videos/{filename} returns 404 for non-existent video."""
    resp = client.delete("/api/videos/completely_non_existent_video_123.mp4")
    assert resp.status_code == 404


def test_delete_video_invalid_format(client):
    """Test DELETE /api/videos/{filename} rejects non-mp4 files."""
    resp = client.delete("/api/videos/critical_code.py")
    assert resp.status_code == 400


