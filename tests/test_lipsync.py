"""
Unit tests for AI Lip Synchronization Engine (Phase 4).
"""

import os
import tempfile
import cv2
import numpy as np
import pytest
from scipy.io import wavfile

from dubber.lipsync import AILipSynchronizer


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


def create_test_audio_wav(filepath: str, duration_sec: float = 1.0, sr: int = 16000):
    """Create a 16kHz WAV file with harmonic speech-like tones."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)
    data = (signal * 32767).astype(np.int16)
    wavfile.write(filepath, sr, data)


def create_test_video_mp4(filepath: str, num_frames: int = 15, fps: float = 25.0, w: int = 160, h: int = 120):
    """Create a small MP4 video with a drawn synthetic face and mouth."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

    for i in range(num_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 200  # Gray background
        # Draw face oval
        cv2.ellipse(frame, (w // 2, h // 2), (30, 40), 0, 0, 360, (220, 180, 150), -1)
        # Draw eyes
        cv2.circle(frame, (w // 2 - 12, h // 2 - 10), 4, (50, 30, 20), -1)
        cv2.circle(frame, (w // 2 + 12, h // 2 - 10), 4, (50, 30, 20), -1)
        # Draw mouth
        cv2.ellipse(frame, (w // 2, h // 2 + 18), (10, 4), 0, 0, 360, (80, 50, 180), -1)
        writer.write(frame)

    writer.release()


def test_mel_filterbank_generation():
    sync = AILipSynchronizer()
    fb = sync._build_mel_filterbank(sr=16000, n_fft=800, n_mels=80)
    assert fb.shape == (80, 401)
    assert np.all(fb >= 0)
    assert np.max(fb) > 0.5


def test_compute_mel_spectrogram(temp_dir):
    wav_path = os.path.join(temp_dir, "test.wav")
    create_test_audio_wav(wav_path, duration_sec=1.0)

    sync = AILipSynchronizer()
    mel_spec, dur = sync.compute_mel_spectrogram(wav_path)

    assert mel_spec.shape[0] == 80  # 80 mel channels
    assert mel_spec.shape[1] > 10
    assert 0.9 <= dur <= 1.1


def test_mel_slice_windowing(temp_dir):
    wav_path = os.path.join(temp_dir, "test.wav")
    create_test_audio_wav(wav_path, duration_sec=1.5)

    sync = AILipSynchronizer()
    mel_spec, dur = sync.compute_mel_spectrogram(wav_path)

    # 16-frame window (0.2s duration)
    window = sync._get_mel_slice_for_time(mel_spec, time_sec=0.5, window_frames=16)
    assert window.shape == (80, 16)


def test_smooth_box_filter():
    sync = AILipSynchronizer()
    prev = (100, 100, 50, 50)
    curr = (120, 120, 60, 60)

    # With alpha=0.70: 0.70*100 + 0.30*120 = 106
    smoothed = sync._smooth_box(prev, curr, alpha=0.70)
    assert smoothed == (106, 106, 53, 53)

    # If prev is None, returns curr directly
    assert sync._smooth_box(None, curr) == curr


def test_feathered_blending_preserves_bounds():
    sync = AILipSynchronizer()
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
    mouth_crop = np.ones((30, 40, 3), dtype=np.uint8) * 200

    blended = sync._blend_feathered(frame, mouth_crop, mx=20, my=40, mw=40, mh=30)
    assert blended.shape == (100, 100, 3)
    assert blended.dtype == np.uint8
    assert np.all(blended >= 0) and np.all(blended <= 255)


def test_speech_selective_timing():
    sync = AILipSynchronizer()
    segments = [
        {"start": 1.0, "end": 2.5},
        {"start": 5.0, "end": 7.0},
    ]

    assert sync._is_time_in_speech(1.5, segments) is True
    assert sync._is_time_in_speech(2.0, segments) is True
    assert sync._is_time_in_speech(0.2, segments) is False
    assert sync._is_time_in_speech(3.5, segments) is False
    assert sync._is_time_in_speech(6.0, segments) is True
    assert sync._is_time_in_speech(9.0, segments) is False


def test_synchronize_video_lips_e2e(temp_dir):
    video_path = os.path.join(temp_dir, "test_vid.mp4")
    audio_path = os.path.join(temp_dir, "test_aud.wav")
    out_path = os.path.join(temp_dir, "synced_vid.mp4")

    create_test_video_mp4(video_path, num_frames=15, fps=25.0)
    create_test_audio_wav(audio_path, duration_sec=0.6)

    segments = [{"start": 0.1, "end": 0.5, "text": "Testing lip synchronization"}]

    sync = AILipSynchronizer()
    stats = sync.synchronize_video_lips(video_path, audio_path, segments, out_path)

    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0
    assert stats["total_frames"] == 15
    assert "synced_frames" in stats
    assert "skipped_frames" in stats
