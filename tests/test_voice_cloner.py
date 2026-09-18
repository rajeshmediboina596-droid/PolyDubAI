"""
Unit tests for Speaker Voice Preservation & Conversion Engine (Phase 3).
"""

import os
import tempfile
import numpy as np
import pytest
from scipy.io import wavfile

from dubber.voice_cloner import SpeakerVoicePreserver, AcousticProfile


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


def create_synthetic_voice_wav(filepath: str, freq_hz: float = 130.0, duration_sec: float = 1.0, sr: int = 16000):
    """Generate a clean synthetic voiced harmonic signal with fundamental frequency freq_hz."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental + 2nd & 3rd harmonics for realistic voice spectrum
    signal = (
        0.6 * np.sin(2 * np.pi * freq_hz * t)
        + 0.3 * np.sin(2 * np.pi * (2 * freq_hz) * t)
        + 0.1 * np.sin(2 * np.pi * (3 * freq_hz) * t)
    )
    # Scale to 16-bit PCM
    audio_int16 = (signal * 32767 * 0.8).astype(np.int16)
    wavfile.write(filepath, sr, audio_int16)


def test_f0_pitch_estimation_male(temp_dir):
    wav_path = os.path.join(temp_dir, "male_test.wav")
    create_synthetic_voice_wav(wav_path, freq_hz=130.0, duration_sec=1.0)

    preserver = SpeakerVoicePreserver(sample_rate=16000)
    profile = preserver.analyze_speaker_profile(wav_path)

    # 130 Hz should be estimated within 5% error
    assert 120.0 <= profile.median_f0 <= 140.0
    assert profile.gender_hint == "male"
    assert profile.energy_rms > 0.01


def test_f0_pitch_estimation_female(temp_dir):
    wav_path = os.path.join(temp_dir, "female_test.wav")
    create_synthetic_voice_wav(wav_path, freq_hz=235.0, duration_sec=1.0)

    preserver = SpeakerVoicePreserver(sample_rate=16000)
    profile = preserver.analyze_speaker_profile(wav_path)

    # 235 Hz should be estimated within 5% error
    assert 220.0 <= profile.median_f0 <= 250.0
    assert profile.gender_hint == "female"


def test_compute_prosody_deltas():
    preserver = SpeakerVoicePreserver()
    profile = AcousticProfile(
        median_f0=145.0,  # +25 Hz above 120 Hz male baseline
        mean_f0=145.0,
        min_f0=120.0,
        max_f0=170.0,
        pitch_std=12.0,
        speaking_rate=4.2,  # Faster than 3.5 baseline
        energy_rms=0.08,
        spectral_centroid=1800.0,
        gender_hint="male",
    )

    deltas = preserver.compute_prosody_deltas(profile, base_gender="male")
    assert deltas["pitch"] == "+25Hz"
    assert deltas["rate"].startswith("+")  # Faster rate
    assert "Hz" in deltas["pitch"]


def test_preserve_voices_for_segments(temp_dir):
    wav_path = os.path.join(temp_dir, "speech_sample.wav")
    create_synthetic_voice_wav(wav_path, freq_hz=140.0, duration_sec=2.0)

    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello world", "gender": "male", "voice": "en-US-ChristopherNeural"},
        {"start": 1.0, "end": 2.0, "text": "Testing voice preservation", "gender": "male", "voice": "en-US-ChristopherNeural"},
    ]

    preserver = SpeakerVoicePreserver()
    enhanced_segments = preserver.preserve_voices_for_segments(segments, wav_path)

    assert len(enhanced_segments) == 2
    for seg in enhanced_segments:
        assert "pitch" in seg
        assert "rate" in seg
        assert "voice_profile" in seg
        assert seg["voice_preservation_mode"] == "adaptive_prosody"
        assert seg["voice_profile"]["median_f0"] > 0


def test_apply_formant_and_timbre_transfer(temp_dir):
    in_wav = os.path.join(temp_dir, "input.wav")
    out_wav = os.path.join(temp_dir, "timbre_out.wav")
    create_synthetic_voice_wav(in_wav, freq_hz=150.0, duration_sec=0.5)

    preserver = SpeakerVoicePreserver()
    profile = preserver.analyze_speaker_profile(in_wav)

    res_path = preserver.apply_formant_and_timbre_transfer(in_wav, profile, out_wav)
    assert os.path.isfile(res_path)
    assert os.path.getsize(res_path) > 0
