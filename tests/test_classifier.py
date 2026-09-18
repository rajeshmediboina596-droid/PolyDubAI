"""
Unit tests for SpeakerGenderClassifier (pitch detection & gender voice assignment).
"""

import os
import tempfile
import wave
import numpy as np
import pytest

from dubber.classifier import SpeakerGenderClassifier


def generate_harmonic_tone(f0: float, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Synthesize a voiced harmonic speech-like signal with fundamental frequency f0."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + first 2 harmonics
    signal = 0.6 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    # Convert to 16-bit PCM scale
    signal = (signal * 20000).astype(np.int16)
    return signal


def test_pitch_detection_male():
    """Verify that a 115 Hz fundamental frequency tone is classified as male."""
    classifier = SpeakerGenderClassifier()
    samples = generate_harmonic_tone(f0=115.0, duration=1.0)
    gender, f0_hz, conf = classifier.analyze_audio_samples(samples, sample_rate=16000)

    assert gender == "male"
    assert abs(f0_hz - 115.0) < 5.0
    assert conf > 0.60


def test_pitch_detection_female():
    """Verify that a 210 Hz fundamental frequency tone is classified as female."""
    classifier = SpeakerGenderClassifier()
    samples = generate_harmonic_tone(f0=210.0, duration=1.0)
    gender, f0_hz, conf = classifier.analyze_audio_samples(samples, sample_rate=16000)

    assert gender == "female"
    assert abs(f0_hz - 210.0) < 5.0
    assert conf > 0.60


def test_classify_segments_dialogue():
    """Verify multi-speaker dialogue segment classification with male and female voices."""
    classifier = SpeakerGenderClassifier()
    sr = 16000

    # Create a 4-second WAV: 0-2s male (110Hz), 2-4s female (220Hz)
    part1_male = generate_harmonic_tone(f0=110.0, duration=2.0, sr=sr)
    part2_female = generate_harmonic_tone(f0=220.0, duration=2.0, sr=sr)
    combined = np.concatenate([part1_male, part2_female])

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "dialogue.wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(combined.tobytes())

        segments = [
            {"id": 1, "start": 0.2, "end": 1.8, "text": "Hello, I am the first speaker."},
            {"id": 2, "start": 2.2, "end": 3.8, "text": "Hi there, I am the second speaker replying."},
        ]

        classified = classifier.classify_segments(
            segments=segments,
            source_audio_path=wav_path,
            male_voice="en-US-ChristopherNeural",
            female_voice="en-US-JennyNeural",
        )

        assert len(classified) == 2
        # Segment 1 should be male -> Christopher
        assert classified[0]["gender"] == "male"
        assert classified[0]["voice"] == "en-US-ChristopherNeural"

        # Segment 2 should be female -> Jenny
        assert classified[1]["gender"] == "female"
        assert classified[1]["voice"] == "en-US-JennyNeural"
