"""
Unit tests for Person-Matching Speaker Consistency and speaker_mode overrides.
Ensures audio matches the on-screen person consistently without random flipping.
"""

import os
import tempfile
import wave
import numpy as np
import pytest

from dubber.classifier import SpeakerGenderClassifier
from tests.test_classifier import generate_harmonic_tone


def test_person_matching_male_only_override():
    """Verify that speaker_mode='male_only' locks all segments to male voice."""
    classifier = SpeakerGenderClassifier()
    segments = [
        {"id": 1, "start": 0.0, "end": 1.0, "text": "Sentence 1"},
        {"id": 2, "start": 1.5, "end": 2.5, "text": "Sentence 2"},
    ]
    res = classifier.classify_segments(
        segments=segments,
        source_audio_path="",
        male_voice="en-US-ChristopherNeural",
        female_voice="en-US-JennyNeural",
        speaker_mode="male_only",
    )
    for s in res:
        assert s["gender"] == "male"
        assert s["voice"] == "en-US-ChristopherNeural"
        assert s["detection_source"] == "user_override_male"


def test_person_matching_female_only_override():
    """Verify that speaker_mode='female_only' locks all segments to female voice."""
    classifier = SpeakerGenderClassifier()
    segments = [
        {"id": 1, "start": 0.0, "end": 1.0, "text": "Sentence 1"},
        {"id": 2, "start": 1.5, "end": 2.5, "text": "Sentence 2"},
    ]
    res = classifier.classify_segments(
        segments=segments,
        source_audio_path="",
        male_voice="en-US-ChristopherNeural",
        female_voice="en-US-JennyNeural",
        speaker_mode="female_only",
    )
    for s in res:
        assert s["gender"] == "female"
        assert s["voice"] == "en-US-JennyNeural"
        assert s["detection_source"] == "user_override_female"


def test_person_matching_single_speaker_lock():
    """Verify that a single-speaker male video locks cutaways/b-roll to the speaker's voice."""
    classifier = SpeakerGenderClassifier()
    sr = 16000
    male_audio = generate_harmonic_tone(f0=120.0, duration=3.0, sr=sr)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "male_speech.wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(male_audio.tobytes())

        segments = [
            {"id": 1, "start": 0.1, "end": 0.9, "text": "Intro talking to camera"},
            {"id": 2, "start": 1.1, "end": 1.9, "text": "B-roll cutaway of workout bench"},
            {"id": 3, "start": 2.1, "end": 2.9, "text": "Conclusion talking to camera"},
        ]

        res = classifier.classify_segments(
            segments=segments,
            source_audio_path=wav_path,
            video_path=None,
            male_voice="en-US-ChristopherNeural",
            female_voice="en-US-JennyNeural",
            speaker_mode="auto",
        )

        assert len(res) == 3
        # All 3 segments MUST be locked to male voice without flipping!
        for s in res:
            assert s["gender"] == "male"
            assert s["voice"] == "en-US-ChristopherNeural"
