"""
Unit tests for Computer Vision Face Detection & Multimodal Gender Classification.
"""

import os
import numpy as np
import pytest
from PIL import Image

from dubber.visual_classifier import VisualGenderClassifier
from dubber.classifier import SpeakerGenderClassifier


def test_visual_classifier_initialization():
    """Test that VisualGenderClassifier initializes properly."""
    classifier = VisualGenderClassifier()
    assert classifier.model_name == "rizvandwiki/gender-classification-2"
    assert classifier._classifier_pipeline is None


def test_face_cascade_loads():
    """Test that OpenCV Haar cascade face detector loads successfully."""
    classifier = VisualGenderClassifier()
    classifier._ensure_face_detector()
    assert classifier._face_cascade is not None
    assert not classifier._face_cascade.empty()


def test_classify_blank_image():
    """Test that classify_image handles images without crashes."""
    classifier = VisualGenderClassifier()
    # Create temporary blank RGB image
    os.makedirs("output/temp", exist_ok=True)
    temp_img_path = "output/temp/test_blank.jpg"
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    img.save(temp_img_path)

    res = classifier.classify_image(temp_img_path)
    assert "face_detected" in res
    assert "gender" in res
    assert res["gender"] in ("male", "female")
    assert "confidence" in res


def test_multimodal_classifier_fallback_to_audio():
    """Test that SpeakerGenderClassifier falls back to audio F0 when no video_path is provided."""
    classifier = SpeakerGenderClassifier()
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Testing multimodal audio fallback"}
    ]
    # Synthetic male audio (120 Hz)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 120.0 * t) * 16000).astype(np.int16)
    import wave
    wav_path = "output/temp/test_multimodal_male.wav"
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())

    res = classifier.classify_segments(
        segments=segments,
        source_audio_path=wav_path,
        video_path=None,  # No video provided
        male_voice="en-US-ChristopherNeural",
        female_voice="en-US-JennyNeural",
    )

    assert len(res) == 1
    assert res[0]["gender"] == "male"
    assert res[0]["voice"] == "en-US-ChristopherNeural"
    assert res[0]["detection_source"] == "audio"
