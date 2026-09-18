"""
Tests for neural speech synthesis with edge-tts.
"""

import os
import tempfile
import pytest
from dubber.synthesizer import SpeechSynthesizer


def test_synthesizer_single_speech():
    """Verify that SpeechSynthesizer generates an audio clip with edge-tts."""
    synth = SpeechSynthesizer(voice="en-US-ChristopherNeural")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_mp3 = os.path.join(tmpdir, "test.mp3")
        dur = synth.synthesize(
            segments=[{"id": 1, "text": "Testing neural speech synthesis."}],
            output_dir=tmpdir,
        )
        assert len(dur) == 1
        assert dur[0]["audio_path"] is not None
        assert os.path.isfile(dur[0]["audio_path"])
        assert dur[0]["tts_duration"] > 0.5
