"""
End-to-end pipeline test with generated audiovisual media.
"""

import asyncio
import os
import subprocess
import tempfile
from unittest.mock import patch
import edge_tts
import pytest

from dubber.config import DubbingConfig, find_ffmpeg_executable
from dubber.pipeline import DubbingPipeline


def test_end_to_end_dubbing_local_video():
    """
    Generate a short foreign video (French), run it through DubbingPipeline,
    and verify the resulting dubbed English video.
    """
    ffmpeg_exe = find_ffmpeg_executable()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Synthesize sample foreign speech
        french_audio = os.path.join(tmpdir, "french_speech.mp3")
        from dubber.synthesizer import SpeechSynthesizer
        french_text = "Bonjour tout le monde."
        synth = SpeechSynthesizer(voice="fr-FR-HenriNeural")
        asyncio.run(synth._synthesize_single(french_text, french_audio))
        assert os.path.isfile(french_audio)

        # Step 2: Create MP4 video with the French audio
        source_video = os.path.join(tmpdir, "source_french_video.mp4")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f", "lavfi",
            "-i", "color=c=navy:s=320x240:d=4:r=25",
            "-i", french_audio,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            source_video,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode == 0
        assert os.path.isfile(source_video)

        # Step 3: Run dubbing pipeline
        cfg = DubbingConfig(
            output_dir=os.path.join(tmpdir, "output"),
            temp_dir=os.path.join(tmpdir, "temp"),
            voice="en-US-ChristopherNeural",
            keep_temp=True,
        )
        pipeline = DubbingPipeline(cfg)

        mock_transcription = {
            "language": "fr",
            "language_probability": 0.98,
            "duration": 4.0,
            "segments": [
                {
                    "id": 1,
                    "start": 0.5,
                    "end": 2.8,
                    "duration": 2.3,
                    "text": "Hello everyone, welcome.",
                }
            ],
        }

        with patch.object(pipeline.transcriber, "transcribe_and_translate", return_value=mock_transcription):
            metrics = pipeline.run(source_video, output_filename="french_dubbed_en.mp4")

        # Step 4: Verify metrics & artifacts
        assert metrics["language"] == "fr"
        assert metrics["segments_count"] == 1
        assert os.path.isfile(metrics["output_video"])
        assert metrics["file_size_mb"] > 0
        assert os.path.isfile(metrics["subtitles_file"])
