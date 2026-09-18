"""
Tests for audio synchronization, atempo filters, and timeline generation.
"""

import os
import tempfile
import pytest
from pydub import AudioSegment
from dubber.synchronizer import AudioSynchronizer


def test_build_atempo_filter():
    """Verify atempo filter string generation for various speed factors."""
    sync = AudioSynchronizer()

    # Factor between 0.5 and 2.0
    f1 = sync._build_atempo_filter(1.25)
    assert f1 == "atempo=1.2500"

    # Factor > 2.0 requires chaining
    f2 = sync._build_atempo_filter(2.5)
    assert "atempo=2.0" in f2
    assert "atempo=1.2500" in f2


def test_time_stretch_audio():
    """Verify that stretching an audio segment adjusts its duration correctly."""
    sync = AudioSynchronizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a 2-second silent tone audio
        in_path = os.path.join(tmpdir, "in.wav")
        out_path = os.path.join(tmpdir, "out.wav")
        tone = AudioSegment.silent(duration=2000, frame_rate=44100)
        tone.export(in_path, format="wav")

        # Speed up by 1.25x -> duration should become ~1.6s
        stretched_dur = sync.time_stretch_audio(in_path, out_path, speed_ratio=1.25)
        assert os.path.isfile(out_path)
        assert 1.4 < stretched_dur < 1.8


def test_synchronize_timeline():
    """Verify timeline assembly stitches segments and pads with silence to target duration."""
    sync = AudioSynchronizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test audio segment (1 second)
        seg_audio = os.path.join(tmpdir, "seg.wav")
        AudioSegment.silent(duration=1000, frame_rate=44100).export(seg_audio, format="wav")

        segments = [
            {
                "id": 1,
                "start": 1.0,
                "end": 2.0,
                "duration": 1.0,
                "text": "First segment",
                "audio_path": seg_audio,
                "tts_duration": 1.0,
            },
            {
                "id": 2,
                "start": 3.0,
                "end": 4.0,
                "duration": 1.0,
                "text": "Second segment",
                "audio_path": seg_audio,
                "tts_duration": 1.0,
            },
        ]

        out_audio = os.path.join(tmpdir, "full_timeline.wav")
        sync_temp = os.path.join(tmpdir, "sync_temp")

        res = sync.synchronize(
            segments=segments,
            total_duration=5.0,
            output_path=out_audio,
            temp_dir=sync_temp,
        )

        assert os.path.isfile(out_audio)
        result_audio = AudioSegment.from_file(out_audio)
        # Verify timeline duration matches ~5.0 seconds
        assert abs(len(result_audio) / 1000.0 - 5.0) < 0.2


def test_strip_silence():
    """Verify that strip_silence removes leading and trailing dead silence."""
    from pydub.generators import Sine
    sync = AudioSynchronizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 300ms leading silence + 600ms tone + 500ms trailing silence = 1400ms total
        lead = AudioSegment.silent(duration=300, frame_rate=44100)
        tone = Sine(440).to_audio_segment(duration=600).set_frame_rate(44100)
        trail = AudioSegment.silent(duration=500, frame_rate=44100)
        padded_audio = lead + tone + trail

        raw_path = os.path.join(tmpdir, "padded.wav")
        clean_path = os.path.join(tmpdir, "clean.wav")
        padded_audio.export(raw_path, format="wav")

        clean_dur = sync.strip_silence(raw_path, clean_path, padding_ms=25)
        assert os.path.isfile(clean_path)
        # The trimmed duration should be around 600ms + 50ms = 650ms, much shorter than 1400ms
        assert clean_dur < 0.85
        assert clean_dur > 0.50


def test_boundary_clamping_zero_bleed():
    """Verify that a long TTS segment never spills into the subsequent segment's start time."""
    from pydub.generators import Sine
    sync = AudioSynchronizer(max_stretch_factor=2.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Segment 1 audio is 4.0 seconds long, but available slot before segment 2 is only 1.5s
        long_tone = Sine(440).to_audio_segment(duration=4000).set_frame_rate(44100)
        seg1_audio = os.path.join(tmpdir, "long_seg.wav")
        long_tone.export(seg1_audio, format="wav")

        segments = [
            {
                "id": 1,
                "start": 0.5,
                "end": 1.5,
                "duration": 1.0,
                "text": "Segment 1",
                "audio_path": seg1_audio,
                "tts_duration": 4.0,
            },
            {
                "id": 2,
                "start": 2.0,
                "end": 3.0,
                "duration": 1.0,
                "text": "Segment 2",
                "audio_path": seg1_audio,
                "tts_duration": 4.0,
            },
        ]

        out_audio = os.path.join(tmpdir, "no_bleed.wav")
        sync_temp = os.path.join(tmpdir, "sync_temp")

        res = sync.synchronize(
            segments=segments,
            total_duration=4.0,
            output_path=out_audio,
            temp_dir=sync_temp,
            mix_original_volume=0.0,
        )

        assert os.path.isfile(out_audio)
        synced = res["synced_segments"]
        assert len(synced) == 2
        # Verify speed ratio adjusted upward
        assert synced[0]["speed_ratio"] > 1.3
        # In the timeline between 1.95s and 2.0s, segment 1 must NOT overlap with segment 2
        timeline_audio = AudioSegment.from_file(out_audio)
        assert len(timeline_audio) == 4000

