"""
Verification script: Creates an alternating multi-speaker (Male & Female) foreign video,
runs the automated dubbing pipeline, and verifies that the system dynamically switches
between male and female neural voices for each speaker.
"""

import asyncio
import os
import subprocess
import edge_tts
from pydub import AudioSegment

from dubber.config import DubbingConfig, find_ffmpeg_executable
from dubber.pipeline import DubbingPipeline


def create_dialogue_video(output_mp4_path: str) -> str:
    ffmpeg_exe = find_ffmpeg_executable()
    temp_dir = os.path.dirname(os.path.abspath(output_mp4_path))
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Synthesize alternating French dialogue:
    # Speaker 1: Male (Henri)
    # Speaker 2: Female (Denise)
    part1_male_mp3 = os.path.join(temp_dir, "dialogue_part1_male.mp3")
    part2_female_mp3 = os.path.join(temp_dir, "dialogue_part2_female.mp3")
    part3_male_mp3 = os.path.join(temp_dir, "dialogue_part3_male.mp3")
    part4_female_mp3 = os.path.join(temp_dir, "dialogue_part4_female.mp3")

    async def gen_dialogue():
        await edge_tts.Communicate(
            "Bonjour à tous, je suis le premier présentateur. Aujourd'hui nous présentons le doublage automatique.",
            "fr-FR-HenriNeural",
        ).save(part1_male_mp3)

        await edge_tts.Communicate(
            "Et moi je suis la co-présentatrice. Je vous montre comment notre système détecte chaque voix automatiquement.",
            "fr-FR-DeniseNeural",
        ).save(part2_female_mp3)

        await edge_tts.Communicate(
            "Exactement. Dès qu'un homme explique, une voix masculine anglaise est sélectionnée.",
            "fr-FR-HenriNeural",
        ).save(part3_male_mp3)

        await edge_tts.Communicate(
            "Et dès qu'une femme intervient, le système bascule immédiatement vers une voix féminine naturelle.",
            "fr-FR-DeniseNeural",
        ).save(part4_female_mp3)

    print("Synthesizing source dialogue clips (French Male & Female)...")
    asyncio.run(gen_dialogue())

    # Convert to standard 16kHz WAVs using FFmpeg (avoids ffprobe)
    clips_wav = []
    for mp3 in [part1_male_mp3, part2_female_mp3, part3_male_mp3, part4_female_mp3]:
        wav = mp3.replace(".mp3", ".wav")
        subprocess.run([ffmpeg_exe, "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav], check=True, capture_output=True)
        clips_wav.append(wav)

    # 2. Combine into a continuous audio track with brief pauses
    pause = AudioSegment.silent(duration=600, frame_rate=16000)
    track = (
        AudioSegment.from_wav(clips_wav[0])
        + pause
        + AudioSegment.from_wav(clips_wav[1])
        + pause
        + AudioSegment.from_wav(clips_wav[2])
        + pause
        + AudioSegment.from_wav(clips_wav[3])
        + pause
    )
    combined_audio = os.path.join(temp_dir, "dialogue_source_audio.wav")
    track.export(combined_audio, format="wav")
    total_dur_sec = len(track) / 1000.0
    print(f"Combined dialogue audio: {total_dur_sec:.1f}s")

    # 3. Create video container with FFmpeg
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "lavfi", "-i", f"color=c=0x1e1b4b:s=1280x720:d={total_dur_sec:.2f}:r=30",
        "-i", combined_audio,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_mp4_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to create dialogue video: {res.stderr}")

    print(f"Created multi-speaker dialogue video: {output_mp4_path}")
    return output_mp4_path


def test_dialogue_pipeline():
    video_source = os.path.join("output", "temp", "multi_speaker_dialogue_source.mp4")
    create_dialogue_video(video_source)

    config = DubbingConfig(
        output_dir="output",
        temp_dir=os.path.join("output", "temp"),
        whisper_model="base",
        auto_gender=True,
        male_voice="en-US-ChristopherNeural",
        female_voice="en-US-JennyNeural",
        keep_temp=True,
    )
    pipeline = DubbingPipeline(config)

    print("\n" + "=" * 60)
    print("RUNNING MULTI-SPEAKER DUBBING PIPELINE (MALE + FEMALE)")
    print("=" * 60)

    metrics = pipeline.run(video_source, output_filename="dialogue_dubbed_multispeaker.mp4")

    print("\n" + "=" * 60)
    print("MULTI-SPEAKER DUBBING COMPLETED!")
    print(f"Output Video: {metrics.get('output_video')}")
    print(f"Male Segments Dubbed: {metrics.get('male_segments')} (Voice: {config.male_voice})")
    print(f"Female Segments Dubbed: {metrics.get('female_segments')} (Voice: {config.female_voice})")
    print("=" * 60)

    assert metrics.get("male_segments", 0) > 0, "Expected at least one male segment detected"
    assert metrics.get("female_segments", 0) > 0, "Expected at least one female segment detected"
    assert os.path.isfile(metrics.get("output_video")), "Output video file was not created"


if __name__ == "__main__":
    test_dialogue_pipeline()
