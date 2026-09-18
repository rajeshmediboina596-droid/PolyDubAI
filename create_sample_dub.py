"""
Script to create a realistic foreign speech video and run it through the full
DubbingPipeline to produce sample dubbed outputs in output/.
"""

import asyncio
import os
import subprocess
import edge_tts

from dubber.config import DubbingConfig, find_ffmpeg_executable
from dubber.pipeline import DubbingPipeline


def create_sample_media(output_video_path: str) -> str:
    ffmpeg_exe = find_ffmpeg_executable()
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    # 1. Synthesize natural multi-sentence French speech
    french_text = (
        "Bonjour et bienvenue à cette démonstration du système de doublage vidéo automatisé. "
        "Ce projet a été développé pour le stage chez IDEALABS DIGITAL. "
        "Notre système utilise l'intelligence artificielle pour transcrire et traduire la parole étrangère directement en anglais. "
        "Ensuite, une voix neuronale naturelle est synthétisée avec une grande clarté. "
        "Grâce à l'alignement temporel dynamique, il n'y a aucune dérive cumulative entre la vidéo et l'audio. "
        "Enfin, le multiplexage FFmpeg préserve la qualité d'origine à cent pour cent. "
        "Merci beaucoup de votre attention."
    )
    audio_path = os.path.join(os.path.dirname(output_video_path), "french_source_speech.mp3")
    asyncio.run(edge_tts.Communicate(french_text, "fr-FR-HenriNeural").save(audio_path))
    print(f"Generated French source speech: {audio_path} ({os.path.getsize(audio_path)} bytes)")

    # 2. Build MP4 video with FFmpeg
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "lavfi", "-i", "color=c=0x0f172a:s=1280x720:d=35:r=30",
        "-i", audio_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_video_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to create source video: {res.stderr}")

    print(f"Created source foreign video: {output_video_path} ({os.path.getsize(output_video_path)} bytes)")
    return output_video_path


def run_sample_dub():
    source_video = os.path.join("output", "temp", "sample_french_source.mp4")
    create_sample_media(source_video)

    config = DubbingConfig(
        output_dir="output",
        temp_dir=os.path.join("output", "temp"),
        whisper_model="base",
        voice="en-US-ChristopherNeural",
        keep_temp=True,
    )
    pipeline = DubbingPipeline(config)

    print("\nRunning DubbingPipeline on sample French video...")
    metrics = pipeline.run(source_video, output_filename="sample_dubbed_en.mp4")
    print("\nDubbing completed successfully!")
    print(f"Final output video: {metrics.get('output_video')}")
    print(f"Subtitles file: {metrics.get('subtitles_file')}")
    print(f"Detected language: {metrics.get('language')} ({metrics.get('lang_confidence')*100:.1f}%)")
    print(f"Total processing time: {metrics.get('total_time')}s")


if __name__ == "__main__":
    run_sample_dub()
