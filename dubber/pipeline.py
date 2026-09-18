"""
Pipeline orchestrator for the Automated Video Dubbing System.
Coordinates download, transcription, synthesis, synchronization, and remuxing.
"""

import json
import os
import shutil
import time
from typing import Any, Callable, Dict, Optional

from .config import DubbingConfig, SUPPORTED_LANGUAGES
from .downloader import YouTubeDownloader
from .transcriber import SpeechTranscriber
from .synthesizer import SpeechSynthesizer, get_media_duration
from .synchronizer import AudioSynchronizer
from .remuxer import VideoRemuxer
from .classifier import SpeakerGenderClassifier
from .ui import (
    console,
    print_banner,
    print_video_info,
    print_step,
    print_segments_preview,
    print_summary,
    get_download_progress,
    get_general_progress,
)


class DubbingPipeline:
    """Orchestrates the entire automated video dubbing pipeline."""

    def __init__(self, config: Optional[DubbingConfig] = None):
        self.config = config or DubbingConfig()
        self.classifier = SpeakerGenderClassifier()
        self.downloader = YouTubeDownloader(
            temp_dir=self.config.temp_dir,
            ffmpeg_path=self.config.ffmpeg_path,
        )
        self.transcriber = SpeechTranscriber(
            model_size=self.config.whisper_model,
            device=self.config.whisper_device,
            compute_type=self.config.compute_type,
        )
        self.synthesizer = SpeechSynthesizer(
            voice=self.config.voice,
            rate=self.config.voice_rate,
            pitch=self.config.voice_pitch,
            volume=self.config.voice_volume,
            ffmpeg_path=self.config.ffmpeg_path,
        )
        self.synchronizer = AudioSynchronizer(
            ffmpeg_path=self.config.ffmpeg_path,
            max_stretch_factor=self.config.max_stretch_factor,
            min_stretch_factor=self.config.min_stretch_factor,
            min_silence_gap=self.config.min_silence_gap,
        )
        self.remuxer = VideoRemuxer(ffmpeg_path=self.config.ffmpeg_path)

    def run(
        self,
        url_or_path: str,
        output_filename: Optional[str] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full dubbing pipeline from YouTube URL or local video file.
        """
        start_time = time.time()
        print_banner()

        # Step 1: Download or load video
        print_step(1, 5, "Fetching Video and Extracting Source Audio")
        t0 = time.time()
        if event_callback:
            event_callback({"stage": 1, "status": "running", "title": "Fetching Video & Extracting Source Audio"})

        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            with get_download_progress() as progress:
                dl_task = progress.add_task("Downloading YouTube Video...", total=100)

                def dl_callback(d: Dict[str, Any]):
                    if d.get("status") == "downloading":
                        total = d.get("total", 1)
                        downloaded = d.get("downloaded", 0)
                        progress.update(dl_task, completed=downloaded, total=total)
                        if event_callback:
                            pct = round((downloaded / total) * 100, 1) if total else 0.0
                            event_callback({"stage": 1, "status": "downloading", "percent": pct, "speed": d.get("speed", 0)})
                    elif d.get("status") == "finished":
                        progress.update(dl_task, completed=100, total=100)
                        if event_callback:
                            event_callback({"stage": 1, "status": "downloading", "percent": 100.0})

                download_res = self.downloader.download(url_or_path, progress_callback=dl_callback)

            video_path = download_res["video_path"]
            audio_path = download_res["audio_path"]
            metadata = download_res["metadata"]
        else:
            # Local video file
            if not os.path.isfile(url_or_path):
                raise FileNotFoundError(f"Local file does not exist: {url_or_path}")
            video_path = url_or_path
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(self.config.temp_dir, f"{base_name}_extracted.wav")
            self.downloader._extract_audio(video_path, audio_path)
            file_dur = get_media_duration(video_path, self.config.ffmpeg_path)
            metadata = {
                "id": base_name,
                "title": base_name,
                "duration": round(file_dur, 2) if file_dur > 0 else 0,
                "uploader": "Local File",
                "url": url_or_path,
            }

        print_video_info(metadata)
        download_time = round(time.time() - t0, 2)
        console.print(f"[green]✓ Video ready in {download_time}s[/green]")
        if event_callback:
            event_callback({"stage": 1, "status": "done", "metadata": metadata, "elapsed": download_time})

        # Step 2: Transcribe & Translate
        target_lang = getattr(self.config, "target_lang", "en")
        lang_info = SUPPORTED_LANGUAGES.get(target_lang, {"name": target_lang.upper()})
        lang_name = lang_info.get("name", "English")

        print_step(2, 5, f"Transcribing & Translating Speech into {lang_name}")
        t0 = time.time()
        if event_callback:
            event_callback({
                "stage": 2,
                "status": "running",
                "title": f"Whisper AI Transcribing & Translating Speech into {lang_name}",
                "target_lang": target_lang,
            })
        video_id = metadata.get("id", "dub")
        cached_segments_path = os.path.join(self.config.temp_dir, f"{video_id}_{target_lang}_segments.json")

        if os.path.isfile(cached_segments_path):
            console.print(f"[cyan]ℹ Using cached transcription from {cached_segments_path}[/cyan]")
            with open(cached_segments_path, "r", encoding="utf-8") as f:
                transcribe_res = json.load(f)
        else:
            with get_general_progress() as progress:
                tx_task = progress.add_task(f"Whisper AI translating audio into {lang_name}...", total=100)

                def tx_callback(d: Dict[str, Any]):
                    pct = d.get("percent", 0.0)
                    progress.update(tx_task, completed=pct, total=100.0)
                    if event_callback:
                        event_callback({"stage": 2, "status": "translating", "percent": round(pct, 1)})

                transcribe_res = self.transcriber.transcribe_and_translate(
                    audio_path,
                    target_lang=target_lang,
                    source_lang=getattr(self.config, "source_lang", "auto"),
                    progress_callback=tx_callback,
                )
                self.transcriber.save_json(transcribe_res, cached_segments_path)

        segments = transcribe_res.get("segments", [])
        detected_lang = transcribe_res.get("language", "unknown")
        lang_prob = transcribe_res.get("language_probability", 0.0)
        
        # Ensure total duration strictly matches the actual video file on disk
        actual_video_dur = get_media_duration(video_path, self.config.ffmpeg_path)
        if actual_video_dur > 0:
            total_audio_duration = actual_video_dur
            metadata["duration"] = round(actual_video_dur, 2)
        else:
            container_dur = metadata.get("duration", 0.0)
            total_audio_duration = container_dur if container_dur > 0 else transcribe_res.get("duration", 0.0)

        console.print(
            f"[green]✓ Detected language: [bold]{detected_lang.upper()}[/bold] "
            f"(confidence {lang_prob*100:.1f}%), extracted {len(segments)} speech segments in {lang_name} (Duration: {total_audio_duration:.2f}s).[/green]"
        )

        # Speaker Gender Detection & Multi-Speaker Voice Assignment
        male_voice = self.config.male_voice
        female_voice = self.config.female_voice

        # If user left voices as default English but selected an Indian or global language, auto-pick language voice pair
        if target_lang != "en" and target_lang in SUPPORTED_LANGUAGES:
            if male_voice == "en-US-ChristopherNeural":
                male_voice = lang_info.get("male_voice", male_voice)
            if female_voice == "en-US-JennyNeural":
                female_voice = lang_info.get("female_voice", female_voice)

        if self.config.auto_gender:
            console.print("[cyan]🔍 Multimodal AI: Analyzing visual faces & vocal pitch with Person-Matching Consistency...[/cyan]")
            if event_callback:
                event_callback({
                    "stage": 2.5,
                    "status": "running",
                    "title": "Multimodal AI: Analyzing Visual Face (OpenCV + Vision AI) & Vocal Pitch"
                })
            segments = self.classifier.classify_segments(
                segments=segments,
                source_audio_path=audio_path,
                video_path=video_path,
                temp_dir=self.config.temp_dir,
                ffmpeg_path=self.config.ffmpeg_path,
                male_voice=male_voice,
                female_voice=female_voice,
                speaker_mode=getattr(self.config, "speaker_mode", "auto"),
            )
            transcribe_res["segments"] = segments
            self.transcriber.save_json(transcribe_res, cached_segments_path)

            num_male = sum(1 for s in segments if s.get("gender") == "male")
            num_female = sum(1 for s in segments if s.get("gender") == "female")
            is_locked = any("person_locked" in str(s.get("detection_source", "")) for s in segments)
            lock_label = " (🎯 Person Locked)" if is_locked else ""
            console.print(
                f"[green]✓ Speaker Analysis{lock_label}: [bold cyan]{num_male} Male ♂[/bold cyan] ({male_voice}) | "
                f"[bold magenta]{num_female} Female ♀[/bold magenta] ({female_voice})[/green]"
            )
        else:
            for s in segments:
                s["gender"] = "custom"
                s["voice"] = self.config.voice
            num_male = len(segments)
            num_female = 0

        print_segments_preview(segments)

        # Export Subtitles: Target SRT, WebVTT, and Bilingual Dual-Language SRT
        srt_path = os.path.join(self.config.output_dir, f"{video_id}_subtitles_{target_lang}.srt")
        vtt_path = os.path.join(self.config.output_dir, f"{video_id}_subtitles_{target_lang}.vtt")
        bilingual_srt_path = os.path.join(self.config.output_dir, f"{video_id}_subtitles_bilingual.srt")

        self.transcriber.export_srt(segments, srt_path)
        self.transcriber.export_vtt(segments, vtt_path)
        self.transcriber.export_bilingual_srt(segments, bilingual_srt_path)
        console.print(f"[green]✓ Subtitles saved: {srt_path} and {bilingual_srt_path}[/green]")

        transcribe_time = round(time.time() - t0, 2)
        if event_callback:
            event_callback({
                "stage": 2,
                "status": "done",
                "language": detected_lang,
                "target_lang": target_lang,
                "source_lang": getattr(self.config, "source_lang", "auto"),
                "lang_confidence": lang_prob,
                "male_segments": num_male,
                "female_segments": num_female,
                "segments": segments,
                "srt_url": f"/output/{os.path.basename(srt_path)}",
                "vtt_url": f"/output/{os.path.basename(vtt_path)}",
                "bilingual_srt_url": f"/output/{os.path.basename(bilingual_srt_path)}",
                "elapsed": transcribe_time,
            })

        if not segments:
            raise RuntimeError("No speech segments detected in the video audio track.")

        # Step 3: Synthesize Speech
        voice_label = (
            f"Adaptive (♂ {male_voice.split('-')[-1]} / ♀ {female_voice.split('-')[-1]})"
            if self.config.auto_gender
            else self.config.voice
        )
        print_step(3, 5, f"Synthesizing Neural Speech ({voice_label})")
        t0 = time.time()
        synth_dir = os.path.join(self.config.temp_dir, f"{video_id}_synth")
        if event_callback:
            event_callback({"stage": 3, "status": "running", "title": f"Synthesizing Neural Speech ({voice_label})", "total": len(segments)})

        with get_general_progress() as progress:
            synth_task = progress.add_task("Edge TTS synthesizing audio...", total=len(segments))

            def synth_callback(d: Dict[str, Any]):
                progress.update(synth_task, completed=d.get("completed", 0), total=d.get("total", 1))
                if event_callback:
                    event_callback({
                        "stage": 3,
                        "status": "synthesizing",
                        "completed": d.get("completed", 0),
                        "total": d.get("total", 1),
                        "percent": round((d.get("completed", 0) / max(1, d.get("total", 1))) * 100, 1),
                    })

            synthesized_segments = self.synthesizer.synthesize(
                segments,
                output_dir=synth_dir,
                progress_callback=synth_callback,
            )

        synth_time = round(time.time() - t0, 2)
        console.print(f"[green]✓ Synthesized {len(synthesized_segments)} segments in {synth_time}s[/green]")
        if event_callback:
            event_callback({"stage": 3, "status": "done", "elapsed": synth_time})

        # Step 4: Audio-Visual Synchronization
        print_step(4, 5, "Synchronizing Audio Timeline with Video Timestamps")
        t0 = time.time()
        synced_audio_path = os.path.join(self.config.temp_dir, f"{video_id}_dubbed_audio.wav")
        if event_callback:
            event_callback({"stage": 4, "status": "running", "title": "Synchronizing Audio Timeline (Zero Drift)", "total": len(synthesized_segments)})

        with get_general_progress() as progress:
            sync_task = progress.add_task("Aligning audio & stretching...", total=len(synthesized_segments))

            def sync_callback(d: Dict[str, Any]):
                progress.update(sync_task, completed=d.get("completed", 0), total=d.get("total", 1))
                if event_callback:
                    event_callback({
                        "stage": 4,
                        "status": "syncing",
                        "completed": d.get("completed", 0),
                        "total": d.get("total", 1),
                        "percent": round(d.get("percent", 0.0), 1),
                    })

            sync_res = self.synchronizer.synchronize(
                segments=synthesized_segments,
                total_duration=total_audio_duration,
                output_path=synced_audio_path,
                temp_dir=os.path.join(self.config.temp_dir, f"{video_id}_sync_temp"),
                original_audio_path=audio_path,
                mix_original_volume=self.config.mix_original_volume,
                progress_callback=sync_callback,
            )

        sync_time = round(time.time() - t0, 2)
        console.print(f"[green]✓ Audio timeline assembled in {sync_time}s[/green]")
        if event_callback:
            event_callback({"stage": 4, "status": "done", "elapsed": sync_time})

        # Step 5: Remux Video
        print_step(5, 5, "Remuxing Final Video Losslessly with FFmpeg")
        t0 = time.time()
        if event_callback:
            event_callback({"stage": 5, "status": "running", "title": "Remuxing Final Video Losslessly with FFmpeg"})
        if not output_filename:
            output_filename = f"{video_id}_dubbed_{target_lang}.mp4"
        final_video_path = os.path.join(self.config.output_dir, output_filename)

        sub_mode = getattr(self.config, "subtitle_mode", "soft")
        sub_to_embed = None
        burn_subs = False

        if sub_mode == "burn":
            sub_to_embed = srt_path
            burn_subs = True
        elif sub_mode == "dual":
            sub_to_embed = bilingual_srt_path
            burn_subs = False
        elif sub_mode == "soft" and self.config.embed_subtitles:
            sub_to_embed = srt_path
            burn_subs = False

        remux_res = self.remuxer.remux(
            video_path=video_path,
            audio_path=synced_audio_path,
            output_path=final_video_path,
            subtitle_path=sub_to_embed,
            burn_subtitles=burn_subs,
            original_audio_path=audio_path,
            include_original_track=False,
        )
        remux_time = round(time.time() - t0, 2)
        console.print(f"[green]✓ Final video remuxed in {remux_time}s[/green]")

        # Clean temp directory if requested
        if not self.config.keep_temp:
            shutil.rmtree(self.config.temp_dir, ignore_errors=True)
            os.makedirs(self.config.temp_dir, exist_ok=True)

        total_time = round(time.time() - start_time, 2)

        summary_data = {
            "title": metadata.get("title"),
            "url": metadata.get("url"),
            "language": detected_lang,
            "target_lang": target_lang,
            "lang_confidence": lang_prob,
            "segments_count": len(segments),
            "voice": voice_label,
            "male_segments": num_male,
            "female_segments": num_female,
            "auto_gender": self.config.auto_gender,
            "video_duration": total_audio_duration,
            "download_time": download_time,
            "transcribe_time": transcribe_time,
            "synth_time": synth_time,
            "sync_time": sync_time,
            "remux_time": remux_time,
            "total_time": total_time,
            "output_video": final_video_path,
            "subtitles_file": srt_path,
            "file_size_mb": remux_res.get("file_size_mb", 0),
        }

        print_summary(summary_data)
        if event_callback:
            event_callback({
                "stage": 5,
                "status": "done",
                "elapsed": remux_time,
                "summary": summary_data,
                "video_url": f"/output/{os.path.basename(final_video_path)}",
                "srt_url": f"/output/{os.path.basename(srt_path)}",
                "vtt_url": f"/output/{os.path.basename(vtt_path)}" if vtt_path else None,
                "bilingual_srt_url": f"/output/{os.path.basename(bilingual_srt_path)}" if bilingual_srt_path else None,
            })
            event_callback({
                "stage": 5,
                "status": "complete",
                "elapsed": remux_time,
                "summary": summary_data,
                "video_url": f"/output/{os.path.basename(final_video_path)}",
                "srt_url": f"/output/{os.path.basename(srt_path)}",
                "vtt_url": f"/output/{os.path.basename(vtt_path)}" if vtt_path else None,
                "bilingual_srt_url": f"/output/{os.path.basename(bilingual_srt_path)}" if bilingual_srt_path else None,
            })
        return summary_data
