"""
Audio timing synchronization and timeline assembler.
Aligns synthesized English audio segments with original video timestamps,
applying tempo adjustments and silence padding to guarantee zero cumulative drift.
"""

import os
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional
import wave
from pydub import AudioSegment

from .config import find_ffmpeg_executable


class AudioSynchronizer:
    """Synchronizes TTS segments with video timestamps using dynamic tempo scaling."""

    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        max_stretch_factor: float = 2.20,
        min_stretch_factor: float = 0.85,
        min_silence_gap: float = 0.05,
    ):
        self.ffmpeg_path = ffmpeg_path or find_ffmpeg_executable()
        self.max_stretch_factor = max_stretch_factor
        self.min_stretch_factor = min_stretch_factor
        self.min_silence_gap = min_silence_gap
        AudioSegment.converter = self.ffmpeg_path

    def _build_atempo_filter(self, speed_ratio: float) -> str:
        """
        Build an FFmpeg atempo filter chain.
        FFmpeg atempo filter requires values between 0.5 and 2.0.
        Values outside this range are factored into multiple chained atempo filters.
        """
        speed = max(0.5, min(speed_ratio, 4.0))
        filters = []
        while speed > 2.0:
            filters.append("atempo=2.0")
            speed /= 2.0
        while speed < 0.5:
            filters.append("atempo=0.5")
            speed /= 0.5
        filters.append(f"atempo={speed:.4f}")
        return ",".join(filters)

    def _get_duration(self, file_path: str) -> float:
        """Extract media duration safely using wave or FFmpeg without ffprobe."""
        if not os.path.isfile(file_path):
            return 0.0
        if file_path.lower().endswith(".wav"):
            try:
                with wave.open(file_path, "rb") as w:
                    return w.getnframes() / float(w.getframerate())
            except Exception:
                pass
        cmd = [self.ffmpeg_path, "-i", file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
        if match:
            hours = float(match.group(1))
            minutes = float(match.group(2))
            seconds = float(match.group(3))
            return hours * 3600 + minutes * 60 + seconds
        return 0.0

    def strip_silence(self, input_path: str, output_path: str, padding_ms: int = 25) -> float:
        """
        Trim leading and trailing silence from synthesized TTS audio.
        Neural TTS engines (like Edge-TTS) consistently prepend 200-300ms of dead silence
        at the start and up to 900ms at the end. Stripping this ensures the dubbed
        voice begins instantaneously with visual mouth movements (precise lip-sync)
        and doesn't artificially spill over into adjacent segment timestamps.
        """
        from pydub import silence

        # Convert input audio to 44.1kHz stereo WAV via FFmpeg
        temp_wav = output_path + ".raw.wav"
        cmd = [self.ffmpeg_path, "-y", "-i", input_path, "-vn", "-ar", "44100", "-ac", "2", temp_wav]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            seg = AudioSegment.from_wav(temp_wav)
            nonsilent = silence.detect_nonsilent(seg, min_silence_len=30, silence_thresh=-42)
            if nonsilent:
                start_trim = max(0, nonsilent[0][0] - padding_ms)
                end_trim = min(len(seg), nonsilent[-1][1] + padding_ms)
                trimmed = seg[start_trim:end_trim]
            else:
                trimmed = seg

            trimmed.export(output_path, format="wav")
        finally:
            if os.path.isfile(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

        return self._get_duration(output_path)

    def time_stretch_audio(
        self,
        input_path: str,
        output_path: str,
        speed_ratio: float,
    ) -> float:
        """
        Time-stretch an audio file without altering pitch using FFmpeg atempo.
        Converts any audio format (MP3, WAV, AAC) to standardized 44.1kHz stereo WAV.
        Returns the duration of the stretched audio in seconds.
        """
        cmd = [self.ffmpeg_path, "-y", "-i", input_path]
        if abs(speed_ratio - 1.0) >= 0.02:
            filter_chain = self._build_atempo_filter(speed_ratio)
            cmd.extend(["-filter:a", filter_chain])

        cmd.extend(["-vn", "-ar", "44100", "-ac", "2", output_path])
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            # Fallback direct transcode
            fallback_cmd = [self.ffmpeg_path, "-y", "-i", input_path, "-vn", "-ar", "44100", "-ac", "2", output_path]
            subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return self._get_duration(output_path)

    def synchronize(
        self,
        segments: List[Dict[str, Any]],
        total_duration: float,
        output_path: str,
        temp_dir: str,
        original_audio_path: Optional[str] = None,
        mix_original_volume: float = 0.0,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Align synthesized speech segments onto a full timeline matching the original video duration.
        Applies leading/trailing silence stripping for tight lip-sync, adaptive stretch
        scaling, and strict boundary clamping to eliminate speech overlap.
        """
        os.makedirs(temp_dir, exist_ok=True)
        target_sr = 44100
        # Initialize blank audio canvas of total_duration with 44100Hz stereo
        timeline = AudioSegment.silent(
            duration=int(total_duration * 1000) + 1000,
            frame_rate=target_sr,
        ).set_channels(2)

        total_segs = len(segments)
        stretched_segments = []

        for idx, seg in enumerate(segments):
            seg_audio_path = seg.get("audio_path")
            if not seg_audio_path or not os.path.isfile(seg_audio_path):
                continue

            target_start = seg["start"]
            target_end = seg["end"]
            slot_duration = max(0.1, target_end - target_start)

            # Available buffer before next segment
            if idx + 1 < total_segs:
                next_start = segments[idx + 1]["start"]
                # Hard limit ensures no spillover into next speaker's start timestamp
                hard_limit = max(0.1, next_start - target_start - self.min_silence_gap)
            else:
                hard_limit = max(0.1, total_duration - target_start)

            # Step 1: Strip leading and trailing dead silence for immediate mouth sync
            clean_path = os.path.join(temp_dir, f"clean_{idx:05d}.wav")
            clean_dur = self.strip_silence(seg_audio_path, clean_path, padding_ms=25)
            if clean_dur <= 0:
                clean_dur = self._get_duration(clean_path)

            # Step 2: Compute ideal speed ratio to fit visual mouth movement window
            if clean_dur > slot_duration:
                # TTS takes longer than visual speech slot
                speed_to_fit_slot = clean_dur / slot_duration
                if speed_to_fit_slot <= 1.35:
                    # Fits comfortably within mouth movement window at a natural speaking rate
                    speed_ratio = speed_to_fit_slot
                else:
                    # Allow slight expansion into available pause before next speaker
                    target_window = min(hard_limit, slot_duration + 0.5 * max(0.0, hard_limit - slot_duration))
                    speed_ratio = clean_dur / max(0.1, target_window)
                speed_ratio = min(speed_ratio, self.max_stretch_factor)
            elif clean_dur < slot_duration * 0.70:
                # Original speaker spoke slowly; gently pace TTS up to min_stretch_factor
                speed_ratio = max(self.min_stretch_factor, clean_dur / slot_duration)
            else:
                speed_ratio = 1.0

            # Step 3: Time stretch audio without pitch distortion
            stretched_path = os.path.join(temp_dir, f"synced_{idx:05d}.wav")
            actual_dur = self.time_stretch_audio(clean_path, stretched_path, speed_ratio)

            # Step 4: Load into pydub and enforce hard-stop boundary (zero overlap guarantee)
            clip = AudioSegment.from_wav(stretched_path).set_frame_rate(target_sr).set_channels(2)
            max_allowed_ms = int(hard_limit * 1000)
            if len(clip) > max_allowed_ms:
                fade_ms = min(25, max(5, max_allowed_ms // 4))
                clip = clip[:max_allowed_ms].fade_out(fade_ms)
                actual_dur = len(clip) / 1000.0

            # Overlay onto timeline at exact start timestamp
            insert_pos_ms = int(target_start * 1000)
            timeline = timeline.overlay(clip, position=insert_pos_ms)

            seg_info = dict(seg)
            seg_info["speed_ratio"] = round(speed_ratio, 3)
            seg_info["synced_duration"] = round(actual_dur, 3)
            seg_info["synced_path"] = stretched_path
            stretched_segments.append(seg_info)

            if progress_callback:
                progress_callback({
                    "completed": idx + 1,
                    "total": total_segs,
                    "percent": ((idx + 1) / total_segs) * 100.0,
                    "speed_ratio": speed_ratio,
                    "start": target_start,
                })

        # Trim timeline to exact video duration
        final_duration_ms = int(total_duration * 1000)
        timeline = timeline[:final_duration_ms]

        # Export dubbed voice track
        temp_dub_wav = os.path.join(temp_dir, "dubbed_voice_raw.wav")
        timeline.export(temp_dub_wav, format="wav")

        # Check if we should mix with original audio:
        # 1. User requested mix_original_volume > 0.0
        # 2. OR dialogue finishes before video ends (e.g. movie song/music in second half)
        last_speech_end = max((s.get("end", 0.0) for s in segments), default=0.0) if segments else 0.0
        has_post_speech_video = (total_duration - last_speech_end) > 2.0

        if (mix_original_volume > 0.0 or has_post_speech_video) and original_audio_path and os.path.isfile(original_audio_path):
            self._mix_with_original(
                dubbed_audio_path=temp_dub_wav,
                original_audio_path=original_audio_path,
                output_path=output_path,
                bg_volume=mix_original_volume,
                segments=segments,
            )
        else:
            # 100% pure dubbed audio (Full replacement - zero leakage)
            if os.path.abspath(temp_dub_wav) != os.path.abspath(output_path):
                import shutil
                shutil.copyfile(temp_dub_wav, output_path)

        return {
            "output_audio_path": output_path,
            "duration": total_duration,
            "synced_segments": stretched_segments,
        }

    def _mix_with_original(
        self,
        dubbed_audio_path: str,
        original_audio_path: str,
        output_path: str,
        bg_volume: float = 0.0,
        segments: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Mix dubbed speech over background audio.
        During speech segments, original foreign audio is completely ducked to volume 0.0
        (guaranteeing zero foreign speech bleed).
        During post-speech video intervals (e.g. background songs/outro), volume returns to 1.0.
        """
        # Natural ducking level: when bg_volume is 0.0 (Clean Dub), original speech is silenced to 0.0
        # guaranteeing zero foreign speech bleed. During pauses: 0.90. After speech ends: 1.0 (song/music!)
        duck_vol = max(0.0, float(bg_volume))
        pause_vol = 0.90
        last_end = max((s.get("end", 0.0) for s in segments), default=0.0) if segments else 0.0

        # Build ducking volume expression
        if segments:
            duck_conditions = []
            for seg in segments:
                s_start = max(0.0, seg.get("start", 0.0) - 0.04)
                s_end = max(s_start + 0.08, seg.get("end", 0.0) + 0.04)
                duck_conditions.append(f"between(t\\,{s_start:.3f}\\,{s_end:.3f})")

            or_expr = "+".join(duck_conditions)
            if last_end > 0:
                # During speech: duck_vol (0.0 for pure English speech!). In pauses: pause_vol (0.90). Post-speech: 1.0 (untouched music/song!)
                bg_filter = f"volume='if(gt({or_expr},0),{duck_vol:.3f},if(gt(t,{last_end+0.25:.3f}),1.0,{pause_vol:.3f}))':eval=frame"
            else:
                bg_filter = f"volume='if(gt({or_expr},0),{duck_vol:.3f},{pause_vol:.3f})':eval=frame"
        else:
            bg_filter = f"volume={duck_vol:.3f}"

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", dubbed_audio_path,
            "-i", original_audio_path,
            "-filter_complex",
            f"[1:a]{bg_filter}[bg];[0:a]volume=1.2[fg];[fg][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]",
            "-map", "[out]",
            "-ac", "2",
            "-ar", "44100",
            output_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            import shutil
            shutil.copyfile(dubbed_audio_path, output_path)
