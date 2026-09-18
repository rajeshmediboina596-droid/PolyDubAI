"""
Speaker Voice Preservation & Conversion Engine for PolyDubAI.
Extracts original speaker acoustic pitch (F0), cadence, and energy,
and modulates speech synthesis to preserve the speaker's vocal characteristics.
"""

from dataclasses import dataclass
import math
import os
import subprocess
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .config import find_ffmpeg_executable


@dataclass
class AcousticProfile:
    """Acoustic characteristics of a speaker extracted from reference audio."""
    median_f0: float
    mean_f0: float
    min_f0: float
    max_f0: float
    pitch_std: float
    speaking_rate: float  # syllables or units per second
    energy_rms: float
    spectral_centroid: float
    gender_hint: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "median_f0": round(self.median_f0, 1),
            "mean_f0": round(self.mean_f0, 1),
            "min_f0": round(self.min_f0, 1),
            "max_f0": round(self.max_f0, 1),
            "pitch_std": round(self.pitch_std, 1),
            "speaking_rate": round(self.speaking_rate, 2),
            "energy_rms": round(self.energy_rms, 4),
            "spectral_centroid": round(self.spectral_centroid, 1),
            "gender_hint": self.gender_hint,
        }


class SpeakerVoicePreserver:
    """
    Analyzes the original speaker's vocal pitch, cadence, and timbre,
    and applies adaptive prosody and spectral transfer so that the translated
    dubbing matches the original speaker's tone and style.
    """

    # Baseline fundamental frequencies for standard neural voices
    MALE_BASELINE_F0: float = 120.0    # Typical male neural voice baseline (Hz)
    FEMALE_BASELINE_F0: float = 210.0  # Typical female neural voice baseline (Hz)
    BASELINE_SPEAKING_RATE: float = 3.5  # Standard syllables per second

    def __init__(self, ffmpeg_path: Optional[str] = None, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.ffmpeg_path = ffmpeg_path or find_ffmpeg_executable()

    def extract_audio_samples(
        self,
        audio_path: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Extract audio samples as a mono float32 numpy array.
        Uses wave/scipy for WAV or FFmpeg conversion if compressed.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # If already a 16kHz WAV file, read directly
        if audio_path.lower().endswith(".wav"):
            try:
                sr, data = wavfile.read(audio_path)
                if data.ndim > 1:
                    data = data.mean(axis=1)  # Convert to mono
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype != np.float32:
                    data = data.astype(np.float32)

                if start is not None or end is not None:
                    s_idx = int((start or 0.0) * sr)
                    e_idx = int((end * sr) if end is not None else len(data))
                    data = data[s_idx:e_idx]

                if sr == self.sample_rate:
                    return data, sr
                else:
                    # Resample to target sample rate
                    num_samples = int(len(data) * self.sample_rate / sr)
                    if num_samples > 0:
                        data = signal.resample(data, num_samples)
                    return data.astype(np.float32), self.sample_rate
            except Exception:
                pass

        # Fallback to FFmpeg pipe to extract clean PCM 16kHz mono float
        cmd = [self.ffmpeg_path, "-y"]
        if start is not None:
            cmd += ["-ss", str(max(0.0, start))]
        cmd += ["-i", audio_path]
        if end is not None and start is not None:
            dur = max(0.05, end - start)
            cmd += ["-t", str(dur)]
        cmd += [
            "-vn",
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-f", "f32le",
            "-"
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            return np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate

        samples = np.frombuffer(proc.stdout, dtype=np.float32)
        return samples, self.sample_rate

    def estimate_pitch_contour(
        self,
        samples: np.ndarray,
        sample_rate: int,
        frame_size_ms: float = 40.0,
        hop_size_ms: float = 20.0,
    ) -> np.ndarray:
        """
        Estimate pitch (F0 in Hz) across audio frames using Normalized Autocorrelation.
        Returns an array of F0 values (0 for unvoiced/silent frames).
        """
        if len(samples) == 0:
            return np.array([], dtype=np.float32)

        frame_len = int(frame_size_ms * sample_rate / 1000.0)
        hop_len = int(hop_size_ms * sample_rate / 1000.0)

        # Bounds for human speech pitch: 65 Hz (deep male) to 450 Hz (high female/child)
        min_lag = int(sample_rate / 450.0)
        max_lag = int(sample_rate / 65.0)

        f0_list = []
        num_frames = max(1, (len(samples) - frame_len) // hop_len + 1)

        for i in range(num_frames):
            idx = i * hop_len
            frame = samples[idx : idx + frame_len]
            if len(frame) < frame_len:
                break

            # Check energy (silence rejection)
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < 0.015:  # Silence threshold
                f0_list.append(0.0)
                continue

            # Remove DC bias and apply Hann window
            frame_centered = (frame - np.mean(frame)) * np.hanning(len(frame))

            # Autocorrelation
            corr = signal.correlate(frame_centered, frame_centered, mode="full")
            corr = corr[len(corr) // 2 :]

            if max_lag >= len(corr):
                f0_list.append(0.0)
                continue

            # Search within human voice pitch range
            search_region = corr[min_lag:max_lag]
            if len(search_region) == 0:
                f0_list.append(0.0)
                continue

            best_offset = np.argmax(search_region)
            peak_lag = min_lag + best_offset
            peak_val = corr[peak_lag]

            # Voice periodicity threshold: peak correlation should be substantial
            if peak_val > 0.3 * corr[0]:
                # Parabolic peak interpolation for sub-sample accuracy
                if 0 < peak_lag < len(corr) - 1:
                    alpha = corr[peak_lag - 1]
                    beta = corr[peak_lag]
                    gamma = corr[peak_lag + 1]
                    denom = 2.0 * (2.0 * beta - alpha - gamma)
                    delta = (alpha - gamma) / denom if abs(denom) > 1e-6 else 0.0
                    refined_lag = peak_lag + delta
                else:
                    refined_lag = float(peak_lag)

                est_f0 = sample_rate / max(1.0, refined_lag)
                if 60.0 <= est_f0 <= 450.0:
                    f0_list.append(est_f0)
                else:
                    f0_list.append(0.0)
            else:
                f0_list.append(0.0)

        return np.array(f0_list, dtype=np.float32)

    def analyze_speaker_profile(
        self,
        audio_path: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        text: Optional[str] = None,
    ) -> AcousticProfile:
        """
        Analyze audio and return the complete acoustic profile including
        median pitch, variation, speaking rate, vocal energy, and spectral centroid.
        """
        samples, sr = self.extract_audio_samples(audio_path, start, end)
        if len(samples) == 0:
            return AcousticProfile(
                median_f0=120.0,
                mean_f0=120.0,
                min_f0=100.0,
                max_f0=150.0,
                pitch_std=15.0,
                speaking_rate=3.5,
                energy_rms=0.05,
                spectral_centroid=1500.0,
                gender_hint="unknown",
            )

        # F0 estimation
        f0_contour = self.estimate_pitch_contour(samples, sr)
        voiced = f0_contour[f0_contour > 0]

        if len(voiced) > 0:
            median_f0 = float(np.median(voiced))
            mean_f0 = float(np.mean(voiced))
            min_f0 = float(np.percentile(voiced, 5))
            max_f0 = float(np.percentile(voiced, 95))
            pitch_std = float(np.std(voiced))
        else:
            median_f0 = 130.0
            mean_f0 = 130.0
            min_f0 = 100.0
            max_f0 = 160.0
            pitch_std = 15.0

        # Energy RMS
        energy_rms = float(np.sqrt(np.mean(samples ** 2)))

        # Spectral Centroid (Vocal Brightness / Timbre)
        # Compute magnitude spectrum
        fft_vals = np.abs(np.fft.rfft(samples[:min(len(samples), 32768)]))
        freqs = np.fft.rfftfreq(min(len(samples), 32768), 1.0 / sr)
        sum_fft = np.sum(fft_vals)
        if sum_fft > 1e-6:
            spectral_centroid = float(np.sum(freqs * fft_vals) / sum_fft)
        else:
            spectral_centroid = 1500.0

        # Speaking rate
        duration = len(samples) / float(sr)
        if text and duration > 0.2:
            # Estimate syllables: rough count based on vowels
            vowel_count = len([c for c in text.lower() if c in "aeiouy"])
            word_count = len(text.strip().split())
            syllables = max(word_count, vowel_count)
            speaking_rate = float(syllables / duration)
        else:
            speaking_rate = 3.5

        # Gender Hint
        if median_f0 < 165.0:
            gender_hint = "male"
        elif median_f0 >= 165.0:
            gender_hint = "female"
        else:
            gender_hint = "unknown"

        return AcousticProfile(
            median_f0=median_f0,
            mean_f0=mean_f0,
            min_f0=min_f0,
            max_f0=max_f0,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            energy_rms=energy_rms,
            spectral_centroid=spectral_centroid,
            gender_hint=gender_hint,
        )

    def compute_prosody_deltas(
        self,
        profile: AcousticProfile,
        base_gender: str = "male",
        target_lang: str = "en",
    ) -> Dict[str, str]:
        """
        Compute pitch, rate, and volume SSML adjustments to make neural TTS
        match the speaker's vocal pitch, pace, and intensity.
        """
        # Select appropriate baseline F0
        if base_gender == "female" or profile.gender_hint == "female":
            base_f0 = self.FEMALE_BASELINE_F0
        else:
            base_f0 = self.MALE_BASELINE_F0

        # Pitch delta in Hz: clamped safely between -50Hz and +50Hz
        raw_pitch_delta = profile.median_f0 - base_f0
        clamped_pitch_delta = int(np.clip(raw_pitch_delta, -45.0, 45.0))
        pitch_str = f"{clamped_pitch_delta:+d}Hz"

        # Rate delta in percentage: clamped between -20% and +30%
        if profile.speaking_rate > 0.1:
            rate_ratio = profile.speaking_rate / self.BASELINE_SPEAKING_RATE
            raw_rate_pct = (rate_ratio - 1.0) * 100.0
            clamped_rate_pct = int(np.clip(raw_rate_pct, -20.0, 30.0))
            rate_str = f"{clamped_rate_pct:+d}%"
        else:
            rate_str = "+0%"

        # Volume adjustment based on vocal energy RMS
        if profile.energy_rms > 0.15:
            vol_str = "+10%"
        elif profile.energy_rms < 0.03:
            vol_str = "-10%"
        else:
            vol_str = "+0%"

        return {
            "pitch": pitch_str,
            "rate": rate_str,
            "volume": vol_str,
            "pitch_hz": str(clamped_pitch_delta),
            "rate_pct": str(clamped_rate_pct if "clamped_rate_pct" in locals() else 0),
        }

    def preserve_voices_for_segments(
        self,
        segments: List[Dict[str, Any]],
        source_audio_path: str,
        target_lang: str = "en",
        mode: str = "adaptive_prosody",
    ) -> List[Dict[str, Any]]:
        """
        Process speech segments, analyze speaker vocal characteristics,
        and assign voice preservation parameters (pitch, rate, timbre) to each segment.
        """
        if not segments:
            return []

        # Analyze global audio profile across the entire speaker audio
        global_profile = self.analyze_speaker_profile(source_audio_path)

        for i, seg in enumerate(segments):
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", seg_start + 1.0)
            orig_text = seg.get("original_text", seg.get("text", ""))
            gender = seg.get("gender", "male")

            # Try local segment acoustic analysis for fine-grained speaker adaptation
            seg_duration = max(0.0, seg_end - seg_start)
            if seg_duration >= 0.6 and os.path.isfile(source_audio_path):
                try:
                    seg_profile = self.analyze_speaker_profile(
                        source_audio_path,
                        start=seg_start,
                        end=seg_end,
                        text=orig_text,
                    )
                except Exception:
                    seg_profile = global_profile
            else:
                seg_profile = global_profile

            # Compute SSML prosody deltas
            deltas = self.compute_prosody_deltas(seg_profile, base_gender=gender, target_lang=target_lang)

            # Assign prosody modulation parameters to segment
            seg["pitch"] = deltas["pitch"]
            seg["rate"] = deltas["rate"]
            seg["volume"] = deltas["volume"]
            seg["voice_profile"] = seg_profile.to_dict()
            seg["voice_preservation_mode"] = mode

        return segments

    def apply_formant_and_timbre_transfer(
        self,
        audio_path: str,
        profile: AcousticProfile,
        output_path: str,
    ) -> str:
        """
        Applies gentle formant EQ and spectral shaping to the synthesized audio
        to transfer the original speaker's vocal color, brightness, and timbre.
        Operates with 0 MB VRAM and minimal CPU overhead.
        """
        if not os.path.isfile(audio_path):
            return audio_path

        # Determine formant shifts based on spectral centroid difference
        # Center frequency for speech formants: F1 (~500Hz), F2 (~1500Hz), F3 (~2500Hz)
        centroid = profile.spectral_centroid
        if centroid > 2200.0:
            # Brighter/higher timbre: slight high-shelf boost (+2dB at 3kHz)
            filter_chain = "equalizer=f=3000:t=q:w=1.0:g=2.5,equalizer=f=300:t=q:w=1.0:g=-1.0"
        elif centroid < 1200.0:
            # Deeper/warmer timbre: bass presence boost (+3dB at 150Hz) and high-cut
            filter_chain = "equalizer=f=150:t=q:w=1.0:g=3.0,equalizer=f=3500:t=q:w=1.0:g=-2.0"
        else:
            # Balanced: subtle presence enhancement
            filter_chain = "equalizer=f=1200:t=q:w=1.5:g=1.5"

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", audio_path,
            "-af", filter_chain,
            "-ac", "1",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception:
            pass

        return audio_path
