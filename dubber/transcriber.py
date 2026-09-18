"""
Speech transcription and multi-language translation module.
Transcribes audio with faster-whisper and translates into all Indian and Global languages.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional

from dubber.config import SUPPORTED_LANGUAGES


def format_timestamp(seconds: float, srt_format: bool = True) -> str:
    """Format seconds into standard SRT (00:00:00,000) or VTT (00:00:00.000) timestamp."""
    millis = int(round((seconds - int(seconds)) * 1000))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    separator = "," if srt_format else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def translate_text(text: str, target_lang: str, source_lang: str = "en") -> str:
    """
    Translate text into the desired Indian or global target language.
    Uses MyMemory / Google translation engines.
    """
    if not text.strip() or target_lang.lower() in ("en", "en-us", "en-gb"):
        return text

    target_info = SUPPORTED_LANGUAGES.get(target_lang, {})
    target_locale = target_info.get("locale", target_lang)

    # Attempt 1: MyMemoryTranslator
    try:
        from deep_translator import MyMemoryTranslator
        src = "en-US" if source_lang == "en" else source_lang
        res = MyMemoryTranslator(source=src, target=target_locale).translate(text)
        if res and res.strip() and not res.startswith("MYMEMORY WARNING"):
            return res.strip()
    except Exception:
        pass

    # Attempt 2: GoogleTranslator fallback
    try:
        from deep_translator import GoogleTranslator
        clean_target = target_lang.split("-")[0]
        res = GoogleTranslator(source="auto", target=clean_target).translate(text)
        if res and res.strip():
            return res.strip()
    except Exception:
        pass

    return text


class SpeechTranscriber:
    """Transcribes and translates speech into any selected Indian or global target language."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel
            from .model_downloader import ensure_model_downloaded
            model_path = ensure_model_downloaded(self.model_size)
            threads = min(8, max(2, os.cpu_count() or 4))
            self._model = WhisperModel(
                model_path,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=threads,
                num_workers=2,
            )
        return self._model

    def transcribe_and_translate(
        self,
        audio_path: str,
        target_lang: str = "en",
        source_lang: Optional[str] = "auto",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio and translate to the target language (Indian or Global).
        Supports explicit source language (or auto-detect), extracting segment-level
        timestamps, original text, and translated dubbing script.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        transcribe_kwargs = dict(
            beam_size=2,
            best_of=2,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=100),
        )

        # Force explicit source language if specified
        if source_lang and source_lang != "auto":
            transcribe_kwargs["language"] = source_lang

        # Whisper AI speech transcription / translation
        # task="translate" translates foreign speech directly into English as pivot
        task_mode = "translate" if target_lang != source_lang else "transcribe"
        segments_iter, info = self.model.transcribe(
            audio_path,
            task=task_mode,
            **transcribe_kwargs,
        )

        detected_lang = info.language
        lang_prob = info.language_probability
        duration = info.duration

        segments: List[Dict[str, Any]] = []
        for i, segment in enumerate(segments_iter):
            raw_text = segment.text.strip()
            if not raw_text:
                continue

            # Translate to target language if needed
            if target_lang != "en" and target_lang != "en-US":
                translated_text = translate_text(raw_text, target_lang=target_lang, source_lang="en")
            else:
                translated_text = raw_text

            seg_data = {
                "id": i + 1,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "duration": round(segment.end - segment.start, 3),
                "text": translated_text,
                "orig_text": raw_text,
                "text_en": raw_text if task_mode == "translate" else translated_text,
            }
            segments.append(seg_data)

            if progress_callback:
                progress_callback({
                    "type": "segment",
                    "segment": seg_data,
                    "processed_duration": segment.end,
                    "total_duration": duration,
                    "percent": min(100.0, (segment.end / duration) * 100.0) if duration else 0.0,
                })

        return {
            "language": detected_lang,
            "language_probability": round(lang_prob, 4),
            "source_language": source_lang or detected_lang,
            "target_language": target_lang,
            "duration": round(duration, 3),
            "segments": segments,
        }

    @staticmethod
    def export_srt(segments: List[Dict[str, Any]], output_path: str, text_key: str = "text") -> None:
        """Export segments to SubRip (.srt) subtitle file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"{seg['id']}\n")
                f.write(
                    f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
                )
                f.write(f"{seg.get(text_key, seg.get('text', ''))}\n\n")

    @staticmethod
    def export_vtt(segments: List[Dict[str, Any]], output_path: str, text_key: str = "text") -> None:
        """Export segments to WebVTT (.vtt) subtitle file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                start_vtt = format_timestamp(seg['start']).replace(",", ".")
                end_vtt = format_timestamp(seg['end']).replace(",", ".")
                f.write(f"{seg['id']}\n")
                f.write(f"{start_vtt} --> {end_vtt}\n")
                f.write(f"{seg.get(text_key, seg.get('text', ''))}\n\n")

    @staticmethod
    def export_bilingual_srt(segments: List[Dict[str, Any]], output_path: str) -> None:
        """Export dual-language stacked subtitles (Original on top, Translated below)."""
        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"{seg['id']}\n")
                f.write(
                    f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
                )
                orig = seg.get("orig_text") or seg.get("text_en") or ""
                dub = seg.get("text") or ""
                if orig and orig != dub:
                    f.write(f"{orig}\n{dub}\n\n")
                else:
                    f.write(f"{dub}\n\n")

    @staticmethod
    def optimize_pacing(text: str, target_duration: float, target_lang: str = "en") -> str:
        """
        Auto-tune sentence phrasing and length to fit within the visual mouth movement window.
        Trims redundant filler words if words-per-second exceeds natural speech pace (> 3.5 wps).
        """
        words = text.strip().split()
        if not words or target_duration <= 0:
            return text

        words_per_sec = len(words) / target_duration
        if words_per_sec <= 3.2:
            return text  # Comfortable natural pace

        # Filter common English filler words if English
        fillers = {"um", "uh", "like", "you know", "actually", "basically", "literally", "well", "so"}
        filtered = [w for w in words if w.lower() not in fillers]
        if len(filtered) >= 2 and len(filtered) < len(words):
            return " ".join(filtered)

        return text

    @staticmethod
    def save_json(data: Dict[str, Any], output_path: str) -> None:
        """Save transcription and segments metadata to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
