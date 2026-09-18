"""
Configuration management for the Automated Video Dubbing System.
Supports all Indian and Global languages, multi-speaker voice pairs, and system tooling.
"""

from dataclasses import dataclass, field
import os
import shutil
from typing import Dict, List, Optional


SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    # --- Indian Languages ---
    "te": {
        "name": "Telugu (తెలుగు)",
        "locale": "te-IN",
        "male_voice": "te-IN-MohanNeural",
        "female_voice": "te-IN-ShrutiNeural",
        "region": "Indian",
    },
    "hi": {
        "name": "Hindi (हिन्दी)",
        "locale": "hi-IN",
        "male_voice": "hi-IN-MadhurNeural",
        "female_voice": "hi-IN-SwaraNeural",
        "region": "Indian",
    },
    "ta": {
        "name": "Tamil (தமிழ்)",
        "locale": "ta-IN",
        "male_voice": "ta-IN-ValluvarNeural",
        "female_voice": "ta-IN-PallaviNeural",
        "region": "Indian",
    },
    "kn": {
        "name": "Kannada (ಕನ್ನಡ)",
        "locale": "kn-IN",
        "male_voice": "kn-IN-GaganNeural",
        "female_voice": "kn-IN-SapnaNeural",
        "region": "Indian",
    },
    "ml": {
        "name": "Malayalam (മലയാളം)",
        "locale": "ml-IN",
        "male_voice": "ml-IN-MidhunNeural",
        "female_voice": "ml-IN-SobhanaNeural",
        "region": "Indian",
    },
    "mr": {
        "name": "Marathi (मराठी)",
        "locale": "mr-IN",
        "male_voice": "mr-IN-ManoharNeural",
        "female_voice": "mr-IN-AarohiNeural",
        "region": "Indian",
    },
    "bn": {
        "name": "Bengali (বাংলা)",
        "locale": "bn-IN",
        "male_voice": "bn-IN-BashkarNeural",
        "female_voice": "bn-IN-TanishaaNeural",
        "region": "Indian",
    },
    "gu": {
        "name": "Gujarati (ગુજરાતી)",
        "locale": "gu-IN",
        "male_voice": "gu-IN-NiranjanNeural",
        "female_voice": "gu-IN-DhwaniNeural",
        "region": "Indian",
    },
    "ur": {
        "name": "Urdu (اردو)",
        "locale": "ur-IN",
        "male_voice": "ur-IN-SalmanNeural",
        "female_voice": "ur-IN-GulNeural",
        "region": "Indian",
    },
    "en-IN": {
        "name": "English (India)",
        "locale": "en-IN",
        "male_voice": "en-IN-PrabhatNeural",
        "female_voice": "en-IN-NeerjaNeural",
        "region": "Indian",
    },

    # --- Global Languages ---
    "en": {
        "name": "English (US)",
        "locale": "en-US",
        "male_voice": "en-US-ChristopherNeural",
        "female_voice": "en-US-JennyNeural",
        "region": "Global",
    },
    "en-GB": {
        "name": "English (UK)",
        "locale": "en-GB",
        "male_voice": "en-GB-RyanNeural",
        "female_voice": "en-GB-SoniaNeural",
        "region": "Global",
    },
    "es": {
        "name": "Spanish (Español)",
        "locale": "es-ES",
        "male_voice": "es-ES-AlvaroNeural",
        "female_voice": "es-ES-ElviraNeural",
        "region": "Global",
    },
    "fr": {
        "name": "French (Français)",
        "locale": "fr-FR",
        "male_voice": "fr-FR-HenriNeural",
        "female_voice": "fr-FR-DeniseNeural",
        "region": "Global",
    },
    "de": {
        "name": "German (Deutsch)",
        "locale": "de-DE",
        "male_voice": "de-DE-ConradNeural",
        "female_voice": "de-DE-KatjaNeural",
        "region": "Global",
    },
    "ja": {
        "name": "Japanese (日本語)",
        "locale": "ja-JP",
        "male_voice": "ja-JP-KeitaNeural",
        "female_voice": "ja-JP-NanamiNeural",
        "region": "Global",
    },
    "ko": {
        "name": "Korean (한국어)",
        "locale": "ko-KR",
        "male_voice": "ko-KR-InJoonNeural",
        "female_voice": "ko-KR-SunHiNeural",
        "region": "Global",
    },
    "zh": {
        "name": "Chinese (中文)",
        "locale": "zh-CN",
        "male_voice": "zh-CN-YunxiNeural",
        "female_voice": "zh-CN-XiaoxiaoNeural",
        "region": "Global",
    },
    "ar": {
        "name": "Arabic (العربية)",
        "locale": "ar-SA",
        "male_voice": "ar-SA-HamedNeural",
        "female_voice": "ar-SA-ZariyahNeural",
        "region": "Global",
    },
    "pt": {
        "name": "Portuguese (Português)",
        "locale": "pt-BR",
        "male_voice": "pt-BR-AntonioNeural",
        "female_voice": "pt-BR-FranciscaNeural",
        "region": "Global",
    },
    "ru": {
        "name": "Russian (Русский)",
        "locale": "ru-RU",
        "male_voice": "ru-RU-DmitryNeural",
        "female_voice": "ru-RU-SvetlanaNeural",
        "region": "Global",
    },
    "it": {
        "name": "Italian (Italiano)",
        "locale": "it-IT",
        "male_voice": "it-IT-DiegoNeural",
        "female_voice": "it-IT-ElsaNeural",
        "region": "Global",
    },
}


AVAILABLE_VOICES: Dict[str, Dict[str, str]] = {
    # English (US / UK / IN)
    "en-US-ChristopherNeural": {
        "gender": "Male",
        "locale": "en-US",
        "language": "English (US)",
        "description": "Natural, clear American male voice (Recommended for documentaries/tutorials)",
    },
    "en-US-JennyNeural": {
        "gender": "Female",
        "locale": "en-US",
        "language": "English (US)",
        "description": "Natural, articulate American female voice (Recommended for general content)",
    },
    "en-US-GuyNeural": {
        "gender": "Male",
        "locale": "en-US",
        "language": "English (US)",
        "description": "Conversational, energetic American male voice",
    },
    "en-US-AriaNeural": {
        "gender": "Female",
        "locale": "en-US",
        "language": "English (US)",
        "description": "Professional, engaging American female voice",
    },
    "en-GB-RyanNeural": {
        "gender": "Male",
        "locale": "en-GB",
        "language": "English (UK)",
        "description": "British male accent, refined and clear",
    },
    "en-GB-SoniaNeural": {
        "gender": "Female",
        "locale": "en-GB",
        "language": "English (UK)",
        "description": "British female accent, warm and professional",
    },
    "en-IN-PrabhatNeural": {
        "gender": "Male",
        "locale": "en-IN",
        "language": "English (India)",
        "description": "Indian English male accent, natural and clear",
    },
    "en-IN-NeerjaNeural": {
        "gender": "Female",
        "locale": "en-IN",
        "language": "English (India)",
        "description": "Indian English female accent, expressive and warm",
    },

    # Telugu
    "te-IN-MohanNeural": {
        "gender": "Male",
        "locale": "te-IN",
        "language": "Telugu (తెలుగు)",
        "description": "Natural Telugu male neural voice",
    },
    "te-IN-ShrutiNeural": {
        "gender": "Female",
        "locale": "te-IN",
        "language": "Telugu (తెలుగు)",
        "description": "Warm Telugu female neural voice",
    },

    # Hindi
    "hi-IN-MadhurNeural": {
        "gender": "Male",
        "locale": "hi-IN",
        "language": "Hindi (हिन्दी)",
        "description": "Expressive Hindi male neural voice",
    },
    "hi-IN-SwaraNeural": {
        "gender": "Female",
        "locale": "hi-IN",
        "language": "Hindi (हिन्दी)",
        "description": "Natural Hindi female neural voice",
    },

    # Tamil
    "ta-IN-ValluvarNeural": {
        "gender": "Male",
        "locale": "ta-IN",
        "language": "Tamil (தமிழ்)",
        "description": "Natural Tamil male neural voice",
    },
    "ta-IN-PallaviNeural": {
        "gender": "Female",
        "locale": "ta-IN",
        "language": "Tamil (தமிழ்)",
        "description": "Expressive Tamil female neural voice",
    },

    # Kannada
    "kn-IN-GaganNeural": {
        "gender": "Male",
        "locale": "kn-IN",
        "language": "Kannada (ಕನ್ನಡ)",
        "description": "Clear Kannada male neural voice",
    },
    "kn-IN-SapnaNeural": {
        "gender": "Female",
        "locale": "kn-IN",
        "language": "Kannada (ಕನ್ನಡ)",
        "description": "Warm Kannada female neural voice",
    },

    # Malayalam
    "ml-IN-MidhunNeural": {
        "gender": "Male",
        "locale": "ml-IN",
        "language": "Malayalam (മലയാളം)",
        "description": "Natural Malayalam male neural voice",
    },
    "ml-IN-SobhanaNeural": {
        "gender": "Female",
        "locale": "ml-IN",
        "language": "Malayalam (മലയാളം)",
        "description": "Expressive Malayalam female neural voice",
    },

    # Marathi
    "mr-IN-ManoharNeural": {
        "gender": "Male",
        "locale": "mr-IN",
        "language": "Marathi (मराठी)",
        "description": "Natural Marathi male neural voice",
    },
    "mr-IN-AarohiNeural": {
        "gender": "Female",
        "locale": "mr-IN",
        "language": "Marathi (मराठी)",
        "description": "Clear Marathi female neural voice",
    },

    # Bengali
    "bn-IN-BashkarNeural": {
        "gender": "Male",
        "locale": "bn-IN",
        "language": "Bengali (বাংলা)",
        "description": "Natural Bengali male neural voice",
    },
    "bn-IN-TanishaaNeural": {
        "gender": "Female",
        "locale": "bn-IN",
        "language": "Bengali (বাংলা)",
        "description": "Expressive Bengali female neural voice",
    },

    # Gujarati
    "gu-IN-NiranjanNeural": {
        "gender": "Male",
        "locale": "gu-IN",
        "language": "Gujarati (ગુજરાતી)",
        "description": "Natural Gujarati male neural voice",
    },
    "gu-IN-DhwaniNeural": {
        "gender": "Female",
        "locale": "gu-IN",
        "language": "Gujarati (ગુજરાતી)",
        "description": "Warm Gujarati female neural voice",
    },

    # Urdu
    "ur-IN-SalmanNeural": {
        "gender": "Male",
        "locale": "ur-IN",
        "language": "Urdu (اردو)",
        "description": "Natural Urdu male neural voice",
    },
    "ur-IN-GulNeural": {
        "gender": "Female",
        "locale": "ur-IN",
        "language": "Urdu (اردو)",
        "description": "Clear Urdu female neural voice",
    },

    # Spanish
    "es-ES-AlvaroNeural": {
        "gender": "Male",
        "locale": "es-ES",
        "language": "Spanish (Español)",
        "description": "Natural Spanish male neural voice",
    },
    "es-ES-ElviraNeural": {
        "gender": "Female",
        "locale": "es-ES",
        "language": "Spanish (Español)",
        "description": "Warm Spanish female neural voice",
    },

    # French
    "fr-FR-HenriNeural": {
        "gender": "Male",
        "locale": "fr-FR",
        "language": "French (Français)",
        "description": "Natural French male neural voice",
    },
    "fr-FR-DeniseNeural": {
        "gender": "Female",
        "locale": "fr-FR",
        "language": "French (Français)",
        "description": "Expressive French female neural voice",
    },

    # German
    "de-DE-ConradNeural": {
        "gender": "Male",
        "locale": "de-DE",
        "language": "German (Deutsch)",
        "description": "Natural German male neural voice",
    },
    "de-DE-KatjaNeural": {
        "gender": "Female",
        "locale": "de-DE",
        "language": "German (Deutsch)",
        "description": "Clear German female neural voice",
    },

    # Japanese
    "ja-JP-KeitaNeural": {
        "gender": "Male",
        "locale": "ja-JP",
        "language": "Japanese (日本語)",
        "description": "Natural Japanese male neural voice",
    },
    "ja-JP-NanamiNeural": {
        "gender": "Female",
        "locale": "ja-JP",
        "language": "Japanese (日本語)",
        "description": "Expressive Japanese female neural voice",
    },

    # Korean
    "ko-KR-InJoonNeural": {
        "gender": "Male",
        "locale": "ko-KR",
        "language": "Korean (한국어)",
        "description": "Natural Korean male neural voice",
    },
    "ko-KR-SunHiNeural": {
        "gender": "Female",
        "locale": "ko-KR",
        "language": "Korean (한국어)",
        "description": "Warm Korean female neural voice",
    },

    # Chinese
    "zh-CN-YunxiNeural": {
        "gender": "Male",
        "locale": "zh-CN",
        "language": "Chinese (中文)",
        "description": "Natural Mandarin male neural voice",
    },
    "zh-CN-XiaoxiaoNeural": {
        "gender": "Female",
        "locale": "zh-CN",
        "language": "Chinese (中文)",
        "description": "Warm Mandarin female neural voice",
    },

    # Arabic
    "ar-SA-HamedNeural": {
        "gender": "Male",
        "locale": "ar-SA",
        "language": "Arabic (العربية)",
        "description": "Natural Arabic male neural voice",
    },
    "ar-SA-ZariyahNeural": {
        "gender": "Female",
        "locale": "ar-SA",
        "language": "Arabic (العربية)",
        "description": "Expressive Arabic female neural voice",
    },

    # Portuguese
    "pt-BR-AntonioNeural": {
        "gender": "Male",
        "locale": "pt-BR",
        "language": "Portuguese (Português)",
        "description": "Natural Brazilian Portuguese male voice",
    },
    "pt-BR-FranciscaNeural": {
        "gender": "Female",
        "locale": "pt-BR",
        "language": "Portuguese (Português)",
        "description": "Warm Brazilian Portuguese female voice",
    },

    # Russian
    "ru-RU-DmitryNeural": {
        "gender": "Male",
        "locale": "ru-RU",
        "language": "Russian (Русский)",
        "description": "Natural Russian male neural voice",
    },
    "ru-RU-SvetlanaNeural": {
        "gender": "Female",
        "locale": "ru-RU",
        "language": "Russian (Русский)",
        "description": "Expressive Russian female neural voice",
    },

    # Italian
    "it-IT-DiegoNeural": {
        "gender": "Male",
        "locale": "it-IT",
        "language": "Italian (Italiano)",
        "description": "Natural Italian male neural voice",
    },
    "it-IT-ElsaNeural": {
        "gender": "Female",
        "locale": "it-IT",
        "language": "Italian (Italiano)",
        "description": "Warm Italian female neural voice",
    },
}


def find_ffmpeg_executable(custom_path: Optional[str] = None) -> str:
    """Find a valid FFmpeg executable path across system PATH, env vars, or imageio-ffmpeg."""
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            dir_name = os.path.dirname(exe)
            alias_path = os.path.join(dir_name, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if not os.path.isfile(alias_path):
                try:
                    shutil.copyfile(exe, alias_path)
                except Exception:
                    pass
            if os.path.isfile(alias_path):
                return alias_path
            return exe
    except Exception:
        pass

    raise RuntimeError(
        "FFmpeg executable not found! Please install FFmpeg or install imageio-ffmpeg."
    )


@dataclass
class DubbingConfig:
    """Master configuration for a video dubbing pipeline run."""

    # Video input & output
    output_dir: str = "output"
    temp_dir: str = os.path.join("output", "temp")
    keep_temp: bool = False

    # Language & Translation
    source_lang: str = "auto"  # 'auto' for auto-detection or specific code (e.g. te, hi, en, es, etc.)
    target_lang: str = "en"    # e.g., 'en', 'te', 'hi', 'ta', 'es', 'fr', 'de', etc.

    # Whisper Transcription & Translation
    whisper_model: str = "small"  # options: tiny, base, small, medium, large-v3
    whisper_device: str = "cpu"   # cpu or cuda
    compute_type: str = "int8"    # int8 is fast and memory efficient for CPU
    models_dir: str = "models"

    # Neural Speech Synthesis
    voice: str = "en-US-ChristopherNeural"
    auto_gender: bool = True  # Automatically detect speaker gender and adapt voice
    speaker_mode: str = "auto"  # 'auto' (person-matching lock), 'male_only', 'female_only', 'multi_speaker'
    male_voice: str = "en-US-ChristopherNeural"  # Male neural voice
    female_voice: str = "en-US-JennyNeural"      # Female neural voice
    voice_rate: str = "+0%"
    voice_pitch: str = "+0Hz"
    voice_volume: str = "+0%"

    # Speaker Voice Preservation & Conversion (Phase 3)
    voice_preservation_mode: str = "adaptive_prosody"  # 'adaptive_prosody', 'formant_transfer', 'none'
    enable_voice_preservation: bool = True

    # Timing Synchronization & AI Lip Sync (Phase 4)
    lip_sync_mode: str = "visual_adaptive"  # 'visual_adaptive', 'strict_lock', 'natural'
    enable_lipsync: bool = True             # Run AI Lip Synchronization on speaking faces
    lipsync_model: str = "wav2lip"          # 'wav2lip', 'phoneme_morphing', 'visual_adaptive'
    lipsync_batch_size: int = 4             # Optimized for 8 GB RAM laptops
    max_stretch_factor: float = 2.20  # Max speech speed-up (up to 2.2x via chained atempo)
    min_stretch_factor: float = 0.85  # Max speech slow-down (15%)
    min_silence_gap: float = 0.05     # Minimum silence gap between segments

    # Audio Mixing & Subtitle Studio
    mix_original_volume: float = 0.0  # Ducked background volume (0.0 = clean dub)
    embed_subtitles: bool = True       # Generate and mux subtitles
    subtitle_mode: str = "soft"       # 'soft' (CC stream), 'burn' (hardsubs), 'dual' (bilingual), 'none'
    proofread_mode: bool = False      # Pause after transcription to allow user proofreading & editing

    # System Tools
    ffmpeg_path: Optional[str] = None

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        if not self.ffmpeg_path:
            try:
                self.ffmpeg_path = find_ffmpeg_executable()
            except RuntimeError:
                self.ffmpeg_path = "ffmpeg"


# Alias for backward compatibility
DubberConfig = DubbingConfig
