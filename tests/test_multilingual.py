"""
Unit tests for All Indian and Global Languages catalog and translation functions.
"""

import pytest
from starlette.testclient import TestClient

from dubber.config import AVAILABLE_VOICES, SUPPORTED_LANGUAGES, DubbingConfig
from dubber.transcriber import translate_text
from server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_supported_languages_catalog():
    """Verify all Indian and Global languages exist in the catalog."""
    indian_langs = ["te", "hi", "ta", "kn", "ml", "mr", "bn", "gu", "ur", "en-IN"]
    global_langs = ["en", "en-GB", "es", "fr", "de", "ja", "ko", "zh", "ar", "pt", "ru", "it"]

    for code in indian_langs:
        assert code in SUPPORTED_LANGUAGES, f"Indian language '{code}' missing from catalog"
        assert SUPPORTED_LANGUAGES[code]["region"] == "Indian"

    for code in global_langs:
        assert code in SUPPORTED_LANGUAGES, f"Global language '{code}' missing from catalog"
        assert SUPPORTED_LANGUAGES[code]["region"] == "Global"


def test_every_language_has_valid_voice_pairs():
    """Verify that every language has male and female voices defined in AVAILABLE_VOICES."""
    for code, info in SUPPORTED_LANGUAGES.items():
        male_v = info["male_voice"]
        female_v = info["female_voice"]
        assert male_v in AVAILABLE_VOICES, f"Voice '{male_v}' for {code} not in AVAILABLE_VOICES"
        assert female_v in AVAILABLE_VOICES, f"Voice '{female_v}' for {code} not in AVAILABLE_VOICES"
        assert AVAILABLE_VOICES[male_v]["gender"] == "Male"
        assert AVAILABLE_VOICES[female_v]["gender"] == "Female"


def test_translate_text_english_identity():
    """Verify English text is unchanged."""
    text = "Hello world"
    assert translate_text(text, "en") == text
    assert translate_text(text, "en-US") == text


def test_translate_text_to_indian_language():
    """Verify translation to Telugu or Hindi produces non-empty translated string."""
    text = "Welcome to the festival"
    res_te = translate_text(text, "te")
    assert isinstance(res_te, str)
    assert len(res_te) > 0


def test_api_languages_endpoint(client):
    """Test /api/languages returns full list of Indian and global languages."""
    response = client.get("/api/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    langs = data["languages"]
    assert len(langs) >= 20

    regions = {l["region"] for l in langs}
    assert "Indian" in regions
    assert "Global" in regions
