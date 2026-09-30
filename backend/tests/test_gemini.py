import json
from unittest.mock import MagicMock, patch
from backend.ai_provider import call_gemini, call_ai, call_vision


def test_call_gemini_fallback_when_no_key(monkeypatch):
    from backend import config, ai_provider
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(ai_provider, "GEMINI_API_KEY", "")

    result = call_gemini({"messages": [{"role": "user", "content": "Hello"}]})
    assert result["provider"] == "local-fallback"
    assert result["text"]


def test_call_gemini_mocked_success(monkeypatch):
    from backend import config, ai_provider
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test_key_123")
    monkeypatch.setattr(ai_provider, "GEMINI_API_KEY", "test_key_123")

    fake_response = json.dumps({
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Gemini response text"}]
                }
            }
        ]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = call_gemini({"messages": [{"role": "user", "content": "Test prompt"}]})
        assert res["provider"] == "gemini"
        assert res["text"] == "Gemini response text"


def test_call_ai_routes_to_gemini(monkeypatch):
    from backend import config, ai_provider
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test_key_123")
    monkeypatch.setattr(ai_provider, "GEMINI_API_KEY", "test_key_123")

    fake_response = json.dumps({
        "candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = call_ai({"provider": "gemini", "messages": [{"role": "user", "content": "Hi"}]})
        assert res["provider"] == "gemini"
        assert res["text"] == "Hello from Gemini"
