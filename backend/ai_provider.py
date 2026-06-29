import json
import urllib.error
import urllib.request

from .config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from .database import database


def fallback_response(payload):
    messages = payload.get("messages") or []
    user_text = ""
    if messages:
        content = messages[-1].get("content", "")
        user_text = content if isinstance(content, str) else json.dumps(content)

    system = payload.get("system", "").lower()
    if "json array of strings" in system:
        return json.dumps(
            [
                "This connects strongly with the point you shared. Thanks for putting it into words.",
                "I like how this brings the main idea forward without overcomplicating it.",
                "That perspective feels useful, especially for people thinking through this topic.",
                "This is a thoughtful post and it gives people something real to respond to.",
                "The message is clear and grounded. Appreciate you sharing it.",
            ]
        )

    if "sentiment analysis expert" in system:
        text = user_text.lower()
        positive = any(word in text for word in ["good", "great", "love", "happy", "best", "success"])
        negative = any(word in text for word in ["bad", "sad", "angry", "hate", "failed", "worried"])
        sentiment = "positive" if positive and not negative else "negative" if negative else "neutral"
        return json.dumps(
            {
                "sentiment": sentiment,
                "confidence": 0.72,
                "sarcasm_detected": False,
                "tone": "informative",
                "topics": ["social media", "engagement"],
                "explanation": "Fallback analysis used because no Anthropic API key is configured.",
                "intensity": "moderate",
            }
        )

    if "content moderation assistant" in system:
        return "Fallback moderation note: the local keyword filter made the safety decision, and no cloud API key is configured."

    if "scheduling expert" in system:
        return json.dumps(
            {
                "slot": "Evening (6-8 PM)",
                "best_day": "weekday",
                "predicted_engagement": 180,
                "reasoning": "Fallback scheduling favors evening activity windows for social posts.",
                "tip": "Keep the caption specific and invite a short response.",
                "hashtag_suggestions": ["#AI", "#SocialMedia"],
            }
        )

    return (
        "Python backend is running and database logging is enabled. "
        "Set ANTHROPIC_API_KEY to get live Claude responses. "
        f"Your question was: {user_text[:300]}"
    )


def call_anthropic(payload):
    model = payload.get("model") or ANTHROPIC_MODEL
    request_payload = {
        "model": model,
        "max_tokens": int(payload.get("max_tokens") or 1000),
        "system": payload.get("system", ""),
        "messages": payload.get("messages") or [],
    }

    if not ANTHROPIC_API_KEY:
        text = fallback_response(request_payload)
        database.store_api_call("local-fallback", model, request_payload, text, "ok")
        return {"text": text, "provider": "local-fallback", "model": model}

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as res:
            data = json.loads(res.read().decode("utf-8"))
        text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
        database.store_api_call("anthropic", model, request_payload, text, "ok")
        return {"text": text, "provider": "anthropic", "model": model, "usage": data.get("usage")}
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")
        database.store_api_call("anthropic", model, request_payload, None, "error", error)
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {error[:400]}") from exc
