import base64
import json
import logging
import time
import urllib.error
import urllib.request

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from .database import database

logger = logging.getLogger("chatbot.ai")


def fallback_response(payload):
    import re
    from .models.comment_generator_model import comment_generator_model
    from .models.sentiment_model import sentiment_model
    from .models.text_features import extract_dynamic_topics, is_question_intent
    from .models.toxicity_model import toxicity_model

    messages = payload.get("messages") or []
    user_text = ""
    if messages:
        content = messages[-1].get("content", "")
        user_text = content if isinstance(content, str) else json.dumps(content)

    system = payload.get("system", "").lower()
    if "uploaded image" in user_text.lower() or "validated image" in user_text.lower():
        return (
            "🖼️ What's happening in the image:\n"
            "An uploaded photograph featuring a distinct visual subject with balanced composition.\n\n"
            "💬 Generated Comment:\n"
            "A thoughtful visual moment with an inviting energy. Thanks for sharing it with us."
        )

    if "sentiment analysis expert" in system:
        clean_text = user_text
        if clean_text.startswith('Analyze: "') and clean_text.endswith('"'):
            clean_text = clean_text[10:-1]
        result = sentiment_model.predict(clean_text)
        return json.dumps(result)

    if "content moderation assistant" in system:
        clean_text = user_text
        if clean_text.startswith('Text: "') and clean_text.endswith('"'):
            clean_text = clean_text[7:-1]
        result = toxicity_model.predict(clean_text)
        result["model"] = "ai-toxicity"
        return json.dumps(result)

    if "json array of strings" in system or "social media comment generator" in system:
        post_match = re.search(r'Post:\s*"(.*?)"(?:\n|$)', user_text, re.DOTALL)
        post_text = post_match.group(1) if post_match else user_text
        num_match = re.search(r'Generate\s+(\d+)\s+comment', user_text, re.IGNORECASE)
        num_comments = int(num_match.group(1)) if num_match else (5 if "json array of strings" in system else 1)
        sent_match = re.search(r'Sentiment:\s*([a-zA-Z]+)', user_text)
        sentiment_label = sent_match.group(1) if sent_match else "neutral"
        if "sentiment is negative" in system.lower() or sentiment_label.lower() == "negative":
            sentiment_label = "negative"
        toxic_match = re.search(r'Toxic:\s*(True|False)', user_text)
        is_toxic = toxic_match.group(1) == "True" if toxic_match else False
        lang_match = re.search(r'Language:\s*([a-zA-Z_]+)', user_text)
        lang_val = lang_match.group(1) if lang_match else None

        if "json array of strings" in system or num_comments > 1:
            multi = comment_generator_model.generate_multiple(
                post_text, sentiment=sentiment_label, toxicity=is_toxic, num_comments=num_comments, language=lang_val
            )
            return json.dumps(multi["comments"])
        else:
            single = comment_generator_model.predict(post_text, sentiment=sentiment_label, toxicity=is_toxic, language=lang_val)
            return json.dumps(single)

    if "engagement prediction expert" in system:
        words = user_text.split()
        w_count = len(words)
        h_count = user_text.count("#")
        sent_res = sentiment_model.predict(user_text)
        sent_label = sent_res.get("sentiment", "neutral")
        base_engagement = min(950, max(45, 120 + w_count * 3 + h_count * 25 + (60 if sent_label == "positive" else 0)))
        return json.dumps(
            {
                "model": "ai-engagement",
                "predicted_engagement": base_engagement,
                "features": {
                    "word_count": w_count,
                    "char_count": len(user_text),
                    "hashtag_count": h_count,
                    "sentiment": sent_label,
                },
                "source": "nlp-local-engine",
                "explanation": f"Engagement predicted from length ({w_count} words), hashtags ({h_count}), and {sent_label} sentiment.",
            }
        )

    if "scheduling expert" in system:
        topics = extract_dynamic_topics(user_text)
        tags = [f"#{t.replace(' ', '')}" for t in topics[:2]] or ["#SocialMedia", "#Update"]
        return json.dumps(
            {
                "slot": "Evening (6-8 PM)",
                "best_day": "weekday",
                "predicted_engagement": 180,
                "reasoning": f"Peak social engagement window for {topics[0] if topics else 'this topic'}.",
                "tip": "Keep the caption specific and invite a short response.",
                "hashtag_suggestions": tags,
            }
        )

    u_low = user_text.lower()
    if any(k in u_low for k in ["project", "cse400", "thesis", "bot", "what can you do", "features", "overview"]):
        return (
            "This Intelligent Social Bot (BRAC CSE400 thesis project) provides an end-to-end AI social workspace:\n"
            "• Contextual Comment Generation: Tailored archetypes for questions, discussions, food, pets, tech, milestones, and sports (English & Bengali).\n"
            "• Sentiment Analysis: Deep BERT classification with sarcasm detection, contrastive clause evaluation, and tone recognition.\n"
            "• Toxicity Filtering: Robust target-discriminating moderation that separates personal attacks from benign situational frustration.\n"
            "• Scheduling & Analytics: Optimal time-slot prediction and engagement scoring.\n"
            "• Meta Graph API Integration: Direct Facebook Page post fetching and commenting.\n"
            "• SSLCommerz Billing: 4 tier packages with hosted checkout and instant IPN verification."
        )
    if any(k in u_low for k in ["sentiment", "emotion", "vader", "bert sentiment"]):
        return (
            "The sentiment pipeline combines fast VADER lexicon scoring with deep BERT classification. "
            "It analyzes emotional valence, recognizes contrastive conjunctions ('X is good but Y is bad'), "
            "identifies tone (excited, grateful, frustrated, angry, inquisitive), and extracts dynamic entities in English and Bengali."
        )
    if any(k in u_low for k in ["toxicity", "moderate", "filter", "safety", "block"]):
        return (
            "The content moderation engine flags personal insults, violent threats, vulgarity, and deceptive scams. "
            "Importantly, it uses target-discrimination so expressions of frustration at objects (e.g. 'stupid bug', 'hate traffic') "
            "are not falsely flagged, while genuine abuse and threats are blocked with clear explanations."
        )
    if any(k in u_low for k in ["payment", "ssl", "sslcommerz", "pricing", "package"]):
        return (
            "The platform supports SSLCommerz payment integration featuring 4 bot packages (Free, Starter ৳499, Pro ৳1,499, Agency ৳3,999). "
            "It supports both hosted checkout and server-to-server IPN validation to unlock premium features."
        )
    if any(k in u_low for k in ["facebook", "meta", "page"]):
        return (
            "The bot integrates with Meta Graph API v22.0. Pages can authenticate via OAuth 2.0 to fetch posts, "
            "read comments, publish feed posts, and generate automated conversational replies."
        )

    # General conversation fallback
    if is_question_intent(user_text):
        topics = extract_dynamic_topics(user_text)
        topic_str = topics[0] if topics and topics[0] != "social update" else "this topic"
        return (
            f"Regarding {topic_str}: A solid approach is to break down your objective into measurable steps, "
            "validate each step with real-world examples, and iterate based on feedback. "
            "If you need specific guidance on sentiment, toxicity moderation, or comment generation, feel free to ask!"
        )

    return (
        f"I've analyzed your input regarding '{user_text[:120]}...'. "
        "The local NLP pipeline is actively processing your request across sentiment, safety moderation, and contextual response generation."
    )



def call_openai(payload):
    model = payload.get("model") or OPENAI_MODEL
    messages = list(payload.get("messages") or [])
    system = payload.get("system")

    request_payload = {
        "model": model,
        "input": messages,
        "max_output_tokens": int(payload.get("max_tokens") or 1000),
        # Do not retain provider-side response state; conversations are stored
        # locally by this app when the request succeeds.
        "store": False,
    }
    if system:
        request_payload["instructions"] = system

    if not OPENAI_API_KEY:
        text = fallback_response({"system": system, "messages": messages})
        database.store_api_call("local-fallback", model, request_payload, text, "ok")
        return {"text": text, "provider": "local-fallback", "model": model}

    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as res:
            data = json.loads(res.read().decode("utf-8"))
        text = data.get("output_text", "")
        if not text:
            text = "".join(
                part.get("text", "")
                for item in data.get("output", [])
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
        if not text:
            raise RuntimeError("OpenAI returned no text output.")
        database.store_api_call("openai", model, request_payload, text, "ok")
        return {"text": text, "provider": "openai", "model": model, "usage": data.get("usage")}
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")
        database.store_api_call("openai", model, request_payload, None, "error", error)
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {error[:400]}") from exc
    except urllib.error.URLError as exc:
        error = str(exc.reason)
        database.store_api_call("openai", model, request_payload, None, "error", error)
        raise RuntimeError(f"OpenAI connection error: {error}") from exc


def call_gemini(payload):
    """Call Google Gemini API via standard library HTTP with retry and fallback models."""
    requested_model = payload.get("model") or GEMINI_MODEL
    if not GEMINI_API_KEY:
        text = fallback_response(payload)
        database.store_api_call("local-fallback", requested_model, payload, text, "ok")
        return {"text": text, "provider": "local-fallback", "model": "local"}

    system = payload.get("system", "")
    messages = payload.get("messages") or []
    user_prompt = ""
    if messages:
        last = messages[-1].get("content", "")
        user_prompt = last if isinstance(last, str) else json.dumps(last)

    full_prompt = f"{system}\n\n{user_prompt}" if system else user_prompt
    request_payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": float(payload.get("temperature", 0.7)),
            "maxOutputTokens": int(payload.get("max_tokens", 800)),
        },
    }

    models_to_try = [
        requested_model,
        "gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    seen = set()
    candidate_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

    last_error = None
    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        for attempt in range(2):
            req = urllib.request.Request(
                url,
                data=json.dumps(request_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as res:
                    data = json.loads(res.read().decode("utf-8"))
                candidates = data.get("candidates") or []
                text = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    database.store_api_call("gemini", model, request_payload, text, "ok")
                    return {"text": text, "provider": "gemini", "model": model}
            except urllib.error.HTTPError as exc:
                error = exc.read().decode("utf-8", errors="replace")
                last_error = f"Gemini API HTTP {exc.code}: {error[:300]}"
                logger.warning("Gemini (%s, attempt %d) HTTP %d: %s", model, attempt + 1, exc.code, error[:200])
                if exc.code in {429, 500, 502, 503, 504} and attempt == 0:
                    time.sleep(1.0)
                    continue
                break
            except urllib.error.URLError as exc:
                error = str(exc.reason)
                last_error = f"Gemini connection error: {error}"
                logger.warning("Gemini (%s, attempt %d) URLError: %s", model, attempt + 1, error)
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                break
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Gemini (%s, attempt %d) exception: %s", model, attempt + 1, exc)
                break

    database.store_api_call("gemini", requested_model, request_payload, None, "error", last_error or "Unknown error")
    raise RuntimeError(last_error or "Gemini API unavailable across all models.")


def call_ai(payload):
    provider = str(payload.get("provider", "gemini" if GEMINI_API_KEY else "anthropic")).lower()
    try:
        if provider in {"gemini", "google"}:
            return call_gemini(payload)
        if provider in {"anthropic", "claude"}:
            return call_anthropic(payload)
        if provider in {"openai", "chatgpt"}:
            return call_openai(payload)
    except RuntimeError as exc:
        logger.warning("Primary provider '%s' failed (%s). Checking failovers...", provider, exc)
        if GEMINI_API_KEY and provider not in {"gemini", "google"}:
            try:
                return call_gemini(payload)
            except Exception:
                pass
        text = fallback_response(payload)
        return {"text": text, "provider": "local-fallback", "model": "local"}
    raise ValueError(f"Unknown provider '{provider}'. Use 'gemini', 'anthropic', or 'openai'.")


def call_vision(image_bytes, media_type, prompt, provider="openai", filename=""):
    """Ask a configured vision provider to interpret the uploaded image."""
    provider = str(provider or "gemini").lower()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{media_type};base64,{encoded}"

    # Route to Gemini Vision if requested or if OpenAI/Anthropic keys are missing
    if provider in {"gemini", "google"} or (GEMINI_API_KEY and not OPENAI_API_KEY and not ANTHROPIC_API_KEY):
        requested_model = GEMINI_MODEL
        if not GEMINI_API_KEY:
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, filename=filename, user_prompt=prompt)
            database.store_api_call("local-fallback-vision", requested_model, {"model": requested_model, "prompt": prompt}, local_res["text"], "ok")
            return {**local_res, "provider": "local-fallback", "model": "local"}

        models_to_try = [
            requested_model,
            "gemini-3.8-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
        seen = set()
        candidate_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

        request_payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": media_type or "image/png",
                            "data": encoded
                        }
                    }
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": 700,
                "temperature": 0.3
            }
        }

        for c_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{c_model}:generateContent?key={GEMINI_API_KEY}"
            for attempt in range(2):
                try:
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(request_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=30) as res:
                        data = json.loads(res.read().decode("utf-8"))
                    candidates = data.get("candidates") or []
                    text = ""
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts).strip()
                    if text:
                        database.store_api_call("gemini-vision", c_model, {"prompt": prompt}, text, "ok")
                        return {"text": text, "provider": "gemini-vision", "model": c_model}
                except urllib.error.HTTPError as exc:
                    err_body = exc.read().decode("utf-8", errors="replace")
                    logger.warning("Gemini vision (%s, attempt %d) HTTP %d: %s", c_model, attempt + 1, exc.code, err_body[:200])
                    if exc.code in {429, 500, 502, 503, 504} and attempt == 0:
                        time.sleep(1.0)
                        continue
                    break
                except Exception as exc:
                    logger.warning("Gemini vision (%s, attempt %d) error: %s", c_model, attempt + 1, exc)
                    if attempt == 0:
                        time.sleep(1.0)
                        continue
                    break

        logger.warning("All Gemini vision models failed or unavailable. Falling back to local vision engine.")
        from .models.vision_model import local_vision_model
        local_res = local_vision_model.process_image(image_bytes, filename=filename, user_prompt=prompt)
        return {**local_res, "provider": "local-fallback", "model": "local"}

    if provider in {"openai", "chatgpt"}:
        model = OPENAI_MODEL
        if not OPENAI_API_KEY:
            if GEMINI_API_KEY:
                return call_vision(image_bytes, media_type, prompt, provider="gemini", filename=filename)
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, filename=filename, user_prompt=prompt)
            database.store_api_call("local-fallback-vision", model, {"model": model, "prompt": prompt}, local_res["text"], "ok")
            return {**local_res, "provider": "local-fallback", "model": "local"}
        request_payload = {
            "model": model,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ]}],
            "max_output_tokens": 250,
            "store": False,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                data = json.loads(res.read().decode("utf-8"))
            text = data.get("output_text", "")
            if not text:
                raise RuntimeError("OpenAI returned no image comment.")
            database.store_api_call("openai-vision", model, {"model": model, "prompt": prompt}, text, "ok")
            return {"text": text.strip(), "provider": "openai-vision", "model": model}
        except (urllib.error.HTTPError, urllib.error.URLError, Exception) as exc:
            logger.warning("OpenAI vision failed (%s), falling back to available engine", exc)
            if GEMINI_API_KEY:
                return call_vision(image_bytes, media_type, prompt, provider="gemini", filename=filename)
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, filename=filename, user_prompt=prompt)
            return {**local_res, "provider": "local-fallback", "model": "local"}

    if provider in {"anthropic", "claude"}:
        model = ANTHROPIC_MODEL
        if not ANTHROPIC_API_KEY:
            if GEMINI_API_KEY:
                return call_vision(image_bytes, media_type, prompt, provider="gemini", filename=filename)
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, filename=filename, user_prompt=prompt)
            database.store_api_call("local-fallback-vision", model, {"model": model, "prompt": prompt}, local_res["text"], "ok")
            return {**local_res, "provider": "local-fallback", "model": "local"}
        request_payload = {
            "model": model,
            "max_tokens": 250,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}},
                {"type": "text", "text": prompt},
            ]}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                data = json.loads(res.read().decode("utf-8"))
            text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
            if not text:
                raise RuntimeError("Anthropic returned no image comment.")
            database.store_api_call("anthropic-vision", model, {"model": model, "prompt": prompt}, text, "ok")
            return {"text": text.strip(), "provider": "anthropic-vision", "model": model}
        except (urllib.error.HTTPError, urllib.error.URLError, Exception) as exc:
            logger.warning("Anthropic vision failed (%s), falling back to available engine", exc)
            if GEMINI_API_KEY:
                return call_vision(image_bytes, media_type, prompt, provider="gemini", filename=filename)
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, filename=filename, user_prompt=prompt)
            return {**local_res, "provider": "local-fallback", "model": "local"}

    from .models.vision_model import local_vision_model
    local_res = local_vision_model.process_image(image_bytes, filename=filename, user_prompt=prompt)
    return {**local_res, "provider": "local-fallback", "model": "local"}


def call_anthropic(payload):
    model = payload.get("model") or ANTHROPIC_MODEL
    messages = list(payload.get("messages") or [])
    system = payload.get("system")
    request_payload = {
        "model": model,
        "max_tokens": int(payload.get("max_tokens") or 1000),
        "temperature": float(payload.get("temperature") or 0.7),
        "messages": messages,
    }
    if system:
        request_payload["system"] = system

    if not ANTHROPIC_API_KEY:
        text = fallback_response({"system": system, "messages": messages})
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
    except urllib.error.URLError as exc:
        error = str(exc.reason)
        database.store_api_call("anthropic", model, request_payload, None, "error", error)
        raise RuntimeError(f"Anthropic connection error: {error}") from exc
