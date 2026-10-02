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

    if "json array of strings" in system or "comment writer" in system or "comment generator" in system:
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

    return answer_assistant_query(user_text, system)


def answer_assistant_query(user_text, system=""):
    import re
    text = (user_text or "").strip()
    if not text:
        return "Hello! How can I assist you today? Feel free to ask me anything about comment generation, sentiment analysis, toxicity filtering, or scheduling!"

    t_low = text.lower()
    clean = re.sub(r'[^\w\s]', '', t_low).strip()
    words = clean.split()

    # 1. Polite Gratitude & Closures
    gratitude_words = {"thank", "thanks", "thx", "appreciate", "helpful", "ধন্যবাদ"}
    if any(gw in t_low for gw in gratitude_words) and len(words) <= 8:
        return "You're very welcome! 😊 Feel free to ask if you need help with anything else."

    goodbye_words = {"bye", "goodbye", "cya", "see you", "take care", "বিদায়"}
    if any(gw in t_low for gw in goodbye_words) and len(words) <= 6:
        return "Goodbye! 👋 Have a great day and happy posting!"

    acknowledgments = {"ok", "okay", "alright", "got it", "cool", "great", "awesome", "nice", "perfect", "understood"}
    if clean in acknowledgments or (len(words) <= 3 and any(w in acknowledgments for w in words)):
        return "Glad that was helpful! Let me know what you'd like to explore next."

    # 2. Greetings Analysis
    is_salam = any(p in t_low for p in ['assalamu alaikum', 'assalam alaikum', 'assalamualaikum', 'as-salamu alaykum', 'salam', 'salaam', 'সালাম', 'আসসালামু আলাইকুম'])
    is_bengali_greeting = any(bg in t_low for bg in ['হ্যালো', 'কেমন আছেন', 'কেমন আছো', 'কি খবর', 'নমস্কার', 'শুভ সকাল', 'শুভ সন্ধ্যা'])
    is_how_are_you = any(h in t_low for h in ['how are you', 'how are u', 'how r u', 'how are you doing', 'how do you do', "how's it going", 'hows it going', "what's up", 'whats up', 'how have you been'])
    basic_greetings = {'hi', 'hello', 'hey', 'heya', 'heyy', 'howdy', 'yo', 'greetings', 'good morning', 'good afternoon', 'good evening', 'good day'}
    is_basic_greeting = clean in basic_greetings or (words and words[0] in basic_greetings and len(words) <= 3)

    # Pure Greetings
    if is_salam and len(words) <= 4:
        if any(c in text for c in ['সালাম', 'আসসালামু']):
            return "ওয়ালাইকুম আসসালাম! আমি আপনার এআই সোশ্যাল মিডিয়া সহকারী। কমেন্ট তৈরি, সেন্টিমেন্ট বা টক্সিসিটি অ্যানালাইসিস সম্পর্কিত যেকোনো প্রশ্ন করতে পারেন।"
        return "Walaikum Assalam! 👋 I am your AI Social Media Assistant. How can I assist you today? You can ask me anything about comment generation, sentiment analysis, toxicity filtering, or scheduling!"

    if is_bengali_greeting and len(words) <= 5:
        return "হ্যালো! আমি ভালো আছি, ধন্যবাদ! আমি আপনার এআই সহকারী। কমেন্ট জেনারেশন, সেন্টিমেন্ট অ্যানালাইসিস, কিংবা শিডিউলিং নিয়ে যেকোনো প্রশ্ন করতে পারেন।"

    if is_how_are_you and len(words) <= 6:
        return "I'm doing great, thank you for asking! 😊 I'm ready to help you analyze posts, generate comments, moderate toxicity, or schedule content. What would you like to work on today?"

    if is_basic_greeting:
        return "Hello! 👋 I'm your AI Social Assistant. How can I assist you today? Feel free to ask me anything about comment generation, sentiment analysis, toxicity filtering, or scheduling!"

    # Check if input starts with a greeting prefix followed by a question
    greeting_prefix = ""
    if is_salam:
        greeting_prefix = "Walaikum Assalam! "
    elif words and words[0] in basic_greetings:
        greeting_prefix = "Hello! "

    # 3. Bengali questions support
    if re.search(r'[\u0980-\u09FF]', text):
        if "কমেন্ট" in text:
            return greeting_prefix + "💬 **কমেন্ট জেনারেশন**: যে পোস্টটির জন্য কমেন্ট চান তা চ্যাটবক্সে দিন। সিস্টেম পোস্টের কনটেক্সট এবং সেন্টিমেন্ট বিশ্লেষণ করে স্বাভাবিক ও প্রাসঙ্গিক কমেন্ট তৈরি করে দেবে।"
        if "সেন্টিমেন্ট" in text:
            return greeting_prefix + "📊 **সেন্টিমেন্ট অ্যানালাইসিস**: এটি টেক্সটের ইতিবাচক, নেতিবাচক বা নিরপেক্ষ ভাবাবেগ এবং সূক্ষ্ম টোন (VADER + BERT) বিশ্লেষণ করে।"
        if any(k in text for k in ["টক্সিক", "টক্সিসিটি", "খারাপ", "গালি"]):
            return greeting_prefix + "🛡️ **টক্সিসিটি অ্যানালাইসিস**: এটি আক্রমণাত্মক বা আপত্তিকর মন্তব্য শনাক্ত করে। সাধারণ বিরক্তি এবং ক্ষতিকর ব্যক্তিগত আক্রমণের মধ্যে পার্থক্য নিশ্চিত করা হয়।"
        if any(k in text for k in ["শিডিউল", "সময়", "সময়"]):
            return greeting_prefix + "📅 **স্মার্ট শিডিউলিং**: সোশ্যাল মিডিয়ায় সর্বোচ্চ এনগেজমেন্ট ও রিচ পাওয়ার জন্য সঠিক সময় এবং হ্যাশট্যাগ সুপারিশ করে।"
        if any(k in text for k in ["সাহায্য", "কী করতে পার", "কি করতে পার"]):
            return greeting_prefix + "আমি আপনার এআই সহকারী। কমেন্ট তৈরি, সেন্টিমেন্ট অ্যানালাইসিস, টক্সিসিটি চেক কিংবা শিডিউলিং নিয়ে যেকোনো প্রশ্ন বাংলায় করতে পারেন।"

    # 4. Social Media Advice & Strategies (Checked before broad keywords)
    if any(k in t_low for k in ["increase engagement", "grow followers", "more likes", "engagement tip", "boost engagement", "algorithm", "more engagement"]):
        ans = (
            "📈 **Tips to Boost Social Media Engagement**:\n"
            "1. **Hook in the first 2 lines**: Capture interest before the 'See more' fold.\n"
            "2. **Post at peak windows**: Weekday evenings (6–8 PM) generally yield the highest interaction.\n"
            "3. **Ask open-ended questions**: Prompt followers to share personal opinions or stories.\n"
            "4. **Engage in the first hour**: Replying promptly to initial comments signals high community activity to platform algorithms."
        )
        return greeting_prefix + ans

    if any(k in t_low for k in ["negative comment", "bad review", "angry customer", "complaint", "crisis"]):
        ans = (
            "🤝 **Best Practices for Handling Negative Comments**:\n"
            "1. **Respond promptly & calmly**: Do not ignore valid customer frustration or delete comments.\n"
            "2. **Acknowledge and apologize**: Validate their experience without becoming defensive.\n"
            "3. **Take specifics to DM**: Invite them to private messages to exchange sensitive order/account info.\n"
            "4. **Close the loop**: Confirm resolution publicly so other viewers see your proactive care."
        )
        return greeting_prefix + ans

    if any(k in t_low for k in ["hashtag", "hashtags", "tagging"]):
        ans = (
            "🏷️ **Effective Hashtag Strategy**:\n"
            "• **Quantity**: Use 3 to 5 highly relevant hashtags rather than stuffing 20+ generic ones.\n"
            "• **Hierarchy**: Combine 1 broad category tag (#Tech), 2 community tags (#WebDev), and 1 niche/brand tag (#PythonBots).\n"
            "• **Placement**: Place them cleanly at the end of captions to keep text readable."
        )
        return greeting_prefix + ans

    # 5. Specific Feature Questions
    # Toxicity / Moderation / Safety
    if any(k in t_low for k in ["toxicity", "toxic", "safety analysis", "moderation", "moderate", "abusive", "harassment", "offensive", "profanity", "hate speech"]):
        ans = (
            "🛡️ **Toxicity Analysis** evaluates whether text contains harassment, personal attacks, or offensive content.\n"
            "• **Target Discrimination**: Distinguishes between benign situational frustration (e.g., 'stupid bug') and abusive attacks directed at individuals.\n"
            "• **Hybrid Detection**: Combines fine-tuned BERT classification with rule-based profanity filters and contextual explanations.\n"
            "• **Review Queue**: Flagged toxic comments are automatically routed to the manual approval queue."
        )
        return greeting_prefix + ans

    # Sentiment / Emotion / VADER
    if any(k in t_low for k in ["sentiment", "emotion", "vader", "sarcasm", "mood", "feeling"]):
        ans = (
            "📊 **Sentiment Analysis** detects emotional valence and nuanced intent in social copy:\n"
            "• **VADER Scoring**: Provides rapid rule-based lexicon scoring for baseline polarity.\n"
            "• **Deep BERT Embeddings**: Evaluates complex context, sarcasm, and contrastive conjunctions ('loved the UI but the service was terrible').\n"
            "• **Multi-Language**: Fully supports both English and Bengali sentiment parsing."
        )
        return greeting_prefix + ans

    # BERT Architecture in this project
    if "bert" in t_low:
        ans = (
            "🤖 **BERT (Bidirectional Encoder Representations from Transformers)**:\n"
            "In this project, BERT processes comments bidirectionally to capture semantic depth and subtle sarcasm. "
            "It powers both the sentiment analysis pipeline (classifying positive, negative, and neutral tones) "
            "and the target-discriminating toxicity detection engine."
        )
        return greeting_prefix + ans

    # Comment Generation / Archetypes
    if any(k in t_low for k in ["comment generation", "generate comment", "generate comments", "archetype", "archetypes", "suggested comment", "write comment"]):
        ans = (
            "💬 **Comment Generation** produces authentic, human-style responses tailored to social posts:\n"
            "• **7 Archetypes**: Tailors tone for questions, discussions, milestones, food, pets, tech, and sports.\n"
            "• **Context Awareness**: Aligns reply sentiment with the post tone (empathetic for negative posts, celebratory for milestones).\n"
            "• **Bilingual Support**: Generates natural responses in English and authentic Bengali."
        )
        return greeting_prefix + ans

    # Scheduling / Best Time / Engagement / XGBoost
    if any(k in t_low for k in ["schedule", "scheduling", "best time", "engagement", "xgboost", "when to post", "optimal time", "peak hour"]):
        ans = (
            "📅 **Scheduling & Engagement Prediction**:\n"
            "• **XGBoost Regressor**: Predicts an engagement score based on post length, hashtag density, sentiment, and platform features.\n"
            "• **Random Forest Timing**: Recommends the optimal posting window (e.g., weekday evenings 6–8 PM) based on historical peak activity.\n"
            "• **Actionable Tips**: Provides hashtag recommendations and caption improvement advice."
        )
        return greeting_prefix + ans

    # Facebook / Meta Integration
    if any(k in t_low for k in ["facebook", "meta", "fb page", "connect account", "oauth"]):
        ans = (
            "🔗 **Facebook Integration** connects via Meta Graph API v22.0:\n"
            "• Connect your Page securely via OAuth 2.0 in the 'Connect Accounts' tab.\n"
            "• Fetch recent page posts, inspect audience comments, and publish scheduled updates.\n"
            "• Reply to comments directly from the workspace using generated AI responses."
        )
        return greeting_prefix + ans

    # Instagram / X / Twitter / LinkedIn
    if any(k in t_low for k in ["instagram", "ig", "twitter", "x.com", "linkedin", "per post"]):
        ans = (
            "🌐 **Connected Platforms & Pricing**:\n"
            "• **Instagram**: Requires an Instagram Professional account linked to your Meta Page (upgrade to Pro mode to enable).\n"
            "• **X (Twitter)**: Direct post scheduling via X API v2 (subject to tier limits and per-post API pricing, typically ~$0.01/tweet).\n"
            "• **LinkedIn**: Company page publishing via OAuth 2.0 authorization tokens."
        )
        return greeting_prefix + ans

    # Pricing / Packages / SSLCommerz / Plans / Upgrade
    if any(k in t_low for k in ["pricing", "price", "package", "packages", "cost", "plan", "plans", "sslcommerz", "bkash", "nagad", "payment", "upgrade", "pro mode", "subscription"]):
        ans = (
            "⚡ **Bot Packages & Pricing**:\n"
            "• **Free**: 100 comments/mo, basic sentiment & toxicity checks.\n"
            "• **Starter (৳499/mo)**: 500 comments/mo, tone detection & priority queues.\n"
            "• **Pro (৳1,499/mo)**: Unlimited comments, Facebook Page sync, AI scheduler & full model access.\n"
            "• **Agency (৳3,999/mo)**: Multi-account management & custom webhook automation.\n"
            "Payments are securely handled in BDT via SSLCommerz supporting bKash, Nagad, and cards."
        )
        return greeting_prefix + ans

    # Image Comments / Vision
    if any(k in t_low for k in ["image", "photo", "picture", "vision"]):
        ans = (
            "🖼️ **Image Comments**:\n"
            "Upload any photo (JPEG, PNG, WebP up to 10MB) in the 'Image Comment' tab. "
            "The vision model inspects visual composition, subject matter, and mood to generate contextual social media captions and replies."
        )
        return greeting_prefix + ans

    # Thesis / Research / Datasets / Accuracy / CSE400 / BRAC
    if any(k in t_low for k in ["dataset", "datasets", "accuracy", "thesis", "cse400", "brac", "evaluation", "pipeline architecture", "architecture"]):
        ans = (
            "🎓 **BRAC University CSE400 Thesis Architecture**:\n"
            "• **Core Objective**: An intelligent end-to-end social media agent combining deep NLP, content moderation, and engagement optimization.\n"
            "• **Datasets**: Evaluated on Twitter Sentiment140, Jigsaw Toxic Comment Classification, and Bengali social corpora.\n"
            "• **Performance**: Achieved ~92% accuracy on sentiment classification and high precision on target-discriminated toxicity filtering.\n"
            "• **Hybrid Pipeline**: Rule-based filters for sub-second responses coupled with transformer models for nuanced comprehension."
        )
        return greeting_prefix + ans

    # Capabilities / Who are you / Overview
    if any(k in t_low for k in ["who are you", "what are you", "what is your name", "who made you", "what can you do", "features", "capabilities", "help me", "how to use", "overview", "what is this"]):
        ans = (
            "I am the **Intelligent Social Bot Assistant** for social media automation and analysis:\n"
            "• 💬 **Comment Generation**: Craft natural replies across 7 archetypes in English and Bengali.\n"
            "• 📊 **Sentiment Analysis**: Analyze emotional tone, sarcasm, and intent using BERT & VADER.\n"
            "• 🛡️ **Toxicity Analysis**: Filter harassment and abusive language with target discrimination.\n"
            "• 📅 **Smart Scheduling**: Predict post engagement with XGBoost and find peak posting times.\n"
            "• 🔗 **Facebook Sync**: Connect Facebook Pages via Meta Graph API v22.0 to manage posts & replies.\n"
            "What would you like assistance with?"
        )
        return greeting_prefix + ans

    # Caption Writing Request
    if any(k in t_low for k in ["write a caption", "caption for", "create a caption", "caption idea"]):
        topic_match = re.search(r'caption\s+(?:for|about)\s+(.+)', text, re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else "your post"
        ans = (
            f"✍️ **Caption Idea for {topic.title()}**:\n"
            f"\"Bringing fresh energy and passion to {topic}! ✨ What is your favorite part about this? Let us know in the comments below! 👇\"\n\n"
            f"🏷️ **Suggested Tags**: #{re.sub(r'[^a-zA-Z0-9]', '', topic.title())} #SocialUpdate #CommunityEngagement"
        )
        return greeting_prefix + ans

    # General AI / Tech concepts
    if any(k in t_low for k in ["what is ai", "what is artificial intelligence"]):
        ans = "🤖 **Artificial Intelligence (AI)** refers to systems engineered to perform cognitive tasks typically associated with human intelligence—such as understanding natural language, identifying patterns in data, making predictions, and learning from experience."
        return greeting_prefix + ans

    if any(k in t_low for k in ["what is machine learning", "what is ml"]):
        ans = "🧠 **Machine Learning (ML)** is a branch of AI that enables systems to learn and improve performance automatically from data rather than following static, explicitly programmed rules. In this project, ML algorithms like XGBoost and Random Forest predict social media engagement and posting schedules."
        return greeting_prefix + ans

    if any(k in t_low for k in ["what is nlp", "natural language processing"]):
        ans = "💬 **Natural Language Processing (NLP)** combines linguistics and computer science to help software understand, interpret, and generate human language. In this project, NLP powers our sentiment analysis, target-discriminated toxicity filtering, and archetype comment generation."
        return greeting_prefix + ans

    if any(k in t_low for k in ["what is python"]):
        ans = "🐍 **Python** is a high-level, interpreted programming language renowned for readability and an extensive ecosystem in AI/ML (PyTorch, Transformers, Scikit-Learn, Flask). It powers the backend and NLP pipelines of this application."
        return greeting_prefix + ans

    # General Question Answering Fallback
    try:
        from .models.text_features import extract_dynamic_topics, is_question_intent
        topics = extract_dynamic_topics(text)
        topic_str = topics[0] if topics and topics[0] != "social update" else "your query"
        is_q = is_question_intent(text)
    except Exception:
        topic_str = "your query"
        is_q = "?" in text

    if is_q or "?" in text or any(w in words[:2] for w in ["how", "what", "why", "when", "where", "can", "is", "are", "do", "does", "should", "will"]):
        ans = (
            f"Regarding **{topic_str}**: A solid approach is to identify your key objective, validate each step with real audience data, and iterate based on engagement metrics. "
            "If you have questions about sentiment, toxicity analysis, comment generation, or scheduling, feel free to ask!"
        )
        return greeting_prefix + ans

    ans = (
        f"I've received your note regarding **{topic_str}**. "
        "Feel free to ask a question, generate comments, analyze text sentiment, check toxicity, or schedule a post!"
    )
    return greeting_prefix + ans



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
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
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
    provider = str(payload.get("provider", "")).lower().strip()
    if not provider or provider in {"ollama", "local", "offline", "none"}:
        if GEMINI_API_KEY:
            provider = "gemini"
        elif ANTHROPIC_API_KEY:
            provider = "anthropic"
        elif OPENAI_API_KEY:
            provider = "openai"
        else:
            text = fallback_response(payload)
            return {"text": text, "provider": "local-fallback", "model": "local"}

    try:
        if provider in {"gemini", "google"}:
            return call_gemini(payload)
        if provider in {"anthropic", "claude"}:
            return call_anthropic(payload)
        if provider in {"openai", "chatgpt"}:
            return call_openai(payload)
    except Exception as exc:
        logger.warning("Primary provider '%s' failed (%s). Checking failovers...", provider, exc)
        for failover_name, failover_fn, key in [
            ("gemini", call_gemini, GEMINI_API_KEY),
            ("anthropic", call_anthropic, ANTHROPIC_API_KEY),
            ("openai", call_openai, OPENAI_API_KEY),
        ]:
            if key and provider not in {failover_name}:
                try:
                    return failover_fn(payload)
                except Exception:
                    continue
        text = fallback_response(payload)
        return {"text": text, "provider": "local-fallback", "model": "local"}

    # If unrecognized provider, try configured providers or fallback
    if GEMINI_API_KEY:
        try:
            return call_gemini(payload)
        except Exception:
            pass
    if ANTHROPIC_API_KEY:
        try:
            return call_anthropic(payload)
        except Exception:
            pass
    if OPENAI_API_KEY:
        try:
            return call_openai(payload)
        except Exception:
            pass
    text = fallback_response(payload)
    return {"text": text, "provider": "local-fallback", "model": "local"}


def call_vision(image_bytes, media_type, prompt, provider="gemini", filename="", user_prompt="", tone="friendly", platform="facebook"):
    """Ask a configured vision provider to interpret the uploaded image."""
    provider = str(provider or "gemini").lower()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{media_type};base64,{encoded}"

    # Route to Gemini Vision if requested or if OpenAI/Anthropic keys are missing
    if provider in {"gemini", "google"} or (GEMINI_API_KEY and not OPENAI_API_KEY and not ANTHROPIC_API_KEY):
        requested_model = GEMINI_MODEL
        if not GEMINI_API_KEY:
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, tone=tone, platform=platform, filename=filename, user_prompt=user_prompt)
            database.store_api_call("local-fallback-vision", requested_model, {"model": requested_model, "prompt": prompt}, local_res["text"], "ok")
            return {**local_res, "provider": "local-fallback", "model": "local"}

        models_to_try = [
            requested_model,
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
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
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=35) as res:
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
        local_res = local_vision_model.process_image(image_bytes, tone=tone, platform=platform, filename=filename, user_prompt=user_prompt)
        return {**local_res, "provider": "local-fallback", "model": "local"}

    if provider in {"openai", "chatgpt"}:
        model = OPENAI_MODEL
        if not OPENAI_API_KEY:
            if GEMINI_API_KEY:
                return call_vision(image_bytes, media_type, prompt, provider="gemini", filename=filename, user_prompt=user_prompt, tone=tone, platform=platform)
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, tone=tone, platform=platform, filename=filename, user_prompt=user_prompt)
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
                return call_vision(image_bytes, media_type, prompt, provider="gemini", filename=filename, user_prompt=user_prompt, tone=tone, platform=platform)
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, tone=tone, platform=platform, filename=filename, user_prompt=user_prompt)
            return {**local_res, "provider": "local-fallback", "model": "local"}

    if provider in {"anthropic", "claude"}:
        model = ANTHROPIC_MODEL
        if not ANTHROPIC_API_KEY:
            if GEMINI_API_KEY:
                return call_vision(image_bytes, media_type, prompt, provider="gemini", filename=filename, user_prompt=user_prompt, tone=tone, platform=platform)
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, tone=tone, platform=platform, filename=filename, user_prompt=user_prompt)
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
                return call_vision(image_bytes, media_type, prompt, provider="gemini", filename=filename, user_prompt=user_prompt, tone=tone, platform=platform)
            from .models.vision_model import local_vision_model
            local_res = local_vision_model.process_image(image_bytes, tone=tone, platform=platform, filename=filename, user_prompt=user_prompt)
            return {**local_res, "provider": "local-fallback", "model": "local"}

    from .models.vision_model import local_vision_model
    local_res = local_vision_model.process_image(image_bytes, tone=tone, platform=platform, filename=filename, user_prompt=user_prompt)
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
