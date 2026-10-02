import json
import re

from ..ai_provider import call_ai
from .vader_model import analyze as vader_analyze
from .comment_generator_model import comment_generator_model
from .sentiment_model import sentiment_model
from .text_features import detect_language
from .toxicity_model import toxicity_model


def _parse_json_response(text):
    if not isinstance(text, str):
        return text
    cleaned = text.strip()
    if not cleaned:
        return None

    # Strip markdown code blocks
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()

    # Strip malformed brackets with prefixes like [json or {json
    cleaned = re.sub(r'^\[(?:json)?\s*', '[', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\{(?:json)?\s*', '{', cleaned, flags=re.IGNORECASE)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract JSON array
    s_arr = cleaned.find('[')
    e_arr = cleaned.rfind(']')
    if s_arr != -1 and e_arr != -1 and e_arr > s_arr:
        try:
            return json.loads(cleaned[s_arr:e_arr + 1])
        except Exception:
            pass

    # Extract JSON object
    s_obj = cleaned.find('{')
    e_obj = cleaned.rfind('}')
    if s_obj != -1 and e_obj != -1 and e_obj > s_obj:
        try:
            return json.loads(cleaned[s_obj:e_obj + 1])
        except Exception:
            pass

    return cleaned


def sentiment(text, provider='gemini'):
    system = (
        'You are a social media sentiment analysis expert. Detect nuanced emotions, sarcasm, irony. '
        'CONTRASTIVE RULE: When a sentence contains contrastive conjunctions like "but", "however", or "yet" '
        '(e.g. "The design is gorgeous, but the customer service was awful"), the clause following the contrastive conjunction '
        'carries the primary overall sentiment (e.g. negative). '
        'Return ONLY valid JSON: {"sentiment":"positive|negative|neutral","confidence":0.0-1.0,'
        '"sarcasm_detected":true|false,"tone":"excited|angry|sad|hopeful|humorous|informative|frustrated|grateful|anxious",'
        '"topics":["topic1","topic2"],"explanation":"one sentence","intensity":"mild|moderate|strong"}'
    )
    user = f'Analyze: "{text}"'
    response = call_ai({
        'provider': provider,
        'system': system,
        'messages': [{'role': 'user', 'content': user}],
        'max_tokens': 250,
        'temperature': 0.2,
    })
    parsed = _parse_json_response(response.get('text', ''))
    if isinstance(parsed, dict) and 'sentiment' in parsed:
        parsed['source'] = response.get('provider')
        parsed['model'] = response.get('model')
        return parsed
    return sentiment_model.predict(text)


def toxicity(text, provider='gemini'):
    system = (
        'You are a content moderation assistant. Analyze the user text for toxicity, abusive language, hate speech, or harassment. '
        'Return ONLY valid JSON: {"model":"ai-toxicity","toxic":true|false,"label":"toxic|clean","score":0.0-1.0,'
        '"keywords":["term1","term2"],"sensitive_topics":["topic1"],"topic_flagged":true|false,'
        '"explanation":"one sentence"}'
    )
    user = f'Text: "{text}"'
    response = call_ai({
        'provider': provider,
        'system': system,
        'messages': [{'role': 'user', 'content': user}],
        'max_tokens': 250,
        'temperature': 0.2,
    })
    parsed = _parse_json_response(response.get('text', ''))
    if isinstance(parsed, dict) and ('toxic' in parsed or 'label' in parsed):
        parsed['source'] = response.get('provider')
        parsed['model'] = response.get('model')
        return parsed
    return toxicity_model.predict(text)


def engagement(text, sentiment_label='neutral', provider='gemini'):
    system = (
        'You are an engagement prediction expert for social media posts. Estimate how many reactions or interactions a post may receive. '
        'Return ONLY valid JSON: {"model":"ai-engagement","predicted_engagement":0-1000,'
        '"features":{"word_count":0,"char_count":0,"hashtag_count":0,"sentiment":"positive|negative|neutral"},'
        '"source":"...","explanation":"one sentence"}'
    )
    user = f'Post: "{text}"\nSentiment: {sentiment_label}'
    response = call_ai({
        'provider': provider,
        'system': system,
        'messages': [{'role': 'user', 'content': user}],
        'max_tokens': 250,
        'temperature': 0.2,
    })
    parsed = _parse_json_response(response.get('text', ''))
    if isinstance(parsed, dict):
        parsed['source'] = response.get('provider')
        parsed['model'] = response.get('model')
        return parsed
    return {
        'model': response.get('model', 'cloud-engagement'),
        'predicted_engagement': 0,
        'features': {
            'word_count': len(text.split()),
            'char_count': len(text),
            'hashtag_count': text.count('#'),
            'sentiment': sentiment_label,
        },
        'source': response.get('provider'),
        'explanation': 'Could not parse cloud provider engagement response.',
        'raw': response.get('text', ''),
    }


def schedule(text, sentiment_label='neutral', predicted_engagement=None, provider='gemini'):
    system = (
        'You are a social media scheduling expert. Recommend the best posting window based on the post content and sentiment. '
        'Return ONLY valid JSON: {"model":"ai-scheduling","slot":"Morning (8-10 AM)|Afternoon (12-2 PM)|Evening (6-8 PM)|Night (9-11 PM)",'
        '"best_day":"weekday|weekend","predicted_engagement":0-1000,"reasoning":"one sentence","tip":"one actionable tip",'
        '"hashtag_suggestions":["#tag1","#tag2"]}'
    )
    user = f'Post: "{text}"\nSentiment: {sentiment_label}\nPredicted engagement: {predicted_engagement or 0}'
    response = call_ai({
        'provider': provider,
        'system': system,
        'messages': [{'role': 'user', 'content': user}],
        'max_tokens': 250,
        'temperature': 0.2,
    })
    parsed = _parse_json_response(response.get('text', ''))
    if isinstance(parsed, dict):
        parsed['source'] = response.get('provider')
        parsed['model'] = response.get('model')
        return parsed
    return {
        'model': response.get('model', 'cloud-scheduling'),
        'slot': 'Evening (6-8 PM)',
        'best_day': 'weekday',
        'predicted_engagement': predicted_engagement or 0,
        'reasoning': 'Could not parse cloud provider scheduling response.',
        'tip': 'Use a specific caption and invite a short response.',
        'hashtag_suggestions': ['#AI', '#SocialMedia'],
        'source': response.get('provider'),
        'raw': response.get('text', ''),
    }


def comment_generator(text, sentiment_label='neutral', toxicity=False, style='casual', provider='gemini', platform='general', num_comments=1, language=None):
    effective_lang = language if language and language != 'auto' else detect_language(text)

    # Fast check for structured Facebook comment-reply prompts
    reply_direct = comment_generator_model._check_comment_reply_prompt(text, language=effective_lang, sentiment=sentiment_label)
    if reply_direct:
        return {
            "model": "comment-reply-engine",
            "comment": reply_direct,
            "comments": [reply_direct],
            "language": effective_lang,
            "sentiment": sentiment_label,
            "toxic": toxicity,
            "source": "local-rule",
            "explanation": "Contextual reply generated for the user's Facebook comment.",
        }

    if str(sentiment_label or '').lower() == 'neutral':
        fast_s = vader_analyze(text)
        if fast_s.get('sentiment') == 'negative':
            sentiment_label = 'negative'
    lang_note = f'Language: {effective_lang}. If Bengali/Bangla, write strictly in natural Bengali (বাংলা script).'
    system_extra = f'Target platform: {platform}. {lang_note} Ensure comments fit the platform length and style requirements.'
    # If requesting multiple comments, instruct cloud to return a JSON array
    return _comment_generator_cloud(
        text, sentiment_label, toxicity, style=style, provider=provider, platform=platform,
        num_comments=num_comments, system_extra=system_extra, language=effective_lang
    )


def _comment_generator_cloud(text, sentiment_label, toxicity, style='casual', provider='gemini', platform='general', num_comments=1, system_extra='', language='en'):
    sentiment_str = str(sentiment_label or 'neutral').strip().lower()
    if sentiment_str == 'positive':
        sentiment_rule = (
            "SENTIMENT RULE (POSITIVE / HAPPY): The post conveys happiness, success, celebration, or positive energy. "
            "Generate genuinely joyful, enthusiastic, warm, and congratulatory comments that celebrate their good news or joy. "
            "Share in their excitement! NEVER assume or mention a software app, technical tool, or digital platform unless the post explicitly talks about one."
        )
    elif sentiment_str == 'negative':
        sentiment_rule = (
            "SENTIMENT RULE (NEGATIVE): "
            "1) If the post is about personal sadness, grief, tragedy, heartbreak, bereavement, illness, injury, or crying: "
            "Generate deeply compassionate, comforting, caring, and sincere human replies offering warmth, solidarity, or heartfelt condolences. "
            "NEVER offer customer service, support tickets, helpdesk links, or DM requests for personal sadness or grief! Do not be cheerful. "
            "2) If the post is about frustrating disruptions, travel/flight delays, lost luggage, customer service, or business complaints: "
            "Generate an apologetic, understanding, and helpful reply acknowledging how frustrating it is, expressing sincere apology and empathy (e.g. 'So sorry to hear that, flight delays are so frustrating' / 'apologize for the disruption' / 'আমরা আন্তরিকভাবে দুঃখিত'), and offering support or hoping for resolution ('support', 'resolve', 'সমাধান', 'যোগাযোগ', 'ইনবক্স')."
        )
    else:
        sentiment_rule = (
            "SENTIMENT RULE (NEUTRAL): Keep the tone balanced, conversational, and directly relevant to the topic of the post. "
            "NEVER assume or mention a software app, technical tool, or digital platform unless the post explicitly mentions one."
        )

    anti_app_bias_rule = (
        "TOPIC ACCURACY RULE: ONLY talk about an 'app', 'software', or 'digital tool' if the post text EXPLICITLY mentions an app, software, or coding. "
        "For general life posts (e.g. food, pets, family, travel, achievements, sadness, day-to-day moments), react purely to what they are experiencing."
    )

    category_guidelines = (
        "CATEGORY & TOPIC GUIDELINES: "
        "- Questions/Advice: Offer practical, encouraging recommendations or perspectives addressing their goals, projects, or starting journey. If in Bengali, you MUST specifically mention either শেখা (learning), বেসিক (basics), or প্রজেক্ট (projects). "
        "- Food/Cooking: Specifically compliment how delicious or tasty the food or plate looks, or ask for the recipe. If in Bengali, compliment the সুস্বাদু খাবার (delicious food) or দারুণ রান্না (cooking). "
        "- Pets/Animals: Express warmth specifically mentioning the adorable pet, puppy or pup, cuteness, or precious face."
    )

    lang_rule = (
        "LANGUAGE RULE: If the input post or comment is in Bengali / Bangla (বাংলা) or Banglish, or if requested language is 'bn', "
        "you MUST generate the comment(s) strictly in natural, fluent, conversational Bengali (বাংলা script). "
        "If in English, generate in English. Always match the language and cultural tone of the post."
    )
    rule_suffix = f"{sentiment_rule} {category_guidelines} {anti_app_bias_rule} {lang_rule} {system_extra}".strip()

    # Build system prompt expecting either single JSON object or JSON array depending on num_comments
    if num_comments and int(num_comments) > 1:
        system = (
            'You are a social media comment writer specializing in realistic, human replies to real-world posts. RULES: '
            '1) React to one concrete detail from the post 2) Match the platform, tone, and sentiment 3) Use conversational wording and natural pacing '
            '4) Never use generic phrases like "Nice post!" or "Great work!" 5) Never claim personal experiences or facts not present in the post '
            '6) Avoid AI cliches, over-polished marketing language, hashtags, and excessive emojis 7) No toxic/harmful content. '
            f'{rule_suffix} Return ONLY a JSON array of strings. No markdown, no explanation.'
        )
    else:
        system = (
            'You are a social media comment writer specializing in realistic, human replies to real-world posts. RULES: '
            '1) React to one concrete detail from the post 2) Match the platform, tone, and sentiment 3) Use conversational wording and natural pacing '
            '4) Never use generic phrases like "Nice post!" or "Great work!" 5) Never claim personal experiences or facts not present in the post '
            '6) Avoid AI cliches, over-polished marketing language, hashtags, and excessive emojis 7) No toxic/harmful content. '
            f'{rule_suffix} Return ONLY a JSON object with keys: comment, language, sentiment, toxic, source, explanation.'
        )

    user = f'Post: "{text}"\nSentiment: {sentiment_label}\nToxic: {toxicity}\nStyle: {style}\nLanguage: {language}\nGenerate {num_comments} comment(s) that fit {platform}.'

    try:
        response = call_ai({
            'provider': provider,
            'system': system,
            'messages': [{'role': 'user', 'content': user}],
            'max_tokens': 400,
            'temperature': 0.7,
        })
    except Exception as exc:
        logger.warning("Cloud comment generation call failed (%s). Using local fallback.", exc)
        response = {'provider': 'local-fallback', 'model': 'local', 'text': ''}

    # If cloud is unavailable, prefer our diverse local generator
    if response.get('provider') == 'local-fallback':
        if not num_comments or int(num_comments) <= 1:
            return comment_generator_model.predict(
                text, sentiment=sentiment_label, toxicity=toxicity, style=style, platform=platform, language=language
            )
        return comment_generator_model.generate_multiple(
            text, sentiment=sentiment_label, toxicity=toxicity, style=style, platform=platform, num_comments=num_comments, language=language
        )

    text_out = response.get('text', '')
    parsed = _parse_json_response(text_out)
    if isinstance(parsed, str):
        parsed = None

    if parsed is not None:
        # If array, return list of comments
        if isinstance(parsed, list):
            while len(parsed) == 1 and isinstance(parsed[0], list):
                parsed = parsed[0]
            return {
                'model': response.get('model'),
                'comments': parsed,
                'language': language,
                'source': response.get('provider'),
                'raw': text_out,
            }
        # single object or string
        if isinstance(parsed, dict):
            if 'language' not in parsed:
                parsed['language'] = language
            if num_comments and int(num_comments) > 1:
                if 'comments' in parsed and isinstance(parsed['comments'], list):
                    return parsed
                if 'comment' in parsed and isinstance(parsed['comment'], str):
                    parsed['comments'] = [parsed['comment']]
                    return parsed
            return parsed

    # Fallback to local model if cloud returned empty or malformed string
    if not text_out.strip() or "fallback analysis" in text_out.lower():
        if not num_comments or int(num_comments) <= 1:
            return comment_generator_model.predict(
                text, sentiment=sentiment_label, toxicity=toxicity, style=style, platform=platform, language=language
            )
        return comment_generator_model.generate_multiple(
            text, sentiment=sentiment_label, toxicity=toxicity, style=style, platform=platform, num_comments=num_comments, language=language
        )

    if num_comments and int(num_comments) > 1:
        # Check if text contains numbered or bulleted lines
        lines = [line.strip().lstrip('1234567890.-*• ') for line in text_out.splitlines() if line.strip()]
        if len(lines) >= int(num_comments):
            return {
                'model': response.get('model', 'cloud-comment-generator'),
                'comments': lines[:int(num_comments)],
                'language': language,
                'source': response.get('provider'),
                'raw': text_out,
            }
        return comment_generator_model.generate_multiple(
            text, sentiment=sentiment_label, toxicity=toxicity, style=style, platform=platform, num_comments=num_comments, language=language
        )

    return {
        'model': response.get('model', 'cloud-comment-generator'),
        'comment': text_out.strip(),
        'language': language,
        'sentiment': sentiment_label,
        'toxic': toxicity,
        'source': response.get('provider'),
        'explanation': f'Cloud comment generated in {language}.',
        'raw': text_out,
    }


def full_pipeline(text, provider='gemini'):
    sent = sentiment(text, provider=provider)
    sentiment_label = sent.get('sentiment', 'neutral')
    fast_sentiment = vader_analyze(text)
    tox = toxicity(text, provider=provider)
    engage = engagement(text, sentiment_label, provider=provider)
    sched = schedule(text, sentiment_label, engage.get('predicted_engagement'), provider=provider)
    comment = comment_generator(text, sentiment_label, tox.get('toxic', False), style='casual', provider=provider)
    return {
        'sentiment': {'bert': sent, 'vader': fast_sentiment},
        'toxicity': tox,
        'engagement': engage,
        'schedule': sched,
        'comment_generator': comment,
    }
