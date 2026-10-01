import json

from ..ai_provider import call_ai
from .vader_model import analyze as vader_analyze
from .comment_generator_model import comment_generator_model
from .sentiment_model import sentiment_model
from .text_features import detect_language
from .toxicity_model import toxicity_model


def _parse_json_response(text):
    if not isinstance(text, str):
        return text
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return text


def sentiment(text, provider='anthropic'):
    system = (
        'You are a social media sentiment analysis expert. Detect nuanced emotions, sarcasm, irony. '
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


def toxicity(text, provider='anthropic'):
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


def engagement(text, sentiment_label='neutral', provider='anthropic'):
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


def schedule(text, sentiment_label='neutral', predicted_engagement=None, provider='anthropic'):
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


def comment_generator(text, sentiment_label='neutral', toxicity=False, style='casual', provider='anthropic', platform='general', num_comments=1, language=None):
    effective_lang = language if language and language != 'auto' else detect_language(text)
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


def _comment_generator_cloud(text, sentiment_label, toxicity, style='casual', provider='anthropic', platform='general', num_comments=1, system_extra='', language='en'):
    sentiment_str = str(sentiment_label or 'neutral').strip().lower()
    negative_rule = ""
    if sentiment_str == 'negative':
        negative_rule = (
            "System Rule: The user sentiment is NEGATIVE. Generate a professional, apologetic, and empathetic customer service reply. "
            "Do not be cheerful. Do not use emojis like ✨ or 🔥. Acknowledge their frustration and offer to resolve the issue."
        )

    lang_rule = (
        "LANGUAGE RULE: If the input post or comment is in Bengali / Bangla (বাংলা) or Banglish, or if requested language is 'bn', "
        "you MUST generate the comment(s) strictly in natural, fluent, conversational Bengali (বাংলা script). "
        "If in English, generate in English. Always match the language and cultural tone of the post."
    )
    rule_suffix = f"{negative_rule} {lang_rule} {system_extra}".strip()

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
    parsed = None
    try:
        parsed = json.loads(text_out)
    except Exception:
        # best-effort extract
        s = text_out.find('[')
        e = text_out.rfind(']')
        if s != -1 and e != -1 and e > s:
            try:
                parsed = json.loads(text_out[s:e+1])
            except Exception:
                parsed = None

    if parsed is not None:
        # If array, return list of comments
        if isinstance(parsed, list):
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


def full_pipeline(text, provider='anthropic'):
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
