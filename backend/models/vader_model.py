from .text_features import (
    MILD_NEGATIVE_WORDS,
    MILD_POSITIVE_WORDS,
    NEGATION_WORDS,
    STRONG_NEGATIVE_WORDS,
    STRONG_POSITIVE_WORDS,
    tokens,
)


def analyze(text):
    if not text:
        return {
            "model": "vader-fast",
            "label": "neutral",
            "compound": 0.0,
            "confidence": 0.5,
            "positive_hits": 0,
            "negative_hits": 0,
            "source": "rule-based-vader-compatible",
        }

    word_list = tokens(text)
    score = 0.0
    pos_hits = 0
    neg_hits = 0

    for i, word in enumerate(word_list):
        prev_window = word_list[max(0, i - 2):i]
        is_negated = any(p in NEGATION_WORDS for p in prev_window)

        if word in STRONG_POSITIVE_WORDS:
            if is_negated:
                score -= 1.5
                neg_hits += 1
            else:
                score += 1.8
                pos_hits += 1
        elif word in MILD_POSITIVE_WORDS:
            if is_negated:
                score -= 0.8
                neg_hits += 1
            else:
                score += 1.0
                pos_hits += 1
        elif word in STRONG_NEGATIVE_WORDS:
            if is_negated:
                score += 1.0
                pos_hits += 1
            else:
                score -= 1.8
                neg_hits += 1
        elif word in MILD_NEGATIVE_WORDS:
            if is_negated:
                score += 0.8
                pos_hits += 1
            else:
                score -= 1.0
                neg_hits += 1

    # Emoji adjustments
    emoji_pos = ["😊", "😍", "❤️", "🎉", "🥰", "💪", "🚀", "⭐", "🏆", "🔥", "👏", "🙌"]
    emoji_neg = ["😞", "😤", "😢", "😡", "💔", "😠", "😭"]
    for e in emoji_pos:
        if e in text:
            score += 1.0
            pos_hits += 1
    for e in emoji_neg:
        if e in text:
            score -= 1.0
            neg_hits += 1

    compound = max(-1.0, min(1.0, score / 4.0))
    label = "positive" if compound > 0.08 else "negative" if compound < -0.08 else "neutral"
    confidence = min(0.96, 0.55 + abs(compound) * 0.4)

    return {
        "model": "vader-fast",
        "label": label,
        "compound": round(compound, 3),
        "confidence": round(confidence, 3),
        "positive_hits": pos_hits,
        "negative_hits": neg_hits,
        "source": "rule-based-vader-compatible",
    }
