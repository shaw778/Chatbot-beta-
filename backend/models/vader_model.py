from .text_features import NEGATIVE_WORDS, POSITIVE_WORDS, count_matches


def analyze(text):
    pos = count_matches(text, POSITIVE_WORDS)
    neg = count_matches(text, NEGATIVE_WORDS)
    raw = pos - neg
    compound = max(-1.0, min(1.0, raw / 5.0))
    label = "positive" if compound > 0.12 else "negative" if compound < -0.12 else "neutral"
    confidence = min(0.95, 0.55 + abs(compound) * 0.4)
    return {
        "model": "vader-fast",
        "label": label,
        "compound": round(compound, 3),
        "confidence": round(confidence, 3),
        "positive_hits": pos,
        "negative_hits": neg,
        "source": "rule-based-vader-compatible",
    }
