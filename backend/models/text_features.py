import re


POSITIVE_WORDS = {
    "love", "amazing", "great", "awesome", "fantastic", "happy", "excited", "beautiful",
    "excellent", "wonderful", "congrats", "best", "perfect", "incredible", "joy",
    "launch", "grateful", "thank", "success", "brilliant", "outstanding", "superb",
    "delighted", "thrilled", "proud", "good", "helpful", "useful", "win", "wins",
}

NEGATIVE_WORDS = {
    "hate", "terrible", "awful", "worst", "bad", "sad", "angry", "frustrated",
    "disappointed", "horrible", "stupid", "useless", "annoying", "depressed",
    "down", "expensive", "wrong", "broke", "fail", "failed", "disaster", "ugly",
    "boring", "waste", "regret", "miss", "worried", "problem", "issues",
}

TOXIC_WORDS = {
    "idiot", "stupid", "hate you", "kill", "die", "trash", "dumb", "moron",
    "shut up", "loser", "worthless", "scam", "fraud",
}

SENSITIVE_TOPICS = {
    "suicide", "self-harm", "drugs", "terrorism", "violence", "weapon", "porn",
    "gambling", "medication dosage", "legal advice", "medical diagnosis",
}


def tokens(text):
    return re.findall(r"[a-zA-Z0-9#']+", text.lower())


def hashtags(text):
    return re.findall(r"#\w+", text)


def count_matches(text, words):
    lower = text.lower()
    return sum(1 for word in words if word in lower)


def feature_vector(text, sentiment="neutral"):
    words = tokens(text)
    char_count = len(text)
    word_count = len(words)
    hash_count = len(hashtags(text))
    pos = count_matches(text, POSITIVE_WORDS)
    neg = count_matches(text, NEGATIVE_WORDS)
    sentiment_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(sentiment, 0.0)
    return [word_count, char_count, hash_count, pos, neg, sentiment_score]
