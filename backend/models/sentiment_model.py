import re
from pathlib import Path

from ..config import SENTIMENT_MODEL_PATH
from .text_features import (
    CONTRASTIVE_CONJUNCTIONS,
    DIMINISHERS,
    INTENSIFIERS,
    MILD_NEGATIVE_WORDS,
    MILD_POSITIVE_WORDS,
    NEGATION_WORDS,
    POSITIVE_SLANG,
    SARCASM_MARKERS,
    STRONG_NEGATIVE_WORDS,
    STRONG_POSITIVE_WORDS,
    extract_dynamic_topics,
    is_question_intent,
    tokens,
)


class BertSentimentModel:
    def __init__(self):
        self.model_path = Path(SENTIMENT_MODEL_PATH)
        self.pipeline = None
        self.source = "nlp-local-engine"
        self._load()

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            from transformers import pipeline

            self.pipeline = pipeline("text-classification", model=str(self.model_path), tokenizer=str(self.model_path))
            self.source = "transformers-bert"
        except Exception:
            self.pipeline = None
            self.source = "nlp-local-engine"

    def predict(self, text):
        if not text or not str(text).strip():
            return {
                "model": "bert-sentiment" if self.pipeline else "nlp-sentiment-engine",
                "sentiment": "neutral",
                "confidence": 0.5,
                "sarcasm_detected": False,
                "tone": "informative",
                "topics": ["general"],
                "explanation": "No text provided for sentiment analysis.",
                "intensity": "mild",
                "source": self.source,
            }

        # 1. If fine-tuned BERT transformer is available locally, run it
        if self.pipeline:
            try:
                pred = self.pipeline(text[:512])[0]
                label = pred.get("label", "neutral").lower()
                if "pos" in label:
                    sentiment = "positive"
                elif "neg" in label:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                confidence = float(pred.get("score", 0.7))
                topics = extract_dynamic_topics(text)
                tone = self._classify_tone(sentiment, text)
                intensity = "strong" if confidence >= 0.8 else "moderate" if confidence >= 0.62 else "mild"
                return {
                    "model": "bert-sentiment",
                    "sentiment": sentiment,
                    "confidence": round(confidence, 3),
                    "sarcasm_detected": self._detect_sarcasm(text),
                    "tone": tone,
                    "topics": topics,
                    "explanation": f"{sentiment.title()} sentiment detected by transformer pipeline.",
                    "intensity": intensity,
                    "source": "transformers-bert",
                }
            except Exception:
                pass

        # 2. Advanced NLP Rule & Context Engine with Negation Handling
        return self._rule_based_predict(text)

    def _detect_sarcasm(self, text):
        lower = text.lower()
        if any(marker in lower for marker in SARCASM_MARKERS):
            return True
        # Positive word followed immediately by ellipses then negative context
        if re.search(r"\b(great|awesome|fantastic|perfect|brilliant)\s*\.\.\.", lower):
            return True
        return False

    def _rule_based_predict(self, text):
        lower = text.lower()
        word_list = tokens(text)
        sarcasm = self._detect_sarcasm(text)
        is_question = is_question_intent(text)

        score = 0.0
        pos_terms_found = []
        neg_terms_found = []

        # Check positive slang / idioms first
        for slang in POSITIVE_SLANG:
            if slang in lower:
                score += 1.8
                pos_terms_found.append(slang)

        for i, word in enumerate(word_list):
            # Check negation window (looking back up to 2 words)
            prev_window = word_list[max(0, i - 2):i]
            is_negated = any(p in NEGATION_WORDS for p in prev_window)

            # Check intensifier or diminisher directly before
            multiplier = 1.0
            if prev_window:
                if prev_window[-1] in INTENSIFIERS:
                    multiplier = 1.5
                elif prev_window[-1] in DIMINISHERS:
                    multiplier = 0.6

            # Evaluate word valence
            if word in STRONG_POSITIVE_WORDS:
                if is_negated:
                    score -= 1.8 * multiplier
                    neg_terms_found.append(f"not {word}")
                else:
                    score += 2.0 * multiplier
                    pos_terms_found.append(word)
            elif word in MILD_POSITIVE_WORDS:
                if is_negated:
                    score -= 1.0 * multiplier
                    neg_terms_found.append(f"not {word}")
                else:
                    score += 1.0 * multiplier
                    pos_terms_found.append(word)
            elif word in STRONG_NEGATIVE_WORDS:
                if is_negated:
                    score += 1.2 * multiplier
                    pos_terms_found.append(f"not {word}")
                else:
                    score -= 2.0 * multiplier
                    neg_terms_found.append(word)
            elif word in MILD_NEGATIVE_WORDS:
                if is_negated:
                    score += 0.9 * multiplier
                    pos_terms_found.append(f"not {word}")
                else:
                    score -= 1.0 * multiplier
                    neg_terms_found.append(word)

        # Emoji boosts
        emoji_pos = ["😊", "😍", "❤️", "🎉", "🥰", "💪", "🚀", "⭐", "🏆", "🔥", "👏", "🙌", "✨"]
        emoji_neg = ["😞", "😤", "😢", "😡", "💔", "😠", "😭", "😿", "👎", "🤬"]
        for e in emoji_pos:
            if e in text:
                score += 1.2
                pos_terms_found.append(e)
        for e in emoji_neg:
            if e in text:
                score -= 1.2
                neg_terms_found.append(e)

        # Contrastive clause handling ("X is good, BUT Y is terrible")
        has_contrast = any(f" {c} " in f" {lower} " or f",{c}" in lower for c in CONTRASTIVE_CONJUNCTIONS)
        if has_contrast and pos_terms_found and neg_terms_found:
            # Contrastive clauses: the second clause after "but" has dominant pragmatic focus
            parts = re.split(r"\b(?:but|however|although|though|yet|কিন্তু|তবে)\b", lower, maxsplit=1)
            if len(parts) == 2:
                second_clause = parts[1]
                second_neg = any(w in second_clause for w in neg_terms_found)
                second_pos = any(w in second_clause for w in pos_terms_found)
                if second_neg and not second_pos:
                    score = -1.2
                elif second_pos and not second_neg:
                    score = 1.2

        # Sarcasm flips or dampens positive scores
        if sarcasm and score > 0:
            score = -abs(score) * 0.75
            neg_terms_found.append("sarcasm detected")

        # Exclamation emphasis
        exclamations = text.count("!")
        if exclamations > 0:
            if score > 0:
                score += min(1.0, exclamations * 0.3)
            elif score < 0:
                score -= min(1.0, exclamations * 0.3)

        # Questions inquiring about advice/info should be neutral unless strongly emotional
        if is_question and -1.0 <= score <= 1.5:
            sentiment = "neutral"
            confidence = 0.82
            tone = "inquisitive"
        elif score > 0.4:
            sentiment = "positive"
            confidence = min(0.98, 0.62 + min(1.0, score / 4.0) * 0.35)
            tone = self._classify_tone(sentiment, text)
        elif score < -0.4:
            sentiment = "negative"
            confidence = min(0.98, 0.62 + min(1.0, abs(score) / 4.0) * 0.35)
            tone = self._classify_tone(sentiment, text)
        else:
            sentiment = "neutral"
            confidence = 0.70
            tone = self._classify_tone(sentiment, text)

        # Intensity classification
        if confidence >= 0.82 or abs(score) >= 2.5:
            intensity = "strong"
        elif confidence >= 0.68 or abs(score) >= 1.0:
            intensity = "moderate"
        else:
            intensity = "mild"

        # Topic extraction
        topics = extract_dynamic_topics(text)

        # Dynamic explanation
        if has_contrast and pos_terms_found and neg_terms_found:
            explanation = "Mixed sentiment with contrastive phrasing balancing both positive aspects and drawbacks."
        elif is_question and sentiment == "neutral":
            explanation = "Inquisitive inquiry seeking recommendations, advice, or community perspective."
        elif sentiment == "positive":
            terms_str = ", ".join(f"'{w}'" for w in pos_terms_found[:3])
            explanation = f"Positive tone driven by upbeat terms ({terms_str})." if terms_str else "Overall message reflects encouraging and positive sentiment."
        elif sentiment == "negative":
            terms_str = ", ".join(f"'{w}'" for w in neg_terms_found[:3])
            explanation = f"Negative sentiment indicated by expressions of concern or dissatisfaction ({terms_str})." if terms_str else "Overall message reflects critical or unhappy tone."
        else:
            explanation = "Balanced, objective wording without significant emotional polarity."

        return {
            "model": "nlp-sentiment-engine",
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "sarcasm_detected": sarcasm,
            "tone": tone,
            "topics": topics,
            "explanation": explanation,
            "intensity": intensity,
            "source": self.source,
        }

    def _classify_tone(self, sentiment, text):
        lower = text.lower()
        if is_question_intent(text):
            return "inquisitive"
        if any(w in lower for w in ["win", "won", "victory", "champion", "trophy", "incredible", "thrilled", "excited", "congrats", "celebrate", "বিজয়", "অভিনন্দন", "দারুণ"]) or (sentiment == "positive" and "!" in text):
            return "excited"
        if any(w in lower for w in ["thank", "thanks", "grateful", "blessed", "appreciate", "shoutout", "ধন্যবাদ", "কৃতজ্ঞতা"]):
            return "grateful"
        if any(w in lower for w in ["haha", "lol", "lmao", "hilarious", "joke", "funny"]):
            return "humorous"
        if any(w in lower for w in ["hope", "looking forward", "wish", "fingers crossed", "future", "আশা করি", "শুভকামনা"]):
            return "hopeful"
        if sentiment == "negative" and any(w in lower for w in ["furious", "angry", "outrage", "disgusting", "unacceptable", "scam", "জঘন্য", "বাজে", "ফালতু"]):
            return "angry"
        if sentiment == "negative" and any(w in lower for w in ["delay", "delayed", "slow", "broken", "annoying", "tired of", "struggle", "problem", "issue", "দেরি", "বিরক্ত", "সমস্যা"]):
            return "frustrated"
        if sentiment == "negative" and any(w in lower for w in ["heartbroken", "sadly", "loss", "grief", "depressed", "miss", "painful", "কষ্ট", "হতাশ"]):
            return "sad"
        if any(w in lower for w in ["worried", "nervous", "anxious", "scared", "fear", "stress", "চিন্তিত"]):
            return "anxious"
        return "informative"


sentiment_model = BertSentimentModel()

