import json
from pathlib import Path

from ..config import SENTIMENT_MODEL_PATH
from .text_features import hashtags, tokens
from .vader_model import analyze as vader_analyze


class BertSentimentModel:
    def __init__(self):
        self.model_path = Path(SENTIMENT_MODEL_PATH)
        self.pipeline = None
        self.source = "fallback"
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
            self.source = "fallback"

    def predict(self, text):
        if self.pipeline:
            pred = self.pipeline(text[:512])[0]
            label = pred.get("label", "neutral").lower()
            if "pos" in label:
                sentiment = "positive"
            elif "neg" in label:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            confidence = float(pred.get("score", 0.7))
        else:
            vader = vader_analyze(text)
            sentiment = vader["label"]
            confidence = vader["confidence"]

        words = tokens(text)
        topic_tags = [tag.strip("#").lower() for tag in hashtags(text)]
        common = [w for w in words if len(w) > 5 and not w.startswith("#")]
        topics = (topic_tags + common[:3])[:4] or ["general"]
        lower = text.lower()
        sarcasm = any(marker in lower for marker in ["yeah right", "totally...", "as if", "/s"])
        tone = self._tone(sentiment, lower)
        intensity = "strong" if confidence >= 0.8 else "moderate" if confidence >= 0.62 else "mild"
        return {
            "model": "bert-sentiment",
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "sarcasm_detected": sarcasm,
            "tone": tone,
            "topics": topics,
            "explanation": f"{sentiment.title()} sentiment predicted by {self.source}.",
            "intensity": intensity,
            "source": self.source,
        }

    def _tone(self, sentiment, lower):
        if any(w in lower for w in ["!", "excited", "thrilled", "launch"]):
            return "excited"
        if sentiment == "negative" and any(w in lower for w in ["angry", "hate", "worst"]):
            return "angry"
        if sentiment == "negative":
            return "frustrated"
        if sentiment == "positive":
            return "hopeful"
        return "informative"


sentiment_model = BertSentimentModel()
