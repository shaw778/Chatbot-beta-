from pathlib import Path

from ..config import TOXICITY_MODEL_PATH
from .text_features import SENSITIVE_TOPICS, TOXIC_WORDS


class BertToxicityModel:
    def __init__(self):
        self.model_path = Path(TOXICITY_MODEL_PATH)
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
        lower = text.lower()
        keywords = [word for word in TOXIC_WORDS if word in lower]
        sensitive = [topic for topic in SENSITIVE_TOPICS if topic in lower]

        if self.pipeline:
            pred = self.pipeline(text[:512])[0]
            label = pred.get("label", "").lower()
            model_score = float(pred.get("score", 0.0))
            toxic = any(x in label for x in ["toxic", "offensive", "hate"]) and model_score >= 0.5
            score = model_score if toxic else min(0.4, 1 - model_score)
        else:
            score = min(1.0, len(keywords) * 0.35 + len(sensitive) * 0.15)
            toxic = bool(keywords) or score >= 0.5

        return {
            "model": "bert-toxicity",
            "toxic": toxic,
            "label": "toxic" if toxic else "clean",
            "score": round(score, 3),
            "keywords": keywords,
            "sensitive_topics": sensitive,
            "topic_flagged": bool(sensitive),
            "source": self.source,
            "explanation": self._explain(toxic, keywords, sensitive),
        }

    def _explain(self, toxic, keywords, sensitive):
        if toxic and keywords:
            return "The text was flagged because it contains toxic or abusive terms."
        if sensitive:
            return "The text mentions sensitive topics and should be reviewed carefully."
        return "No strong toxicity indicators were found."


toxicity_model = BertToxicityModel()
