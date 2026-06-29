from pathlib import Path

from ..config import ENGAGEMENT_MODEL_PATH
from .text_features import feature_vector, hashtags


class XGBoostEngagementModel:
    def __init__(self):
        self.model_path = Path(ENGAGEMENT_MODEL_PATH)
        self.model = None
        self.source = "fallback"
        self._load()

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            import joblib

            self.model = joblib.load(self.model_path)
            self.source = "xgboost-artifact"
        except Exception:
            self.model = None
            self.source = "fallback"

    def predict(self, text, sentiment="neutral"):
        features = feature_vector(text, sentiment)
        if self.model:
            predicted = float(self.model.predict([features])[0])
        else:
            word_count, char_count, hash_count, pos, neg, sent_score = features
            predicted = 70 + min(word_count, 40) * 3 + min(char_count, 280) * 0.25
            predicted += hash_count * 15 + pos * 18 - neg * 12 + sent_score * 35
        predicted = max(10, min(1000, round(predicted)))
        return {
            "model": "xgboost-engagement",
            "predicted_engagement": predicted,
            "features": {
                "word_count": features[0],
                "char_count": features[1],
                "hashtag_count": len(hashtags(text)),
                "sentiment": sentiment,
            },
            "source": self.source,
        }


engagement_model = XGBoostEngagementModel()
