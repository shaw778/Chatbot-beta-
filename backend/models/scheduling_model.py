from pathlib import Path

from ..config import SCHEDULING_MODEL_PATH
from .text_features import feature_vector


SLOTS = ["Morning (6-10 AM)", "Afternoon (12-2 PM)", "Evening (6-8 PM)", "Night (9-11 PM)"]


class RandomForestSchedulingModel:
    def __init__(self):
        self.model_path = Path(SCHEDULING_MODEL_PATH)
        self.model = None
        self.source = "fallback"
        self._load()

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            import joblib

            self.model = joblib.load(self.model_path)
            self.source = "random-forest-artifact"
        except Exception:
            self.model = None
            self.source = "fallback"

    def predict(self, text, sentiment="neutral", predicted_engagement=None):
        features = feature_vector(text, sentiment)
        if self.model:
            slot = str(self.model.predict([features])[0])
        else:
            word_count, _, hash_count, pos, neg, sent_score = features
            if sent_score > 0 or pos > neg:
                slot = "Evening (6-8 PM)"
            elif hash_count >= 2:
                slot = "Afternoon (12-2 PM)"
            elif word_count < 8:
                slot = "Morning (6-10 AM)"
            else:
                slot = "Night (9-11 PM)"

        best_day = "weekend" if sentiment == "positive" and (predicted_engagement or 0) >= 250 else "weekday"
        return {
            "model": "random-forest-scheduling",
            "slot": slot if slot in SLOTS else "Evening (6-8 PM)",
            "best_day": best_day,
            "confidence": 0.68 if self.source == "fallback" else 0.82,
            "source": self.source,
        }


scheduling_model = RandomForestSchedulingModel()
