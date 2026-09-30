import re
from pathlib import Path

from ..config import TOXICITY_MODEL_PATH
from .text_features import (
    BENIGN_HATE_RE,
    BENIGN_KILL_RE,
    BENIGN_STUPID_RE,
    BENIGN_SUCK_RE,
    BENIGN_TRASH_RE,
    PERSONAL_INSULTS,
    PROFANITIES,
    SCAM_PATTERNS,
    SENSITIVE_TOPICS,
    VIOLENT_THREATS,
)


def _matches_phrase(phrase, text):
    """Check if phrase matches with word boundaries or unicode word edges."""
    if " " in phrase:
        return phrase in text
    pat = r"(?:^|[^\w\u0980-\u09FF])" + re.escape(phrase) + r"(?:[^\w\u0980-\u09FF]|$)"
    return bool(re.search(pat, text))


class BertToxicityModel:
    def __init__(self):
        self.model_path = Path(TOXICITY_MODEL_PATH)
        self.pipeline = None
        self.source = "nlp-local-moderator"
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
            self.source = "nlp-local-moderator"

    def predict(self, text):
        if not text or not str(text).strip():
            return {
                "model": "nlp-toxicity-moderator",
                "toxic": False,
                "label": "clean",
                "score": 0.0,
                "keywords": [],
                "sensitive_topics": [],
                "topic_flagged": False,
                "source": self.source,
                "explanation": "No text provided for moderation.",
            }

        # 1. If fine-tuned BERT transformer is available locally, run it
        if self.pipeline:
            try:
                pred = self.pipeline(text[:512])[0]
                label = pred.get("label", "").lower()
                model_score = float(pred.get("score", 0.0))
                is_toxic = any(x in label for x in ["toxic", "offensive", "hate"]) and model_score >= 0.5
                score = model_score if is_toxic else min(0.35, 1.0 - model_score)
                keywords = self._extract_matched_terms(text)
                sensitive = [s for s in SENSITIVE_TOPICS if _matches_phrase(s, text.lower())]
                return {
                    "model": "bert-toxicity",
                    "toxic": is_toxic,
                    "label": "toxic" if is_toxic else "clean",
                    "score": round(score, 3),
                    "keywords": keywords,
                    "sensitive_topics": sensitive,
                    "topic_flagged": bool(sensitive),
                    "source": "transformers-bert",
                    "explanation": self._explain(is_toxic, keywords, sensitive),
                }
            except Exception:
                pass

        # 2. Comprehensive Multi-Category NLP Moderation Engine
        return self._rule_based_moderate(text)

    def _extract_matched_terms(self, text):
        lower = text.lower()
        matched = []
        for phrase in VIOLENT_THREATS | PERSONAL_INSULTS | PROFANITIES:
            if _matches_phrase(phrase, lower):
                matched.append(phrase)
        return matched

    def _rule_based_moderate(self, text):
        lower = text.lower()
        flagged_keywords = []
        threats_found = []
        insults_found = []
        profanity_found = []
        scams_found = []

        # 1. Check Violent Threats (Highest Severity)
        for threat in VIOLENT_THREATS:
            if _matches_phrase(threat, lower):
                threats_found.append(threat)
                flagged_keywords.append(threat)

        # 2. Check Personal Insults
        for insult in PERSONAL_INSULTS:
            if _matches_phrase(insult, lower):
                insults_found.append(insult)
                flagged_keywords.append(insult)

        # 3. Check Profanities
        for prof in PROFANITIES:
            if _matches_phrase(prof, lower):
                profanity_found.append(prof)
                flagged_keywords.append(prof)

        # 4. Check Scams
        for pattern in SCAM_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                scams_found.append(match.group(0))
                flagged_keywords.append("suspicious financial solicitation")

        # 5. Check Sensitive Topics (suicide, drugs, weapons, etc.)
        sensitive = [s for s in SENSITIVE_TOPICS if _matches_phrase(s, lower)]

        # 6. Apply Anti-False-Positive Filter for benign context
        # Check if text contains benign phrases like "hate when it rains", "take out trash", "killing it", "stupid bug"
        is_benign_hate = bool(BENIGN_HATE_RE.search(lower))
        is_benign_trash = bool(BENIGN_TRASH_RE.search(lower))
        is_benign_kill = bool(BENIGN_KILL_RE.search(lower))
        is_benign_stupid = bool(BENIGN_STUPID_RE.search(lower))
        is_benign_suck = bool(BENIGN_SUCK_RE.search(lower))

        # Filter out accidental keywords that were actually benign
        if is_benign_hate and not any(re.search(r"\bhate\s+you\b", lower) for _ in [1]):
            threats_found = [t for t in threats_found if "hate" not in t]
            insults_found = [i for i in insults_found if "hate" not in i]
            flagged_keywords = [k for k in flagged_keywords if k != "hate"]
        if is_benign_trash:
            insults_found = [i for i in insults_found if i != "trash"]
            flagged_keywords = [k for k in flagged_keywords if k != "trash"]
        if is_benign_kill and not any(k in lower for k in ["kill yourself", "kys", "i will kill you", "want to kill you", "মেরে ফেলব", "মর তুই"]):
            threats_found = [t for t in threats_found if "kill" not in t]
            flagged_keywords = [k for k in flagged_keywords if "kill" not in k]
        if is_benign_stupid:
            insults_found = [i for i in insults_found if i != "stupid"]
            flagged_keywords = [k for k in flagged_keywords if k != "stupid"]
        if is_benign_suck:
            insults_found = [i for i in insults_found if "suck" not in i]
            flagged_keywords = [k for k in flagged_keywords if "suck" not in k]


        # Calculate toxicity score and boolean
        if threats_found:
            score = 0.98
            toxic = True
        elif insults_found:
            score = min(0.95, 0.72 + len(insults_found) * 0.08)
            toxic = True
        elif profanity_found:
            score = min(0.85, 0.65 + len(profanity_found) * 0.08)
            toxic = True
        elif scams_found:
            score = 0.75
            toxic = True
        elif sensitive:
            score = 0.45
            # Sensitive topics alone without insults are flagged for review, but may be informational
            toxic = any(s in ["suicide", "self-harm", "terrorism", "weapon"] for s in sensitive)
        else:
            # Benign expressions of frustration or garbage
            if is_benign_hate or is_benign_trash or is_benign_kill:
                score = 0.05
            else:
                score = 0.02
            toxic = False

        label = "toxic" if toxic else "clean"
        unique_keywords = list(dict.fromkeys(flagged_keywords))[:5]
        explanation = self._explain(toxic, unique_keywords, sensitive, threats_found, insults_found, scams_found)

        return {
            "model": "nlp-toxicity-moderator",
            "toxic": toxic,
            "label": label,
            "score": round(score, 3),
            "keywords": unique_keywords,
            "sensitive_topics": sensitive,
            "topic_flagged": bool(sensitive),
            "source": self.source,
            "explanation": explanation,
        }

    def _explain(self, toxic, keywords, sensitive, threats=None, insults=None, scams=None):
        if threats:
            return f"Blocked for severe violent threats or incitement: {', '.join(repr(k) for k in threats[:2])}."
        if insults:
            return f"Flagged for direct personal attacks or abusive insults: {', '.join(repr(k) for k in insults[:3])}."
        if scams:
            return "Flagged for deceptive financial or spam solicitation."
        if toxic and keywords:
            return f"Flagged for inappropriate or toxic content: {', '.join(repr(k) for k in keywords[:3])}."
        if sensitive:
            return f"Contains sensitive reference to {', '.join(repr(s) for s in sensitive[:2])}; requires review."
        return "Clean: No toxic, abusive, or threatening language detected."


toxicity_model = BertToxicityModel()
