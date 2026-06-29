from .engagement_model import engagement_model
from .scheduling_model import scheduling_model
from .sentiment_model import sentiment_model
from .toxicity_model import toxicity_model
from .vader_model import analyze as vader_analyze


def sentiment(text):
    return {
        "vader": vader_analyze(text),
        "bert": sentiment_model.predict(text),
    }


def toxicity(text):
    return toxicity_model.predict(text)


def engagement(text, sentiment_label="neutral"):
    return engagement_model.predict(text, sentiment_label)


def schedule(text, sentiment_label="neutral", predicted_engagement=None):
    return scheduling_model.predict(text, sentiment_label, predicted_engagement)


def full_pipeline(text):
    sent = sentiment(text)
    sentiment_label = sent["bert"]["sentiment"]
    tox = toxicity(text)
    engage = engagement(text, sentiment_label)
    sched = schedule(text, sentiment_label, engage["predicted_engagement"])
    return {
        "sentiment": sent,
        "toxicity": tox,
        "engagement": engage,
        "schedule": sched,
    }
