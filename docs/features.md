# Feature Organization

The UI still presents all 40 thesis features from `frontend/index.html`, while the backend is organized by responsibility.

## Frontend Feature Groups

- Preprocessing features: hashtag extraction, tokenization, cleanup, lemmatization, emoji/slang handling, counters.
- Sentiment features: VADER-style fast scoring, BERT-labeled deep sentiment, sarcasm/tone/topics, branch selection.
- Generation features: candidate comments, ranking, repeat penalty, platform-length optimization, context checks.
- Safety features: toxicity keyword filter, sensitive-topic flags, transparency notes, manual-review queue.
- Scheduling features: engagement prediction, timing recommendation, frequency cap, simulated posting delay.
- Optimization features: logs, approvals, drafts, preference learning, analytics, time-saved dashboard.

## Backend Feature Files

- `backend/server.py`: API endpoints and frontend serving.
- `backend/ai_provider.py`: Claude request handling and local fallback responses.
- `backend/database.py`: SQLite and MySQL database support.
- `backend/config.py`: environment variables and paths.
- `backend/models/sentiment_model.py`: BERT sentiment wrapper with fallback inference.
- `backend/models/toxicity_model.py`: BERT toxicity wrapper with fallback inference.
- `backend/models/engagement_model.py`: XGBoost engagement wrapper with fallback inference.
- `backend/models/scheduling_model.py`: Random Forest scheduling wrapper with fallback inference.
- `backend/models/vader_model.py`: VADER-compatible fast sentiment scorer.

## Database Tables

- `conversations`: stores chat mode, user text, assistant text, and timestamp.
- `api_calls`: stores provider, model, request JSON, response text, status, errors, and timestamp.
- `model_predictions`: stores model name, input text, output JSON, source, and timestamp.
