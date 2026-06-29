# Model Artifacts

Put trained model files here when you have them.

Expected paths:

```text
models/sentiment_bert/              Hugging Face BERT sentiment model folder
models/toxicity_bert/               Hugging Face BERT toxicity model folder
models/engagement_xgboost.pkl       joblib-saved XGBoost regressor
models/scheduling_random_forest.pkl joblib-saved Random Forest classifier
```

The backend already implements all model wrappers and endpoints. If these files are missing, it uses deterministic fallback models so the app still runs.

Endpoints:

```text
POST /api/models/sentiment
POST /api/models/toxicity
POST /api/models/engagement
POST /api/models/schedule
POST /api/models/pipeline
```

Payload:

```json
{"text": "Your social media post here"}
```
