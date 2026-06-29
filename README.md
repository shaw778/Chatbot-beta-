# Intelligent Social Bot

Python backend, organized frontend, and database persistence for the BRAC CSE400 chatbot.

## Project Structure

```text
backend/
  models/            Model wrappers for BERT, XGBoost, RF, and VADER
  ai_provider.py      Claude API + local fallback responses
  config.py           Environment/config settings
  database.py         SQLite/MySQL persistence layer
  server.py           HTTP API and static frontend server
frontend/
  index.html          Chatbot UI and 40 client-side features
data/
  chatbot.db          Local SQLite database, created automatically
docs/
  features.md         Feature grouping and project notes
models/
  README.md           Where trained model artifacts should be placed
app.py                Main launcher
```

## Run

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

## Claude API

Set your Anthropic key before starting the server:

```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
python app.py
```

Without a key, the backend uses local fallback responses so the chatbot still works and still records data.

## Database Options

Default SQLite:

```powershell
$env:DB_ENGINE="sqlite"
$env:DB_PATH="data/chatbot.db"
python app.py
```

MySQL:

```powershell
pip install mysql-connector-python
$env:DB_ENGINE="mysql"
$env:MYSQL_HOST="127.0.0.1"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_mysql_password"
$env:MYSQL_DATABASE="chatbot"
python app.py
```

Create the MySQL database first:

```sql
CREATE DATABASE chatbot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Tables are created automatically:

- `conversations`: user messages and assistant responses
- `api_calls`: provider requests, responses, status, and errors
- `model_predictions`: model inputs/outputs for audit/debugging

## Model Endpoints

The backend now exposes all model endpoints:

```text
POST /api/models/sentiment   BERT sentiment + VADER fast score
POST /api/models/toxicity    BERT toxicity
POST /api/models/engagement  XGBoost engagement prediction
POST /api/models/schedule    Random Forest scheduling
POST /api/models/pipeline    Full model chain
```

Trained artifacts are optional. Put them in:

```text
models/sentiment_bert/
models/toxicity_bert/
models/engagement_xgboost.pkl
models/scheduling_random_forest.pkl
```

If artifacts are missing, deterministic fallback models are used so the chatbot remains runnable.
