import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

DB_ENGINE = os.environ.get("DB_ENGINE", "sqlite").lower()
DB_PATH = Path(os.environ.get("DB_PATH", DATA_DIR / "chatbot.db"))

MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "chatbot"),
}

SENTIMENT_MODEL_PATH = os.environ.get("SENTIMENT_MODEL_PATH", str(MODEL_DIR / "sentiment_bert"))
TOXICITY_MODEL_PATH = os.environ.get("TOXICITY_MODEL_PATH", str(MODEL_DIR / "toxicity_bert"))
ENGAGEMENT_MODEL_PATH = os.environ.get("ENGAGEMENT_MODEL_PATH", str(MODEL_DIR / "engagement_xgboost.pkl"))
SCHEDULING_MODEL_PATH = os.environ.get("SCHEDULING_MODEL_PATH", str(MODEL_DIR / "scheduling_random_forest.pkl"))
