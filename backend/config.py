import os
from pathlib import Path


def load_local_env():
    """Load development-only .env values without replacing real environment variables."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"


def resolve_project_path(raw_value, fallback):
    """Resolve relative filesystem paths from the project root rather than the caller's CWD."""
    path = Path(raw_value) if raw_value is not None and str(raw_value).strip() else Path(fallback)
    path = path.expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_IMAGE_REQUESTS = int(os.environ.get("RATE_LIMIT_IMAGE_REQUESTS", "10"))

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# A current Responses API model. Override this in .env if your account uses a
# different approved model.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
META_APP_ID = os.environ.get("META_APP_ID", "")
META_APP_SECRET = os.environ.get("META_APP_SECRET", "")
META_VERIFY_TOKEN = os.environ.get("META_VERIFY_TOKEN", "")
META_PAGE_ACCESS_TOKEN = os.environ.get("META_PAGE_ACCESS_TOKEN", "")
META_PAGE_ID = os.environ.get("META_PAGE_ID", "")
META_LOGIN_CONFIG_ID = os.environ.get("META_LOGIN_CONFIG_ID", "")
META_OAUTH_SCOPES = os.environ.get(
    "META_OAUTH_SCOPES",
    ",".join([
        "public_profile",
        "email",
        "pages_show_list",
        "pages_read_engagement",
        "pages_read_user_content",
        "pages_manage_metadata",
        "pages_manage_posts",
        "pages_messaging",
        "pages_manage_engagement",
        "pages_manage_ads",
        "read_insights",
        "business_management",
        "instagram_basic",
        "instagram_content_publish",
        "instagram_manage_comments",
        "instagram_manage_insights",
        "instagram_manage_messages",
        "page_events",
        "leads_retrieval",
    ]),
)
META_REQUIRE_SIGNATURE = os.environ.get("META_REQUIRE_SIGNATURE", "false").lower() == "true"
META_REDIRECT_URI = os.environ.get(
    "META_REDIRECT_URI",
    "https://rights-diameter-katrina-actress.trycloudflare.com/auth/meta/callback",
)
X_CLIENT_ID = os.environ.get("X_CLIENT_ID", "")
X_CLIENT_SECRET = os.environ.get("X_CLIENT_SECRET", "")
X_REDIRECT_URI = os.environ.get("X_REDIRECT_URI", "http://127.0.0.1:8000/auth/x/callback")
TOKEN_ENCRYPTION_KEY = os.environ.get("TOKEN_ENCRYPTION_KEY", "")
OAUTH_STATE_SECRET = os.environ.get("OAUTH_STATE_SECRET", TOKEN_ENCRYPTION_KEY)

DB_ENGINE = os.environ.get("DB_ENGINE", "sqlite").lower()
DB_PATH = resolve_project_path(os.environ.get("DB_PATH", DATA_DIR / "chatbot.db"), DATA_DIR / "chatbot.db")

MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "chatbot"),
}

SENTIMENT_MODEL_PATH = str(resolve_project_path(os.environ.get("SENTIMENT_MODEL_PATH", MODEL_DIR / "sentiment_bert"), MODEL_DIR / "sentiment_bert"))
TOXICITY_MODEL_PATH = str(resolve_project_path(os.environ.get("TOXICITY_MODEL_PATH", MODEL_DIR / "toxicity_bert"), MODEL_DIR / "toxicity_bert"))
ENGAGEMENT_MODEL_PATH = str(resolve_project_path(os.environ.get("ENGAGEMENT_MODEL_PATH", MODEL_DIR / "engagement_xgboost.pkl"), MODEL_DIR / "engagement_xgboost.pkl"))
SCHEDULING_MODEL_PATH = str(resolve_project_path(os.environ.get("SCHEDULING_MODEL_PATH", MODEL_DIR / "scheduling_random_forest.pkl"), MODEL_DIR / "scheduling_random_forest.pkl"))
COMMENT_GENERATOR_MODEL_PATH = str(resolve_project_path(os.environ.get("COMMENT_GENERATOR_MODEL_PATH", MODEL_DIR / "comment_generator_bert"), MODEL_DIR / "comment_generator_bert"))

SSLCOMMERZ_STORE_ID = os.environ.get("SSLCOMMERZ_STORE_ID", "testbox")
SSLCOMMERZ_STORE_PASSWD = os.environ.get("SSLCOMMERZ_STORE_PASSWD", "qwerty")
SSLCOMMERZ_IS_SANDBOX = os.environ.get("SSLCOMMERZ_IS_SANDBOX", "true").lower() == "true"
APP_BASE_URL = os.environ.get("APP_BASE_URL", f"http://{HOST}:{PORT}")
