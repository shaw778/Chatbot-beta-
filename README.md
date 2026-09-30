# Intelligent Social Bot

Python backend, organized frontend, and database persistence for the BRAC CSE400 chatbot.

The project uses a dependency-light threaded HTTP server rather than FastAPI. Cloud AI
providers are optional: with no keys configured, deterministic local fallbacks keep the
demo and tests runnable offline.

## Project Structure

```text
backend/
  models/            Model wrappers for BERT, XGBoost, RF, and VADER
  ai_provider.py      Claude/OpenAI text and vision adapters + local fallback
  config.py           Environment/config settings
  database.py         SQLite/MySQL persistence layer
  server.py           HTTP API and static frontend server
frontend/
  index.html          Chatbot UI and analysis workflows
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

Install dependencies and run the focused checks first:

```powershell
\.venv\Scripts\python.exe -m pip install -r requirements.txt
\.venv\Scripts\python.exe -m pytest backend/tests -q
```

## Image Comment Workflow

`POST /api/comment/image` accepts `multipart/form-data` with `image_file`, `tone`,
`platform`, and an optional `provider`. The server:

- limits the request and image payload to 10 MB;
- requires an image MIME type and verifies the decoded file with Pillow;
- rejects images smaller than 100x100 pixels or with grayscale standard deviation below 5;
- treats supplied image instructions as passive data and rejects known instruction-injection text;
- records the filename and generated result in the existing conversation audit table.

When a funded OpenAI or Anthropic vision key is configured, the actual image bytes are sent
to that provider and the prompt asks for a specific, evidence-based comment. If the vision
provider is unavailable, the API returns a clear `503` instead of inventing visual details.
Text generation can still use the deterministic local fallback.

## Meta Webhook Contract

`GET /api/facebook/webhook` completes the Meta challenge when `META_VERIFY_TOKEN` matches.
`POST /api/facebook/webhook` accepts Page Messenger events, verifies `X-Hub-Signature-256`,
asks the configured bot provider for a reply, and sends it back through the Graph API.
Set `META_REQUIRE_SIGNATURE=true` for production; in that mode a missing app secret or
signature is rejected. Keep all values in `.env`; never put them in the frontend or commit them.

Configure the Facebook Page connection with:

```text
META_APP_ID=your_meta_app_id
META_APP_SECRET=your_meta_app_secret
META_VERIFY_TOKEN=random_webhook_verify_value
META_PAGE_ACCESS_TOKEN=your_page_access_token
META_LOGIN_CONFIG_ID=your_facebook_login_for_business_config_id
META_REDIRECT_URI=https://your-public-host/auth/meta/callback
OAUTH_STATE_SECRET=random_long_secret
META_REQUIRE_SIGNATURE=true
```

In the Meta developer dashboard, create a **Facebook Login for Business** configuration
with `pages_show_list`, `pages_read_engagement`, `pages_manage_posts`, and any other
Page permissions the app uses (such as `pages_messaging` for Messenger replies), then set its ID as
`META_LOGIN_CONFIG_ID`. Add `https://your-public-host/auth/meta/callback` as the OAuth
redirect URI, set the Page webhook callback to
`https://your-public-host/api/facebook/webhook`, use the same verify token, and subscribe
the Page to `messages`. A local `127.0.0.1` URL is not reachable by Meta; use an HTTPS
deployment or a temporary HTTPS tunnel while testing. The existing Connect Accounts button
starts OAuth. If `META_LOGIN_CONFIG_ID` is omitted, the app falls back to
`META_OAUTH_SCOPES`. The Page feed reader requires `pages_read_engagement`, while publishing
uses the separate `pages_manage_posts` permission. After adding a permission to the login
configuration, reconnect the Page so Meta issues a token with the updated grant. App users
who are not listed as app roles may require Advanced Access approved through Meta App Review.
If `META_PAGE_ACCESS_TOKEN` is configured directly, replace it with a token issued after
granting the permission. For local development, a successful Facebook OAuth connection
writes the Page token and Page ID to the git-ignored `.env` file. This file stores secrets
as plaintext; do not use this persistence approach in production.

For a public Meta callback, deploy behind HTTPS or use a temporary tunnel such as ngrok.
Localhost alone cannot receive callbacks from Meta.

## Social Account Connections

The UI provides square connection buttons for Facebook, Instagram, X, and LinkedIn. The
Facebook/Instagram and X buttons use OAuth authorization redirects; they never collect a
social-media password in this application. Configure `META_APP_ID`, `META_REDIRECT_URI`,
`X_CLIENT_ID`, `X_REDIRECT_URI`, and `OAUTH_STATE_SECRET` before enabling those providers.

The backend signs and validates the OAuth `state` value to prevent login-CSRF attacks.
Facebook Page credentials are persisted to `.env` for local development only; X tokens are
not persisted. Production deployments must use encrypted token storage. LinkedIn is shown
as unavailable until a LinkedIn OAuth application is configured.

## Claude API

Set your Anthropic key before starting the server:

```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
python app.py
```

## ChatGPT / OpenAI API

Set your OpenAI key before starting the server:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python app.py
```

Call the new ChatGPT endpoint at:

```text
POST /api/chatgpt
```

Request payload example:

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a friendly response."}
  ]
}
```

Without a key, the backend uses the same local fallback behavior as Claude.

Without a key, the backend uses local fallback responses so the chatbot still works and still records data.

## Reliability and Operations

- Every API response includes an `X-Request-ID` for support and log correlation.
- Requests are logged as structured JSON with method context, client address, and request ID.
- In-memory per-client rate limits protect POST endpoints; image uploads have a stricter limit.
- Configure limits with `RATE_LIMIT_WINDOW_SECONDS`, `RATE_LIMIT_REQUESTS`, and
  `RATE_LIMIT_IMAGE_REQUESTS`.
- Set `META_REQUIRE_SIGNATURE=true` in production to reject unsigned Meta webhooks.

## Privacy and Retention

See [docs/privacy-and-retention.md](docs/privacy-and-retention.md) for stored data,
deletion guidance, and the recommended 30-day audit retention policy. The application
processes uploaded image bytes in memory and does not persist the original media.

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

## SSLCommerz Payment & Bot Packages

The bot includes complete SSLCommerz payment integration to purchase and unlock 4 subscription packages:

### 1. The 4 Bot Packages
- **Starter Bot (Basic)** — `৳500 BDT` (~$5): 500 AI comments/mo, BERT Sentiment & Sarcasm, real-time toxicity filter, 1 connected social account.
- **Creator Bot (Pro)** — `৳1,500 BDT` (~$15): 2,500 AI comments/mo, Image Vision understanding (BLIP + AI), smart posting scheduler (RF + XGBoost), Facebook & Instagram feed auto-reply.
- **Business Bot (Agency)** — `৳3,500 BDT` (~$35): 10,000 AI comments/mo, unlimited Facebook pages & IG profiles, batch generation, automated reply engine, approval workflow.
- **Enterprise Bot (Unlimited)** — `৳7,500 BDT` (~$75): Unlimited AI comments, fine-tuned BERT & LLM pipeline, instant IPN hooks, zero rate limits, 24/7 dedicated support.

### 2. Backend Flow & Endpoints
- `POST /initiate-payment` (or `/api/payment/initiate`): Initiates SSLCommerz session via `sslcommerz-lib`, stores a `PENDING` payment in the database, and returns the `GatewayPageURL` and `sessionkey`.
- `POST /payment/ipn`: Server-to-server Instant Payment Notification (IPN) listener from SSLCommerz. Securely queries SSLCommerz validation API with `val_id` to confirm funds before unlocking features.
- `POST /payment/success` and `GET /payment/success`: Hosted Checkout and EasyCheckout return handler. Validates `val_id` and redirects user to active plan view.
- `POST /payment/fail` and `POST /payment/cancel`: Gateway callback failure and cancellation handlers.
- `GET /api/payment/packages`: Returns current packages and active subscription tier.
- `GET /api/payment/status`: Returns current bot tier and recent payment transactions.
- `GET /api/payment/history`: Returns transaction history.
- `POST /api/payment/simulate-test`: Developer endpoint to simulate successful IPN validation for local sandbox testing.

### 3. Frontend Experience
- **Hosted Checkout**: Directs user to SSLCommerz's official payment page (bKash, Nagad, Rocket, Cards, NetBanking) and redirects back on completion.
- **EasyCheckout (Seamless Modal)**: Opens SSLCommerz gateway in an in-app popup window with real-time status polling.
- **Transaction History**: Displays past payments with status badges (`PENDING`, `VALID`, `FAILED`) and verification details.

### 4. Configuration (.env)
```text
SSLCOMMERZ_STORE_ID=testbox
SSLCOMMERZ_STORE_PASSWD=qwerty
SSLCOMMERZ_IS_SANDBOX=true
APP_BASE_URL=http://127.0.0.1:8000
```
