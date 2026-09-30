# Project Documentation Manual

## System

**Intelligent Social Bot** is a decoupled browser client and Python HTTP API for comment
generation, text analysis, scheduling, persistence, and image-upload validation.

## Architecture

```text
Browser UI (frontend/index.html)
        | JSON and multipart/form-data
        v
Threaded Python HTTP server (backend/server.py)
        |-- AI provider adapters with offline fallback
        |-- model pipeline and validation
        `-- SQLite/MySQL persistence
```

The browser never receives provider credentials. The backend owns provider calls, logging,
database writes, file validation, and webhook authentication.

## Security Controls

1. Provider secrets are read from `.env` and `.env.example` contains placeholders only.
2. Image requests are bounded, decoded with Pillow, and rejected when corrupt, too small,
   or nearly uniform. Valid images are sent to the configured OpenAI or Anthropic vision adapter.
3. Known instruction-injection phrases are rejected in request text and filenames. Text
   found inside pixels is not treated as executable instructions by the provider prompt.
4. Meta POST events use HMAC-SHA256 with constant-time comparison. Set
        `META_REQUIRE_SIGNATURE=true` to require this check in production.
5. Local fallback mode avoids requiring network access during classroom demonstrations.

6. Every request receives a correlation ID and structured JSON log entry. Per-client POST
        rate limits protect the API, with a stricter limit for image uploads.

## Demonstration Checklist

```powershell
\.venv\Scripts\python.exe -m pip install -r requirements.txt
\.venv\Scripts\python.exe -m pytest backend/tests -q
\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8000`, choose **Image Comment**, and test a readable image followed
by a blank or very small image. The second request should return a clear validation error.
Also demonstrate a successful text comment, then show the truthful provider-failure response
when a vision provider has no credits. Do not present that failure as successful AI analysis.

## Honest Scope

The repository integrates Anthropic and OpenAI text APIs, OpenAI/Anthropic vision request
adapters, deterministic text fallbacks, and explicit `503` errors when image understanding
cannot run because provider credits or configuration are unavailable.

Privacy and retention details are documented in [docs/privacy-and-retention.md](privacy-and-retention.md).

## Q&A Vocabulary

- **Separation of concerns:** the browser, API, model adapters, and database have distinct responsibilities.
- **Multipart form data:** the HTTP encoding used to send binary files with regular fields.
- **Image variance:** a low-cost signal for detecting blank or severely low-contrast images.
- **Prompt injection:** untrusted text attempting to override model instructions.
- **HMAC-SHA256:** a keyed digest used to authenticate webhook payloads.
- **Offline fallback:** deterministic behavior used when a cloud provider is unavailable.