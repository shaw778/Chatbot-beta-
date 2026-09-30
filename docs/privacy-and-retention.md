# Privacy and Data Retention

## What the application receives

The application may receive text submitted for analysis, image files uploaded to the image-comment endpoint, and optional webhook payloads from connected Meta services.

Social account login uses provider-hosted OAuth pages. The application must never request,
receive, or store a user's Facebook, Instagram, X, or LinkedIn password. Authorization codes
must be exchanged server-side, and access/refresh tokens must be encrypted at rest before
publishing features are enabled.

## What is stored

- `conversations` stores submitted text or image filenames and generated responses.
- `api_calls` stores provider name, model, request metadata, response text, status, and errors. API keys are never written to this table.
- `model_predictions` stores model inputs and structured outputs for audit and debugging.
- `scheduled_comments` stores scheduled comment text, platform, and timestamps.
- Uploaded image bytes are processed in memory and are not stored by this application.
- Local Facebook OAuth writes the Page access token and Page ID to the git-ignored `.env`
	file as plaintext. This development convenience is not suitable for production.

## Retention

SQLite data remains on the local machine until an administrator deletes it. For classroom or development use, delete `data/chatbot.db` after the demonstration. Production deployments should implement a scheduled retention job, encrypted storage, access controls, and a documented retention period before accepting real user data.

Recommended starting policy:

- Keep generated audit records for 30 days.
- Delete uploaded media immediately after provider processing.
- Delete provider request and response text after 30 days unless required for an active investigation.
- Allow users to request deletion of their conversation records.
- Do not store passwords or unnecessary personal identifiers. Production access tokens must
	use encrypted storage; local `.env` token persistence is plaintext and development-only.

## User transparency

Before production use, provide a public privacy notice explaining the data categories, provider transfers, retention period, deletion contact, and Meta data-deletion callback. The current project provides the storage behavior and documentation foundation; deployment owners must publish the final legal notice for their jurisdiction.
