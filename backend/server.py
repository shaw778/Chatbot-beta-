import json
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .ai_provider import call_anthropic
from .config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, DB_ENGINE, FRONTEND_DIR, HOST, PORT
from .database import database
from .models import pipeline


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length).decode("utf-8") if length else "{}"
    return json.loads(raw or "{}")


def send_json(handler, payload, status=HTTPStatus.OK):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class ChatbotHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path == "/":
            return str(FRONTEND_DIR / "index.html")
        return str(FRONTEND_DIR / path.lstrip("/"))

    def log_message(self, format, *args):
        print("[%s] %s" % (self.log_date_time_string(), format % args))

    def do_GET(self):
        if self.path == "/api/health":
            return send_json(
                self,
                {
                    "ok": True,
                    "database_engine": DB_ENGINE,
                    "database": database.label(),
                    "anthropic_key_configured": bool(ANTHROPIC_API_KEY),
                    "model": ANTHROPIC_MODEL,
                },
            )

        if self.path == "/api/conversations":
            return send_json(self, {"items": database.recent_conversations()})

        return super().do_GET()

    def do_POST(self):
        try:
            if self.path.startswith("/api/models/"):
                return self.handle_model_endpoint()

            if self.path == "/api/claude":
                payload = read_json(self)
                mode = payload.pop("mode", "chat")
                result = call_anthropic(payload)
                user_text = ""
                messages = payload.get("messages") or []
                if messages:
                    user_text = str(messages[-1].get("content", ""))
                database.store_conversation(mode, user_text, result.get("text", ""))
                return send_json(self, result)

            return send_json(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            traceback.print_exc()
            return send_json(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_model_endpoint(self):
        payload = read_json(self)
        text = payload.get("text", "")
        sentiment_label = payload.get("sentiment", "neutral")

        if self.path == "/api/models/sentiment":
            result = pipeline.sentiment(text)
            database.store_model_prediction("sentiment", text, result, result["bert"].get("source"))
            return send_json(self, result)

        if self.path == "/api/models/toxicity":
            result = pipeline.toxicity(text)
            database.store_model_prediction("toxicity", text, result, result.get("source"))
            return send_json(self, result)

        if self.path == "/api/models/engagement":
            result = pipeline.engagement(text, sentiment_label)
            database.store_model_prediction("engagement", text, result, result.get("source"))
            return send_json(self, result)

        if self.path == "/api/models/schedule":
            result = pipeline.schedule(text, sentiment_label, payload.get("predicted_engagement"))
            database.store_model_prediction("schedule", text, result, result.get("source"))
            return send_json(self, result)

        if self.path == "/api/models/pipeline":
            result = pipeline.full_pipeline(text)
            database.store_model_prediction("full_pipeline", text, result, "mixed")
            return send_json(self, result)

        return send_json(self, {"error": "Model endpoint not found"}, HTTPStatus.NOT_FOUND)


def main():
    database.init()
    server = ThreadingHTTPServer((HOST, PORT), ChatbotHandler)
    print(f"Chatbot running at http://{HOST}:{PORT}")
    print(f"Database engine: {DB_ENGINE}")
    print(f"Database: {database.label()}")
    server.serve_forever()
