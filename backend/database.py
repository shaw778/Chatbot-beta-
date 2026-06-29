import json
import sqlite3
import time
from contextlib import contextmanager

from .config import DB_ENGINE, DB_PATH, MYSQL_CONFIG


def now_ms():
    return int(time.time() * 1000)


class Database:
    def __init__(self):
        self.engine = DB_ENGINE
        if self.engine not in {"sqlite", "mysql"}:
            raise ValueError("DB_ENGINE must be sqlite or mysql")

    def label(self):
        if self.engine == "mysql":
            return f"mysql://{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
        return str(DB_PATH)

    @contextmanager
    def connect(self):
        if self.engine == "mysql":
            conn = self._connect_mysql()
        else:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _connect_mysql(self):
        try:
            import mysql.connector
        except ImportError as exc:
            raise RuntimeError(
                "MySQL selected but mysql-connector-python is not installed. "
                "Run: pip install mysql-connector-python"
            ) from exc
        return mysql.connector.connect(**MYSQL_CONFIG)

    def init(self):
        if self.engine == "mysql":
            self._init_mysql()
        else:
            self._init_sqlite()

    def _init_sqlite(self):
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_text TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                """
            )

    def _init_mysql(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    mode VARCHAR(50) NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text MEDIUMTEXT,
                    created_at BIGINT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    provider VARCHAR(100) NOT NULL,
                    model VARCHAR(100) NOT NULL,
                    request_json MEDIUMTEXT NOT NULL,
                    response_text MEDIUMTEXT,
                    status VARCHAR(50) NOT NULL,
                    error MEDIUMTEXT,
                    created_at BIGINT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    input_text TEXT NOT NULL,
                    output_json MEDIUMTEXT NOT NULL,
                    source VARCHAR(100) NOT NULL,
                    created_at BIGINT NOT NULL
                )
                """
            )
            cur.close()

    def param(self):
        return "%s" if self.engine == "mysql" else "?"

    def store_api_call(self, provider, model, request_payload, response_text, status, error=None):
        p = self.param()
        sql = (
            "INSERT INTO api_calls(provider, model, request_json, response_text, status, error, created_at) "
            f"VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p})"
        )
        values = (
            provider,
            model,
            json.dumps(request_payload, ensure_ascii=False),
            response_text,
            status,
            error,
            now_ms(),
        )
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, values)
            cur.close()

    def store_conversation(self, mode, user_text, assistant_text):
        p = self.param()
        sql = f"INSERT INTO conversations(mode, user_text, assistant_text, created_at) VALUES ({p}, {p}, {p}, {p})"
        values = (mode or "chat", user_text or "", assistant_text or "", now_ms())
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, values)
            cur.close()

    def store_model_prediction(self, model_name, input_text, output, source):
        p = self.param()
        sql = (
            "INSERT INTO model_predictions(model_name, input_text, output_json, source, created_at) "
            f"VALUES ({p}, {p}, {p}, {p}, {p})"
        )
        values = (model_name, input_text or "", json.dumps(output, ensure_ascii=False), source or "unknown", now_ms())
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, values)
            cur.close()

    def recent_conversations(self, limit=50):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, mode, user_text, assistant_text, created_at "
                "FROM conversations ORDER BY id DESC LIMIT " + str(int(limit))
            )
            columns = [col[0] for col in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
            cur.close()
        return rows


database = Database()
