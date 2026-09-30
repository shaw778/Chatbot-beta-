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

                CREATE TABLE IF NOT EXISTS scheduled_comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comment_text TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    scheduled_at INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'scheduled',
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tran_id TEXT UNIQUE NOT NULL,
                    val_id TEXT,
                    package_id TEXT NOT NULL,
                    package_name TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'BDT',
                    cus_name TEXT NOT NULL,
                    cus_email TEXT NOT NULL,
                    cus_phone TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    payment_method TEXT,
                    bank_tran_id TEXT,
                    validation_payload TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bot_subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default_user',
                    package_id TEXT NOT NULL,
                    package_name TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    tran_id TEXT NOT NULL,
                    activated_at INTEGER NOT NULL,
                    expires_at INTEGER
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
                CREATE TABLE IF NOT EXISTS scheduled_comments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    comment_text TEXT NOT NULL,
                    platform VARCHAR(30) NOT NULL,
                    scheduled_at BIGINT NOT NULL,
                    status VARCHAR(30) NOT NULL DEFAULT 'scheduled',
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS payments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    tran_id VARCHAR(100) UNIQUE NOT NULL,
                    val_id VARCHAR(100),
                    package_id VARCHAR(50) NOT NULL,
                    package_name VARCHAR(100) NOT NULL,
                    amount DECIMAL(10,2) NOT NULL,
                    currency VARCHAR(10) NOT NULL DEFAULT 'BDT',
                    cus_name VARCHAR(100) NOT NULL,
                    cus_email VARCHAR(100) NOT NULL,
                    cus_phone VARCHAR(50) NOT NULL,
                    status VARCHAR(30) NOT NULL DEFAULT 'PENDING',
                    payment_method VARCHAR(50),
                    bank_tran_id VARCHAR(100),
                    validation_payload MEDIUMTEXT,
                    created_at BIGINT NOT NULL,
                    updated_at BIGINT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_subscriptions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL DEFAULT 'default_user',
                    package_id VARCHAR(50) NOT NULL,
                    package_name VARCHAR(100) NOT NULL,
                    tier VARCHAR(50) NOT NULL,
                    status VARCHAR(30) NOT NULL DEFAULT 'ACTIVE',
                    tran_id VARCHAR(100) NOT NULL,
                    activated_at BIGINT NOT NULL,
                    expires_at BIGINT
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

    def create_scheduled_comment(self, comment_text, platform, scheduled_at):
        p = self.param()
        sql = (
            "INSERT INTO scheduled_comments(comment_text, platform, scheduled_at, status, created_at) "
            f"VALUES ({p}, {p}, {p}, {p}, {p})"
        )
        values = (comment_text, platform, scheduled_at, "scheduled", now_ms())
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, values)
            item_id = cur.lastrowid
            cur.close()
        return {"id": item_id, "comment_text": comment_text, "platform": platform,
                "scheduled_at": scheduled_at, "status": "scheduled"}

    def scheduled_comments(self, limit=100):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, comment_text, platform, scheduled_at, status, created_at "
                "FROM scheduled_comments ORDER BY scheduled_at ASC LIMIT " + str(int(limit))
            )
            columns = [col[0] for col in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
            cur.close()
        return rows

    def create_payment(self, tran_id, package_id, package_name, amount, currency, cus_name, cus_email, cus_phone):
        p = self.param()
        now = now_ms()
        sql = (
            "INSERT INTO payments(tran_id, package_id, package_name, amount, currency, "
            "cus_name, cus_email, cus_phone, status, created_at, updated_at) "
            f"VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, 'PENDING', {p}, {p})"
        )
        values = (tran_id, package_id, package_name, float(amount), currency, cus_name, cus_email, cus_phone, now, now)
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, values)
            item_id = cur.lastrowid
            cur.close()
        return {
            "id": item_id,
            "tran_id": tran_id,
            "package_id": package_id,
            "package_name": package_name,
            "amount": amount,
            "currency": currency,
            "cus_name": cus_name,
            "cus_email": cus_email,
            "cus_phone": cus_phone,
            "status": "PENDING",
            "created_at": now
        }

    def update_payment(self, tran_id, status, val_id=None, payment_method=None, bank_tran_id=None, validation_payload=None):
        p = self.param()
        now = now_ms()
        val_str = json.dumps(validation_payload, ensure_ascii=False) if validation_payload is not None and not isinstance(validation_payload, str) else validation_payload
        sql = (
            f"UPDATE payments SET status = {p}, val_id = COALESCE({p}, val_id), "
            f"payment_method = COALESCE({p}, payment_method), bank_tran_id = COALESCE({p}, bank_tran_id), "
            f"validation_payload = COALESCE({p}, validation_payload), updated_at = {p} "
            f"WHERE tran_id = {p}"
        )
        values = (status, val_id, payment_method, bank_tran_id, val_str, now, tran_id)
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, values)
            cur.close()

    def get_payment(self, tran_id):
        p = self.param()
        sql = f"SELECT * FROM payments WHERE tran_id = {p} LIMIT 1"
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, (tran_id,))
            columns = [col[0] for col in cur.description]
            row = cur.fetchone()
            cur.close()
        return dict(zip(columns, row)) if row else None

    def list_payments(self, limit=50):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, tran_id, val_id, package_id, package_name, amount, currency, "
                "cus_name, cus_email, cus_phone, status, payment_method, bank_tran_id, created_at, updated_at "
                "FROM payments ORDER BY id DESC LIMIT " + str(int(limit))
            )
            columns = [col[0] for col in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
            cur.close()
        return rows

    def activate_subscription(self, package_id, package_name, tier, tran_id, user_id='default_user', duration_days=30):
        p = self.param()
        now = now_ms()
        expires = now + (duration_days * 24 * 3600 * 1000) if duration_days else None
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(f"UPDATE bot_subscriptions SET status = 'SUPERSEDED' WHERE user_id = {p} AND status = 'ACTIVE'", (user_id,))
            sql = (
                "INSERT INTO bot_subscriptions(user_id, package_id, package_name, tier, status, tran_id, activated_at, expires_at) "
                f"VALUES ({p}, {p}, {p}, {p}, 'ACTIVE', {p}, {p}, {p})"
            )
            cur.execute(sql, (user_id, package_id, package_name, tier, tran_id, now, expires))
            sub_id = cur.lastrowid
            cur.close()
        return {
            "id": sub_id,
            "user_id": user_id,
            "package_id": package_id,
            "package_name": package_name,
            "tier": tier,
            "status": "ACTIVE",
            "tran_id": tran_id,
            "activated_at": now,
            "expires_at": expires
        }

    def get_active_subscription(self, user_id='default_user'):
        p = self.param()
        sql = f"SELECT * FROM bot_subscriptions WHERE user_id = {p} AND status = 'ACTIVE' ORDER BY id DESC LIMIT 1"
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, (user_id,))
            if not cur.description:
                cur.close()
                return None
            columns = [col[0] for col in cur.description]
            row = cur.fetchone()
            cur.close()
        return dict(zip(columns, row)) if row else None


database = Database()
