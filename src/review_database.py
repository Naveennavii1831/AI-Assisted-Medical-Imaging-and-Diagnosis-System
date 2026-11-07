
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "review_audit.db"


def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                decision TEXT NOT NULL
                    CHECK (decision IN ('approved', 'rejected')),
                reviewer TEXT NOT NULL,
                comments TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)

        connection.commit()


def save_review(
    review_id,
    filename,
    decision,
    reviewer,
    comments="",
):
    timestamp = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO reviews (
                review_id,
                filename,
                decision,
                reviewer,
                comments,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            review_id,
            filename,
            decision,
            reviewer,
            comments,
            timestamp,
        ))

        connection.commit()

        saved = connection.execute(
            "SELECT * FROM reviews WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    return dict(saved)


def get_reviews(limit=100):
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT
                id,
                review_id,
                filename,
                decision,
                reviewer,
                comments,
                created_at
            FROM reviews
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]