"""Opt-in local SQLite archive of research conversations.

Nothing is saved until the user clicks Save in the Streamlit sidebar.
The database stays on the machine and is excluded from Git.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / 'data' / 'research_history.sqlite3'


def save_session(kind: str, label: str, messages: list[dict], path: Path = DB_PATH) -> int:
    if kind not in {'sec', 'pdf'}:
        raise ValueError('Unknown research session type.')
    if not messages:
        raise ValueError('No conversation messages to save.')
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_messages = [{'role': item['role'], 'content': item['content']} for item in messages]
    with sqlite3.connect(path) as connection:
        connection.execute('''CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_utc TEXT NOT NULL,
            kind TEXT NOT NULL,
            label TEXT NOT NULL,
            messages_json TEXT NOT NULL
        )''')
        cursor = connection.execute(
            'INSERT INTO sessions (created_utc, kind, label, messages_json) VALUES (?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(timespec='seconds'), kind, label[:150], json.dumps(safe_messages)),
        )
        return int(cursor.lastrowid)


def list_sessions(path: Path = DB_PATH, limit: int = 15) -> list[dict]:
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute('SELECT id, created_utc, kind, label FROM sessions ORDER BY id DESC LIMIT ?',
                                  (max(1, min(int(limit), 100)),)).fetchall()
        return [dict(row) for row in rows]
