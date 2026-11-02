from pathlib import Path
import json
import sqlite3
import threading
import time
import uuid


TERMINAL = {"imported", "complete", "failed", "cancelled"}


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "sessions.sqlite3"
        self.lock = threading.RLock()
        with self.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL NOT NULL, payload TEXT NOT NULL)")
        for job in self.list():
            if job["status"] not in TERMINAL:
                self.update(job["id"], status="failed", error="Processing was interrupted by a server restart. Import again to retry.")

    def connection(self):
        return sqlite3.connect(self.database, timeout=15)

    def create(self, filename, options):
        identifier = uuid.uuid4().hex
        job = {"id": identifier, "filename": filename, "created_at": time.time(), "status": "queued",
               "progress": 0., "duration": None, "sample_rate": 8000, "speakers": [],
               "detected_count": None, "flags": [], "transcript": [], "options": options, "error": None}
        self.directory(identifier).mkdir()
        with self.lock, self.connection() as db:
            db.execute("INSERT INTO jobs VALUES (?, ?, ?)", (identifier, job["created_at"], json.dumps(job)))
        return job

    def directory(self, identifier):
        if not identifier or any(c not in "0123456789abcdef" for c in identifier) or len(identifier) != 32:
            raise ValueError("Invalid session identifier")
        return self.root / identifier

    def get(self, identifier):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (identifier,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self):
        with self.connection() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM jobs ORDER BY created DESC")]

    def update(self, identifier, **updates):
        with self.lock, self.connection() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            job = json.loads(row[0])
            job.update(updates)
            db.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(job), identifier))
        return job

    def remove(self, identifier):
        with self.lock, self.connection() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (identifier,))
