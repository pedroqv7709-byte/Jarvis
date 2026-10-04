"""SQLite-backed state store (standard library only).

Schema changes live in ``jarvis/state/migrations/NNNN_name.sql`` and are applied in order on
startup; applied versions are tracked in ``schema_migrations``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

from jarvis.core.ids import utcnow
from jarvis.core.task import Task, TaskState
from jarvis.state.store import ConcurrencyError, TaskNotFoundError

MIGRATIONS_PACKAGE = "jarvis.state.migrations"


def _load_migrations() -> list[tuple[int, str, str]]:
    migrations = []
    for entry in resources.files(MIGRATIONS_PACKAGE).iterdir():
        name = entry.name
        if not name.endswith(".sql"):
            continue
        prefix = name.split("_", 1)[0]
        if not prefix.isdigit():
            raise RuntimeError(f"migration {name!r} must start with a numeric version")
        migrations.append((int(prefix), name, entry.read_text(encoding="utf-8")))
    migrations.sort()
    versions = [v for v, _, _ in migrations]
    if len(versions) != len(set(versions)):
        raise RuntimeError("duplicate migration versions")
    return migrations


class SqliteStateStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def migrate(self) -> list[int]:
        """Apply pending migrations. Returns the versions applied by this call."""
        applied_now: list[int] = []
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
            )
            done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
            for version, name, sql in _load_migrations():
                if version in done:
                    continue
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for statement in _split_sql(sql):
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name, applied_at) "
                        "VALUES (?, ?, ?)",
                        (version, name, utcnow().isoformat()),
                    )
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                applied_now.append(version)
        return applied_now

    def applied_migrations(self) -> list[int]:
        with self._connect() as conn:
            return [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]

    def save(self, task: Task) -> None:
        new_version = task.version + 1
        payload = task.model_copy(update={"version": new_version}).model_dump_json()
        row = (
            task.state.value,
            task.request.user_id,
            task.created_at.isoformat(),
            task.updated_at.isoformat(),
            new_version,
            payload,
        )
        with self._connect() as conn:
            if task.version == 0:
                try:
                    conn.execute(
                        "INSERT INTO tasks (state, user_id, created_at, updated_at, version, "
                        "payload, id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (*row, task.id),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ConcurrencyError(f"task {task.id} already exists") from exc
            else:
                cursor = conn.execute(
                    "UPDATE tasks SET state = ?, user_id = ?, created_at = ?, updated_at = ?, "
                    "version = ?, payload = ? WHERE id = ? AND version = ?",
                    (*row, task.id, task.version),
                )
                if cursor.rowcount != 1:
                    raise ConcurrencyError(f"task {task.id} was modified concurrently")
        task.version = new_version

    def get(self, task_id: str) -> Task:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise TaskNotFoundError(task_id)
        return Task.model_validate_json(row[0])

    def list(self, state: TaskState | None = None, limit: int = 50) -> list[Task]:
        query = "SELECT payload FROM tasks"
        params: tuple[object, ...] = ()
        if state is not None:
            query += " WHERE state = ?"
            params = (state.value,)
        query += " ORDER BY created_at DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(query, (*params, limit)).fetchall()
        return [Task.model_validate_json(r[0]) for r in rows]


def _split_sql(sql: str) -> list[str]:
    """Split a migration into statements. Migrations must not contain ';' inside literals."""
    lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
