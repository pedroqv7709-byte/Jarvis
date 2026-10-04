-- Tasks are stored as JSON documents (payload) with the columns needed for lookup and
-- optimistic concurrency extracted alongside.
CREATE TABLE tasks (
    id          TEXT PRIMARY KEY,
    state       TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    version     INTEGER NOT NULL,
    payload     TEXT NOT NULL
);

CREATE INDEX idx_tasks_state ON tasks (state);
CREATE INDEX idx_tasks_created_at ON tasks (created_at);
