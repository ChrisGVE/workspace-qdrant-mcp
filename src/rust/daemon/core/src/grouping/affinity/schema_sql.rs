/// SQL to create the project_embeddings table (schema v25).
pub const CREATE_PROJECT_EMBEDDINGS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS project_embeddings (
    tenant_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    label TEXT,
    updated_at TEXT NOT NULL
)
"#;

/// SQL to create the affinity_labels table (schema v25).
pub const CREATE_AFFINITY_LABELS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS affinity_labels (
    group_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    category TEXT NOT NULL,
    score REAL NOT NULL,
    updated_at TEXT NOT NULL
)
"#;
