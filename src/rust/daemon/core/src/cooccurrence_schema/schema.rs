//! SQL schema constants for the symbol_cooccurrence table.

/// SQL to create the symbol_cooccurrence table (schema v23).
pub const CREATE_SYMBOL_COOCCURRENCE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS symbol_cooccurrence (
    symbol_a TEXT NOT NULL,
    symbol_b TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    cooccurrence_count INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol_a, symbol_b, tenant_id, collection)
)
"#;

/// SQL to create indexes on symbol_cooccurrence.
pub const CREATE_COOCCURRENCE_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_cooccurrence_tenant ON symbol_cooccurrence(tenant_id, collection)",
    "CREATE INDEX IF NOT EXISTS idx_cooccurrence_symbol_a ON symbol_cooccurrence(symbol_a, tenant_id, collection)",
    "CREATE INDEX IF NOT EXISTS idx_cooccurrence_symbol_b ON symbol_cooccurrence(symbol_b, tenant_id, collection)",
];
