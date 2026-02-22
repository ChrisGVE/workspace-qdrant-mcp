//! Symbol co-occurrence schema and operations.
//!
//! Stores pairwise symbol co-occurrence counts per tenant/collection.
//! Used by the keyword extraction pipeline to compute degree centrality
//! for concept tag boosting.
//!
//! Edges are stored with canonical ordering (`symbol_a < symbol_b`)
//! to halve storage and simplify lookups.

use sqlx::SqlitePool;
use tracing::debug;

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

/// Upsert co-occurrence pairs in batch, incrementing counts for existing edges.
///
/// Each pair `(a, b)` must be in canonical order (`a < b`).
pub async fn upsert_cooccurrences(
    pool: &SqlitePool,
    tenant_id: &str,
    collection: &str,
    pairs: &[(String, String)],
) -> Result<(), sqlx::Error> {
    if pairs.is_empty() {
        return Ok(());
    }

    let now = wqm_common::timestamps::now_utc();

    // Batch in chunks of 50 to avoid SQLite variable limits
    for chunk in pairs.chunks(50) {
        let mut tx = pool.begin().await?;
        for (sym_a, sym_b) in chunk {
            sqlx::query(
                "INSERT INTO symbol_cooccurrence (symbol_a, symbol_b, tenant_id, collection, cooccurrence_count, updated_at)
                 VALUES (?1, ?2, ?3, ?4, 1, ?5)
                 ON CONFLICT (symbol_a, symbol_b, tenant_id, collection)
                 DO UPDATE SET cooccurrence_count = cooccurrence_count + 1, updated_at = ?5",
            )
            .bind(sym_a)
            .bind(sym_b)
            .bind(tenant_id)
            .bind(collection)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
    }

    debug!(
        "Upserted {} co-occurrence pairs for {}/{}",
        pairs.len(),
        tenant_id,
        collection
    );
    Ok(())
}

/// Compute degree centrality for all symbols in a tenant/collection.
///
/// Returns a map from symbol name to normalized centrality score in [0, 1].
/// Degree centrality = sum of edge weights for a node / max sum across all nodes.
pub async fn get_degree_centrality(
    pool: &SqlitePool,
    tenant_id: &str,
    collection: &str,
) -> Result<std::collections::HashMap<String, f64>, sqlx::Error> {
    // Aggregate edge weights per symbol (appears as either symbol_a or symbol_b)
    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT symbol, SUM(weight) as total_weight FROM (
            SELECT symbol_a AS symbol, cooccurrence_count AS weight
            FROM symbol_cooccurrence
            WHERE tenant_id = ?1 AND collection = ?2
            UNION ALL
            SELECT symbol_b AS symbol, cooccurrence_count AS weight
            FROM symbol_cooccurrence
            WHERE tenant_id = ?1 AND collection = ?2
        ) GROUP BY symbol",
    )
    .bind(tenant_id)
    .bind(collection)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(std::collections::HashMap::new());
    }

    let max_weight = rows.iter().map(|(_, w)| *w).max().unwrap_or(1) as f64;

    let mut centrality = std::collections::HashMap::with_capacity(rows.len());
    for (symbol, weight) in rows {
        centrality.insert(symbol, weight as f64 / max_weight);
    }

    Ok(centrality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table_sql_syntax() {
        // Verify the SQL string parses (basic sanity check)
        assert!(CREATE_SYMBOL_COOCCURRENCE_SQL.contains("CREATE TABLE"));
        assert!(CREATE_SYMBOL_COOCCURRENCE_SQL.contains("symbol_a TEXT NOT NULL"));
        assert!(CREATE_SYMBOL_COOCCURRENCE_SQL.contains("PRIMARY KEY"));
    }

    #[test]
    fn test_indexes_count() {
        assert_eq!(CREATE_COOCCURRENCE_INDEXES_SQL.len(), 3);
        for sql in CREATE_COOCCURRENCE_INDEXES_SQL {
            assert!(sql.starts_with("CREATE INDEX"));
        }
    }

    #[tokio::test]
    async fn test_upsert_increments() {
        // In-memory SQLite for testing
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let pairs = vec![
            ("alpha".to_string(), "beta".to_string()),
            ("alpha".to_string(), "gamma".to_string()),
        ];

        // First insert
        upsert_cooccurrences(&pool, "t1", "projects", &pairs).await.unwrap();

        let count: i64 = sqlx::query_scalar(
            "SELECT cooccurrence_count FROM symbol_cooccurrence WHERE symbol_a = 'alpha' AND symbol_b = 'beta'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);

        // Second insert should increment
        upsert_cooccurrences(&pool, "t1", "projects", &pairs).await.unwrap();

        let count: i64 = sqlx::query_scalar(
            "SELECT cooccurrence_count FROM symbol_cooccurrence WHERE symbol_a = 'alpha' AND symbol_b = 'beta'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_degree_centrality() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        // Create a small graph: A-B(3), A-C(1), B-C(2)
        // A degree = 3+1=4, B degree = 3+2=5, C degree = 1+2=3
        let now = wqm_common::timestamps::now_utc();
        sqlx::query(
            "INSERT INTO symbol_cooccurrence VALUES ('a', 'b', 't1', 'projects', 3, ?1)"
        ).bind(&now).execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO symbol_cooccurrence VALUES ('a', 'c', 't1', 'projects', 1, ?1)"
        ).bind(&now).execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO symbol_cooccurrence VALUES ('b', 'c', 't1', 'projects', 2, ?1)"
        ).bind(&now).execute(&pool).await.unwrap();

        let centrality = get_degree_centrality(&pool, "t1", "projects").await.unwrap();

        // B has max degree (5), so B=1.0
        assert!((centrality["b"] - 1.0).abs() < 1e-6);
        // A has degree 4/5=0.8
        assert!((centrality["a"] - 0.8).abs() < 1e-6);
        // C has degree 3/5=0.6
        assert!((centrality["c"] - 0.6).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_empty_centrality() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let centrality = get_degree_centrality(&pool, "t1", "projects").await.unwrap();
        assert!(centrality.is_empty());
    }

    #[tokio::test]
    async fn test_tenant_isolation() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let pairs = vec![("x".to_string(), "y".to_string())];
        upsert_cooccurrences(&pool, "t1", "projects", &pairs).await.unwrap();
        upsert_cooccurrences(&pool, "t2", "projects", &pairs).await.unwrap();

        let c1 = get_degree_centrality(&pool, "t1", "projects").await.unwrap();
        let c2 = get_degree_centrality(&pool, "t2", "projects").await.unwrap();

        // Both should have entries, isolated by tenant
        assert!(c1.contains_key("x"));
        assert!(c2.contains_key("x"));

        // Count should be 1 each (not 2)
        let count: i64 = sqlx::query_scalar(
            "SELECT cooccurrence_count FROM symbol_cooccurrence WHERE tenant_id = 't1'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);
    }
}
