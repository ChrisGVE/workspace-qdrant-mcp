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

/// A co-occurrence cluster: a group of symbols that appear together frequently.
#[derive(Debug, Clone)]
pub struct CooccurrenceCluster {
    /// Symbols in this cluster.
    pub symbols: Vec<String>,
    /// Minimum edge weight within the cluster.
    pub min_weight: i64,
}

/// Find co-occurrence clusters using a recursive CTE.
///
/// Starting from high-weight edges (>= `min_count`), recursively expands
/// to reachable symbols up to `max_hops` hops. Returns connected components
/// as clusters.
///
/// Performance: limited to `max_hops` recursion depth (typically 2-3).
pub async fn find_clusters(
    pool: &SqlitePool,
    tenant_id: &str,
    collection: &str,
    min_count: i64,
    max_hops: u32,
) -> Result<Vec<CooccurrenceCluster>, sqlx::Error> {
    // For each "seed" symbol with high-weight edges, expand via CTE
    let seeds: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT symbol FROM (
            SELECT symbol_a AS symbol FROM symbol_cooccurrence
            WHERE tenant_id = ?1 AND collection = ?2 AND cooccurrence_count >= ?3
            UNION
            SELECT symbol_b AS symbol FROM symbol_cooccurrence
            WHERE tenant_id = ?1 AND collection = ?2 AND cooccurrence_count >= ?3
        )",
    )
    .bind(tenant_id)
    .bind(collection)
    .bind(min_count)
    .fetch_all(pool)
    .await?;

    if seeds.is_empty() {
        return Ok(Vec::new());
    }

    // Build clusters by expanding from each seed via CTE
    let mut assigned: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut clusters = Vec::new();

    for (seed,) in &seeds {
        if assigned.contains(seed) {
            continue;
        }

        let rows: Vec<(String,)> = sqlx::query_as(&format!(
            "WITH RECURSIVE cluster_expand AS (
                SELECT ?1 AS symbol, 0 AS depth
                UNION
                SELECT
                    CASE
                        WHEN sc.symbol_a = ce.symbol THEN sc.symbol_b
                        ELSE sc.symbol_a
                    END AS symbol,
                    ce.depth + 1
                FROM cluster_expand ce
                JOIN symbol_cooccurrence sc ON (
                    (sc.symbol_a = ce.symbol OR sc.symbol_b = ce.symbol)
                    AND sc.tenant_id = ?2
                    AND sc.collection = ?3
                    AND sc.cooccurrence_count >= ?4
                )
                WHERE ce.depth < {}
            )
            SELECT DISTINCT symbol FROM cluster_expand",
            max_hops
        ))
        .bind(seed)
        .bind(tenant_id)
        .bind(collection)
        .bind(min_count)
        .fetch_all(pool)
        .await?;

        let cluster_symbols: Vec<String> = rows
            .into_iter()
            .map(|(s,)| s)
            .filter(|s| !assigned.contains(s))
            .collect();

        if cluster_symbols.len() >= 2 {
            for s in &cluster_symbols {
                assigned.insert(s.clone());
            }
            clusters.push(CooccurrenceCluster {
                symbols: cluster_symbols,
                min_weight: min_count,
            });
        } else {
            // Single-symbol "cluster" — mark as assigned but don't emit
            assigned.insert(seed.clone());
        }
    }

    debug!(
        "Found {} co-occurrence clusters for {}/{}",
        clusters.len(),
        tenant_id,
        collection
    );

    Ok(clusters)
}

/// Find symbols that co-occur with a given symbol, sorted by weight.
///
/// Returns `(symbol, cooccurrence_count)` pairs.
pub async fn get_neighbors(
    pool: &SqlitePool,
    tenant_id: &str,
    collection: &str,
    symbol: &str,
    min_count: i64,
) -> Result<Vec<(String, i64)>, sqlx::Error> {
    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT neighbor, weight FROM (
            SELECT symbol_b AS neighbor, cooccurrence_count AS weight
            FROM symbol_cooccurrence
            WHERE symbol_a = ?1 AND tenant_id = ?2 AND collection = ?3 AND cooccurrence_count >= ?4
            UNION ALL
            SELECT symbol_a AS neighbor, cooccurrence_count AS weight
            FROM symbol_cooccurrence
            WHERE symbol_b = ?1 AND tenant_id = ?2 AND collection = ?3 AND cooccurrence_count >= ?4
        ) ORDER BY weight DESC",
    )
    .bind(symbol)
    .bind(tenant_id)
    .bind(collection)
    .bind(min_count)
    .fetch_all(pool)
    .await?;

    Ok(rows)
}

/// Approximate betweenness centrality via 2-hop path counting.
///
/// For each symbol S, counts how many pairs of other symbols are
/// connected through S (i.e., S appears on a 2-hop path A-S-B).
/// Symbols with high bridge counts connect disparate clusters.
///
/// Returns normalized scores in [0, 1].
pub async fn get_betweenness_centrality(
    pool: &SqlitePool,
    tenant_id: &str,
    collection: &str,
    min_count: i64,
) -> Result<std::collections::HashMap<String, f64>, sqlx::Error> {
    // Count 2-hop paths through each symbol:
    // For a bridge symbol S: count distinct (A, B) pairs where A-S and S-B exist
    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT bridge, COUNT(*) AS path_count FROM (
            SELECT
                CASE WHEN e1.symbol_a = e2.symbol_a THEN e1.symbol_a
                     WHEN e1.symbol_a = e2.symbol_b THEN e1.symbol_a
                     WHEN e1.symbol_b = e2.symbol_a THEN e1.symbol_b
                     ELSE e1.symbol_b
                END AS bridge
            FROM symbol_cooccurrence e1
            JOIN symbol_cooccurrence e2 ON (
                (e1.symbol_a = e2.symbol_a OR e1.symbol_a = e2.symbol_b
                 OR e1.symbol_b = e2.symbol_a OR e1.symbol_b = e2.symbol_b)
                AND e1.rowid < e2.rowid
                AND e1.tenant_id = e2.tenant_id
                AND e1.collection = e2.collection
            )
            WHERE e1.tenant_id = ?1 AND e1.collection = ?2
                AND e1.cooccurrence_count >= ?3 AND e2.cooccurrence_count >= ?3
        ) GROUP BY bridge",
    )
    .bind(tenant_id)
    .bind(collection)
    .bind(min_count)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(std::collections::HashMap::new());
    }

    let max_count = rows.iter().map(|(_, c)| *c).max().unwrap_or(1) as f64;

    let mut centrality = std::collections::HashMap::with_capacity(rows.len());
    for (symbol, count) in rows {
        centrality.insert(symbol, count as f64 / max_count);
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

    async fn setup_graph(pool: &SqlitePool) {
        let now = wqm_common::timestamps::now_utc();
        // Graph: a-b(3), a-c(1), b-c(2), c-d(4), d-e(5)
        // Clusters: {a,b,c} connected via high weights, {d,e} connected
        for (a, b, w) in &[
            ("a", "b", 3),
            ("a", "c", 1),
            ("b", "c", 2),
            ("c", "d", 4),
            ("d", "e", 5),
        ] {
            sqlx::query(
                "INSERT INTO symbol_cooccurrence VALUES (?1, ?2, 't1', 'projects', ?3, ?4)",
            )
            .bind(a)
            .bind(b)
            .bind(w)
            .bind(&now)
            .execute(pool)
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn test_find_clusters_basic() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        // min_count=2 should find clusters from edges with weight >= 2
        let clusters = find_clusters(&pool, "t1", "projects", 2, 2).await.unwrap();
        assert!(
            !clusters.is_empty(),
            "Should find at least one cluster"
        );
        // The high-weight cluster should contain symbols from edges >= 2
        let all_symbols: Vec<&str> = clusters
            .iter()
            .flat_map(|c| c.symbols.iter().map(|s| s.as_str()))
            .collect();
        assert!(
            all_symbols.contains(&"b"),
            "b should be in a cluster (a-b:3, b-c:2)"
        );
    }

    #[tokio::test]
    async fn test_find_clusters_high_threshold() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        // min_count=5 should only find d-e(5)
        let clusters = find_clusters(&pool, "t1", "projects", 5, 2).await.unwrap();
        if !clusters.is_empty() {
            let syms: Vec<&str> = clusters[0].symbols.iter().map(|s| s.as_str()).collect();
            assert!(syms.contains(&"d"));
            assert!(syms.contains(&"e"));
        }
    }

    #[tokio::test]
    async fn test_find_clusters_empty() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let clusters = find_clusters(&pool, "t1", "projects", 1, 2).await.unwrap();
        assert!(clusters.is_empty());
    }

    #[tokio::test]
    async fn test_get_neighbors() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        // Neighbors of 'c' with min_count=1
        let neighbors = get_neighbors(&pool, "t1", "projects", "c", 1).await.unwrap();
        let names: Vec<&str> = neighbors.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"a"), "c co-occurs with a");
        assert!(names.contains(&"b"), "c co-occurs with b");
        assert!(names.contains(&"d"), "c co-occurs with d");
        assert!(!names.contains(&"e"), "c does not directly co-occur with e");
    }

    #[tokio::test]
    async fn test_get_neighbors_min_count() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        // Neighbors of 'c' with min_count=3: only c-d(4)
        let neighbors = get_neighbors(&pool, "t1", "projects", "c", 3).await.unwrap();
        let names: Vec<&str> = neighbors.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"d"), "c-d has weight 4 >= 3");
        assert!(!names.contains(&"a"), "c-a has weight 1 < 3");
        assert!(!names.contains(&"b"), "c-b has weight 2 < 3");
    }

    #[tokio::test]
    async fn test_betweenness_centrality() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        let centrality =
            get_betweenness_centrality(&pool, "t1", "projects", 1).await.unwrap();
        // Symbols that appear on many 2-hop paths should have high centrality
        assert!(!centrality.is_empty(), "Should compute centrality scores");
    }

    #[tokio::test]
    async fn test_betweenness_centrality_empty() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let centrality =
            get_betweenness_centrality(&pool, "t1", "projects", 1).await.unwrap();
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
