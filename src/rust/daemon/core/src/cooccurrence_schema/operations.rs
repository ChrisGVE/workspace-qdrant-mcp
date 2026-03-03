//! Database operations for symbol co-occurrence.

use sqlx::SqlitePool;
use tracing::debug;

use super::types::CooccurrenceCluster;

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
