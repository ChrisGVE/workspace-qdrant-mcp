//! FTS5 full-text search over `fts_content` (AC-F10.3, AC-F10.5).
//!
//! File: `wqm-storage/src/fts/search.rs`
//! Location: `src/rust/storage/src/fts/` (read crate)
//! Context: workspace-qdrant-mcp branch-storage model (arch §5.2, §6.2, §6.5).
//!   This is the SOLE FTS5 module in the read crate (arch §9 FP-2 / AC-F10.5).
//!   The facade method `fts_search` is the only entry point for FTS5 queries;
//!   `facade/read/fts.rs` MUST NOT EXIST (enforced by structural test).
//!
//! Query pattern (arch §5.2, AC-F10.3):
//!   `fts_content MATCH ? JOIN fts_branch_membership USING (blob_id)
//!    WHERE branch_id = ?`
//!
//!   `fts_branch_membership(branch_id, blob_id)` carries an index on
//!   `(branch_id, blob_id)` so the branch filter is a fast indexed scalar scan
//!   (NOT json_each). Proven by an EXPLAIN QUERY PLAN test in the test module.
//!
//! FTS5 sanitization contract (arch §6.5 A5):
//!   - All MATCH values are bound parameters, never string-interpolated.
//!   - Before binding, the query is sanitized via `sanitize_fts_query`, which
//!     wraps the entire input in double quotes (phrase search) unless the query
//!     is already a valid FTS5 quoted phrase, escaping any inner double-quote
//!     characters. This eliminates operator/colon ambiguity with one rule.
//!   - A malformed raw MATCH string (unmatched quotes, bareword operators) cannot
//!     reach the SQLite FTS5 engine.
//!
//! Neighbors: `crate::types::results::FtsResult` (output type), `crate::schema`
//!   (column name constants), `crate::connection::open_store_readonly` (pool).

use sqlx::SqlitePool;
use wqm_common::error::StorageError;

use crate::types::results::FtsResult;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run a branch-scoped FTS5 query against `fts_content`.
///
/// `query` is the raw user-supplied query string. It is sanitized and wrapped
/// as a phrase search before binding (arch §6.5 A5). `branch_id` scopes
/// results to one branch via the indexed `fts_branch_membership` junction.
/// `limit` caps the result count.
///
/// # Errors
///
/// Returns `StorageError::Sqlite` on SQL errors. A sanitization-induced
/// transformation is transparent to callers — the operation always succeeds
/// as long as the DB is reachable.
pub async fn fts_search(
    pool: &SqlitePool,
    query: &str,
    branch_id: &str,
    limit: u32,
) -> Result<Vec<FtsResult>, StorageError> {
    let safe_query = sanitize_fts_query(query);

    // FTS5 external-content restriction: snippet() can only be used when the
    // FTS5 virtual table is the direct outermost FROM with no JOINs. Joining
    // fts_content with other tables causes "unable to use function snippet in
    // the requested context". The workaround is a two-pass approach:
    //   Pass 1: query fts_content alone for (blob_id, rank, snip).
    //   Pass 2: JOIN blob_refs + files to filter by branch_id and add path.
    // This is correct because branch membership is enforced in pass 2.

    // Pass 1 — collect all FTS5 hits with snippet (no JOINs, large enough
    // limit to not miss branch-filtered results).
    let fts_limit = (limit as i64).saturating_mul(10).max(200);
    let fts_hits = sqlx::query_as::<_, FtsHitRow>(
        r#"
        SELECT rowid                                                      AS blob_id,
               rank,
               snippet(fts_content, 0, '<b>', '</b>', '...', 12)        AS snip
        FROM   fts_content
        WHERE  fts_content MATCH ?
        ORDER BY rank
        LIMIT ?
        "#,
    )
    .bind(&safe_query)
    .bind(fts_limit)
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("fts_search pass1 failed: {e}")))?;

    if fts_hits.is_empty() {
        return Ok(vec![]);
    }

    // Pass 2 — enrich with branch membership + file path using
    // idx_fts_branch (branch_id, blob_id) to scope to the requested branch.
    let blob_ids: Vec<i64> = fts_hits.iter().map(|h| h.blob_id).collect();
    // Build IN clause placeholders for blob_ids. branch_id is bound three
    // times (one per ? occurrence) then each blob_id once.
    let placeholders: String = std::iter::repeat_n("?", blob_ids.len())
        .collect::<Vec<_>>()
        .join(", ");

    let sql = format!(
        r#"
        SELECT
            f.file_id,
            f.relative_path    AS path,
            m.blob_id
        FROM fts_branch_membership m
        JOIN blob_refs br ON br.blob_id = m.blob_id AND br.branch_id = ?
        JOIN files f      ON f.file_id  = br.file_id AND f.branch_id = ?
        WHERE m.branch_id = ?
          AND m.blob_id IN ({placeholders})
        GROUP BY m.blob_id, f.file_id
        "#
    );

    // Bind: branch_id x3 (three ? in JOIN/WHERE), then blob_ids.
    let mut q = sqlx::query_as::<_, EnrichRow>(&sql)
        .bind(branch_id)
        .bind(branch_id)
        .bind(branch_id);
    for id in &blob_ids {
        q = q.bind(id);
    }
    let enriched = q
        .fetch_all(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("fts_search pass2 failed: {e}")))?;

    // Build a blob_id -> (file_id, path) map from pass 2.
    let enrich_map: std::collections::HashMap<i64, (i64, String)> = enriched
        .into_iter()
        .map(|r| (r.blob_id, (r.file_id, r.path)))
        .collect();

    // Merge pass 1 hits with pass 2 enrichment; apply branch filter implicitly
    // (hits absent from enrich_map are not on the requested branch).
    let mut rows: Vec<FtsResult> = fts_hits
        .into_iter()
        .filter_map(|h| {
            let (file_id, path) = enrich_map.get(&h.blob_id)?.clone();
            Some(FtsResult {
                file_id,
                path,
                blob_id: h.blob_id,
                score: -h.rank,
                snippet: h.snip,
            })
        })
        .collect();

    // Sort by descending score and cap to limit.
    rows.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit as usize);
    Ok(rows)
}

// ---------------------------------------------------------------------------
// FTS5 sanitization (arch §6.5 A5)
// ---------------------------------------------------------------------------

/// Sanitize a raw user query into a safe FTS5 MATCH string.
///
/// Strategy: wrap the entire input in double quotes (phrase search) after
/// escaping any double-quote characters inside it. This converts any sequence
/// of words into a phrase query and eliminates all FTS5 operator ambiguity
/// (`AND`, `OR`, `NOT`, `NEAR`, `^`, `*`, `-`, `(`, `)`, `:`).
///
/// The normalised form is `"<escaped input>"`. An empty input string yields
/// `""` which FTS5 treats as an empty phrase (returns zero rows harmlessly).
pub fn sanitize_fts_query(raw: &str) -> String {
    // Escape inner double quotes by doubling them (FTS5 phrase literal rule).
    let escaped = raw.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

// ---------------------------------------------------------------------------
// Internal row types for sqlx deserialization
// ---------------------------------------------------------------------------

/// Pass-1 row: raw FTS5 hit with snippet (no JOINs, snippet() works here).
#[derive(sqlx::FromRow)]
struct FtsHitRow {
    blob_id: i64,
    rank: f32,
    snip: Option<String>,
}

/// Pass-2 row: branch-scoped enrichment from blob_refs + files.
#[derive(sqlx::FromRow)]
struct EnrichRow {
    file_id: i64,
    path: String,
    blob_id: i64,
}

// ---------------------------------------------------------------------------
// Tests (AC-F10.3, AC-F10.5)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
