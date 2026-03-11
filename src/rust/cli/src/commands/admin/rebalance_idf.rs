//! Rebalance-IDF subcommand: correct IDF drift in stored sparse vectors.
//!
//! Sparse vectors are computed at ingest time using the corpus size N at that
//! moment. As the collection grows, IDF weights drift — older points have
//! over-scored rare terms compared to newer ones. This command applies a
//! correction factor to bring all points to the current N.
//!
//! Guard: only runs when N has grown by more than `--min-growth-pct` (default
//! 10%) since the last correction. Stores `last_corrected_n` in
//! `corpus_statistics` so repeated runs are idempotent.

use std::collections::HashMap;

use anyhow::{Context, Result};

use crate::output;

/// IDF formula: ln((N - df + 0.5) / (df + 0.5)), floored at 0.
///
/// Mirrors the formula in `embedding/bm25.rs`.
fn bm25_idf(n: u64, df: u64) -> f64 {
    if n == 0 || df == 0 {
        return 0.0;
    }
    let n = n as f64;
    let df = df as f64;
    ((n - df + 0.5) / (df + 0.5)).ln().max(0.0)
}

/// Correction factor to apply to an IDF weight that was computed at `old_n`
/// to bring it up to `new_n`.
///
/// Returns 1.0 when no correction is needed (old_n == new_n or old_idf ≈ 0).
pub fn idf_correction(old_n: u64, new_n: u64, df: u64) -> f32 {
    if old_n == new_n || df == 0 {
        return 1.0;
    }
    let old_idf = bm25_idf(old_n, df);
    let new_idf = bm25_idf(new_n, df);
    if old_idf < 1e-10 {
        return 1.0;
    }
    (new_idf / old_idf) as f32
}

/// Open state.db in read-write mode (as opposed to the read-only `open_state_db()`).
fn open_state_db_rw() -> Result<rusqlite::Connection> {
    let db_path =
        crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }

    let conn = rusqlite::Connection::open(&db_path)
        .context("Failed to open state database read-write")?;
    conn.execute_batch("PRAGMA busy_timeout=10000; PRAGMA journal_mode=WAL;")
        .context("Failed to configure SQLite connection")?;
    Ok(conn)
}

/// Run IDF drift correction for one or all collections.
pub async fn execute(
    collection: Option<String>,
    dry_run: bool,
    min_growth_pct: f64,
) -> Result<()> {
    output::section("IDF Drift Correction");

    if dry_run {
        output::info("Dry-run mode — no vectors will be updated");
    }

    // Open SQLite for corpus_statistics and sparse_vocabulary queries.
    let conn = match open_state_db_rw() {
        Ok(c) => c,
        Err(e) => {
            output::error(format!("Cannot open state database: {}", e));
            return Ok(());
        }
    };

    // Determine which collections to process.
    let collections: Vec<String> = match collection {
        Some(c) => vec![c],
        None => {
            let mut stmt = conn
                .prepare(
                    "SELECT collection FROM corpus_statistics WHERE total_documents > 0",
                )
                .context("Failed to query corpus_statistics")?;
            // Collect eagerly so the iterator (which borrows stmt) is dropped
            // before stmt goes out of scope.
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .context("Failed to read collections")?;
            rows.collect::<Result<Vec<_>, _>>()
                .context("Failed to collect collections")?
        }
    };

    if collections.is_empty() {
        output::info("No collections with corpus statistics found — nothing to do.");
        return Ok(());
    }

    // Build Qdrant storage client using environment config.
    let storage_config = build_storage_config();
    let storage_client = std::sync::Arc::new(
        workspace_qdrant_core::storage::StorageClient::with_config(storage_config),
    );

    let mut total_updated = 0u64;

    for coll in &collections {
        match rebalance_collection(
            &conn,
            &storage_client,
            coll,
            dry_run,
            min_growth_pct,
        )
        .await
        {
            Ok(updated) => total_updated += updated,
            Err(e) => output::error(format!("  {} — failed: {}", coll, e)),
        }
    }

    output::separator();
    if dry_run {
        output::info(format!(
            "Dry-run complete. Would update {} point(s).",
            total_updated
        ));
    } else {
        output::success(format!("Updated {} point(s) total.", total_updated));
    }
    Ok(())
}

/// Process IDF drift correction for a single collection.
async fn rebalance_collection(
    conn: &rusqlite::Connection,
    storage_client: &workspace_qdrant_core::storage::StorageClient,
    collection: &str,
    dry_run: bool,
    min_growth_pct: f64,
) -> Result<u64> {
    // Read corpus statistics.
    let (current_n, last_corrected_n): (i64, i64) = conn
        .query_row(
            "SELECT total_documents, last_corrected_n \
             FROM corpus_statistics WHERE collection = ?1",
            rusqlite::params![collection],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .context("Failed to read corpus_statistics")?;

    let current_n = current_n as u64;
    let last_corrected_n = last_corrected_n as u64;

    output::info(format!(
        "Collection '{}': current_n={}, last_corrected_n={}",
        collection, current_n, last_corrected_n
    ));

    // Check growth threshold.
    if last_corrected_n > 0 {
        let growth = (current_n as f64 - last_corrected_n as f64) / last_corrected_n as f64
            * 100.0;
        if growth < min_growth_pct {
            output::info(format!(
                "  Growth {:.1}% < {:.1}% threshold — skipping",
                growth, min_growth_pct
            ));
            return Ok(0);
        }
        output::info(format!("  Growth {:.1}% — proceeding with correction", growth));
    }

    // Load term_id → df mapping for this collection.
    let df_map: HashMap<u32, u64> = {
        let mut stmt = conn
            .prepare(
                "SELECT term_id, document_count \
                 FROM sparse_vocabulary WHERE collection = ?1",
            )
            .context("Failed to prepare sparse_vocabulary query")?;
        let rows = stmt
            .query_map(rusqlite::params![collection], |row| {
                Ok((row.get::<_, i64>(0)? as u32, row.get::<_, i64>(1)? as u64))
            })
            .context("Failed to read sparse_vocabulary")?;
        // Collect eagerly so the iterator (borrowing stmt) drops before stmt.
        rows.collect::<Result<HashMap<_, _>, _>>()
            .context("Failed to collect df_map")?
    };

    if df_map.is_empty() {
        output::info(format!("  No vocabulary entries for '{}' — skipping", collection));
        return Ok(0);
    }

    output::info(format!("  Loaded {} term entries", df_map.len()));

    // Scroll through Qdrant points (with sparse vectors) and apply corrections.
    let batch_size = 200u32;
    let mut offset: Option<String> = None;
    let mut total_updated = 0u64;
    let mut page = 0u32;

    loop {
        let (points, next_cursor) = storage_client
            .scroll_with_sparse_vectors(collection, batch_size, offset.clone())
            .await
            .context("Scroll failed")?;

        if points.is_empty() {
            break;
        }

        page += 1;
        let mut batch_updates: Vec<(String, HashMap<u32, f32>)> = Vec::new();
        let mut updated_epochs: Vec<(String, u64)> = Vec::new();

        for point in &points {
            // Read idf_epoch from payload.
            let old_n = match point.idf_epoch {
                Some(n) if n > 0 && n != current_n => n,
                _ => continue, // no epoch or already up to date
            };

            // Extract sparse vector.
            let sparse_map = match &point.sparse_vector {
                Some(m) if !m.is_empty() => m.clone(),
                _ => continue,
            };

            // Apply per-term correction.
            let corrected: HashMap<u32, f32> = sparse_map
                .into_iter()
                .map(|(term_id, val)| {
                    let df = df_map.get(&term_id).copied().unwrap_or(1);
                    let factor = idf_correction(old_n, current_n, df);
                    (term_id, val * factor)
                })
                .collect();

            batch_updates.push((point.id.clone(), corrected));
            updated_epochs.push((point.id.clone(), current_n));
        }

        let updated_in_batch = batch_updates.len() as u64;

        if !dry_run && !batch_updates.is_empty() {
            // Update sparse vectors.
            storage_client
                .update_named_sparse_vectors(collection, batch_updates)
                .await
                .context("Failed to update sparse vectors")?;

            // Update idf_epoch in payload for each corrected point.
            // Use set_payload on a per-point-ID filter.
            for (pid, new_epoch) in updated_epochs {
                let _ = update_idf_epoch_payload(
                    storage_client,
                    collection,
                    &pid,
                    new_epoch,
                )
                .await;
            }
        }

        total_updated += updated_in_batch;

        if page % 10 == 0 {
            output::info(format!(
                "  Page {}: {} corrected so far",
                page, total_updated
            ));
        }

        match next_cursor {
            Some(cursor) => offset = Some(cursor),
            None => break,
        }
    }

    output::info(format!(
        "  '{}': {} point(s) {}",
        collection,
        total_updated,
        if dry_run { "would be updated" } else { "updated" }
    ));

    // Update last_corrected_n in SQLite (not in dry-run).
    if !dry_run && total_updated > 0 {
        conn.execute(
            "UPDATE corpus_statistics SET last_corrected_n = ?1 WHERE collection = ?2",
            rusqlite::params![current_n as i64, collection],
        )
        .context("Failed to update last_corrected_n")?;
    }

    Ok(total_updated)
}

/// Update the `idf_epoch` payload field on a single point (best-effort).
///
/// Errors are silently ignored by the caller since the sparse vector
/// has already been corrected.
async fn update_idf_epoch_payload(
    storage_client: &workspace_qdrant_core::storage::StorageClient,
    collection: &str,
    point_id: &str,
    new_epoch: u64,
) -> anyhow::Result<()> {
    let mut payload = HashMap::new();
    payload.insert("idf_epoch".to_string(), serde_json::json!(new_epoch));

    storage_client
        .set_payload_on_point(collection, point_id, payload)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(())
}

/// Build a StorageConfig from the environment (mirrors daemon startup logic).
fn build_storage_config() -> workspace_qdrant_core::storage::StorageConfig {
    let mut config = workspace_qdrant_core::storage::StorageConfig::default();
    if let Ok(url) = std::env::var("QDRANT_URL") {
        config.url = url;
    }
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        config.api_key = Some(key);
    }
    config.check_compatibility = false;
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idf_correction_no_change_when_equal_n() {
        // When old_n == new_n, correction factor must be exactly 1.0.
        assert_eq!(idf_correction(1000, 1000, 5), 1.0);
        assert_eq!(idf_correction(500, 500, 1), 1.0);
    }

    #[test]
    fn test_idf_correction_grows_with_corpus() {
        // As N increases (df stays constant), IDF increases for rare terms
        // (numerator grows faster than denominator when df << N).
        // Correction factor > 1.0: upscale stored weights to current IDF.
        let factor = idf_correction(1000, 10000, 5);
        assert!(
            factor > 1.0,
            "factor {factor} should be > 1.0 when N grows 10x for a rare term"
        );
    }

    #[test]
    fn test_idf_correction_formula_accuracy() {
        // Given old_N=1000, new_N=10000, df=5:
        // old_idf = ln((1000 - 5 + 0.5) / (5 + 0.5)) = ln(995.5 / 5.5) ≈ ln(180.9) ≈ 5.198
        // new_idf = ln((10000 - 5 + 0.5) / (5 + 0.5)) = ln(9995.5 / 5.5) ≈ ln(1817.4) ≈ 7.505
        // correction ≈ 7.505 / 5.198 ≈ 1.444 (NOT < 1.0 — I was wrong above)
        // Actually wait: when N grows, IDF grows for rare terms (df << N).
        // IDF = ln((N - df + 0.5) / (df + 0.5)).
        // For df=5, N=1000: (994.5/5.5)=180.8, ln=5.197
        // For df=5, N=10000: (9994.5/5.5)=1817.2, ln=7.505
        // correction = 7.505/5.197 ≈ 1.444 > 1.0 (upscale for rare terms)
        let factor = idf_correction(1000, 10000, 5) as f64;
        let expected = {
            let old_idf = bm25_idf(1000, 5);
            let new_idf = bm25_idf(10000, 5);
            new_idf / old_idf
        };
        assert!(
            (factor - expected as f32 as f64).abs() < 0.01,
            "factor {} should be close to expected {:.4}",
            factor,
            expected
        );
    }

    #[test]
    fn test_idf_correction_zero_df_returns_one() {
        // df=0 is degenerate — return 1.0 to avoid division by zero.
        assert_eq!(idf_correction(1000, 2000, 0), 1.0);
    }

    #[test]
    fn test_bm25_idf_zero_n_returns_zero() {
        assert_eq!(bm25_idf(0, 5), 0.0);
    }

    #[test]
    fn test_bm25_idf_floored_at_zero() {
        // When df > N/2, numerator < denominator → ln < 0. Must be clamped to 0.
        let idf = bm25_idf(10, 9); // ln((10-9+0.5)/(9+0.5)) = ln(1.5/9.5) < 0
        assert_eq!(idf, 0.0, "IDF should be floored at 0 for common terms");
    }
}
