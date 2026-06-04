//! IDF drift correction engine (WI-f1, #82) — daemon-side.
//!
//! Sparse vectors are computed at ingest time using the corpus size N at that
//! moment. As the collection grows, IDF weights drift — older points over-score
//! rare terms relative to newer ones. This engine applies a correction factor to
//! bring all points to the current N.
//!
//! Moved out of the CLI (`cli/admin/rebalance_idf.rs`) so the CLI no longer does
//! direct Qdrant writes / opens state.db read-write: the daemon owns the engine
//! (reads `corpus_statistics`/`sparse_vocabulary` via the SQLite pool, writes
//! Qdrant via [`StorageClient`]), and persistence of `last_corrected_n` stays on
//! the single-writer path (the gRPC handler calls the WriteActor). Guard: a
//! collection is only corrected when N has grown by more than `min_growth_pct`
//! since the last correction (idempotent).

use std::collections::HashMap;

use sqlx::{Row, SqlitePool};

use crate::storage::{SparsePointData, StorageClient, StorageError};

/// Scroll page size for Qdrant point retrieval.
const SCROLL_BATCH_SIZE: u32 = 200;

/// Errors from the rebalance engine.
#[derive(Debug, thiserror::Error)]
pub enum RebalanceError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("qdrant error: {0}")]
    Storage(#[from] StorageError),
}

/// Per-collection rebalance outcome.
#[derive(Debug, Clone)]
pub struct CollectionRebalance {
    pub collection: String,
    pub current_n: u64,
    /// Points whose sparse vectors were (or, in dry-run, would be) corrected.
    pub updated: u64,
    /// Set when the collection was skipped (e.g. growth below threshold).
    pub skipped_reason: Option<String>,
    /// True when the caller should persist `last_corrected_n = current_n`
    /// (non-dry-run with at least one correction).
    pub persist_n: bool,
}

/// Full rebalance report across the processed collections.
#[derive(Debug, Clone, Default)]
pub struct RebalanceReport {
    pub per_collection: Vec<CollectionRebalance>,
    pub total_updated: u64,
    pub dry_run: bool,
}

// ── IDF math (moved verbatim from the CLI; mirrors embedding/bm25.rs) ─────────

/// IDF formula: `ln((N - df + 0.5) / (df + 0.5))`, floored at 0.
fn bm25_idf(n: u64, df: u64) -> f64 {
    // `df > n` (corrupt / partially-updated vocabulary) makes the numerator
    // negative → `ln()` is NaN, and `NaN.max(0.0)` is NaN in Rust (not 0), which
    // would silently poison the stored sparse weights. Treat it as 0.
    if n == 0 || df == 0 || df > n {
        return 0.0;
    }
    let n = n as f64;
    let df = df as f64;
    ((n - df + 0.5) / (df + 0.5)).ln().max(0.0)
}

/// Correction factor applied to an IDF weight computed at `old_n` to bring it to
/// `new_n`. Returns `1.0` when no correction is needed.
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

// ── Engine ───────────────────────────────────────────────────────────────────

/// Run IDF drift correction for one collection (when `Some`) or every collection
/// with corpus statistics. Reads SQLite via `pool`, writes Qdrant via `storage`.
/// Does NOT persist `last_corrected_n` (the caller does, via the WriteActor).
pub async fn rebalance_idf(
    pool: &SqlitePool,
    storage: &StorageClient,
    collection: Option<String>,
    dry_run: bool,
    min_growth_pct: f64,
) -> Result<RebalanceReport, RebalanceError> {
    let collections = match collection {
        Some(c) => vec![c],
        None => list_collections(pool).await?,
    };

    let mut report = RebalanceReport {
        dry_run,
        ..Default::default()
    };

    for coll in collections {
        let outcome = rebalance_collection(pool, storage, &coll, dry_run, min_growth_pct).await?;
        report.total_updated += outcome.updated;
        report.per_collection.push(outcome);
    }

    Ok(report)
}

/// Collections that have corpus statistics worth correcting.
async fn list_collections(pool: &SqlitePool) -> Result<Vec<String>, RebalanceError> {
    let rows = sqlx::query("SELECT collection FROM corpus_statistics WHERE total_documents > 0")
        .fetch_all(pool)
        .await?;
    Ok(rows
        .into_iter()
        .map(|r| r.get::<String, _>("collection"))
        .collect())
}

/// Process IDF drift correction for a single collection.
async fn rebalance_collection(
    pool: &SqlitePool,
    storage: &StorageClient,
    collection: &str,
    dry_run: bool,
    min_growth_pct: f64,
) -> Result<CollectionRebalance, RebalanceError> {
    let (current_n, last_corrected_n) = read_corpus_n(pool, collection).await?;

    // Growth guard (idempotency): skip when the corpus has not grown enough.
    if last_corrected_n > 0 {
        // Shrinkage / no growth: a NEGATIVE growth would satisfy
        // `growth < min_growth_pct` and wrongly trigger a correction run with
        // old_n > new_n. Skip explicitly before computing the percentage.
        if current_n <= last_corrected_n {
            return Ok(CollectionRebalance {
                collection: collection.to_string(),
                current_n,
                updated: 0,
                skipped_reason: Some("corpus has not grown since last correction".to_string()),
                persist_n: false,
            });
        }
        let growth = (current_n - last_corrected_n) as f64 / last_corrected_n as f64 * 100.0;
        if growth < min_growth_pct {
            return Ok(CollectionRebalance {
                collection: collection.to_string(),
                current_n,
                updated: 0,
                skipped_reason: Some(format!(
                    "growth {growth:.1}% < {min_growth_pct:.1}% threshold"
                )),
                persist_n: false,
            });
        }
    }

    let df_map = load_df_map(pool, collection).await?;
    if df_map.is_empty() {
        return Ok(CollectionRebalance {
            collection: collection.to_string(),
            current_n,
            updated: 0,
            skipped_reason: Some("no vocabulary entries".to_string()),
            persist_n: false,
        });
    }

    let updated = scroll_and_correct(storage, collection, current_n, &df_map, dry_run).await?;

    Ok(CollectionRebalance {
        collection: collection.to_string(),
        current_n,
        updated,
        skipped_reason: None,
        persist_n: !dry_run && updated > 0,
    })
}

/// Read `(total_documents, last_corrected_n)` for a collection.
async fn read_corpus_n(pool: &SqlitePool, collection: &str) -> Result<(u64, u64), RebalanceError> {
    let row = sqlx::query(
        "SELECT total_documents, last_corrected_n FROM corpus_statistics WHERE collection = ?1",
    )
    .bind(collection)
    .fetch_one(pool)
    .await?;
    let current_n: i64 = row.get("total_documents");
    let last: i64 = row.get("last_corrected_n");
    Ok((current_n.max(0) as u64, last.max(0) as u64))
}

/// Load the `term_id → document_frequency` map for a collection.
async fn load_df_map(
    pool: &SqlitePool,
    collection: &str,
) -> Result<HashMap<u32, u64>, RebalanceError> {
    let rows =
        sqlx::query("SELECT term_id, document_count FROM sparse_vocabulary WHERE collection = ?1")
            .bind(collection)
            .fetch_all(pool)
            .await?;
    Ok(rows
        .into_iter()
        .map(|r| {
            let term_id: i64 = r.get("term_id");
            let df: i64 = r.get("document_count");
            (term_id.max(0) as u32, df.max(0) as u64)
        })
        .collect())
}

/// Scroll all Qdrant points and apply IDF correction factors. Returns the number
/// of points corrected (or, in dry-run, that would be corrected).
async fn scroll_and_correct(
    storage: &StorageClient,
    collection: &str,
    current_n: u64,
    df_map: &HashMap<u32, u64>,
    dry_run: bool,
) -> Result<u64, RebalanceError> {
    let mut offset: Option<String> = None;
    let mut total_updated = 0u64;

    loop {
        let (points, next_cursor) = storage
            .scroll_with_sparse_vectors(collection, SCROLL_BATCH_SIZE, offset.clone())
            .await?;

        // Process this page only when non-empty, but DON'T terminate on an empty
        // page: a page can legitimately come back empty (all points filtered) while
        // a `next_cursor` remains — terminating here would skip later pages.
        if !points.is_empty() {
            let (batch_updates, updated_epochs) =
                build_correction_batch(&points, current_n, df_map);
            total_updated += batch_updates.len() as u64;

            if !dry_run && !batch_updates.is_empty() {
                storage
                    .update_named_sparse_vectors(collection, batch_updates)
                    .await?;
                for (pid, new_epoch) in updated_epochs {
                    // Best-effort: the sparse vector is already corrected. A failed
                    // epoch write leaves a stale idf_epoch → the next run would
                    // re-correct (compounding); warn so operators can detect it.
                    let mut payload = HashMap::new();
                    payload.insert("idf_epoch".to_string(), serde_json::json!(new_epoch));
                    if let Err(e) = storage
                        .set_payload_on_point(collection, &pid, payload)
                        .await
                    {
                        tracing::warn!(point = %pid, error = %e, "idf_epoch payload write failed; sparse vector corrected but epoch is stale");
                    }
                }
            }
        }

        match next_cursor {
            Some(cursor) => offset = Some(cursor),
            None => break,
        }
    }

    Ok(total_updated)
}

/// Build per-point correction vectors + epoch updates for one scroll page.
fn build_correction_batch(
    points: &[SparsePointData],
    current_n: u64,
    df_map: &HashMap<u32, u64>,
) -> (Vec<(String, HashMap<u32, f32>)>, Vec<(String, u64)>) {
    let mut batch_updates = Vec::new();
    let mut updated_epochs = Vec::new();

    for point in points {
        let old_n = match point.idf_epoch {
            Some(n) if n > 0 && n != current_n => n,
            _ => continue,
        };
        let sparse_map = match &point.sparse_vector {
            Some(m) if !m.is_empty() => m.clone(),
            _ => continue,
        };
        let corrected: HashMap<u32, f32> = sparse_map
            .into_iter()
            .map(|(term_id, val)| {
                // A term absent from sparse_vocabulary (pruned / cross-collection)
                // must be left UNCHANGED, not corrected with a fabricated df=1.
                // df=0 makes idf_correction return 1.0 via its `df == 0` guard.
                let df = df_map.get(&term_id).copied().unwrap_or(0);
                (term_id, val * idf_correction(old_n, current_n, df))
            })
            .collect();
        batch_updates.push((point.id.clone(), corrected));
        updated_epochs.push((point.id.clone(), current_n));
    }

    (batch_updates, updated_epochs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idf_correction_no_change_when_equal_n() {
        assert_eq!(idf_correction(1000, 1000, 5), 1.0);
        assert_eq!(idf_correction(500, 500, 1), 1.0);
    }

    #[test]
    fn idf_correction_grows_with_corpus() {
        let factor = idf_correction(1000, 10000, 5);
        assert!(
            factor > 1.0,
            "factor {factor} should be > 1.0 when N grows 10x for a rare term"
        );
    }

    #[test]
    fn idf_correction_formula_accuracy() {
        let factor = idf_correction(1000, 10000, 5) as f64;
        let expected = {
            let old_idf = bm25_idf(1000, 5);
            let new_idf = bm25_idf(10000, 5);
            new_idf / old_idf
        };
        assert!(
            (factor - expected as f32 as f64).abs() < 0.01,
            "factor {factor} should be close to expected {expected:.4}"
        );
    }

    #[test]
    fn idf_correction_zero_df_returns_one() {
        assert_eq!(idf_correction(1000, 2000, 0), 1.0);
    }

    #[test]
    fn bm25_idf_zero_n_returns_zero() {
        assert_eq!(bm25_idf(0, 5), 0.0);
    }

    #[test]
    fn bm25_idf_floored_at_zero() {
        let idf = bm25_idf(10, 9); // ln((10-9+0.5)/(9+0.5)) < 0
        assert_eq!(idf, 0.0, "IDF should be floored at 0 for common terms");
    }

    #[test]
    fn bm25_idf_df_exceeds_n_returns_zero_not_nan() {
        // df > n (corrupt vocabulary) would make ln() NaN; NaN.max(0.0) is NaN in
        // Rust, so the guard must short-circuit to 0.
        let idf = bm25_idf(5, 20);
        assert!(!idf.is_nan(), "df > n must not yield NaN");
        assert_eq!(idf, 0.0);
        // And the correction factor stays finite (1.0) for df > n.
        let f = idf_correction(100, 200, 250);
        assert!(f.is_finite() && (f - 1.0).abs() < 1e-9);
    }

    #[test]
    fn build_batch_skips_uncorrectable_points() {
        let df_map = HashMap::from([(1u32, 5u64)]);
        let points = vec![
            // No idf_epoch → skip.
            SparsePointData {
                id: "a".into(),
                idf_epoch: None,
                sparse_vector: Some(HashMap::from([(1u32, 2.0f32)])),
            },
            // idf_epoch == current_n → skip.
            SparsePointData {
                id: "b".into(),
                idf_epoch: Some(1000),
                sparse_vector: Some(HashMap::from([(1u32, 2.0f32)])),
            },
            // Correctable.
            SparsePointData {
                id: "c".into(),
                idf_epoch: Some(100),
                sparse_vector: Some(HashMap::from([(1u32, 2.0f32)])),
            },
        ];
        let (updates, epochs) = build_correction_batch(&points, 1000, &df_map);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].0, "c");
        assert_eq!(epochs, vec![("c".to_string(), 1000)]);
    }

    #[test]
    fn build_batch_leaves_unknown_vocab_terms_unchanged() {
        // term 99 is NOT in df_map → must be left unchanged (factor 1.0), not
        // corrected with a fabricated df.
        let df_map = HashMap::from([(1u32, 5u64)]);
        let points = vec![SparsePointData {
            id: "p".into(),
            idf_epoch: Some(100),
            sparse_vector: Some(HashMap::from([(99u32, 3.5f32)])),
        }];
        let (updates, _) = build_correction_batch(&points, 1000, &df_map);
        assert_eq!(updates.len(), 1);
        assert_eq!(
            updates[0].1.get(&99),
            Some(&3.5f32),
            "unknown term unchanged"
        );
    }
}
