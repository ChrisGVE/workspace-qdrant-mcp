//! Mirror repair — rebuild missing `qdrant_chunks` rows from live Qdrant
//! points.
//!
//! A tracked file that claims chunks (`chunk_count > 0`, `base_point` set)
//! but has no mirror rows is invisible to every per-file Qdrant deletion:
//! `execute_update_deletion`, `delete_qdrant_points`, and
//! `cleanup_missing_file` all gate on a non-empty point list, and queue
//! triage drops failed `file|delete` items as "effectively done". Missing
//! rows therefore leak points. This module rebuilds them by scrolling the
//! file's `base_point` (payload only) and translating each point back into
//! a `qdrant_chunks` tuple.
//!
//! Used by the [`super::orphan_cleanup::OrphanCleanupTask`] repair phase —
//! which also heals the mass deletion its own pre-fix cleanup sweep caused
//! (see that module's docs).

use std::collections::HashSet;

use qdrant_client::qdrant::{Condition, Filter};
use tracing::{debug, warn};

use crate::idle::task::MaintenanceContext;
use crate::tracked_files_schema::{self, ChunkType};

/// Chunk tuple accepted by `tracked_files_schema::insert_qdrant_chunks`.
type ChunkTuple = (
    String,
    i32,
    String,
    Option<ChunkType>,
    Option<String>,
    Option<i32>,
    Option<i32>,
);

/// Single-page scroll cap. Per-file chunk counts are bounded by the
/// chunking config (typically < 100); 1024 leaves a wide margin while
/// keeping the payload fetch a single round trip.
const REPAIR_SCROLL_LIMIT: u32 = 1024;

/// Fetch the next batch of repair candidates: tracked files that claim
/// chunks (`chunk_count > 0`, `base_point` set) but have no mirror rows.
/// Keyset pagination on `file_id`, same rationale as the cleanup phase.
pub(crate) async fn fetch_repair_batch(
    pool: &sqlx::SqlitePool,
    after_file_id: i64,
    limit: i64,
) -> Result<Vec<(i64, String, String, String)>, sqlx::Error> {
    sqlx::query_as(
        "SELECT tf.file_id, tf.relative_path, tf.collection, tf.base_point
         FROM tracked_files tf
         WHERE tf.file_id > ?1
           AND tf.chunk_count > 0
           AND tf.base_point IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM qdrant_chunks qc WHERE qc.file_id = tf.file_id)
         ORDER BY tf.file_id
         LIMIT ?2",
    )
    .bind(after_file_id)
    .bind(limit)
    .fetch_all(pool)
    .await
}

/// Rebuild a file's missing `qdrant_chunks` rows from its live Qdrant
/// points. Returns the number of rows rebuilt (0 when there was nothing to
/// rebuild or the attempt failed — both already logged).
pub(crate) async fn repair_file(
    ctx: &MaintenanceContext<'_>,
    file_id: i64,
    relative_path: &str,
    collection: &str,
    base_point: &str,
) -> u64 {
    let filter = Filter::must([Condition::matches("base_point", base_point.to_string())]);
    let points = match ctx
        .storage_client
        .scroll_with_filter(collection, filter, REPAIR_SCROLL_LIMIT, None)
        .await
    {
        Ok(p) => p,
        Err(e) => {
            warn!(
                "Mirror repair scroll failed for {} ({}): {}",
                relative_path, collection, e
            );
            return 0;
        }
    };

    if points.is_empty() {
        // Tracked as having chunks but Qdrant holds nothing under this
        // base_point: there is nothing to mirror. The file re-enters the
        // index through the normal ingest path on its next change.
        debug!(
            "Mirror repair: no Qdrant points for {} (base_point {}) — skipping",
            relative_path, base_point
        );
        return 0;
    }
    if points.len() as u32 == REPAIR_SCROLL_LIMIT {
        warn!(
            "Mirror repair: {} hit the {}-point scroll cap; rebuilt mirror may be partial",
            relative_path, REPAIR_SCROLL_LIMIT
        );
    }

    // UNIQUE(file_id, chunk_index): keep the first tuple per index.
    let mut seen_indexes = HashSet::new();
    let chunks: Vec<ChunkTuple> = points
        .iter()
        .filter_map(rebuild_chunk_tuple)
        .filter(|t| seen_indexes.insert(t.1))
        .collect();

    if chunks.is_empty() {
        warn!(
            "Mirror repair: {} returned {} points but none were decodable — skipping",
            relative_path,
            points.len()
        );
        return 0;
    }

    match tracked_files_schema::insert_qdrant_chunks(ctx.pool, file_id, &chunks).await {
        Ok(()) => {
            debug!(
                "Mirror repair: rebuilt {} qdrant_chunks rows for {}",
                chunks.len(),
                relative_path
            );
            chunks.len() as u64
        }
        Err(e) => {
            warn!("Mirror repair insert failed for {}: {}", relative_path, e);
            0
        }
    }
}

/// Translate a scrolled Qdrant point into a `qdrant_chunks` tuple.
///
/// The point ID is stored in the bare-hex form `compute_point_id` produces,
/// not the hyphenated form Qdrant renders, so the mirror keeps one format.
/// Chunk metadata comes from the ingest payload (`chunk_index`, `content`,
/// and the forwarded `chunk_*` fields, which are stored as strings).
fn rebuild_chunk_tuple(p: &qdrant_client::qdrant::RetrievedPoint) -> Option<ChunkTuple> {
    use qdrant_client::qdrant::point_id::PointIdOptions;
    use qdrant_client::qdrant::value::Kind;

    let point_id = match p.id.as_ref()?.point_id_options.as_ref()? {
        PointIdOptions::Uuid(u) => u
            .chars()
            .filter(|c| *c != '-')
            .map(|c| c.to_ascii_lowercase())
            .collect::<String>(),
        PointIdOptions::Num(n) => n.to_string(),
    };

    let int_of = |v: &qdrant_client::qdrant::Value| match v.kind.as_ref()? {
        Kind::IntegerValue(n) => Some(*n),
        Kind::DoubleValue(f) => Some(*f as i64),
        Kind::StringValue(s) => s.parse::<i64>().ok(),
        _ => None,
    };
    let str_of = |v: &qdrant_client::qdrant::Value| match v.kind.as_ref()? {
        Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    };

    let chunk_index = i32::try_from(p.payload.get("chunk_index").and_then(int_of)?).ok()?;
    let content_hash = p
        .payload
        .get("content")
        .and_then(str_of)
        .map(|c| tracked_files_schema::compute_content_hash(&c))
        .unwrap_or_default();
    let chunk_type = p
        .payload
        .get("chunk_chunk_type")
        .and_then(str_of)
        .as_deref()
        .and_then(ChunkType::from_str);
    let symbol_name = p.payload.get("chunk_symbol_name").and_then(str_of);
    let start_line = p
        .payload
        .get("chunk_start_line")
        .and_then(int_of)
        .and_then(|n| i32::try_from(n).ok());
    let end_line = p
        .payload
        .get("chunk_end_line")
        .and_then(int_of)
        .and_then(|n| i32::try_from(n).ok());

    Some((
        point_id,
        chunk_index,
        content_hash,
        chunk_type,
        symbol_name,
        start_line,
        end_line,
    ))
}

#[cfg(test)]
mod tests {
    use crate::tracked_files_schema::{self, ProcessingStatus};

    use super::super::orphan_cleanup::tests_support::{seed_file, test_pool};
    use super::{fetch_repair_batch, rebuild_chunk_tuple};

    /// The repair phase must target exactly the drifted files: chunks
    /// claimed (`chunk_count > 0`, `base_point` set) but zero mirror rows.
    #[tokio::test]
    async fn repair_batch_selects_only_zero_row_files_with_base_point() {
        let pool = test_pool().await;
        let _healthy = seed_file(&pool, 0, true, true).await;
        let drifted = seed_file(&pool, 1, false, true).await;
        let no_base_point = seed_file(&pool, 2, false, false).await;
        // chunk_count = 0: legitimately has no rows.
        let empty = tracked_files_schema::insert_tracked_file(
            &pool,
            "wf-test",
            "src/empty.rs",
            Some("main"),
            None,
            None,
            "2026-06-10T00:00:00Z",
            "hash-empty",
            0,
            None,
            ProcessingStatus::None,
            ProcessingStatus::None,
            Some("projects"),
            None,
            false,
            Some("bp-empty"),
            None,
        )
        .await
        .unwrap();

        let candidates = fetch_repair_batch(&pool, 0, 50).await.unwrap();
        let candidate_ids: Vec<i64> = candidates.iter().map(|c| c.0).collect();
        assert_eq!(candidate_ids, vec![drifted]);
        assert!(!candidate_ids.contains(&no_base_point));
        assert!(!candidate_ids.contains(&empty));

        let (_, relative_path, collection, base_point) = &candidates[0];
        assert_eq!(relative_path, "src/file1.rs");
        assert_eq!(collection, "projects");
        assert_eq!(base_point, "bp1");

        // Keyset cursor past the candidate ends the phase.
        let rest = fetch_repair_batch(&pool, drifted, 50).await.unwrap();
        assert!(rest.is_empty());
    }

    /// Rebuilt rows must store the bare-hex point ID (`compute_point_id`
    /// format), not the hyphenated form Qdrant returns, and decode the
    /// stringly-typed `chunk_*` payload metadata.
    #[test]
    fn rebuild_chunk_tuple_normalizes_id_and_reads_payload() {
        use qdrant_client::qdrant::value::Kind;
        use qdrant_client::qdrant::{PointId, RetrievedPoint, Value};

        let mut payload = std::collections::HashMap::new();
        payload.insert(
            "chunk_index".to_string(),
            Value {
                kind: Some(Kind::IntegerValue(3)),
            },
        );
        payload.insert(
            "content".to_string(),
            Value {
                kind: Some(Kind::StringValue("fn main() {}".to_string())),
            },
        );
        payload.insert(
            "chunk_symbol_name".to_string(),
            Value {
                kind: Some(Kind::StringValue("main".to_string())),
            },
        );
        payload.insert(
            "chunk_chunk_type".to_string(),
            Value {
                kind: Some(Kind::StringValue("function".to_string())),
            },
        );
        payload.insert(
            "chunk_start_line".to_string(),
            Value {
                kind: Some(Kind::StringValue("10".to_string())),
            },
        );
        payload.insert(
            "chunk_end_line".to_string(),
            Value {
                kind: Some(Kind::StringValue("20".to_string())),
            },
        );

        let point = RetrievedPoint {
            id: Some(PointId::from("00006C52-38E7-1A6E-7DDD-4CF06EC2A0C8")),
            payload,
            ..Default::default()
        };

        let (point_id, chunk_index, content_hash, chunk_type, symbol_name, start_line, end_line) =
            rebuild_chunk_tuple(&point).expect("decodable point");
        assert_eq!(point_id, "00006c5238e71a6e7ddd4cf06ec2a0c8");
        assert_eq!(chunk_index, 3);
        assert_eq!(
            content_hash,
            tracked_files_schema::compute_content_hash("fn main() {}")
        );
        assert_eq!(chunk_type, Some(tracked_files_schema::ChunkType::Function));
        assert_eq!(symbol_name.as_deref(), Some("main"));
        assert_eq!(start_line, Some(10));
        assert_eq!(end_line, Some(20));
    }

    #[test]
    fn rebuild_chunk_tuple_requires_chunk_index() {
        use qdrant_client::qdrant::{PointId, RetrievedPoint};

        let point = RetrievedPoint {
            id: Some(PointId::from("00006c52-38e7-1a6e-7ddd-4cf06ec2a0c8")),
            payload: std::collections::HashMap::new(),
            ..Default::default()
        };
        assert!(rebuild_chunk_tuple(&point).is_none());
    }
}
