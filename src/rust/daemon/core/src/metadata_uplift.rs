//! Metadata Uplifting Background Process (Task 18).
//!
//! When the queue is empty and the system is idle, scans Qdrant for chunks
//! with failed/partial LSP enrichment or missing tags, and re-attempts
//! enrichment without re-chunking or re-embedding.
//!
//! Tracks `uplift_generation` per point to avoid infinite re-processing.
//! Pauses immediately if new queue items appear.

use std::collections::HashMap;
use std::sync::Arc;

use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::{Condition, Filter, PointId, PointsIdsList};
use tracing::{debug, info, warn};

use crate::lexicon::LexiconManager;
use crate::storage::{StorageClient, StorageError};

/// Configuration for the metadata uplift process.
#[derive(Debug, Clone)]
pub struct UpliftConfig {
    /// Maximum points to process per uplift batch.
    pub batch_size: u32,
    /// Minimum seconds between uplift attempts.
    pub min_interval_secs: u64,
    /// Current uplift generation (fixed for the daemon's lifetime; a no-update
    /// pass means the collection is caught up, not that the generation moved).
    pub current_generation: u64,
}

impl Default for UpliftConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            min_interval_secs: 300, // 5 minutes
            current_generation: 1,
        }
    }
}

/// Result of a single uplift pass.
#[derive(Debug, Clone, Default)]
pub struct UpliftStats {
    /// Points scanned (scrolled from Qdrant).
    pub scanned: u64,
    /// Points updated with new metadata.
    pub updated: u64,
    /// Points skipped (already at current generation or no improvement possible).
    pub skipped: u64,
    /// Errors encountered.
    pub errors: u64,
}

/// Scroll Qdrant for points needing metadata uplift.
///
/// Finds points where `lsp_enrichment_status` is 'failed', 'partial', or
/// 'pending' AND `uplift_generation` < current_generation. Points that lack
/// the `uplift_generation` key count as generation 0 and always match.
///
/// The generation condition lives in the Qdrant filter (server-side), so
/// already-uplifted points are never returned; the scroll pages via its
/// next-page offset until `batch_size` candidates are collected or the
/// collection is exhausted (GitHub #292 residual: a single non-paging batch
/// stranded every point after the first `batch_size` ids).
///
/// Returns point IDs and their current payloads.
pub async fn find_points_needing_uplift(
    storage_client: &StorageClient,
    collection: &str,
    config: &UpliftConfig,
) -> Result<Vec<UpliftCandidate>, StorageError> {
    let filter = uplift_candidate_filter(config.current_generation);

    let mut candidates = Vec::new();
    let mut offset: Option<qdrant_client::qdrant::PointId> = None;

    loop {
        let (points, next_offset) = storage_client
            .scroll_with_filter_paged(collection, filter.clone(), config.batch_size, offset)
            .await?;
        offset = next_offset;

        for point in points {
            let point_id = match &point.id {
                Some(id) => format_point_id(id),
                None => continue,
            };

            let mut payload_map: HashMap<String, serde_json::Value> = HashMap::new();
            for (key, value) in &point.payload {
                payload_map.insert(key.clone(), qdrant_value_to_json(value));
            }

            candidates.push(UpliftCandidate {
                point_id,
                collection: collection.to_string(),
                payload: payload_map,
            });

            if candidates.len() >= config.batch_size as usize {
                return Ok(candidates);
            }
        }

        if offset.is_none() {
            break;
        }
    }

    Ok(candidates)
}

/// Build the server-side candidate filter for one uplift pass.
///
/// `should`: `lsp_enrichment_status` in ['failed', 'partial', 'pending']
/// ('pending' = code file where the LSP server wasn't ready during initial
/// processing).
///
/// `must_not`: range `uplift_generation` >= current_generation — points
/// already uplifted at the current generation are excluded server-side.
/// Points lacking the key do not match the range, so `must_not` keeps them,
/// which is exactly the "generation 0" semantics.
fn uplift_candidate_filter(current_generation: u64) -> Filter {
    Filter {
        should: vec![
            Condition::matches("lsp_enrichment_status", "failed".to_string()),
            Condition::matches("lsp_enrichment_status", "partial".to_string()),
            Condition::matches("lsp_enrichment_status", "pending".to_string()),
        ],
        must_not: vec![Condition::range(
            "uplift_generation",
            qdrant_client::qdrant::Range {
                gte: Some(current_generation as f64),
                ..Default::default()
            },
        )],
        ..Default::default()
    }
}

/// A point that needs metadata uplift.
#[derive(Debug, Clone)]
pub struct UpliftCandidate {
    pub point_id: String,
    pub collection: String,
    pub payload: HashMap<String, serde_json::Value>,
}

/// Run one uplift pass: find candidates and update their metadata.
///
/// Returns stats about what was processed.
pub async fn run_uplift_pass(
    storage_client: &Arc<StorageClient>,
    lexicon_manager: &Arc<LexiconManager>,
    collections: &[String],
    config: &UpliftConfig,
) -> UpliftStats {
    let mut stats = UpliftStats::default();

    for collection in collections {
        let candidates = match find_points_needing_uplift(storage_client, collection, config).await
        {
            Ok(c) => c,
            Err(e) => {
                debug!("Skipping uplift for '{}': {}", collection, e);
                continue;
            }
        };

        if candidates.is_empty() {
            debug!("No points need uplift in '{}'", collection);
            continue;
        }

        info!(
            "Found {} points needing uplift in '{}'",
            candidates.len(),
            collection
        );

        for candidate in &candidates {
            stats.scanned += 1;

            match uplift_single_point(storage_client, lexicon_manager, candidate, config).await {
                Ok(true) => stats.updated += 1,
                Ok(false) => stats.skipped += 1,
                Err(e) => {
                    warn!(
                        "Failed to uplift point {} in '{}': {}",
                        candidate.point_id, collection, e
                    );
                    stats.errors += 1;
                }
            }
        }
    }

    stats
}

/// Uplift a single point's metadata.
///
/// Updates:
/// - Applies new tags from dynamic lexicon if concept_tags is empty
/// - Sets uplift_generation to current
///
/// Returns true if the point was updated, false if skipped.
async fn uplift_single_point(
    storage_client: &Arc<StorageClient>,
    lexicon_manager: &Arc<LexiconManager>,
    candidate: &UpliftCandidate,
    config: &UpliftConfig,
) -> Result<bool, StorageError> {
    let mut updates: HashMap<String, serde_json::Value> = HashMap::new();
    let mut changed = false;

    // Check if concept_tags is missing or empty
    let has_tags = candidate
        .payload
        .get("concept_tags")
        .and_then(|v| v.as_array())
        .map(|arr| !arr.is_empty())
        .unwrap_or(false);

    if !has_tags {
        // Try to generate tags from the content using lexicon
        if let Some(content) = candidate.payload.get("content").and_then(|v| v.as_str()) {
            let tokens: Vec<String> = content
                .split_whitespace()
                .map(|w| {
                    w.to_lowercase()
                        .trim_matches(|c: char| !c.is_alphanumeric())
                        .to_string()
                })
                .filter(|w| w.len() >= 2)
                .collect();

            // Find distinctive terms (high DF relative to corpus)
            let corpus_size = lexicon_manager.corpus_size(&candidate.collection).await;
            if corpus_size > 0 {
                let mut term_scores: Vec<(String, f64)> = Vec::new();
                for term in &tokens {
                    let df = lexicon_manager
                        .document_frequency(&candidate.collection, term)
                        .await;
                    if df > 0 {
                        // IDF score: ln(N / (1 + df))
                        let idf = (corpus_size as f64 / (1.0 + df as f64)).ln();
                        if idf > 0.5 {
                            // Only distinctive terms
                            term_scores.push((term.clone(), idf));
                        }
                    }
                }
                term_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_tags: Vec<String> =
                    term_scores.into_iter().take(5).map(|(t, _)| t).collect();

                if !top_tags.is_empty() {
                    updates.insert("concept_tags".to_string(), serde_json::json!(top_tags));
                    changed = true;
                }
            }
        }
    }

    // Always update uplift_generation
    updates.insert(
        "uplift_generation".to_string(),
        serde_json::json!(config.current_generation),
    );

    if !changed && has_tags {
        // Only update the generation marker if nothing else changed
        // This prevents re-scanning the same point
    }

    storage_client
        .set_payload_on_selector(
            &candidate.collection,
            uplift_write_selector(candidate),
            updates,
        )
        .await?;

    Ok(changed)
}

/// Build the selector addressing the one point an uplift write updates.
///
/// The candidate's `point_id` is the Qdrant point **id** (from the scroll
/// response), so the write addresses it as an id.  It used to be matched
/// against a `chunk_id` payload key instead — a key no point carries — so
/// every uplift write matched zero points, the `uplift_generation` marker
/// never landed, the same candidates were re-selected on every pass, and each
/// no-op cost two unindexed scans of the whole collection (GitHub #292).
fn uplift_write_selector(candidate: &UpliftCandidate) -> PointsSelectorOneOf {
    let id = PointId {
        point_id_options: Some(PointIdOptions::Uuid(candidate.point_id.clone())),
    };
    PointsSelectorOneOf::Points(PointsIdsList { ids: vec![id] })
}

/// Convert a Qdrant PointId to a string.
fn format_point_id(id: &qdrant_client::qdrant::PointId) -> String {
    match &id.point_id_options {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid.clone(),
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => num.to_string(),
        None => String::new(),
    }
}

/// Convert a Qdrant Value to serde_json::Value.
fn qdrant_value_to_json(value: &qdrant_client::qdrant::Value) -> serde_json::Value {
    match &value.kind {
        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
            serde_json::Value::String(s.clone())
        }
        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => {
            serde_json::json!(*i)
        }
        Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => {
            serde_json::json!(*d)
        }
        Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => {
            serde_json::json!(*b)
        }
        Some(qdrant_client::qdrant::value::Kind::ListValue(list)) => {
            let items: Vec<serde_json::Value> =
                list.values.iter().map(qdrant_value_to_json).collect();
            serde_json::Value::Array(items)
        }
        Some(qdrant_client::qdrant::value::Kind::StructValue(s)) => {
            let map: serde_json::Map<String, serde_json::Value> = s
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
        Some(qdrant_client::qdrant::value::Kind::NullValue(_)) | None => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uplift_config_defaults() {
        let config = UpliftConfig::default();
        assert_eq!(config.batch_size, 10);
        assert_eq!(config.min_interval_secs, 300);
        assert_eq!(config.current_generation, 1);
    }

    #[test]
    fn test_uplift_stats_default() {
        let stats = UpliftStats::default();
        assert_eq!(stats.scanned, 0);
        assert_eq!(stats.updated, 0);
        assert_eq!(stats.skipped, 0);
        assert_eq!(stats.errors, 0);
    }

    fn candidate(point_id: &str) -> UpliftCandidate {
        UpliftCandidate {
            point_id: point_id.to_string(),
            collection: "projects".to_string(),
            payload: HashMap::new(),
        }
    }

    #[test]
    fn uplift_write_addresses_the_point_by_its_id() {
        // GitHub #292: the write used to select on a `chunk_id` payload key
        // that no point carries, so every uplift matched zero points and the
        // `uplift_generation` marker never landed.
        let selector = uplift_write_selector(&candidate("000000b2-619a-d36c-e584-b2ed3515e831"));
        match selector {
            PointsSelectorOneOf::Points(ids) => {
                assert_eq!(ids.ids.len(), 1, "exactly the candidate point");
                assert_eq!(
                    ids.ids[0].point_id_options,
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(
                        "000000b2-619a-d36c-e584-b2ed3515e831".to_string()
                    ))
                );
            }
            other => panic!("uplift must address points by id, got {:?}", other),
        }
    }

    #[test]
    fn uplift_write_never_selects_on_a_payload_filter() {
        // A payload filter is what made the write a full unindexed scan that
        // matched nothing; an id selector is O(1) and always matches.
        let selector = uplift_write_selector(&candidate("abc-123"));
        assert!(
            !matches!(selector, PointsSelectorOneOf::Filter(_)),
            "uplift must not address its point through a payload filter"
        );
    }

    #[test]
    fn uplift_filter_excludes_current_generation_server_side() {
        // GitHub #292 residual: the generation condition must be part of the
        // Qdrant filter (must_not range uplift_generation >= current), not an
        // in-memory skip after a single non-paging batch.
        use qdrant_client::qdrant::condition::ConditionOneOf;

        let filter = uplift_candidate_filter(7);

        assert_eq!(filter.must_not.len(), 1, "exactly one must_not condition");
        match &filter.must_not[0].condition_one_of {
            Some(ConditionOneOf::Field(field)) => {
                assert_eq!(field.key, "uplift_generation");
                let range = field.range.as_ref().expect("range on the condition");
                assert_eq!(range.gte, Some(7.0));
                assert_eq!(range.gt, None);
                assert_eq!(range.lte, None);
                assert_eq!(range.lt, None);
            }
            other => panic!("expected a field range condition, got {:?}", other),
        }
    }

    #[test]
    fn uplift_filter_matches_the_three_incomplete_statuses() {
        use qdrant_client::qdrant::condition::ConditionOneOf;
        use qdrant_client::qdrant::r#match::MatchValue;

        let filter = uplift_candidate_filter(1);

        assert!(!filter.should.is_empty());
        let mut statuses = Vec::new();
        for condition in &filter.should {
            match &condition.condition_one_of {
                Some(ConditionOneOf::Field(field)) => {
                    assert_eq!(field.key, "lsp_enrichment_status");
                    match field.r#match.as_ref().and_then(|m| m.match_value.as_ref()) {
                        Some(MatchValue::Keyword(kw)) => statuses.push(kw.clone()),
                        other => panic!("expected keyword match, got {:?}", other),
                    }
                }
                other => panic!("expected field condition, got {:?}", other),
            }
        }
        assert_eq!(statuses, vec!["failed", "partial", "pending"]);
    }

    #[test]
    fn test_format_point_id_uuid() {
        let id = qdrant_client::qdrant::PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(
                "abc-123".to_string(),
            )),
        };
        assert_eq!(format_point_id(&id), "abc-123");
    }

    #[test]
    fn test_format_point_id_num() {
        let id = qdrant_client::qdrant::PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(42)),
        };
        assert_eq!(format_point_id(&id), "42");
    }

    #[test]
    fn test_qdrant_value_to_json_string() {
        let value = qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                "hello".to_string(),
            )),
        };
        assert_eq!(qdrant_value_to_json(&value), serde_json::json!("hello"));
    }

    #[test]
    fn test_qdrant_value_to_json_int() {
        let value = qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(42)),
        };
        assert_eq!(qdrant_value_to_json(&value), serde_json::json!(42));
    }

    #[test]
    fn test_qdrant_value_to_json_null() {
        let value = qdrant_client::qdrant::Value { kind: None };
        assert_eq!(qdrant_value_to_json(&value), serde_json::Value::Null);
    }
}
