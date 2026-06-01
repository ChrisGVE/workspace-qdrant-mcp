//! Search operations
//!
//! Dense, sparse, and hybrid (RRF) search implementations using
//! Qdrant's QueryPoints API with named vector support.
//!
//! When `SearchParams::diversity_penalty` is set, the search method
//! applies post-retrieval source diversity re-ranking to reduce
//! result clustering from a single file or project.

use std::collections::HashMap;

use qdrant_client::qdrant::{Condition, Filter, QueryPointsBuilder};
use serde_json::Value;
use tracing::debug;

use super::client::StorageClient;
use super::convert::convert_qdrant_value_to_json;
use super::types::{
    HybridSearchMode, HybridSearchParams, SearchParams, SearchResult, StorageError,
};
use crate::source_diversity::apply_diversity_penalty;

/// Convert the flat `field -> JSON value` filter map used by callers into a
/// Qdrant `Filter` of `must` match conditions.
///
/// Each entry produces one `Condition::matches(field, value)` clause. Values
/// are converted using their native representation:
///
/// - `String`            -> match by string keyword
/// - `Number` (integer)  -> match by integer
/// - `Bool`              -> match by boolean
/// - `Number` (float)    -> match by string (stringified) — Qdrant has no
///   float-match semantics; preserves the value without crashing.
/// - `Array` (strings)   -> match any (Qdrant `match_any`)
/// - `null` / `Object`   -> skipped (no equivalent single-value match)
///
/// Returns `None` when every entry was skipped or the input map was empty.
pub fn build_filter_from_json(filter: &HashMap<String, Value>) -> Option<Filter> {
    let mut conditions: Vec<Condition> = Vec::with_capacity(filter.len());
    for (key, value) in filter {
        let (effective_key, effective_value);
        if key == "branch" || key == "branches" {
            // Cross-branch wildcard: omit branch filter entirely
            if matches!(value, Value::String(s) if s == "*") {
                continue;
            }
            effective_key = "branches";
            effective_value = match value {
                Value::String(_) => std::borrow::Cow::Owned(Value::Array(vec![value.clone()])),
                _ => std::borrow::Cow::Borrowed(value),
            };
        } else {
            effective_key = key.as_str();
            effective_value = std::borrow::Cow::Borrowed(value);
        }
        let value = effective_value.as_ref();
        let cond = match value {
            Value::String(s) => Condition::matches(effective_key, s.clone()),
            Value::Bool(b) => Condition::matches(effective_key, *b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Condition::matches(effective_key, i)
                } else {
                    Condition::matches(effective_key, n.to_string())
                }
            }
            Value::Array(arr) => {
                let string_values: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                if string_values.is_empty() {
                    continue;
                }
                Condition::matches(effective_key, string_values)
            }
            Value::Null | Value::Object(_) => continue,
        };
        conditions.push(cond);
    }
    if conditions.is_empty() {
        None
    } else {
        Some(Filter::must(conditions))
    }
}

/// Build a Qdrant filter that matches tags across both `concept_tags` and `tags` fields.
pub fn build_tag_filter(tags: &[String]) -> Option<Filter> {
    if tags.is_empty() {
        return None;
    }

    let conditions: Vec<Condition> = if tags.len() == 1 {
        let tag = tags[0].clone();
        vec![
            Condition::matches(wqm_common::constants::field::CONCEPT_TAGS, tag.clone()),
            Condition::matches(wqm_common::constants::field::TAGS, tag),
        ]
    } else {
        let tag_vec = tags.to_vec();
        vec![
            Condition::matches(wqm_common::constants::field::CONCEPT_TAGS, tag_vec.clone()),
            Condition::matches(wqm_common::constants::field::TAGS, tag_vec),
        ]
    };

    Some(Filter::should(conditions))
}

/// Merge a tag filter into a base filter as a nested `must` condition.
pub fn merge_tag_filter_into(
    base_filter: Option<Filter>,
    tag_filter: Option<Filter>,
) -> Option<Filter> {
    match (base_filter, tag_filter) {
        (Some(base), Some(tags)) => {
            let tag_condition = Condition::from(tags);
            let mut must = base.must;
            must.push(tag_condition);
            Some(Filter {
                must,
                should: base.should,
                must_not: base.must_not,
                min_should: base.min_should,
            })
        }
        (Some(base), None) => Some(base),
        (None, Some(tags)) => Some(Filter::must([Condition::from(tags)])),
        (None, None) => None,
    }
}

impl StorageClient {
    /// Perform hybrid search with dense/sparse vector fusion
    ///
    /// When `params.diversity_penalty` is `Some`, applies post-retrieval
    /// source diversity re-ranking to penalize consecutive results from the
    /// same file or project before returning.
    #[tracing::instrument(
        name = "qdrant.search",
        skip_all,
        fields(collection = %collection_name, mode = ?params.search_mode, limit = params.limit)
    )]
    pub async fn search(
        &self,
        collection_name: &str,
        params: SearchParams,
    ) -> Result<Vec<SearchResult>, StorageError> {
        debug!("Performing search in collection: {}", collection_name);
        let started = std::time::Instant::now();

        // Capture RED-metric labels before `params` fields move into the
        // per-mode branches below (B6).
        let mode_label = match &params.search_mode {
            HybridSearchMode::Dense => "dense",
            HybridSearchMode::Sparse => "sparse",
            HybridSearchMode::Hybrid { .. } => "hybrid",
        };
        let tenant_label = params
            .filter
            .as_ref()
            .and_then(|f| f.get("tenant_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let diversity_config = params.diversity_penalty.clone();

        let results = match params.search_mode {
            HybridSearchMode::Dense => {
                if let Some(vector) = params.dense_vector {
                    self.search_dense(
                        collection_name,
                        vector,
                        params.limit,
                        params.score_threshold,
                        params.filter,
                    )
                    .await?
                } else {
                    return Err(StorageError::Search(
                        "Dense vector required for dense search".to_string(),
                    ));
                }
            }
            HybridSearchMode::Sparse => {
                if let Some(vector) = params.sparse_vector {
                    self.search_sparse(
                        collection_name,
                        vector,
                        params.limit,
                        params.score_threshold,
                        params.filter,
                    )
                    .await?
                } else {
                    return Err(StorageError::Search(
                        "Sparse vector required for sparse search".to_string(),
                    ));
                }
            }
            HybridSearchMode::Hybrid {
                dense_weight,
                sparse_weight,
            } => {
                let hybrid_params = HybridSearchParams {
                    dense_vector: params.dense_vector,
                    sparse_vector: params.sparse_vector,
                    dense_weight,
                    sparse_weight,
                    limit: params.limit,
                    score_threshold: params.score_threshold,
                    filter: params.filter,
                };
                self.search_hybrid(collection_name, hybrid_params).await?
            }
        };

        // Apply diversity penalty re-ranking when configured.
        let results = if let Some(ref penalty_config) = diversity_config {
            debug!(
                same_file_penalty = penalty_config.same_file_penalty,
                same_project_penalty = penalty_config.same_project_penalty,
                "Applying source diversity penalty re-ranking"
            );
            apply_diversity_penalty(results, penalty_config)
        } else {
            results
        };

        debug!("Search completed, returned {} results", results.len());
        let elapsed = started.elapsed();
        crate::monitoring::metrics_core::METRICS.record_qdrant("search", elapsed, None);
        crate::monitoring::metrics_core::METRICS.record_search(
            collection_name,
            mode_label,
            &tenant_label,
            results.len(),
            elapsed,
        );
        Ok(results)
    }

    /// Dense vector search using Qdrant's QueryPoints API with named dense vectors
    async fn search_dense(
        &self,
        collection_name: &str,
        dense_vector: Vec<f32>,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let mut query_builder = QueryPointsBuilder::new(collection_name)
            .query(dense_vector)
            .using("dense")
            .limit(limit as u64)
            .with_payload(true);

        if let Some(filter_map) = filter.as_ref() {
            if let Some(qdrant_filter) = build_filter_from_json(filter_map) {
                query_builder = query_builder.filter(qdrant_filter);
            }
        }
        if let Some(threshold) = score_threshold {
            query_builder = query_builder.score_threshold(threshold);
        }

        let response = self
            .retry_operation(|| async {
                self.client
                    .query(query_builder.clone())
                    .await
                    .map_err(|e| StorageError::Search(e.to_string()))
            })
            .await?;

        let results = response
            .result
            .into_iter()
            .map(|scored_point| {
                let json_payload: HashMap<String, serde_json::Value> = scored_point
                    .payload
                    .into_iter()
                    .map(|(k, v)| (k, convert_qdrant_value_to_json(v)))
                    .collect();

                let id = extract_point_id(&scored_point.id);

                SearchResult {
                    id,
                    score: scored_point.score,
                    payload: json_payload,
                    dense_vector: None,
                    sparse_vector: None,
                }
            })
            .collect();

        Ok(results)
    }

    /// Sparse vector search using Qdrant's QueryPoints API with named sparse vectors
    async fn search_sparse(
        &self,
        collection_name: &str,
        sparse_vector: HashMap<u32, f32>,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        if sparse_vector.is_empty() {
            debug!("Empty sparse vector, returning no results");
            return Ok(vec![]);
        }

        let sparse_pairs: Vec<(u32, f32)> = sparse_vector.into_iter().collect();

        let mut query_builder = QueryPointsBuilder::new(collection_name)
            .query(sparse_pairs)
            .using("sparse")
            .limit(limit as u64)
            .with_payload(true);

        if let Some(filter_map) = filter.as_ref() {
            if let Some(qdrant_filter) = build_filter_from_json(filter_map) {
                query_builder = query_builder.filter(qdrant_filter);
            }
        }
        if let Some(threshold) = score_threshold {
            query_builder = query_builder.score_threshold(threshold);
        }

        let response = self
            .retry_operation(|| async {
                self.client
                    .query(query_builder.clone())
                    .await
                    .map_err(|e| StorageError::Search(e.to_string()))
            })
            .await?;

        let results = response
            .result
            .into_iter()
            .map(|scored_point| {
                let json_payload: HashMap<String, serde_json::Value> = scored_point
                    .payload
                    .into_iter()
                    .map(|(k, v)| (k, convert_qdrant_value_to_json(v)))
                    .collect();

                let id = extract_point_id(&scored_point.id);

                SearchResult {
                    id,
                    score: scored_point.score,
                    payload: json_payload,
                    dense_vector: None,
                    sparse_vector: None,
                }
            })
            .collect();

        Ok(results)
    }

    /// Hybrid search with RRF (Reciprocal Rank Fusion)
    async fn search_hybrid(
        &self,
        collection_name: &str,
        params: HybridSearchParams,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let mut all_results = HashMap::new();

        // Perform dense search if vector is provided
        if let Some(vector) = params.dense_vector {
            let dense_results = self
                .search_dense(
                    collection_name,
                    vector,
                    params.limit * 2,
                    params.score_threshold,
                    params.filter.clone(),
                )
                .await?;

            for (rank, result) in dense_results.into_iter().enumerate() {
                let rrf_score = params.dense_weight / (60.0 + (rank + 1) as f32);
                let entry = all_results
                    .entry(result.id.clone())
                    .or_insert_with(|| (result, 0.0));
                entry.1 += rrf_score;
            }
        }

        // Perform sparse search if vector is provided
        if let Some(vector) = params.sparse_vector {
            let sparse_results = self
                .search_sparse(
                    collection_name,
                    vector,
                    params.limit * 2,
                    params.score_threshold,
                    params.filter,
                )
                .await?;

            for (rank, result) in sparse_results.into_iter().enumerate() {
                let rrf_score = params.sparse_weight / (60.0 + (rank + 1) as f32);
                let entry = all_results
                    .entry(result.id.clone())
                    .or_insert_with(|| (result, 0.0));
                entry.1 += rrf_score;
            }
        }

        // Sort by combined RRF score and take top results
        let mut final_results: Vec<_> = all_results
            .into_iter()
            .map(|(_, (mut result, rrf_score))| {
                result.score = rrf_score;
                result
            })
            .collect();

        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        final_results.truncate(params.limit);

        Ok(final_results)
    }
}

/// Extract point ID string from an optional Qdrant PointId
fn extract_point_id(id: &Option<qdrant_client::qdrant::PointId>) -> String {
    match id.as_ref().and_then(|pid| pid.point_id_options.as_ref()) {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid.clone(),
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => num.to_string(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    //! Filter-conversion unit tests for the JSON-to-Qdrant adapter used by
    //! `search_dense` and `search_sparse`. Regression coverage for F-003
    //! (Rust storage search ignores filter and score_threshold).
    //!
    //! Live Qdrant round-trip behaviour is exercised in
    //! `tests/storage_search_filter_integration.rs`; here we verify the
    //! filter-building plumbing that previously discarded its inputs.

    use super::*;
    use serde_json::json;

    #[test]
    fn build_filter_from_json_empty_map_returns_none() {
        let filter: HashMap<String, Value> = HashMap::new();
        assert!(build_filter_from_json(&filter).is_none());
    }

    #[test]
    fn build_filter_from_json_string_value_produces_filter() {
        let mut filter = HashMap::new();
        filter.insert("tenant_id".to_string(), json!("alpha"));
        let result = build_filter_from_json(&filter).expect("filter expected");
        assert_eq!(
            result.must.len(),
            1,
            "single string entry must produce exactly one match condition"
        );
    }

    #[test]
    fn build_filter_from_json_multiple_fields_all_present() {
        let mut filter = HashMap::new();
        filter.insert("tenant_id".to_string(), json!("alpha"));
        filter.insert("source_collection".to_string(), json!("docs"));
        let result = build_filter_from_json(&filter).expect("filter expected");
        assert_eq!(
            result.must.len(),
            2,
            "every supported entry must yield a must condition"
        );
    }

    #[test]
    fn build_filter_from_json_bool_and_int_values_supported() {
        let mut filter = HashMap::new();
        filter.insert("deleted".to_string(), json!(false));
        filter.insert("version".to_string(), json!(42));
        let result = build_filter_from_json(&filter).expect("filter expected");
        assert_eq!(result.must.len(), 2);
    }

    #[test]
    fn build_filter_from_json_unsupported_types_are_skipped() {
        let mut filter = HashMap::new();
        filter.insert("ok".to_string(), json!("yes"));
        filter.insert("nullish".to_string(), json!(null));
        filter.insert("arr".to_string(), json!(["a", "b"]));
        filter.insert("obj".to_string(), json!({"k": "v"}));
        let result = build_filter_from_json(&filter).expect("filter expected");
        assert_eq!(
            result.must.len(),
            2,
            "string + string-array entries produce conditions; null + obj are skipped"
        );
    }

    #[test]
    fn build_filter_from_json_all_unsupported_returns_none() {
        let mut filter = HashMap::new();
        filter.insert("nullish".to_string(), json!(null));
        filter.insert("obj".to_string(), json!({"k": "v"}));
        assert!(build_filter_from_json(&filter).is_none());
    }

    #[test]
    fn build_filter_from_json_string_array_produces_match_any() {
        let mut filter = HashMap::new();
        filter.insert("tags".to_string(), json!(["rust", "path:models"]));
        let result = build_filter_from_json(&filter).expect("filter expected");
        assert_eq!(
            result.must.len(),
            1,
            "string array should produce one match-any condition"
        );
    }

    #[test]
    fn build_filter_from_json_empty_array_skipped() {
        let mut filter = HashMap::new();
        filter.insert("tags".to_string(), json!([]));
        assert!(
            build_filter_from_json(&filter).is_none(),
            "empty array should be skipped"
        );
    }

    #[test]
    fn build_filter_from_json_mixed_array_only_strings() {
        let mut filter = HashMap::new();
        filter.insert("tags".to_string(), json!(["rust", 42, "test"]));
        let result = build_filter_from_json(&filter).expect("filter expected");
        assert_eq!(result.must.len(), 1);
    }

    #[test]
    fn build_filter_from_json_non_string_array_skipped() {
        let mut filter = HashMap::new();
        filter.insert("nums".to_string(), json!([1, 2, 3]));
        assert!(
            build_filter_from_json(&filter).is_none(),
            "array of non-strings should be skipped"
        );
    }

    #[test]
    fn build_filter_branch_string_rewritten_to_branches_array() {
        let mut filter = HashMap::new();
        filter.insert("branch".to_string(), json!("main"));
        let result = build_filter_from_json(&filter).expect("filter expected");
        assert_eq!(
            result.must.len(),
            1,
            "legacy branch string should produce one condition on branches"
        );
    }

    #[test]
    fn build_filter_branches_array_passed_through() {
        let mut filter = HashMap::new();
        filter.insert("branches".to_string(), json!(["main", "dev"]));
        let result = build_filter_from_json(&filter).expect("filter expected");
        assert_eq!(
            result.must.len(),
            1,
            "branches array should produce one match-any condition"
        );
    }

    #[test]
    fn build_filter_branch_wildcard_omits_branch_condition() {
        let mut filter = HashMap::new();
        filter.insert("tenant_id".to_string(), json!("proj1"));
        filter.insert("branch".to_string(), json!("*"));
        let result = build_filter_from_json(&filter).expect("filter expected");
        assert_eq!(
            result.must.len(),
            1,
            "branch='*' should be skipped, leaving only tenant_id"
        );
    }

    #[test]
    fn build_filter_branches_wildcard_omits_branch_condition() {
        let mut filter = HashMap::new();
        filter.insert("branches".to_string(), json!("*"));
        assert!(
            build_filter_from_json(&filter).is_none(),
            "branches='*' alone should produce no conditions"
        );
    }
}
