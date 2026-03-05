//! Search operations
//!
//! Dense, sparse, and hybrid (RRF) search implementations using
//! Qdrant's QueryPoints API with named vector support.

use std::collections::HashMap;

use qdrant_client::qdrant::QueryPointsBuilder;
use tracing::debug;

use super::client::StorageClient;
use super::convert::convert_qdrant_value_to_json;
use super::types::{
    HybridSearchMode, HybridSearchParams, SearchParams, SearchResult, StorageError,
};

impl StorageClient {
    /// Perform hybrid search with dense/sparse vector fusion
    pub async fn search(
        &self,
        collection_name: &str,
        params: SearchParams,
    ) -> Result<Vec<SearchResult>, StorageError> {
        debug!("Performing search in collection: {}", collection_name);

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

        debug!("Search completed, returned {} results", results.len());
        Ok(results)
    }

    /// Dense vector search using Qdrant's QueryPoints API with named dense vectors
    async fn search_dense(
        &self,
        collection_name: &str,
        dense_vector: Vec<f32>,
        limit: usize,
        _score_threshold: Option<f32>,
        _filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let query_builder = QueryPointsBuilder::new(collection_name)
            .query(dense_vector)
            .using("dense")
            .limit(limit as u64)
            .with_payload(true);

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
        _score_threshold: Option<f32>,
        _filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        if sparse_vector.is_empty() {
            debug!("Empty sparse vector, returning no results");
            return Ok(vec![]);
        }

        let sparse_pairs: Vec<(u32, f32)> = sparse_vector.into_iter().collect();

        let query_builder = QueryPointsBuilder::new(collection_name)
            .query(sparse_pairs)
            .using("sparse")
            .limit(limit as u64)
            .with_payload(true);

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
