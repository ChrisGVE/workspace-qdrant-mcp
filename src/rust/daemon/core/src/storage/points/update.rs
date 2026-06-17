//! Point update operations
//!
//! Update sparse vectors and payload fields on existing Qdrant points.

use qdrant_client::qdrant::Filter;
use tracing::info;

use crate::storage::client::StorageClient;
use crate::storage::convert::convert_json_to_qdrant_value;
use crate::storage::types::StorageError;

impl StorageClient {
    /// Update only the sparse named vector for a batch of points.
    ///
    /// Leaves the dense vector and payload untouched. Used by `rebalance-idf`
    /// to apply IDF correction factors without re-embedding dense vectors.
    pub async fn update_named_sparse_vectors(
        &self,
        collection_name: &str,
        updates: Vec<(String, std::collections::HashMap<u32, f32>)>,
    ) -> Result<(), StorageError> {
        use qdrant_client::qdrant::{
            point_id, vector, vectors, NamedVectors, PointVectors, SparseVector,
            UpdatePointVectorsBuilder, Vector, Vectors,
        };

        if updates.is_empty() {
            return Ok(());
        }

        let point_vectors: Vec<PointVectors> = updates
            .into_iter()
            .map(|(id, sparse_map)| {
                let mut entries: Vec<(u32, f32)> = sparse_map.into_iter().collect();
                entries.sort_by_key(|(idx, _)| *idx);
                let indices: Vec<u32> = entries.iter().map(|(i, _)| *i).collect();
                let values: Vec<f32> = entries.iter().map(|(_, v)| *v).collect();

                let sparse_vec = Vector {
                    vector: Some(vector::Vector::Sparse(SparseVector { indices, values })),
                    ..Default::default()
                };
                let mut named = std::collections::HashMap::new();
                named.insert("sparse".to_string(), sparse_vec);

                PointVectors {
                    id: Some(qdrant_client::qdrant::PointId {
                        point_id_options: Some(point_id::PointIdOptions::Uuid(id)),
                    }),
                    vectors: Some(Vectors {
                        vectors_options: Some(vectors::VectorsOptions::Vectors(NamedVectors {
                            vectors: named,
                        })),
                    }),
                }
            })
            .collect();

        let builder = UpdatePointVectorsBuilder::new(collection_name, point_vectors).wait(true);

        self.retry_operation(|| async {
            self.client
                .update_vectors(builder.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to update sparse vectors: {}", e)))
        })
        .await?;

        Ok(())
    }

    /// Update payload fields on a single point identified by UUID.
    ///
    /// Convenience wrapper that avoids exposing `qdrant_client::Filter` to
    /// callers in the CLI layer.
    pub async fn set_payload_on_point(
        &self,
        collection_name: &str,
        point_id: &str,
        payload: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<(), StorageError> {
        use qdrant_client::qdrant::{point_id, PointId, SetPayloadPointsBuilder};

        let id = PointId {
            point_id_options: Some(point_id::PointIdOptions::Uuid(point_id.to_string())),
        };

        let qdrant_payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> =
            payload
                .into_iter()
                .map(|(k, v)| (k, convert_json_to_qdrant_value(v)))
                .collect();

        // Vec<PointId> implements Into<PointsIdsList> which implements Into<PointsSelectorOneOf>.
        let set_payload_request = SetPayloadPointsBuilder::new(collection_name, qdrant_payload)
            .points_selector(vec![id])
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .set_payload(set_payload_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to set payload on point: {}", e)))
        })
        .await?;

        Ok(())
    }

    /// Update payload fields on all points matching a filter.
    ///
    /// Used for cascade renames where tenant_id needs to be updated.
    pub async fn set_payload_by_filter(
        &self,
        collection_name: &str,
        filter: Filter,
        payload: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<(), StorageError> {
        use qdrant_client::qdrant::SetPayloadPointsBuilder;

        if !self.collection_exists(collection_name).await? {
            return Err(StorageError::Collection(format!(
                "Collection not found: {}",
                collection_name
            )));
        }

        let qdrant_payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> =
            payload
                .into_iter()
                .map(|(k, v)| (k, convert_json_to_qdrant_value(v)))
                .collect();

        let count = self
            .count_points_with_filter(collection_name, filter.clone())
            .await?;
        info!(
            "Updating payload on {} point(s) in collection '{}'",
            count, collection_name
        );

        let set_payload_request = SetPayloadPointsBuilder::new(collection_name, qdrant_payload)
            .points_selector(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .set_payload(set_payload_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to set payload: {}", e)))
        })
        .await?;

        info!(
            "Updated payload on {} point(s) in '{}'",
            count, collection_name
        );

        Ok(())
    }

    /// Rewrite the absolute path payload keys of a tenant's points by prefix,
    /// for the recover/re-point command (#140).
    ///
    /// Scrolls every point in `collection_name` whose `tenant_id` payload
    /// matches `tenant_id`, and for the absolute path keys `file_path` and
    /// `absolute_path`, replaces a leading `old_prefix` with `new_prefix`. The
    /// prefix match is anchored at a path boundary: a stored path matches only
    /// when it equals `old_prefix` exactly or continues with a `/` (so the
    /// sibling directory `/old/projfoo/x.rs` is never rewritten by an
    /// `/old/proj` move). The `relative_path` payload is watch-root-relative and
    /// is left untouched, so only the chunk's absolute location is corrected. A
    /// per-point set_payload is issued only when at least one key actually
    /// changed, so re-running on an already-correct tenant is a cheap no-op
    /// (idempotent).
    ///
    /// When `dry_run` is true, no writes are issued — the return value is the
    /// number of points that *would* change. When applying, the return value is
    /// the number of points actually written (the two agree on success).
    pub async fn rewrite_path_payload_prefix_by_tenant(
        &self,
        collection_name: &str,
        tenant_id: &str,
        old_prefix: &str,
        new_prefix: &str,
        dry_run: bool,
    ) -> Result<u64, StorageError> {
        use qdrant_client::qdrant::{value::Kind, Condition, Filter, PointId, ScrollPointsBuilder};

        if !self.collection_exists(collection_name).await? {
            // A tenant may have no media/scratchpad collection — treat as
            // nothing to do rather than an error.
            return Ok(0);
        }

        const PATH_KEYS: [&str; 2] = ["file_path", "absolute_path"];
        let filter = Filter::must([Condition::matches("tenant_id", tenant_id.to_string())]);

        // `changed` counts points that *would* change (the dry-run answer);
        // `written` counts points actually persisted. They agree on a clean
        // apply, but tracking them separately keeps the return value honest if a
        // write is ever skipped (e.g. a non-UUID point id).
        let mut changed: u64 = 0;
        let mut written: u64 = 0;
        let mut offset: Option<PointId> = None;
        let batch_size = 100u32;

        loop {
            let f = filter.clone();
            let o = offset.clone();
            let response = self
                .retry_operation(|| {
                    let f = f.clone();
                    let o = o.clone();
                    async move {
                        let mut builder = ScrollPointsBuilder::new(collection_name)
                            .filter(f)
                            .limit(batch_size)
                            .with_payload(true)
                            .with_vectors(false);
                        if let Some(id) = o {
                            builder = builder.offset(id);
                        }
                        self.client.scroll(builder).await.map_err(|e| {
                            StorageError::Search(format!("Scroll for repath failed: {}", e))
                        })
                    }
                })
                .await?;

            for point in &response.result {
                let mut rewritten: std::collections::HashMap<String, serde_json::Value> =
                    std::collections::HashMap::new();
                for key in PATH_KEYS {
                    if let Some(Kind::StringValue(path)) =
                        point.payload.get(key).and_then(|v| v.kind.as_ref())
                    {
                        if let Some(new_path) =
                            rewrite_path_under_prefix(path, old_prefix, new_prefix)
                        {
                            rewritten
                                .insert(key.to_string(), serde_json::Value::String(new_path));
                        }
                    }
                }

                if rewritten.is_empty() {
                    continue;
                }
                changed += 1;

                if dry_run {
                    continue;
                }
                if let Some(id) = point.id.as_ref() {
                    if let Some(id_str) = point_id_to_uuid(id) {
                        self.set_payload_on_point(collection_name, &id_str, rewritten)
                            .await?;
                        written += 1;
                    }
                }
            }

            match response.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }

        let reported = if dry_run { changed } else { written };
        info!(
            "Repath {} point(s) in '{}' ({} -> {}, dry_run={})",
            reported, collection_name, old_prefix, new_prefix, dry_run
        );
        Ok(reported)
    }
}

/// Rewrite `path` when it sits under `old_prefix`, returning the new path.
///
/// The match is anchored at a path boundary: `path` matches only when it equals
/// `old_prefix` exactly or continues with a `/` separator, so the sibling
/// directory `/old/projfoo/x.rs` is never rewritten by an `/old/proj` move.
/// Returns `None` when `path` is outside the old root, or when the rewrite would
/// be a no-op (old == new prefix) — the caller then issues no write.
fn rewrite_path_under_prefix(path: &str, old_prefix: &str, new_prefix: &str) -> Option<String> {
    let rest = path
        .strip_prefix(old_prefix)
        .filter(|rest| rest.is_empty() || rest.starts_with('/'))?;
    let new_path = format!("{}{}", new_prefix, rest);
    (new_path != path).then_some(new_path)
}

/// Extract a UUID string from a Qdrant point id (the only id form
/// `set_payload_on_point` accepts). Numeric ids are not used by this codebase
/// for content points and are skipped.
fn point_id_to_uuid(id: &qdrant_client::qdrant::PointId) -> Option<String> {
    use qdrant_client::qdrant::point_id::PointIdOptions;
    match id.point_id_options.as_ref()? {
        PointIdOptions::Uuid(uuid) => Some(uuid.clone()),
        PointIdOptions::Num(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::rewrite_path_under_prefix;

    #[test]
    fn rewrites_a_child_path() {
        assert_eq!(
            rewrite_path_under_prefix("/old/proj/src/a.rs", "/old/proj", "/new/home"),
            Some("/new/home/src/a.rs".to_string())
        );
    }

    #[test]
    fn rewrites_the_root_itself() {
        assert_eq!(
            rewrite_path_under_prefix("/old/proj", "/old/proj", "/new/home"),
            Some("/new/home".to_string())
        );
    }

    #[test]
    fn leaves_a_sibling_directory_untouched() {
        // `/old/proj` must not match `/old/projfoo/...` (the regression guard).
        assert_eq!(
            rewrite_path_under_prefix("/old/projfoo/x.rs", "/old/proj", "/new/home"),
            None
        );
    }

    #[test]
    fn leaves_an_unrelated_path_untouched() {
        assert_eq!(
            rewrite_path_under_prefix("/elsewhere/x.rs", "/old/proj", "/new/home"),
            None
        );
    }

    #[test]
    fn no_op_when_prefix_is_unchanged() {
        // Already correct: re-running recover finds nothing to write (idempotent).
        assert_eq!(
            rewrite_path_under_prefix("/old/proj/src/a.rs", "/old/proj", "/old/proj"),
            None
        );
    }
}
