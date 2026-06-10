//! Point count and existence check operations
//!
//! Query point counts and check existence of specific point IDs.

use qdrant_client::qdrant::{Condition, CountPointsBuilder, Filter, GetPointsBuilder};
use tracing::debug;

use crate::storage::client::StorageClient;
use crate::storage::types::StorageError;

impl StorageClient {
    /// Count points matching a specific filter (private helper)
    pub(crate) async fn count_points_with_filter(
        &self,
        collection_name: &str,
        filter: Filter,
    ) -> Result<u64, StorageError> {
        let builder = CountPointsBuilder::new(collection_name)
            .filter(filter)
            .exact(true);

        let count = self
            .retry_operation(|| async {
                self.client
                    .count(builder.clone())
                    .await
                    .map_err(|e| StorageError::Collection(e.to_string()))
            })
            .await?;

        Ok(count.result.map(|r| r.count).unwrap_or(0))
    }

    /// Count points in a collection, optionally filtered by tenant_id
    pub async fn count_points(
        &self,
        collection_name: &str,
        tenant_id: Option<&str>,
    ) -> Result<u64, StorageError> {
        debug!(
            "Counting points in collection: {} (tenant: {:?})",
            collection_name, tenant_id
        );

        let mut builder = CountPointsBuilder::new(collection_name).exact(true);

        if let Some(tid) = tenant_id {
            builder = builder.filter(Filter::must([Condition::matches(
                "tenant_id",
                tid.to_string(),
            )]));
        }

        let count = self
            .retry_operation(|| async {
                self.client
                    .count(builder.clone())
                    .await
                    .map_err(|e| StorageError::Collection(e.to_string()))
            })
            .await?;

        Ok(count.result.map(|r| r.count).unwrap_or(0))
    }

    /// Check which point UUIDs exist in a collection.
    ///
    /// Fetches points by ID with no payload or vectors (minimal overhead).
    /// Returns the set of IDs that actually exist in Qdrant, expressed in
    /// the exact string form the caller passed in.
    ///
    /// Qdrant parses string point IDs as UUIDs and renders them back in
    /// canonical hyphenated form: a point upserted with the bare 32-hex ID
    /// `compute_point_id` produces (`"00006c52…a0c8"`) comes back as
    /// `"00006c52-38e7-…-a0c8"`. A raw string comparison therefore reported
    /// every stored point as missing, and the orphan-cleanup idle task mass-
    /// deleted valid `qdrant_chunks` rows. Existence is matched on the
    /// canonical form and translated back to the caller's representation.
    pub async fn check_points_exist(
        &self,
        collection_name: &str,
        point_ids: &[String],
    ) -> Result<std::collections::HashSet<String>, StorageError> {
        use qdrant_client::qdrant::PointId;
        use std::collections::HashSet;

        if point_ids.is_empty() {
            return Ok(HashSet::new());
        }

        let ids: Vec<PointId> = point_ids
            .iter()
            .map(|id| PointId::from(id.as_str()))
            .collect();

        let response = self
            .retry_operation(|| async {
                let builder = GetPointsBuilder::new(collection_name, ids.clone())
                    .with_payload(false)
                    .with_vectors(false);
                self.client
                    .get_points(builder)
                    .await
                    .map_err(|e| StorageError::Point(e.to_string()))
            })
            .await?;

        let returned = response.result.into_iter().filter_map(|p| {
            p.id.and_then(|pid| {
                use qdrant_client::qdrant::point_id::PointIdOptions;
                match pid.point_id_options {
                    Some(PointIdOptions::Uuid(u)) => Some(u),
                    Some(PointIdOptions::Num(n)) => Some(n.to_string()),
                    None => None,
                }
            })
        });

        let existing: HashSet<String> = match_returned_ids_to_inputs(point_ids, returned);

        debug!(
            "check_points_exist: {}/{} points exist in {}",
            existing.len(),
            point_ids.len(),
            collection_name
        );

        Ok(existing)
    }
}

/// Canonical comparison form for a point ID string: lowercase with UUID
/// hyphens stripped, so the hyphenated form Qdrant returns and the bare
/// 32-hex form `compute_point_id` stores compare equal. Numeric IDs pass
/// through unchanged (no hyphens to strip).
fn canonical_point_id(id: &str) -> String {
    id.chars()
        .filter(|c| *c != '-')
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Map the IDs Qdrant returned back to the caller's input strings.
///
/// Returned IDs with no corresponding input (a Qdrant-side normalization we
/// cannot reverse) are dropped — callers only ever test membership of their
/// own inputs.
fn match_returned_ids_to_inputs(
    inputs: &[String],
    returned: impl IntoIterator<Item = String>,
) -> std::collections::HashSet<String> {
    let by_canon: std::collections::HashMap<String, &String> =
        inputs.iter().map(|s| (canonical_point_id(s), s)).collect();
    returned
        .into_iter()
        .filter_map(|r| by_canon.get(&canonical_point_id(&r)).map(|s| (*s).clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_strips_hyphens_and_lowercases() {
        assert_eq!(
            canonical_point_id("00006C52-38E7-1A6E-7DDD-4CF06EC2A0C8"),
            "00006c5238e71a6e7ddd4cf06ec2a0c8"
        );
        assert_eq!(
            canonical_point_id("00006c5238e71a6e7ddd4cf06ec2a0c8"),
            "00006c5238e71a6e7ddd4cf06ec2a0c8"
        );
        assert_eq!(canonical_point_id("42"), "42");
    }

    /// Regression for the orphan-cleanup mass-deletion: bare-hex stored IDs
    /// must match the hyphenated canonical UUIDs Qdrant returns, and the
    /// result must be in the caller's (stored) form.
    #[test]
    fn bare_hex_inputs_match_hyphenated_returns() {
        let inputs = vec![
            "00006c5238e71a6e7ddd4cf06ec2a0c8".to_string(),
            "5a6a64b1796e420ecd790ea97ff3944a".to_string(),
        ];
        let returned = vec![
            "00006c52-38e7-1a6e-7ddd-4cf06ec2a0c8".to_string(),
            "5a6a64b1-796e-420e-cd79-0ea97ff3944a".to_string(),
        ];
        let existing = match_returned_ids_to_inputs(&inputs, returned);
        assert!(existing.contains(inputs[0].as_str()));
        assert!(existing.contains(inputs[1].as_str()));
        assert_eq!(existing.len(), 2);
    }

    #[test]
    fn missing_points_stay_missing() {
        let inputs = vec![
            "00006c5238e71a6e7ddd4cf06ec2a0c8".to_string(),
            "5a6a64b1796e420ecd790ea97ff3944a".to_string(),
        ];
        let returned = vec!["00006c52-38e7-1a6e-7ddd-4cf06ec2a0c8".to_string()];
        let existing = match_returned_ids_to_inputs(&inputs, returned);
        assert!(existing.contains(inputs[0].as_str()));
        assert!(!existing.contains(inputs[1].as_str()));
    }

    #[test]
    fn hyphenated_inputs_and_numeric_ids_round_trip() {
        let inputs = vec![
            "00006c52-38e7-1a6e-7ddd-4cf06ec2a0c8".to_string(),
            "42".to_string(),
        ];
        let returned = vec![
            "00006C52-38E7-1A6E-7DDD-4CF06EC2A0C8".to_string(),
            "42".to_string(),
        ];
        let existing = match_returned_ids_to_inputs(&inputs, returned);
        assert!(existing.contains(inputs[0].as_str()));
        assert!(existing.contains("42"));
    }
}
