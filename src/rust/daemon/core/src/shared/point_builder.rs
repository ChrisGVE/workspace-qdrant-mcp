//! Builder for constructing `DocumentPoint` structs consistently.
//!
//! Wraps `PayloadBuilder` and adds dense/sparse vector fields to produce
//! a complete `DocumentPoint` ready for Qdrant upsert.

use std::collections::HashMap;
use serde_json::Value;

use crate::storage::DocumentPoint;
use super::payload_builder::PayloadBuilder;

/// Fluent builder for `DocumentPoint` construction.
///
/// Combines a point ID, dense vector, optional sparse vector, and a
/// payload built via `PayloadBuilder` into a complete `DocumentPoint`.
///
/// # Example
/// ```ignore
/// let point = PointBuilder::new("point-id-abc")
///     .dense_vector(vec![0.1, 0.2, 0.3])
///     .sparse_vector(Some(sparse_map))
///     .payload(PayloadBuilder::new().tenant_id("t").content("text").build())
///     .build();
/// ```
pub struct PointBuilder {
    id: String,
    dense_vector: Vec<f32>,
    sparse_vector: Option<HashMap<u32, f32>>,
    payload: HashMap<String, Value>,
}

impl PointBuilder {
    /// Create a new point builder with the given point ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            dense_vector: Vec::new(),
            sparse_vector: None,
            payload: HashMap::new(),
        }
    }

    /// Set the dense embedding vector.
    pub fn dense_vector(mut self, vector: Vec<f32>) -> Self {
        self.dense_vector = vector;
        self
    }

    /// Set the sparse BM25 vector.
    pub fn sparse_vector(mut self, sparse: Option<HashMap<u32, f32>>) -> Self {
        self.sparse_vector = sparse;
        self
    }

    /// Set the payload from a pre-built HashMap.
    pub fn payload(mut self, payload: HashMap<String, Value>) -> Self {
        self.payload = payload;
        self
    }

    /// Set the payload using a PayloadBuilder (convenience method).
    pub fn with_payload_builder(mut self, builder: PayloadBuilder) -> Self {
        self.payload = builder.build();
        self
    }

    /// Consume the builder and produce a `DocumentPoint`.
    pub fn build(self) -> DocumentPoint {
        DocumentPoint {
            id: self.id,
            dense_vector: self.dense_vector,
            sparse_vector: self.sparse_vector,
            payload: self.payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_builder_basic() {
        let point = PointBuilder::new("test-id")
            .dense_vector(vec![0.1, 0.2, 0.3])
            .build();

        assert_eq!(point.id, "test-id");
        assert_eq!(point.dense_vector.len(), 3);
        assert!(point.sparse_vector.is_none());
        assert!(point.payload.is_empty());
    }

    #[test]
    fn test_point_builder_with_payload() {
        let payload = PayloadBuilder::new()
            .tenant_id("tenant-1")
            .content("hello")
            .build();

        let point = PointBuilder::new("p1")
            .dense_vector(vec![1.0])
            .payload(payload)
            .build();

        assert_eq!(point.payload["tenant_id"], serde_json::json!("tenant-1"));
        assert_eq!(point.payload["content"], serde_json::json!("hello"));
    }

    #[test]
    fn test_point_builder_with_sparse() {
        let mut sparse = HashMap::new();
        sparse.insert(5u32, 0.7f32);

        let point = PointBuilder::new("p2")
            .dense_vector(vec![1.0])
            .sparse_vector(Some(sparse))
            .build();

        let sv = point.sparse_vector.expect("should have sparse");
        assert!((sv[&5] - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_point_builder_with_payload_builder() {
        let point = PointBuilder::new("p3")
            .dense_vector(vec![0.5])
            .with_payload_builder(
                PayloadBuilder::new()
                    .tenant_id("t")
                    .item_type("file"),
            )
            .build();

        assert_eq!(point.payload["tenant_id"], serde_json::json!("t"));
        assert_eq!(point.payload["item_type"], serde_json::json!("file"));
    }
}
