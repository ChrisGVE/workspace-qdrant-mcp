//! Qdrant operation types and traits

use crate::qdrant::error::{QdrantError, QdrantResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use qdrant_client::qdrant::{PointId, PointStruct, SearchParams, Distance};

/// Vector operation types
#[derive(Debug, Clone)]
pub enum VectorOperation {
    /// Upsert points (insert or update)
    Upsert {
        collection_name: String,
        points: Vec<PointStruct>,
        wait: bool,
    },
    /// Retrieve points by IDs
    GetPoints {
        collection_name: String,
        point_ids: Vec<PointId>,
        with_payload: bool,
        with_vector: bool,
    },
    /// Delete points by IDs
    DeletePoints {
        collection_name: String,
        point_ids: Vec<PointId>,
        wait: bool,
    },
    /// Update point payload
    UpdatePayload {
        collection_name: String,
        point_id: PointId,
        payload: HashMap<String, serde_json::Value>,
        wait: bool,
    },
    /// Delete payload fields
    DeletePayload {
        collection_name: String,
        point_id: PointId,
        payload_keys: Vec<String>,
        wait: bool,
    },
}

/// Search operation configuration
#[derive(Debug, Clone)]
pub struct SearchOperation {
    /// Collection to search in
    pub collection_name: String,
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of results to return
    pub limit: u64,
    /// Offset for pagination
    pub offset: Option<u64>,
    /// Search parameters
    pub params: Option<SearchParams>,
    /// Filter conditions
    pub filter: Option<serde_json::Value>,
    /// Include payload in results
    pub with_payload: bool,
    /// Include vectors in results
    pub with_vector: bool,
    /// Score threshold
    pub score_threshold: Option<f32>,
}

/// Collection operation types
#[derive(Debug, Clone)]
pub enum CollectionOperation {
    /// Create a new collection
    Create {
        collection_name: String,
        vector_size: u64,
        distance: Distance,
        shard_number: Option<u32>,
        replication_factor: Option<u32>,
        on_disk_vectors: Option<bool>,
    },
    /// Delete a collection
    Delete {
        collection_name: String,
    },
    /// Get collection info
    GetInfo {
        collection_name: String,
    },
    /// List all collections
    List,
    /// Update collection parameters
    Update {
        collection_name: String,
        optimizers_config: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    },
    /// Create alias for collection
    CreateAlias {
        collection_name: String,
        alias_name: String,
    },
    /// Delete alias
    DeleteAlias {
        alias_name: String,
    },
}

/// Batch operation for processing multiple operations efficiently
#[derive(Debug, Clone)]
pub struct BatchOperation {
    /// Collection name for batch operations
    pub collection_name: String,
    /// Vector operations to perform in batch
    pub operations: Vec<VectorOperation>,
    /// Maximum batch size
    pub batch_size: usize,
    /// Parallel execution configuration
    pub parallel: bool,
    /// Wait for operation completion
    pub wait: bool,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Point ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Point payload
    pub payload: Option<HashMap<String, serde_json::Value>>,
    /// Point vector
    pub vector: Option<Vec<f32>>,
}

/// Collection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    /// Collection name
    pub name: String,
    /// Vector configuration
    pub vectors_config: VectorConfig,
    /// Number of points
    pub points_count: u64,
    /// Collection status
    pub status: String,
    /// Optimization status
    pub optimizer_status: String,
    /// Index status
    pub indexed_vectors_count: u64,
}

/// Vector configuration for collections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    /// Vector size
    pub size: u64,
    /// Distance metric
    pub distance: String,
    /// On disk storage
    pub on_disk: Option<bool>,
}

/// Point data structure for upsert operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    /// Unique point identifier
    pub id: String,
    /// Vector data
    pub vector: Vec<f32>,
    /// Metadata payload
    pub payload: HashMap<String, serde_json::Value>,
}

impl Point {
    /// Create a new point
    pub fn new(id: String, vector: Vec<f32>) -> Self {
        Self {
            id,
            vector,
            payload: HashMap::new(),
        }
    }

    /// Add payload field
    pub fn with_payload<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.payload.insert(key.into(), value.into());
        self
    }

    /// Convert to qdrant PointStruct
    pub fn to_point_struct(&self) -> QdrantResult<PointStruct> {
        let point_id = PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(
                self.id.clone()
            )),
        };

        let vectors = qdrant_client::qdrant::vectors::VectorsOptions::Vector(
            qdrant_client::qdrant::Vector {
                data: self.vector.clone(),
                indices: None,
                vector: None,
                vectors_count: None,
            }
        );

        // Convert payload to qdrant format
        let mut payload_map = HashMap::new();
        for (key, value) in &self.payload {
            let qdrant_value = json_to_qdrant_value(value)
                .map_err(|e| QdrantError::Serialization { message: e.to_string() })?;
            payload_map.insert(key.clone(), qdrant_value);
        }

        Ok(PointStruct {
            id: Some(point_id),
            vectors: Some(qdrant_client::qdrant::Vectors {
                vectors_options: Some(vectors),
            }),
            payload: payload_map,
        })
    }
}

/// Convert JSON value to Qdrant value
pub fn json_to_qdrant_value(value: &serde_json::Value) -> Result<qdrant_client::qdrant::Value, serde_json::Error> {
    use qdrant_client::qdrant::{Value, value::Kind};

    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Kind::IntegerValue(i)
            } else if let Some(f) = n.as_f64() {
                Kind::DoubleValue(f)
            } else {
                return Err(serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Invalid number"
                )));
            }
        },
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => {
            let mut list_values = Vec::new();
            for item in arr {
                list_values.push(json_to_qdrant_value(item)?);
            }
            Kind::ListValue(qdrant_client::qdrant::ListValue {
                values: list_values,
            })
        },
        serde_json::Value::Object(obj) => {
            let mut struct_fields = HashMap::new();
            for (key, val) in obj {
                struct_fields.insert(key.clone(), json_to_qdrant_value(val)?);
            }
            Kind::StructValue(qdrant_client::qdrant::Struct {
                fields: struct_fields,
            })
        },
    };

    Ok(Value { kind: Some(kind) })
}

impl SearchOperation {
    /// Create a new search operation
    pub fn new(collection_name: String, vector: Vec<f32>, limit: u64) -> Self {
        Self {
            collection_name,
            vector,
            limit,
            offset: None,
            params: None,
            filter: None,
            with_payload: true,
            with_vector: false,
            score_threshold: None,
        }
    }

    /// Set search offset for pagination
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Set search parameters
    pub fn with_params(mut self, params: SearchParams) -> Self {
        self.params = Some(params);
        self
    }

    /// Set filter conditions
    pub fn with_filter(mut self, filter: serde_json::Value) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Include vectors in search results
    pub fn with_vector(mut self) -> Self {
        self.with_vector = true;
        self
    }

    /// Set score threshold
    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }

    /// Validate the search operation parameters
    pub fn validate(&self) -> QdrantResult<()> {
        if self.collection_name.is_empty() {
            return Err(QdrantError::InvalidParameters {
                message: "Collection name cannot be empty".to_string(),
            });
        }

        if self.vector.is_empty() {
            return Err(QdrantError::InvalidParameters {
                message: "Search vector cannot be empty".to_string(),
            });
        }

        if self.limit == 0 {
            return Err(QdrantError::InvalidParameters {
                message: "Search limit must be greater than 0".to_string(),
            });
        }

        if let Some(threshold) = self.score_threshold {
            if threshold < 0.0 || threshold > 1.0 {
                return Err(QdrantError::InvalidParameters {
                    message: "Score threshold must be between 0.0 and 1.0".to_string(),
                });
            }
        }

        Ok(())
    }
}

impl BatchOperation {
    /// Create a new batch operation
    pub fn new(collection_name: String, operations: Vec<VectorOperation>) -> Self {
        Self {
            collection_name,
            operations,
            batch_size: 100, // Default batch size
            parallel: false,
            wait: true,
        }
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable parallel execution
    pub fn with_parallel_execution(mut self) -> Self {
        self.parallel = true;
        self
    }

    /// Set wait behavior
    pub fn with_wait(mut self, wait: bool) -> Self {
        self.wait = wait;
        self
    }

    /// Validate batch operation
    pub fn validate(&self) -> QdrantResult<()> {
        if self.collection_name.is_empty() {
            return Err(QdrantError::InvalidParameters {
                message: "Collection name cannot be empty".to_string(),
            });
        }

        if self.operations.is_empty() {
            return Err(QdrantError::InvalidParameters {
                message: "Batch operations cannot be empty".to_string(),
            });
        }

        if self.batch_size == 0 {
            return Err(QdrantError::InvalidParameters {
                message: "Batch size must be greater than 0".to_string(),
            });
        }

        // Validate that all operations are for the same collection
        for operation in &self.operations {
            let op_collection = match operation {
                VectorOperation::Upsert { collection_name, .. } => collection_name,
                VectorOperation::GetPoints { collection_name, .. } => collection_name,
                VectorOperation::DeletePoints { collection_name, .. } => collection_name,
                VectorOperation::UpdatePayload { collection_name, .. } => collection_name,
                VectorOperation::DeletePayload { collection_name, .. } => collection_name,
            };

            if op_collection != &self.collection_name {
                return Err(QdrantError::InvalidParameters {
                    message: format!(
                        "All operations must be for the same collection. Expected: {}, found: {}",
                        self.collection_name, op_collection
                    ),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let point = Point::new("test-id".to_string(), vec![1.0, 2.0, 3.0])
            .with_payload("text", "test document")
            .with_payload("category", "test");

        assert_eq!(point.id, "test-id");
        assert_eq!(point.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(point.payload.len(), 2);
    }

    #[test]
    fn test_point_to_point_struct() {
        let point = Point::new("test-id".to_string(), vec![1.0, 2.0, 3.0])
            .with_payload("text", "test document");

        let point_struct = point.to_point_struct().unwrap();
        assert!(point_struct.id.is_some());
        assert!(point_struct.vectors.is_some());
        assert!(!point_struct.payload.is_empty());
    }

    #[test]
    fn test_search_operation_validation() {
        let search = SearchOperation::new("test-collection".to_string(), vec![1.0, 2.0], 10);
        assert!(search.validate().is_ok());

        // Test invalid collection name
        let invalid_search = SearchOperation::new("".to_string(), vec![1.0, 2.0], 10);
        assert!(invalid_search.validate().is_err());

        // Test empty vector
        let invalid_search = SearchOperation::new("test".to_string(), vec![], 10);
        assert!(invalid_search.validate().is_err());

        // Test zero limit
        let invalid_search = SearchOperation::new("test".to_string(), vec![1.0], 0);
        assert!(invalid_search.validate().is_err());
    }

    #[test]
    fn test_batch_operation_validation() {
        let operations = vec![
            VectorOperation::Upsert {
                collection_name: "test".to_string(),
                points: vec![],
                wait: true,
            }
        ];

        let batch = BatchOperation::new("test".to_string(), operations);
        assert!(batch.validate().is_ok());

        // Test empty operations
        let empty_batch = BatchOperation::new("test".to_string(), vec![]);
        assert!(empty_batch.validate().is_err());

        // Test mismatched collection names
        let mixed_operations = vec![
            VectorOperation::Upsert {
                collection_name: "test1".to_string(),
                points: vec![],
                wait: true,
            },
            VectorOperation::Upsert {
                collection_name: "test2".to_string(),
                points: vec![],
                wait: true,
            },
        ];
        let mixed_batch = BatchOperation::new("test1".to_string(), mixed_operations);
        assert!(mixed_batch.validate().is_err());
    }

    #[test]
    fn test_search_operation_builder() {
        let search = SearchOperation::new("test".to_string(), vec![1.0, 2.0], 10)
            .with_offset(5)
            .with_vector()
            .with_score_threshold(0.8);

        assert_eq!(search.offset, Some(5));
        assert_eq!(search.with_vector, true);
        assert_eq!(search.score_threshold, Some(0.8));
    }

    #[test]
    fn test_batch_operation_builder() {
        let operations = vec![
            VectorOperation::Upsert {
                collection_name: "test".to_string(),
                points: vec![],
                wait: true,
            }
        ];

        let batch = BatchOperation::new("test".to_string(), operations)
            .with_batch_size(50)
            .with_parallel_execution()
            .with_wait(false);

        assert_eq!(batch.batch_size, 50);
        assert_eq!(batch.parallel, true);
        assert_eq!(batch.wait, false);
    }
}