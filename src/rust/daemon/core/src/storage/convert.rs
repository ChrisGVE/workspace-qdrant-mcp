//! Value conversion helpers
//!
//! Converts between `serde_json::Value` and Qdrant protobuf `Value` types,
//! and converts `DocumentPoint` to Qdrant `PointStruct`.

use std::collections::HashMap;
use qdrant_client::qdrant::{
    DenseVector, PointStruct, SparseVector,
};

use super::types::{DocumentPoint, StorageError};

/// Convert a `DocumentPoint` to a Qdrant `PointStruct`.
///
/// Uses named vectors: "dense" for semantic vectors, "sparse" for BM25-style
/// keyword vectors.
pub(crate) fn convert_to_qdrant_point(point: DocumentPoint) -> Result<PointStruct, StorageError> {
    let payload = point.payload.into_iter()
        .map(|(k, v)| (k, convert_json_to_qdrant_value(v)))
        .collect();

    // Build named vectors with "dense" and optionally "sparse"
    let mut named_vectors = HashMap::new();

    // Add dense vector
    named_vectors.insert(
        "dense".to_string(),
        DenseVector { data: point.dense_vector }.into(),
    );

    // Add sparse vector if present
    if let Some(sparse_map) = point.sparse_vector {
        // Convert HashMap<u32, f32> to (indices, values) for SparseVector
        let mut indices: Vec<u32> = Vec::with_capacity(sparse_map.len());
        let mut values: Vec<f32> = Vec::with_capacity(sparse_map.len());

        // Sort by index for consistent ordering
        let mut entries: Vec<_> = sparse_map.into_iter().collect();
        entries.sort_by_key(|(idx, _)| *idx);

        for (idx, val) in entries {
            indices.push(idx);
            values.push(val);
        }

        named_vectors.insert(
            "sparse".to_string(),
            SparseVector { indices, values }.into(),
        );
    }

    Ok(PointStruct {
        id: Some(qdrant_client::qdrant::PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(point.id)),
        }),
        vectors: Some(qdrant_client::qdrant::Vectors {
            vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vectors(
                qdrant_client::qdrant::NamedVectors {
                    vectors: named_vectors,
                }
            )),
        }),
        payload,
    })
}

/// Convert a JSON value to a Qdrant protobuf value
pub(crate) fn convert_json_to_qdrant_value(value: serde_json::Value) -> qdrant_client::qdrant::Value {
    match value {
        serde_json::Value::Null => qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::NullValue(0)),
        },
        serde_json::Value::Bool(b) => qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::BoolValue(b)),
        },
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)),
                }
            } else if let Some(f) = n.as_f64() {
                qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)),
                }
            } else {
                qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::StringValue(n.to_string())),
                }
            }
        },
        serde_json::Value::String(s) => qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(s)),
        },
        serde_json::Value::Array(arr) => {
            let list_value = qdrant_client::qdrant::ListValue {
                values: arr.into_iter()
                    .map(convert_json_to_qdrant_value)
                    .collect(),
            };
            qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::ListValue(list_value)),
            }
        },
        serde_json::Value::Object(obj) => {
            let struct_value = qdrant_client::qdrant::Struct {
                fields: obj.into_iter()
                    .map(|(k, v)| (k, convert_json_to_qdrant_value(v)))
                    .collect(),
            };
            qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::StructValue(struct_value)),
            }
        }
    }
}

/// Convert a Qdrant protobuf value to a JSON value
pub(crate) fn convert_qdrant_value_to_json(value: qdrant_client::qdrant::Value) -> serde_json::Value {
    match value.kind {
        Some(qdrant_client::qdrant::value::Kind::NullValue(_)) => serde_json::Value::Null,
        Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => serde_json::Value::Bool(b),
        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => serde_json::Value::Number(i.into()),
        Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)) => {
            serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)))
        },
        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => serde_json::Value::String(s),
        Some(qdrant_client::qdrant::value::Kind::ListValue(list)) => {
            serde_json::Value::Array(
                list.values.into_iter()
                    .map(convert_qdrant_value_to_json)
                    .collect()
            )
        },
        Some(qdrant_client::qdrant::value::Kind::StructValue(struct_val)) => {
            serde_json::Value::Object(
                struct_val.fields.into_iter()
                    .map(|(k, v)| (k, convert_qdrant_value_to_json(v)))
                    .collect()
            )
        },
        None => serde_json::Value::Null,
    }
}
