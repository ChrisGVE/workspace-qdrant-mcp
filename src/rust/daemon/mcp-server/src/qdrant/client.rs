//! Read-only Qdrant client for the MCP server.
//!
//! `QdrantReadClient` wraps `qdrant_client::Qdrant` and exposes only the
//! read operations required by the MCP search/retrieve tools:
//!
//! - `search`            — named-vector query (dense or sparse)
//! - `scroll`            — paginated scan with optional filter
//! - `retrieve`          — fetch specific point IDs with payload
//! - `collection_exists` — check whether a collection is present
//!
//! The API key is stored as `secrecy::SecretString` and is never written to
//! any log output (neither via `Debug` nor `Display`).
//!
//! **No write operations are ever called from this module.**

use std::collections::HashMap;

use anyhow::{Context, Result};
use qdrant_client::config::QdrantConfig;
use qdrant_client::qdrant::{
    point_id::PointIdOptions, Filter, GetPointsBuilder, PointId, QueryPointsBuilder, ScoredPoint,
    ScrollPointsBuilder,
};
use qdrant_client::Qdrant;
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;
use tracing::debug;

/// A scored point retrieved from Qdrant, with its payload decoded to JSON.
///
/// This is the canonical result type used throughout the `mcp-server` qdrant
/// module.  It is NOT a re-export of `qdrant_client::qdrant::ScoredPoint` —
/// instead it carries a decoded `payload` map so that callers do not need to
/// depend on Qdrant's internal proto types.
#[derive(Debug, Clone)]
pub struct QdrantPoint {
    /// Point identifier (UUID string or numeric string).
    pub id: String,
    /// Search relevance score (raw float from Qdrant; replaced by RRF in
    /// hybrid mode).
    pub score: f64,
    /// Decoded payload fields.
    pub payload: HashMap<String, Value>,
}

/// A retrieved point (no score) from `retrieve` / scroll.
#[derive(Debug, Clone)]
pub struct QdrantRetrievedPoint {
    pub id: String,
    pub payload: HashMap<String, Value>,
}

/// Read-only Qdrant client.
///
/// Created via [`QdrantReadClient::new`].  The underlying `Qdrant` client is
/// `Clone`-able; `QdrantReadClient` wraps it in an `Arc` so cheap clones do
/// not duplicate the connection.
#[derive(Clone)]
pub struct QdrantReadClient {
    inner: std::sync::Arc<Qdrant>,
}

impl std::fmt::Debug for QdrantReadClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantReadClient").finish_non_exhaustive()
    }
}

impl QdrantReadClient {
    /// Create a new read-only client.
    ///
    /// The `api_key` is consumed as a [`SecretString`] and is only exposed
    /// (via `expose_secret()`) when building the underlying Qdrant config.
    /// It is never written to logs or debug output.
    ///
    /// The `url` is REST-style (`:6333`, the TypeScript server's convention);
    /// it is translated to the gRPC port (`:6334`) via
    /// [`grpc_endpoint`](super::endpoint::grpc_endpoint) because the
    /// `qdrant_client` crate speaks gRPC, not REST.
    pub fn new(url: String, api_key: Option<SecretString>) -> Self {
        let url = super::endpoint::grpc_endpoint(&url);
        let mut config = QdrantConfig::from_url(&url)
            // Disable the Qdrant version-compatibility check to avoid the
            // client printing to stdout when the server is unreachable.  In
            // stdio-mode the MCP protocol owns stdout; any extraneous writes
            // break JSON-RPC framing (AC-T2 stdout purity).
            .skip_compatibility_check();
        if let Some(ref key) = api_key {
            config = config.api_key(key.expose_secret());
        }

        let inner = Qdrant::new(config)
            .unwrap_or_else(|_| panic!("Failed to create Qdrant client for URL: {url}"));

        Self {
            inner: std::sync::Arc::new(inner),
        }
    }

    /// Perform a named-vector query search.
    ///
    /// Uses `QueryPointsBuilder` with a named vector and optional filter.
    /// Equivalent to `qdrantClient.search(collection, { vector: { name, vector },
    /// limit, score_threshold, with_payload: true, filter? })` in TypeScript.
    pub async fn search(
        &self,
        collection: &str,
        vector_name: &str,
        vector: Vec<f32>,
        limit: u64,
        score_threshold: Option<f32>,
        filter: Option<Filter>,
    ) -> Result<Vec<QdrantPoint>> {
        debug!(collection, vector_name, limit, "QdrantReadClient::search");

        let mut builder = QueryPointsBuilder::new(collection)
            .query(vector)
            .using(vector_name)
            .limit(limit)
            .with_payload(true);

        if let Some(f) = filter {
            builder = builder.filter(f);
        }
        if let Some(threshold) = score_threshold {
            builder = builder.score_threshold(threshold);
        }

        let response = self
            .inner
            .query(builder)
            .await
            .with_context(|| format!("Qdrant search failed on collection={collection}"))?;

        Ok(response
            .result
            .into_iter()
            .map(scored_point_to_qdrant_point)
            .collect())
    }

    /// Perform a sparse named-vector search.
    ///
    /// The sparse vector is supplied as `(index, value)` pairs.
    pub async fn search_sparse(
        &self,
        collection: &str,
        vector_name: &str,
        indices: Vec<u32>,
        values: Vec<f32>,
        limit: u64,
        score_threshold: Option<f32>,
        filter: Option<Filter>,
    ) -> Result<Vec<QdrantPoint>> {
        debug!(
            collection,
            vector_name, limit, "QdrantReadClient::search_sparse"
        );

        if indices.is_empty() {
            return Ok(vec![]);
        }

        let pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();

        let mut builder = QueryPointsBuilder::new(collection)
            .query(pairs)
            .using(vector_name)
            .limit(limit)
            .with_payload(true);

        if let Some(f) = filter {
            builder = builder.filter(f);
        }
        if let Some(threshold) = score_threshold {
            builder = builder.score_threshold(threshold);
        }

        let response =
            self.inner.query(builder).await.with_context(|| {
                format!("Qdrant sparse search failed on collection={collection}")
            })?;

        Ok(response
            .result
            .into_iter()
            .map(scored_point_to_qdrant_point)
            .collect())
    }

    /// Paginated scroll through a collection with an optional filter.
    ///
    /// Returns up to `limit` points starting from `offset`.  The
    /// `next_page_offset` field of the scroll response is returned as the
    /// second element of the tuple — callers use it for subsequent pages.
    pub async fn scroll(
        &self,
        collection: &str,
        filter: Option<Filter>,
        limit: u32,
        offset: Option<PointId>,
    ) -> Result<(Vec<QdrantRetrievedPoint>, Option<PointId>)> {
        debug!(collection, limit, "QdrantReadClient::scroll");

        let mut builder = ScrollPointsBuilder::new(collection)
            .limit(limit)
            .with_payload(true)
            .with_vectors(false);

        if let Some(f) = filter {
            builder = builder.filter(f);
        }
        if let Some(off) = offset {
            builder = builder.offset(off);
        }

        let response = self
            .inner
            .scroll(builder)
            .await
            .with_context(|| format!("Qdrant scroll failed on collection={collection}"))?;

        let points = response
            .result
            .into_iter()
            .map(|p| QdrantRetrievedPoint {
                id: extract_id_from_option(p.id.as_ref()),
                payload: decode_payload(p.payload),
            })
            .collect();

        Ok((points, response.next_page_offset))
    }

    /// Retrieve specific points by their IDs.
    ///
    /// Equivalent to `qdrantClient.retrieve(collection, { ids, with_payload: true })`.
    pub async fn retrieve(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> Result<Vec<QdrantRetrievedPoint>> {
        debug!(collection, count = ids.len(), "QdrantReadClient::retrieve");

        if ids.is_empty() {
            return Ok(vec![]);
        }

        let point_ids: Vec<PointId> = ids.iter().map(|s| PointId::from(s.as_str())).collect();

        let builder = GetPointsBuilder::new(collection, point_ids).with_payload(true);

        let response = self
            .inner
            .get_points(builder)
            .await
            .with_context(|| format!("Qdrant retrieve failed on collection={collection}"))?;

        Ok(response
            .result
            .into_iter()
            .map(|p| QdrantRetrievedPoint {
                id: extract_id_from_option(p.id.as_ref()),
                payload: decode_payload(p.payload),
            })
            .collect())
    }

    /// Check whether a named collection exists.
    pub async fn collection_exists(&self, collection: &str) -> Result<bool> {
        debug!(collection, "QdrantReadClient::collection_exists");

        let exists = self
            .inner
            .collection_exists(collection)
            .await
            .with_context(|| {
                format!("Qdrant collection_exists check failed for collection={collection}")
            })?;

        Ok(exists)
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Convert a Qdrant `ScoredPoint` proto to our `QdrantPoint` value type.
fn scored_point_to_qdrant_point(sp: ScoredPoint) -> QdrantPoint {
    QdrantPoint {
        id: extract_id_from_option(sp.id.as_ref()),
        score: sp.score as f64,
        payload: decode_payload(sp.payload),
    }
}

/// Extract a string ID from an `Option<PointId>` reference.
pub(crate) fn extract_id_from_option(id: Option<&PointId>) -> String {
    match id.and_then(|pid| pid.point_id_options.as_ref()) {
        Some(PointIdOptions::Uuid(uuid)) => uuid.clone(),
        Some(PointIdOptions::Num(num)) => num.to_string(),
        None => String::new(),
    }
}

/// Decode a Qdrant payload map into a `HashMap<String, serde_json::Value>`.
pub(crate) fn decode_payload(
    payload: HashMap<String, qdrant_client::qdrant::Value>,
) -> HashMap<String, Value> {
    payload
        .into_iter()
        .map(|(k, v)| (k, qdrant_value_to_json(v)))
        .collect()
}

/// Convert a single `qdrant_client::qdrant::Value` to `serde_json::Value`.
fn qdrant_value_to_json(v: qdrant_client::qdrant::Value) -> Value {
    use qdrant_client::qdrant::value::Kind;
    match v.kind {
        Some(Kind::NullValue(_)) | None => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(b),
        Some(Kind::IntegerValue(i)) => Value::Number(i.into()),
        Some(Kind::DoubleValue(d)) => serde_json::Number::from_f64(d)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Some(Kind::StringValue(s)) => Value::String(s),
        Some(Kind::ListValue(list)) => {
            Value::Array(list.values.into_iter().map(qdrant_value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => {
            let map = s
                .fields
                .into_iter()
                .map(|(k, v)| (k, qdrant_value_to_json(v)))
                .collect();
            Value::Object(map)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::{point_id::PointIdOptions, PointId, Value as QValue};

    // ---- extract_id_from_option ----

    #[test]
    fn extract_id_uuid_returns_string() {
        let pid = PointId {
            point_id_options: Some(PointIdOptions::Uuid("abc-123".to_string())),
        };
        assert_eq!(extract_id_from_option(Some(&pid)), "abc-123");
    }

    #[test]
    fn extract_id_num_returns_decimal_string() {
        let pid = PointId {
            point_id_options: Some(PointIdOptions::Num(42)),
        };
        assert_eq!(extract_id_from_option(Some(&pid)), "42");
    }

    #[test]
    fn extract_id_none_returns_empty() {
        assert_eq!(extract_id_from_option(None), "");
    }

    #[test]
    fn extract_id_no_options_returns_empty() {
        let pid = PointId {
            point_id_options: None,
        };
        assert_eq!(extract_id_from_option(Some(&pid)), "");
    }

    // ---- qdrant_value_to_json ----

    fn wrap(kind: qdrant_client::qdrant::value::Kind) -> QValue {
        QValue { kind: Some(kind) }
    }

    #[test]
    fn qdrant_null_maps_to_json_null() {
        let v = QValue { kind: None };
        assert_eq!(qdrant_value_to_json(v), Value::Null);
    }

    #[test]
    fn qdrant_bool_maps_to_json_bool() {
        use qdrant_client::qdrant::value::Kind;
        assert_eq!(
            qdrant_value_to_json(wrap(Kind::BoolValue(true))),
            Value::Bool(true)
        );
        assert_eq!(
            qdrant_value_to_json(wrap(Kind::BoolValue(false))),
            Value::Bool(false)
        );
    }

    #[test]
    fn qdrant_integer_maps_to_json_number() {
        use qdrant_client::qdrant::value::Kind;
        let v = qdrant_value_to_json(wrap(Kind::IntegerValue(99)));
        assert_eq!(v, Value::Number(99.into()));
    }

    #[test]
    fn qdrant_string_maps_to_json_string() {
        use qdrant_client::qdrant::value::Kind;
        let v = qdrant_value_to_json(wrap(Kind::StringValue("hello".to_string())));
        assert_eq!(v, Value::String("hello".to_string()));
    }

    #[test]
    fn qdrant_double_maps_to_json_number() {
        use qdrant_client::qdrant::value::Kind;
        let v = qdrant_value_to_json(wrap(Kind::DoubleValue(3.14)));
        assert!(matches!(v, Value::Number(_)));
    }

    #[test]
    fn qdrant_list_maps_to_json_array() {
        use qdrant_client::qdrant::{value::Kind, ListValue};
        let list = ListValue {
            values: vec![wrap(Kind::StringValue("a".to_string()))],
        };
        let v = qdrant_value_to_json(wrap(Kind::ListValue(list)));
        assert!(matches!(v, Value::Array(_)));
    }

    #[test]
    fn qdrant_struct_maps_to_json_object() {
        use qdrant_client::qdrant::{value::Kind, Struct};
        let mut fields = std::collections::HashMap::new();
        fields.insert("k".to_string(), wrap(Kind::StringValue("v".to_string())));
        let s = Struct { fields };
        let v = qdrant_value_to_json(wrap(Kind::StructValue(s)));
        assert!(matches!(v, Value::Object(_)));
    }

    // ---- decode_payload ----

    #[test]
    fn decode_payload_converts_all_fields() {
        use qdrant_client::qdrant::value::Kind;
        let mut raw = HashMap::new();
        raw.insert(
            "name".to_string(),
            wrap(Kind::StringValue("test".to_string())),
        );
        raw.insert("count".to_string(), wrap(Kind::IntegerValue(7)));
        let decoded = decode_payload(raw);
        assert_eq!(
            decoded.get("name"),
            Some(&Value::String("test".to_string()))
        );
        assert_eq!(decoded.get("count"), Some(&Value::Number(7.into())));
    }

    // ---- QdrantReadClient::new (smoke test — does not dial network) ----

    #[test]
    fn client_new_does_not_panic_without_api_key() {
        // The constructor creates the Qdrant handle synchronously without
        // establishing a connection (connection is lazy).
        let _client = QdrantReadClient::new("http://localhost:6333".to_string(), None);
    }

    #[test]
    fn client_new_does_not_panic_with_api_key() {
        let key = SecretString::new("my-secret-key".into());
        let _client = QdrantReadClient::new("http://localhost:6333".to_string(), Some(key));
    }

    #[test]
    fn client_debug_does_not_expose_url_or_key() {
        let key = SecretString::new("super-secret".into());
        let client = QdrantReadClient::new("http://localhost:6333".to_string(), Some(key));
        let debug_str = format!("{:?}", client);
        assert!(
            !debug_str.contains("super-secret"),
            "debug output must not contain the api key"
        );
    }
}
