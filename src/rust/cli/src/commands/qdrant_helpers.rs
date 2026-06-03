//! Qdrant read helpers for orphan detection and tenant introspection.
//!
//! Used by admin, library, project, and collections commands.
//!
//! Reads go through [`QdrantReader`], which prefers the shared gRPC
//! [`wqm_client::QdrantReadClient`] (single read client, WI-d3 #82) and falls
//! back to the local REST path if the gRPC call fails. The REST primitives
//! ([`build_qdrant_http_client`], [`qdrant_base_url`]) remain public because a
//! couple of ad-hoc health probes (`GET /collections`) still use them directly.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use secrecy::SecretString;
use wqm_client::qdrant::PointId;
use wqm_client::QdrantReadClient;
use wqm_common::constants::COLLECTION_LIBRARIES;

/// Get Qdrant REST base URL. Priority: `QDRANT_URL` env > active cli-config
/// profile > workspace default (`http://localhost:6333`).
pub fn qdrant_base_url() -> String {
    crate::config::resolve_qdrant_url()
}

/// Build an HTTP client for Qdrant REST API with optional API key.
pub fn build_qdrant_http_client() -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(30));

    if let Some(api_key) = crate::config::resolve_qdrant_api_key() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "api-key",
            reqwest::header::HeaderValue::from_str(&api_key)
                .context("Invalid QDRANT_API_KEY header value")?,
        );
        builder = builder.default_headers(headers);
    }

    builder.build().context("Failed to build HTTP client")
}

/// Get the Qdrant payload field name used as tenant key for a collection.
pub fn tenant_field_for_collection(collection: &str) -> &'static str {
    if collection == COLLECTION_LIBRARIES {
        "library_name"
    } else {
        "tenant_id"
    }
}

/// Get known tenant_ids (or library_names) from SQLite for a specific collection.
pub fn get_known_tenants_for_collection(
    conn: &rusqlite::Connection,
    collection: &str,
) -> Result<HashSet<String>> {
    let mut known = HashSet::new();

    // From watch_folders
    let mut stmt = conn
        .prepare("SELECT DISTINCT tenant_id FROM watch_folders WHERE collection = ?1")
        .context("Failed to query watch_folders")?;

    let rows: Vec<String> = stmt
        .query_map(rusqlite::params![collection], |row| row.get::<_, String>(0))
        .context("Failed to read watch_folders rows")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to parse watch_folders rows")?;
    known.extend(rows);

    // From tracked_files (table may not exist yet)
    if let Ok(mut stmt2) =
        conn.prepare("SELECT DISTINCT tenant_id FROM tracked_files WHERE collection = ?1")
    {
        let rows2: Vec<String> = stmt2
            .query_map(rusqlite::params![collection], |row| row.get::<_, String>(0))
            .context("Failed to read tracked_files rows")?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to parse tracked_files rows")?;
        known.extend(rows2);
    }

    Ok(known)
}

const SCROLL_BATCH: u32 = 100;

/// Qdrant reader: gRPC-primary, REST-fallback.
///
/// Each aggregation tries the shared [`QdrantReadClient`] (gRPC :6334) first and
/// falls back to the local REST path (:6333) if the gRPC call errors. This keeps
/// a single canonical read client while preserving resilience if only one of the
/// two Qdrant ports is reachable.
pub struct QdrantReader {
    grpc: QdrantReadClient,
    rest: reqwest::Client,
    base_url: String,
}

impl QdrantReader {
    /// Build a reader from the resolved Qdrant URL + API key.
    pub fn from_config() -> Result<Self> {
        let base_url = qdrant_base_url();
        let api_key = crate::config::resolve_qdrant_api_key().map(|k| SecretString::new(k.into()));
        let grpc = QdrantReadClient::new(base_url.clone(), api_key);
        let rest = build_qdrant_http_client()?;
        Ok(Self {
            grpc,
            rest,
            base_url,
        })
    }

    /// The resolved REST base URL (for display).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Unique values of `field` across all points of `collection`.
    pub async fn unique_field_values(
        &self,
        collection: &str,
        field: &str,
    ) -> Result<HashSet<String>> {
        match self.grpc_unique_field_values(collection, field).await {
            Ok(v) => Ok(v),
            Err(e) => {
                tracing::debug!(error = %e, collection, "gRPC scroll failed; falling back to REST");
                rest_scroll_unique_field_values(&self.rest, &self.base_url, collection, field).await
            }
        }
    }

    /// Point count per unique value of `field` across `collection`.
    pub async fn tenant_point_counts(
        &self,
        collection: &str,
        field: &str,
    ) -> Result<HashMap<String, usize>> {
        match self.grpc_tenant_point_counts(collection, field).await {
            Ok(v) => Ok(v),
            Err(e) => {
                tracing::debug!(error = %e, collection, "gRPC scroll failed; falling back to REST");
                rest_scroll_tenant_point_counts(&self.rest, &self.base_url, collection, field).await
            }
        }
    }

    /// Total point count for `collection` (None if the collection is absent).
    pub async fn collection_point_count(&self, collection: &str) -> Result<Option<u64>> {
        match self.grpc.count(collection).await {
            Ok(n) => Ok(Some(n)),
            Err(e) => {
                tracing::debug!(error = %e, collection, "gRPC count failed; falling back to REST");
                rest_get_collection_point_count(&self.rest, &self.base_url, collection).await
            }
        }
    }

    /// All points of `collection` as `{"id", "payload"}` JSON (recover-state).
    pub async fn all_points(&self, collection: &str) -> Result<Vec<serde_json::Value>> {
        match self.grpc_all_points(collection).await {
            Ok(v) => Ok(v),
            Err(e) => {
                tracing::debug!(error = %e, collection, "gRPC scroll failed; falling back to REST");
                rest_scroll_all_points(&self.rest, &self.base_url, collection).await
            }
        }
    }

    // ── gRPC implementations ────────────────────────────────────────────────

    async fn grpc_unique_field_values(
        &self,
        collection: &str,
        field: &str,
    ) -> Result<HashSet<String>> {
        let mut values = HashSet::new();
        let mut offset: Option<PointId> = None;
        loop {
            let (points, next) = self
                .grpc
                .scroll(collection, None, SCROLL_BATCH, offset)
                .await?;
            if points.is_empty() {
                break;
            }
            for p in &points {
                if let Some(v) = p.payload.get(field).and_then(|v| v.as_str()) {
                    values.insert(v.to_string());
                }
            }
            match next {
                Some(o) => offset = Some(o),
                None => break,
            }
        }
        Ok(values)
    }

    async fn grpc_tenant_point_counts(
        &self,
        collection: &str,
        field: &str,
    ) -> Result<HashMap<String, usize>> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut offset: Option<PointId> = None;
        loop {
            let (points, next) = self
                .grpc
                .scroll(collection, None, SCROLL_BATCH, offset)
                .await?;
            if points.is_empty() {
                break;
            }
            for p in &points {
                if let Some(v) = p.payload.get(field).and_then(|v| v.as_str()) {
                    *counts.entry(v.to_string()).or_insert(0) += 1;
                }
            }
            match next {
                Some(o) => offset = Some(o),
                None => break,
            }
        }
        Ok(counts)
    }

    async fn grpc_all_points(&self, collection: &str) -> Result<Vec<serde_json::Value>> {
        let mut out = Vec::new();
        let mut offset: Option<PointId> = None;
        loop {
            let (points, next) = self
                .grpc
                .scroll(collection, None, SCROLL_BATCH, offset)
                .await?;
            if points.is_empty() {
                break;
            }
            for p in points {
                // Reproduce the REST point shape consumed by recover-state's
                // reconstruction (`point["payload"][field]`).
                let payload = serde_json::Value::Object(p.payload.into_iter().collect());
                out.push(serde_json::json!({ "id": p.id, "payload": payload }));
            }
            match next {
                Some(o) => offset = Some(o),
                None => break,
            }
        }
        Ok(out)
    }
}

// ── REST fallback implementations ───────────────────────────────────────────

/// Scroll a Qdrant collection (REST) and collect unique values of a payload field.
async fn rest_scroll_unique_field_values(
    client: &reqwest::Client,
    base_url: &str,
    collection: &str,
    field: &str,
) -> Result<HashSet<String>> {
    let scroll_url = format!("{}/collections/{}/points/scroll", base_url, collection);
    let mut all_values = HashSet::new();
    let mut offset: Option<serde_json::Value> = None;

    loop {
        let mut body = serde_json::json!({
            "limit": SCROLL_BATCH,
            "with_payload": { "include": [field] },
            "with_vector": false
        });

        if let Some(ref off) = offset {
            body["offset"] = off.clone();
        }

        let resp = client
            .post(&scroll_url)
            .json(&body)
            .send()
            .await
            .context(format!("Failed to scroll Qdrant {} collection", collection))?;

        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 404 {
                return Ok(all_values);
            }
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Qdrant scroll failed for {} ({}): {}",
                collection,
                status,
                text
            );
        }

        let resp_json: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse Qdrant scroll response")?;

        let points = resp_json["result"]["points"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        if points.is_empty() {
            break;
        }

        for point in &points {
            if let Some(val) = point["payload"][field].as_str() {
                all_values.insert(val.to_string());
            }
        }

        let next_offset = &resp_json["result"]["next_page_offset"];
        if next_offset.is_null() {
            break;
        }
        offset = Some(next_offset.clone());
    }

    Ok(all_values)
}

/// Scroll a Qdrant collection (REST) and count points per unique tenant value.
async fn rest_scroll_tenant_point_counts(
    client: &reqwest::Client,
    base_url: &str,
    collection: &str,
    field: &str,
) -> Result<HashMap<String, usize>> {
    let scroll_url = format!("{}/collections/{}/points/scroll", base_url, collection);
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut offset: Option<serde_json::Value> = None;

    loop {
        let mut body = serde_json::json!({
            "limit": SCROLL_BATCH,
            "with_payload": { "include": [field] },
            "with_vector": false
        });

        if let Some(ref off) = offset {
            body["offset"] = off.clone();
        }

        let resp = client
            .post(&scroll_url)
            .json(&body)
            .send()
            .await
            .context(format!("Failed to scroll Qdrant {} collection", collection))?;

        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 404 {
                return Ok(counts);
            }
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Qdrant scroll failed for {} ({}): {}",
                collection,
                status,
                text
            );
        }

        let resp_json: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse Qdrant scroll response")?;

        let points = resp_json["result"]["points"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        if points.is_empty() {
            break;
        }

        for point in &points {
            if let Some(val) = point["payload"][field].as_str() {
                *counts.entry(val.to_string()).or_insert(0) += 1;
            }
        }

        let next_offset = &resp_json["result"]["next_page_offset"];
        if next_offset.is_null() {
            break;
        }
        offset = Some(next_offset.clone());
    }

    Ok(counts)
}

/// Get total point count for a collection via Qdrant collection info API (REST).
async fn rest_get_collection_point_count(
    client: &reqwest::Client,
    base_url: &str,
    collection: &str,
) -> Result<Option<u64>> {
    let url = format!("{}/collections/{}", base_url, collection);
    let resp = client.get(&url).send().await;

    match resp {
        Ok(r) if r.status().is_success() => {
            let json: serde_json::Value = r.json().await.context("Failed to parse response")?;
            Ok(json["result"]["points_count"].as_u64())
        }
        Ok(r) if r.status().as_u16() == 404 => Ok(None),
        Ok(r) => {
            let text = r.text().await.unwrap_or_default();
            anyhow::bail!("Failed to get collection info for {}: {}", collection, text);
        }
        Err(e) => anyhow::bail!("Failed to connect to Qdrant: {}", e),
    }
}

/// Scroll a Qdrant collection (REST) and return all points with full payloads.
///
/// Used by recover-state to reconstruct SQLite from Qdrant data.
async fn rest_scroll_all_points(
    client: &reqwest::Client,
    base_url: &str,
    collection: &str,
) -> Result<Vec<serde_json::Value>> {
    let scroll_url = format!("{}/collections/{}/points/scroll", base_url, collection);
    let mut all_points = Vec::new();
    let mut offset: Option<serde_json::Value> = None;

    loop {
        let mut body = serde_json::json!({
            "limit": SCROLL_BATCH,
            "with_payload": true,
            "with_vector": false
        });

        if let Some(ref off) = offset {
            body["offset"] = off.clone();
        }

        let resp = client
            .post(&scroll_url)
            .json(&body)
            .send()
            .await
            .context(format!("Failed to scroll Qdrant {} collection", collection))?;

        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 404 {
                // Collection doesn't exist — return empty
                return Ok(all_points);
            }
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Qdrant scroll failed for {} ({}): {}",
                collection,
                status,
                text
            );
        }

        let resp_json: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse Qdrant scroll response")?;

        let points = resp_json["result"]["points"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        if points.is_empty() {
            break;
        }

        all_points.extend(points);

        let next_offset = &resp_json["result"]["next_page_offset"];
        if next_offset.is_null() {
            break;
        }
        offset = Some(next_offset.clone());
    }

    Ok(all_points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_field_for_collection() {
        assert_eq!(tenant_field_for_collection("libraries"), "library_name");
        assert_eq!(tenant_field_for_collection("projects"), "tenant_id");
        assert_eq!(tenant_field_for_collection("rules"), "tenant_id");
        assert_eq!(tenant_field_for_collection("scratchpad"), "tenant_id");
    }

    #[test]
    fn test_qdrant_base_url_default() {
        std::env::remove_var("QDRANT_URL");
        let url = qdrant_base_url();
        assert!(url.starts_with("http"));
        assert!(url.contains("6333"));
    }

    #[test]
    fn reader_from_config_builds() {
        // Construction must not panic and must not dial the network.
        let reader = QdrantReader::from_config();
        assert!(reader.is_ok());
    }
}
