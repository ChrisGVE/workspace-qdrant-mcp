//! Shared Qdrant HTTP helpers for orphan detection and tenant introspection.
//!
//! Used by admin, library, project, and collections commands.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use wqm_common::constants::COLLECTION_LIBRARIES;

/// Get Qdrant REST base URL from environment or default.
pub fn qdrant_base_url() -> String {
    std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| wqm_common::constants::DEFAULT_QDRANT_URL.to_string())
}

/// Build an HTTP client for Qdrant REST API with optional API key.
pub fn build_qdrant_http_client() -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(30));

    if let Ok(api_key) = std::env::var("QDRANT_API_KEY") {
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

/// Open the state database in read-only mode.
pub fn open_state_db() -> Result<rusqlite::Connection> {
    let db_path = crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }

    let conn = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context("Failed to open state database")?;

    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;

    Ok(conn)
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

    // From tracked_files
    match conn.prepare("SELECT DISTINCT tenant_id FROM tracked_files WHERE collection = ?1") {
        Ok(mut stmt2) => {
            let rows2: Vec<String> = stmt2
                .query_map(rusqlite::params![collection], |row| row.get::<_, String>(0))
                .context("Failed to read tracked_files rows")?
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to parse tracked_files rows")?;
            known.extend(rows2);
        }
        Err(_) => {} // Table may not exist yet
    }

    Ok(known)
}

/// Scroll a Qdrant collection and collect unique values of a payload field.
pub async fn scroll_unique_field_values(
    client: &reqwest::Client,
    base_url: &str,
    collection: &str,
    field: &str,
) -> Result<HashSet<String>> {
    let scroll_url = format!("{}/collections/{}/points/scroll", base_url, collection);
    let mut all_values = HashSet::new();
    let mut offset: Option<serde_json::Value> = None;
    let batch_size = 100;

    loop {
        let mut body = serde_json::json!({
            "limit": batch_size,
            "with_payload": {
                "include": [field]
            },
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
            .unwrap_or(&Vec::new())
            .clone();

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

/// Scroll a Qdrant collection and count points per unique tenant value.
pub async fn scroll_tenant_point_counts(
    client: &reqwest::Client,
    base_url: &str,
    collection: &str,
    field: &str,
) -> Result<HashMap<String, usize>> {
    let scroll_url = format!("{}/collections/{}/points/scroll", base_url, collection);
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut offset: Option<serde_json::Value> = None;
    let batch_size = 100;

    loop {
        let mut body = serde_json::json!({
            "limit": batch_size,
            "with_payload": {
                "include": [field]
            },
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
            .unwrap_or(&Vec::new())
            .clone();

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

/// Get total point count for a collection via Qdrant collection info API.
pub async fn get_collection_point_count(
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

/// Scroll a Qdrant collection and return all points with full payloads.
///
/// Used by recover-state to reconstruct SQLite from Qdrant data.
pub async fn scroll_all_points(
    client: &reqwest::Client,
    base_url: &str,
    collection: &str,
) -> Result<Vec<serde_json::Value>> {
    let scroll_url = format!("{}/collections/{}/points/scroll", base_url, collection);
    let mut all_points = Vec::new();
    let mut offset: Option<serde_json::Value> = None;
    let batch_size = 100;

    loop {
        let mut body = serde_json::json!({
            "limit": batch_size,
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
            .unwrap_or(&Vec::new())
            .clone();

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
}
