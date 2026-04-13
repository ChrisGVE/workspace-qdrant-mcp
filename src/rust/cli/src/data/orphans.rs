//! Shared orphan detection for projects and libraries.
//!
//! Orphans are tenants that exist in Qdrant but are not registered
//! in the SQLite watch_folders table. This module provides a single
//! implementation used by project list, library list, collections list,
//! and admin cleanup commands.

use std::collections::HashSet;

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::commands::qdrant_helpers;

/// An orphaned tenant found in Qdrant but not in SQLite.
#[derive(Debug, Clone)]
pub struct OrphanInfo {
    pub tenant_id: String,
    pub collection: String,
    pub document_count: usize,
}

/// Detect orphaned tenants for a specific collection.
///
/// Compares tenant IDs in Qdrant against registered watch_folders in SQLite.
/// Returns tenants that exist in Qdrant but not in the database.
pub async fn detect_orphans(conn: &Connection, collection: &str) -> Result<Vec<OrphanInfo>> {
    let tenant_field = qdrant_helpers::tenant_field_for_collection(collection);

    // Get known tenants from SQLite
    let known = qdrant_helpers::get_known_tenants_for_collection(conn, collection)
        .context("Failed to get known tenants from database")?;

    // Get tenants from Qdrant
    let client =
        qdrant_helpers::build_qdrant_http_client().context("Failed to build Qdrant client")?;
    let base_url = qdrant_helpers::qdrant_base_url();

    let qdrant_counts =
        qdrant_helpers::scroll_tenant_point_counts(&client, &base_url, collection, tenant_field)
            .await
            .unwrap_or_default();

    // Find orphans: in Qdrant but not in SQLite
    let mut orphans: Vec<OrphanInfo> = qdrant_counts
        .into_iter()
        .filter(|(id, _)| !known.contains(id))
        .map(|(tenant_id, count)| OrphanInfo {
            tenant_id,
            collection: collection.to_string(),
            document_count: count,
        })
        .collect();

    orphans.sort_by(|a, b| a.tenant_id.cmp(&b.tenant_id));
    Ok(orphans)
}

/// Get just the orphan tenant IDs for a collection (lighter than full detect).
pub async fn get_orphan_ids(conn: &Connection, collection: &str) -> Result<HashSet<String>> {
    let tenant_field = qdrant_helpers::tenant_field_for_collection(collection);
    let known =
        qdrant_helpers::get_known_tenants_for_collection(conn, collection).unwrap_or_default();

    let client = match qdrant_helpers::build_qdrant_http_client() {
        Ok(c) => c,
        Err(_) => return Ok(HashSet::new()),
    };
    let base_url = qdrant_helpers::qdrant_base_url();

    let qdrant_ids =
        qdrant_helpers::scroll_unique_field_values(&client, &base_url, collection, tenant_field)
            .await
            .unwrap_or_default();

    Ok(qdrant_ids
        .into_iter()
        .filter(|id| !known.contains(id))
        .collect())
}
