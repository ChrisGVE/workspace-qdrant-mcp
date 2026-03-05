//! Collections list subcommand handler

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use tabled::Tabled;

use crate::commands::qdrant_helpers;
use crate::output;

use super::{build_client, qdrant_url, VALID_COLLECTIONS};

/// Qdrant collection info from list endpoint
#[derive(Debug, Deserialize, Serialize)]
struct CollectionDescription {
    name: String,
}

/// Qdrant list collections response
#[derive(Debug, Deserialize)]
struct CollectionsListResult {
    collections: Vec<CollectionDescription>,
}

/// Qdrant API response wrapper
#[derive(Debug, Deserialize)]
struct QdrantResponse<T> {
    #[allow(dead_code)]
    status: Option<String>,
    result: T,
}

#[derive(Tabled)]
struct CollectionRow {
    #[tabled(rename = "Name")]
    name: String,
    #[tabled(rename = "Type")]
    ctype: String,
    #[tabled(rename = "Points")]
    points: String,
    #[tabled(rename = "Tenants")]
    tenants: String,
    #[tabled(rename = "Orphans")]
    orphans: String,
}

pub async fn list_collections(json: bool, script: bool, no_headers: bool) -> Result<()> {
    if !json && !script {
        output::section("Qdrant Collections");
    }

    let client = build_client()?;
    let url = format!("{}/collections", qdrant_url());

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body: QdrantResponse<CollectionsListResult> = resp
                .json()
                .await
                .context("Failed to parse Qdrant response")?;

            if json {
                output::print_json(&body.result.collections);
                return Ok(());
            }

            if body.result.collections.is_empty() {
                if !script {
                    output::info("No collections found");
                }
                return Ok(());
            }

            let rows = build_collection_rows(&body.result.collections).await?;
            print_collection_results(&rows, &body.result.collections, script, no_headers);
        }
        Ok(resp) => {
            output::error(format!(
                "Qdrant returned status {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            ));
        }
        Err(e) => {
            output::error(format!(
                "Failed to connect to Qdrant at {}: {}",
                qdrant_url(),
                e
            ));
        }
    }

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async fn build_collection_rows(
    collections: &[CollectionDescription],
) -> Result<Vec<CollectionRow>> {
    let db_conn = qdrant_helpers::open_state_db().ok();
    let qdrant_client = qdrant_helpers::build_qdrant_http_client()?;
    let base_url = qdrant_helpers::qdrant_base_url();

    let mut rows: Vec<CollectionRow> = Vec::new();

    for col in collections {
        let row =
            build_single_collection_row(col, &qdrant_client, &base_url, db_conn.as_ref()).await?;
        rows.push(row);
    }

    Ok(rows)
}

async fn build_single_collection_row(
    col: &CollectionDescription,
    qdrant_client: &reqwest::Client,
    base_url: &str,
    db_conn: Option<&rusqlite::Connection>,
) -> Result<CollectionRow> {
    let is_canonical = VALID_COLLECTIONS.contains(&col.name.as_str());
    let label = if is_canonical { "canonical" } else { "custom" };

    let point_count =
        qdrant_helpers::get_collection_point_count(qdrant_client, base_url, &col.name)
            .await
            .unwrap_or(None);

    let points_str = point_count
        .map(|c| c.to_string())
        .unwrap_or_else(|| "?".to_string());

    if is_canonical {
        build_canonical_row(col, qdrant_client, base_url, db_conn, label, points_str).await
    } else {
        Ok(CollectionRow {
            name: col.name.clone(),
            ctype: label.to_string(),
            points: points_str,
            tenants: "-".to_string(),
            orphans: "-".to_string(),
        })
    }
}

async fn build_canonical_row(
    col: &CollectionDescription,
    qdrant_client: &reqwest::Client,
    base_url: &str,
    db_conn: Option<&rusqlite::Connection>,
    label: &str,
    points_str: String,
) -> Result<CollectionRow> {
    let tenant_field = qdrant_helpers::tenant_field_for_collection(&col.name);
    let qdrant_tenants = qdrant_helpers::scroll_unique_field_values(
        qdrant_client,
        base_url,
        &col.name,
        tenant_field,
    )
    .await
    .unwrap_or_default();

    let known_tenants = db_conn
        .map(|c| qdrant_helpers::get_known_tenants_for_collection(c, &col.name))
        .transpose()?
        .unwrap_or_default();

    let orphan_count = qdrant_tenants
        .iter()
        .filter(|t| !known_tenants.contains(*t))
        .count();

    Ok(CollectionRow {
        name: col.name.clone(),
        ctype: label.to_string(),
        points: points_str,
        tenants: qdrant_tenants.len().to_string(),
        orphans: orphan_count.to_string(),
    })
}

fn print_collection_results(
    rows: &[CollectionRow],
    collections: &[CollectionDescription],
    script: bool,
    no_headers: bool,
) {
    if script {
        output::print_script(rows, !no_headers);
        return;
    }

    for row in rows {
        let orphan_str = row
            .orphans
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .map(|n| format!(", {} orphan(s)", n))
            .unwrap_or_default();

        if row.tenants == "-" {
            output::kv(&row.name, format!("{} | {} points", row.ctype, row.points));
        } else {
            output::kv(
                &row.name,
                format!(
                    "{} | {} points | {} tenant(s){}",
                    row.ctype, row.points, row.tenants, orphan_str
                ),
            );
        }
    }

    output::separator();
    output::info(format!("Total: {} collections", collections.len()));
}
