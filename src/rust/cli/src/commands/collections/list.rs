//! Collections list subcommand handler

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use tabled::Tabled;

use crate::output;
use crate::commands::qdrant_helpers;

use super::{VALID_COLLECTIONS, build_client, qdrant_url};

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

            // Open SQLite for orphan detection (non-fatal)
            let db_conn = qdrant_helpers::open_state_db().ok();
            let qdrant_client = qdrant_helpers::build_qdrant_http_client()?;
            let base_url = qdrant_helpers::qdrant_base_url();

            let mut script_rows: Vec<CollectionRow> = Vec::new();

            for col in &body.result.collections {
                let is_canonical = VALID_COLLECTIONS.contains(&col.name.as_str());
                let label = if is_canonical { "canonical" } else { "custom" };

                // Get point count
                let point_count = qdrant_helpers::get_collection_point_count(
                    &qdrant_client,
                    &base_url,
                    &col.name,
                )
                .await
                .unwrap_or(None);

                let points_str = point_count
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "?".to_string());

                // For canonical collections, compute tenant and orphan counts
                if is_canonical {
                    let tenant_field =
                        qdrant_helpers::tenant_field_for_collection(&col.name);
                    let qdrant_tenants = qdrant_helpers::scroll_unique_field_values(
                        &qdrant_client,
                        &base_url,
                        &col.name,
                        tenant_field,
                    )
                    .await
                    .unwrap_or_default();

                    let known_tenants = db_conn
                        .as_ref()
                        .map(|c| {
                            qdrant_helpers::get_known_tenants_for_collection(c, &col.name)
                        })
                        .transpose()?
                        .unwrap_or_default();

                    let orphan_count = qdrant_tenants
                        .iter()
                        .filter(|t| !known_tenants.contains(*t))
                        .count();

                    if script {
                        script_rows.push(CollectionRow {
                            name: col.name.clone(),
                            ctype: label.to_string(),
                            points: points_str.clone(),
                            tenants: qdrant_tenants.len().to_string(),
                            orphans: orphan_count.to_string(),
                        });
                    } else {
                        let orphan_str = if orphan_count > 0 {
                            format!(", {} orphan(s)", orphan_count)
                        } else {
                            String::new()
                        };

                        output::kv(
                            &col.name,
                            &format!(
                                "{} | {} points | {} tenant(s){}",
                                label,
                                points_str,
                                qdrant_tenants.len(),
                                orphan_str
                            ),
                        );
                    }
                } else if script {
                    script_rows.push(CollectionRow {
                        name: col.name.clone(),
                        ctype: label.to_string(),
                        points: points_str.clone(),
                        tenants: "-".to_string(),
                        orphans: "-".to_string(),
                    });
                } else {
                    output::kv(&col.name, &format!("{} | {} points", label, points_str));
                }
            }

            if script {
                output::print_script(&script_rows, !no_headers);
            } else {
                output::separator();
                output::info(format!(
                    "Total: {} collections",
                    body.result.collections.len()
                ));
            }
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
