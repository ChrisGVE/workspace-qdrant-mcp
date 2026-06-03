//! Collections list subcommand handler.
//!
//! Table template per cli-feedback.md.

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use tabled::Tabled;

use crate::commands::qdrant_helpers;
use crate::output::canvas;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};
use crate::output::render::{render_table, GutterRow};
use crate::output::table::ColumnHints;
use crate::output::{self};

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

/// Table row for collection display.
#[derive(Clone, Tabled, Serialize)]
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

impl ColumnHints for CollectionRow {
    fn content_columns() -> &'static [usize] {
        &[0] // Name is content column
    }

    fn numeric_columns() -> &'static [usize] {
        &[2, 3, 4] // Points, Tenants, Orphans
    }
}

pub async fn list_collections(json: bool, script: bool, no_headers: bool) -> Result<()> {
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
                    canvas::print_title("Qdrant Collections");
                    canvas::print_blank();
                    output::info("No collections found. Register a project to get started.");
                }
                return Ok(());
            }

            let rows = build_collection_rows(&body.result.collections).await?;

            if script {
                output::print_script(&rows, !no_headers);
            } else {
                canvas::print_title("Qdrant Collections");
                canvas::print_blank();
                print_collection_table(&rows);
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

// ─── Helpers ─────────────────────────────────────────────────────────────────

async fn build_collection_rows(
    collections: &[CollectionDescription],
) -> Result<Vec<CollectionRow>> {
    let db_conn = crate::data::db::connect_readonly().ok();
    let reader = qdrant_helpers::QdrantReader::from_config()?;
    let locale = NumberLocale::default();

    let mut rows: Vec<CollectionRow> = Vec::new();

    for col in collections {
        let row = build_single_collection_row(col, &reader, db_conn.as_ref(), &locale).await?;
        rows.push(row);
    }

    Ok(rows)
}

async fn build_single_collection_row(
    col: &CollectionDescription,
    reader: &qdrant_helpers::QdrantReader,
    db_conn: Option<&rusqlite::Connection>,
    locale: &NumberLocale,
) -> Result<CollectionRow> {
    let is_canonical = VALID_COLLECTIONS.contains(&col.name.as_str());
    let label = if is_canonical { "canonical" } else { "custom" };

    let point_count = reader
        .collection_point_count(&col.name)
        .await
        .unwrap_or(None);

    let points_str = point_count
        .map(|c| format_usize(c as usize, locale))
        .unwrap_or_else(|| "?".to_string());

    if is_canonical {
        build_canonical_row(col, reader, db_conn, label, points_str, locale).await
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
    reader: &qdrant_helpers::QdrantReader,
    db_conn: Option<&rusqlite::Connection>,
    label: &str,
    points_str: String,
    locale: &NumberLocale,
) -> Result<CollectionRow> {
    let tenant_field = qdrant_helpers::tenant_field_for_collection(&col.name);
    let qdrant_tenants = reader
        .unique_field_values(&col.name, tenant_field)
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
        tenants: format_usize(qdrant_tenants.len(), locale),
        orphans: if orphan_count > 0 {
            format_usize(orphan_count, locale)
        } else {
            "0".to_string()
        },
    })
}

/// Render collections as a table with gutter indicators.
fn print_collection_table(rows: &[CollectionRow]) {
    let gutter_rows: Vec<GutterRow<CollectionRow>> = rows
        .iter()
        .map(|r| {
            let has_points = r
                .points
                .replace('\'', "")
                .parse::<u64>()
                .ok()
                .is_some_and(|n| n > 0);
            GutterRow {
                gutter: if has_points {
                    Gutter::Sync
                } else {
                    Gutter::None
                },
                data: r.clone(),
            }
        })
        .collect();

    let summary = format!("{} collections", rows.len());
    render_table(&gutter_rows, Some(&summary));
}

#[cfg(test)]
mod tests {
    use crate::output::number::NumberLocale;

    use super::format_usize;

    #[test]
    fn format_usize_small() {
        let locale = NumberLocale::default();
        assert_eq!(format_usize(0, &locale), "0");
        assert_eq!(format_usize(42, &locale), "42");
        assert_eq!(format_usize(999, &locale), "999");
    }

    #[test]
    fn format_usize_thousands() {
        let locale = NumberLocale::default();
        assert_eq!(format_usize(1000, &locale), "1'000");
        assert_eq!(format_usize(104100, &locale), "104'100");
        assert_eq!(format_usize(1234567, &locale), "1'234'567");
    }
}
