//! Stats subcommand handler: `tags stats` — tag extraction statistics per tenant.

use anyhow::Result;
use rusqlite::Connection;
use tabled::Tabled;

use super::db::{open_db, table_exists};
use crate::output;

#[derive(Tabled, serde::Serialize)]
pub(super) struct StatsRow {
    #[tabled(rename = "Tenant")]
    pub tenant_id: String,
    #[tabled(rename = "Documents")]
    pub doc_count: i64,
    #[tabled(rename = "Avg Keywords")]
    pub avg_keywords: String,
    #[tabled(rename = "Avg Tags")]
    pub avg_tags: String,
    #[tabled(rename = "Canonical")]
    pub canonical_count: i64,
}

pub(super) fn show_stats(tenant_id: Option<&str>, collection: &str) -> Result<()> {
    let conn = open_db()?;
    if !table_exists(&conn, "tags") {
        anyhow::bail!("Tags table not found. Ensure daemon schema v16+ is applied.");
    }

    let has_canonical = table_exists(&conn, "canonical_tags");

    let rows: Vec<StatsRow> = if let Some(tid) = tenant_id {
        vec![compute_stats_for_tenant(
            &conn,
            tid,
            collection,
            has_canonical,
        )?]
    } else {
        let mut stmt = conn.prepare(
            "SELECT DISTINCT tenant_id FROM tags WHERE collection = ? ORDER BY tenant_id",
        )?;
        let tenants: Vec<String> = stmt
            .query_map([collection], |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        tenants
            .iter()
            .filter_map(|tid| compute_stats_for_tenant(&conn, tid, collection, has_canonical).ok())
            .collect()
    };

    if rows.is_empty() {
        output::info("No tag data found. Ingest documents to generate keywords and tags.");
        return Ok(());
    }

    output::print_table(&rows);
    Ok(())
}

fn compute_stats_for_tenant(
    conn: &Connection,
    tenant_id: &str,
    collection: &str,
    has_canonical: bool,
) -> Result<StatsRow> {
    let doc_count: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT doc_id) FROM keywords WHERE tenant_id = ? AND collection = ?",
            rusqlite::params![tenant_id, collection],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let avg_kw: f64 = if doc_count > 0 {
        let total_kw: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM keywords WHERE tenant_id = ? AND collection = ?",
                rusqlite::params![tenant_id, collection],
                |row| row.get(0),
            )
            .unwrap_or(0);
        total_kw as f64 / doc_count as f64
    } else {
        0.0
    };

    let avg_tags: f64 = if doc_count > 0 {
        let total_tags: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM tags WHERE tenant_id = ? AND collection = ? AND tag_type = 'concept'",
                rusqlite::params![tenant_id, collection],
                |row| row.get(0),
            )
            .unwrap_or(0);
        total_tags as f64 / doc_count as f64
    } else {
        0.0
    };

    let canonical_count: i64 = if has_canonical {
        conn.query_row(
            "SELECT COUNT(*) FROM canonical_tags WHERE tenant_id = ? AND collection = ?",
            rusqlite::params![tenant_id, collection],
            |row| row.get(0),
        )
        .unwrap_or(0)
    } else {
        0
    };

    Ok(StatsRow {
        tenant_id: tenant_id.to_string(),
        doc_count,
        avg_keywords: format!("{:.1}", avg_kw),
        avg_tags: format!("{:.1}", avg_tags),
        canonical_count,
    })
}
