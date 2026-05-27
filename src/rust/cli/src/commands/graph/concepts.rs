//! `wqm graph concepts` — list concept nodes with relationship counts.
//!
//! Queries graph_nodes (symbol_type = 'concept_node') directly from SQLite
//! and counts IMPLEMENTS_CONCEPT and COVERS_TOPIC edges targeting each concept.

use anyhow::{Context, Result};
use serde::Serialize;
use tabled::Tabled;

use crate::output::canvas;
use crate::output::table::ColumnHints;
use crate::output::{self};

/// Row struct for the concepts table output.
#[derive(Tabled, Serialize, Clone)]
struct ConceptRow {
    #[tabled(rename = "Concept")]
    #[serde(rename = "concept")]
    concept: String,
    #[tabled(rename = "Implements")]
    #[serde(rename = "implements_count")]
    implements: String,
    #[tabled(rename = "Covers")]
    #[serde(rename = "covers_count")]
    covers: String,
}

impl ColumnHints for ConceptRow {
    fn content_columns() -> &'static [usize] {
        &[0] // Concept is content
    }

    fn numeric_columns() -> &'static [usize] {
        &[1, 2] // Implements, Covers
    }
}

/// Full JSON response including metadata.
#[derive(Serialize)]
struct ConceptsJson {
    tenant_id: String,
    total: usize,
    concepts: Vec<ConceptJsonEntry>,
}

#[derive(Serialize)]
struct ConceptJsonEntry {
    concept: String,
    implements_count: i64,
    covers_count: i64,
}

/// Open state.db in read-only mode.
fn open_state_db() -> Result<rusqlite::Connection> {
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

/// Query concept nodes with edge counts, filtered by tenant and optional params.
fn query_concepts(
    conn: &rusqlite::Connection,
    tenant_id: &str,
    concept_filter: Option<&str>,
    depth_filter: Option<&str>,
    limit: u32,
) -> Result<Vec<(String, i64, i64)>> {
    // Build WHERE clauses for optional filters
    let mut where_clauses = vec![
        "n.symbol_type = 'concept_node'".to_string(),
        "n.tenant_id = ?1 OR n.tenant_id = '__global__'".to_string(),
    ];
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(tenant_id.to_string())];

    if let Some(concept) = concept_filter {
        params.push(Box::new(format!("%{}%", concept)));
        where_clauses.push(format!("n.symbol_name LIKE ?{}", params.len()));
    }

    // depth_filter applies to COVERS_TOPIC edges' metadata_json
    let depth_condition = if let Some(depth) = depth_filter {
        format!(
            "AND e.metadata_json LIKE '%\"depth\":\"{}\"%%'",
            depth.replace('\'', "''")
        )
    } else {
        String::new()
    };

    let tenant_edge_param_idx = params.len() + 1;
    params.push(Box::new(tenant_id.to_string()));

    let sql = format!(
        "SELECT n.symbol_name AS concept,
                COUNT(DISTINCT CASE WHEN e.edge_type = 'IMPLEMENTS_CONCEPT' \
                    THEN e.source_node_id END) AS implements_count,
                COUNT(DISTINCT CASE WHEN e.edge_type = 'COVERS_TOPIC' {} \
                    THEN e.source_node_id END) AS covers_count
         FROM graph_nodes n
         LEFT JOIN graph_edges e ON e.target_node_id = n.node_id
              AND e.tenant_id = ?{tenant_edge_param_idx}
         WHERE {}
         GROUP BY n.node_id, n.symbol_name
         ORDER BY (implements_count + covers_count) DESC
         LIMIT ?{}",
        depth_condition,
        where_clauses.join(" AND "),
        params.len() + 1,
    );

    params.push(Box::new(limit));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn
        .prepare(&sql)
        .context("Failed to prepare concepts query")?;
    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .context("Failed to execute concepts query")?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row.context("Failed to read row")?);
    }

    Ok(results)
}

/// Execute the `wqm graph concepts` subcommand.
pub async fn concepts(
    tenant_id: &str,
    concept_filter: Option<&str>,
    depth_filter: Option<&str>,
    top: u32,
    json: bool,
) -> Result<()> {
    let conn = open_state_db()?;
    let results = query_concepts(&conn, tenant_id, concept_filter, depth_filter, top)?;

    if results.is_empty() {
        if json {
            output::print_json(&ConceptsJson {
                tenant_id: tenant_id.to_string(),
                total: 0,
                concepts: Vec::new(),
            });
        } else {
            output::info("No concept nodes found. Concepts populate after tagging pipeline runs.");
        }
        return Ok(());
    }

    if json {
        let entries: Vec<ConceptJsonEntry> = results
            .iter()
            .map(|(concept, implements, covers)| ConceptJsonEntry {
                concept: concept.clone(),
                implements_count: *implements,
                covers_count: *covers,
            })
            .collect();
        output::print_json(&ConceptsJson {
            tenant_id: tenant_id.to_string(),
            total: entries.len(),
            concepts: entries,
        });
    } else {
        canvas::print_title(&format!("Concept Nodes (tenant: {})", tenant_id));
        canvas::print_blank();

        let rows: Vec<ConceptRow> = results
            .iter()
            .map(|(concept, implements, covers)| ConceptRow {
                concept: concept.clone(),
                implements: implements.to_string(),
                covers: covers.to_string(),
            })
            .collect();

        output::print_table_auto(&rows);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concept_row_implements_column_hints() {
        assert_eq!(ConceptRow::content_columns(), &[0]);
        assert_eq!(ConceptRow::numeric_columns(), &[1, 2]);
    }

    #[test]
    fn concepts_json_serializes() {
        let json_out = ConceptsJson {
            tenant_id: "test-tenant".to_string(),
            total: 1,
            concepts: vec![ConceptJsonEntry {
                concept: "async-runtime".to_string(),
                implements_count: 15,
                covers_count: 3,
            }],
        };
        let serialized = serde_json::to_string(&json_out).unwrap();
        assert!(serialized.contains("async-runtime"));
        assert!(serialized.contains("15"));
    }
}
