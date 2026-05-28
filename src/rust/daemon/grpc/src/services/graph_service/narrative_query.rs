//! SQL query logic for the NarrativeQuery RPC.
//!
//! Resolves a symbol name or concept name to graph node IDs, then
//! traverses narrative/concept edges via a recursive CTE to collect
//! narrative nodes (document_section, code_comment, docstring,
//! library_section, concept_node).

use sqlx::{Row, SqlitePool};
use tonic::Status;
use tracing::error;

use crate::proto::NarrativeNode;

/// Narrative node types that the query returns.
const NARRATIVE_TYPES: &[&str] = &[
    "document_section",
    "code_comment",
    "docstring",
    "library_section",
    "concept_node",
];

/// Build the SQL IN-clause literal for narrative node types.
fn narrative_type_filter() -> String {
    NARRATIVE_TYPES
        .iter()
        .map(|t| format!("'{t}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Execute the narrative query against the graph database.
///
/// 1. Resolves seed node IDs from `query_name` (symbol or concept).
/// 2. Traverses outward through edges up to `max_depth` hops.
/// 3. Filters results to narrative node types only.
/// 4. Returns at most `max_results` nodes.
pub(crate) async fn execute_narrative_query(
    pool: &SqlitePool,
    tenant_id: &str,
    query_name: &str,
    is_concept: bool,
    edge_types: Option<&[String]>,
    max_depth: u32,
    max_results: u32,
) -> Result<Vec<NarrativeNode>, Status> {
    // Step 1: Resolve seed node IDs
    let seed_ids = find_seed_nodes(pool, tenant_id, query_name, is_concept).await?;
    if seed_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: Build edge type filter clause
    let edge_type_clause = match edge_types {
        Some(types) if !types.is_empty() => {
            let vals: Vec<String> = types.iter().map(|t| format!("'{t}'")).collect();
            format!("AND e.edge_type IN ({})", vals.join(", "))
        }
        _ => String::new(),
    };

    // Step 3: Build seed placeholders
    let tenant_slot = seed_ids.len() + 1;
    let depth_slot = seed_ids.len() + 2;
    let limit_slot = seed_ids.len() + 3;

    let narr_filter = narrative_type_filter();

    // Step 4: Build a seed-set table and a bidirectional recursive CTE.
    //         Narrative edges point narrative_node → code_symbol and
    //         file → concept_node, so a symbol/concept seed must follow
    //         incoming edges to find narrators. We exclude seeds from
    //         results and from recursive expansion to prevent cycles.
    let seed_values: Vec<String> = (0..seed_ids.len())
        .map(|i| format!("SELECT ?{} AS id", i + 1))
        .collect();
    let seed_cte = seed_values.join(" UNION ALL ");

    // visited tracks "|id|id|..." to prevent cycles via INSTR check.
    let sql = format!(
        "WITH seed_set AS ({seed_cte}),
        narrative_traverse AS (
            SELECT e.target_node_id AS node_id,
                   e.edge_type,
                   e.metadata_json,
                   1 AS depth,
                   e.source_node_id || ' -> ' || e.target_node_id AS path,
                   '|' || e.source_node_id || '|' || e.target_node_id || '|' AS visited
            FROM graph_edges e
            WHERE e.source_node_id IN (SELECT id FROM seed_set)
              AND e.target_node_id NOT IN (SELECT id FROM seed_set)
              AND e.tenant_id = ?{tenant_slot}
              {edge_type_clause}
            UNION ALL
            SELECT e.source_node_id AS node_id,
                   e.edge_type,
                   e.metadata_json,
                   1 AS depth,
                   e.target_node_id || ' <- ' || e.source_node_id AS path,
                   '|' || e.target_node_id || '|' || e.source_node_id || '|' AS visited
            FROM graph_edges e
            WHERE e.target_node_id IN (SELECT id FROM seed_set)
              AND e.source_node_id NOT IN (SELECT id FROM seed_set)
              AND e.tenant_id = ?{tenant_slot}
              {edge_type_clause}
            UNION ALL
            SELECT e.target_node_id,
                   e.edge_type,
                   e.metadata_json,
                   nt.depth + 1,
                   nt.path || ' -> ' || e.target_node_id,
                   nt.visited || e.target_node_id || '|'
            FROM graph_edges e
            INNER JOIN narrative_traverse nt ON e.source_node_id = nt.node_id
            WHERE nt.depth < ?{depth_slot}
              AND INSTR(nt.visited, '|' || e.target_node_id || '|') = 0
              AND e.tenant_id = ?{tenant_slot}
              {edge_type_clause}
            UNION ALL
            SELECT e.source_node_id,
                   e.edge_type,
                   e.metadata_json,
                   nt.depth + 1,
                   nt.path || ' <- ' || e.source_node_id,
                   nt.visited || e.source_node_id || '|'
            FROM graph_edges e
            INNER JOIN narrative_traverse nt ON e.target_node_id = nt.node_id
            WHERE nt.depth < ?{depth_slot}
              AND INSTR(nt.visited, '|' || e.source_node_id || '|') = 0
              AND e.tenant_id = ?{tenant_slot}
              {edge_type_clause}
        )
        SELECT r.node_id, r.edge_type, r.depth, r.path,
               r.metadata_json,
               r.symbol_name, r.symbol_type, r.file_path
        FROM (
            SELECT nt.node_id, nt.edge_type, nt.depth, nt.path,
                   nt.metadata_json,
                   n.symbol_name, n.symbol_type, n.file_path,
                   ROW_NUMBER() OVER (
                       PARTITION BY nt.node_id
                       ORDER BY nt.depth, nt.edge_type
                   ) AS rn
            FROM narrative_traverse nt
            JOIN graph_nodes n ON nt.node_id = n.node_id
            WHERE n.symbol_type IN ({narr_filter})
        ) r
        WHERE r.rn = 1
        ORDER BY r.depth, r.symbol_name
        LIMIT ?{limit_slot}"
    );

    let mut qb = sqlx::query(&sql);
    for id in &seed_ids {
        qb = qb.bind(id);
    }
    qb = qb.bind(tenant_id);
    qb = qb.bind(max_depth as i64);
    qb = qb.bind(max_results as i64);

    let rows = qb.fetch_all(pool).await.map_err(|e| {
        error!("NarrativeQuery traversal failed: {}", e);
        Status::internal(format!("Narrative query failed: {}", e))
    })?;

    let nodes = rows
        .iter()
        .map(|row| NarrativeNode {
            node_id: row.get("node_id"),
            symbol_name: row.get("symbol_name"),
            symbol_type: row.get("symbol_type"),
            file_path: row.get("file_path"),
            edge_type: row.get("edge_type"),
            depth: row.get::<i64, _>("depth") as i32,
            path: row.get("path"),
            metadata_json: row.get("metadata_json"),
        })
        .collect();

    Ok(nodes)
}

/// Find seed node IDs for the query target.
///
/// For symbol queries: looks up nodes by `symbol_name` in the tenant.
/// For concept queries: looks up ConceptNode by name (tenant-agnostic,
/// since concept nodes use a global ID scheme).
async fn find_seed_nodes(
    pool: &SqlitePool,
    tenant_id: &str,
    name: &str,
    is_concept: bool,
) -> Result<Vec<String>, Status> {
    let rows = if is_concept {
        // ConceptNode IDs are computed globally (no tenant prefix),
        // but we still check tenant_id for edges originating from them.
        // Look up by symbol_name with type = concept_node.
        sqlx::query(
            "SELECT node_id FROM graph_nodes
             WHERE symbol_name = ?1 AND symbol_type = 'concept_node'
             LIMIT 100",
        )
        .bind(name)
        .fetch_all(pool)
        .await
    } else {
        sqlx::query(
            "SELECT node_id FROM graph_nodes
             WHERE tenant_id = ?1 AND symbol_name = ?2
               AND symbol_type NOT IN ('document_section', 'code_comment',
                   'docstring', 'library_section', 'concept_node')
             LIMIT 100",
        )
        .bind(tenant_id)
        .bind(name)
        .fetch_all(pool)
        .await
    };

    let rows = rows.map_err(|e| {
        error!("NarrativeQuery seed lookup failed: {}", e);
        Status::internal(format!("Seed node lookup failed: {}", e))
    })?;

    Ok(rows.iter().map(|r| r.get("node_id")).collect())
}
